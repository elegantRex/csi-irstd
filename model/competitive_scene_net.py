import math
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                         nn.GroupNorm(math.gcd(8, cout), cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, groups=cout),
                         nn.SiLU(), nn.Conv2d(cout, cout, 1))


def render_points(offset, scale, amplitude, probability, output_size, radius=8):
    """Differentiable finite-support Gaussian splatting on the stride-4 grid.

    amplitude is integrated image contrast, not confidence. Kernels sum to one
    before cropping at the image boundary. All queries participate, no top-k.
    """
    b, _, h, w = offset.shape
    axis = torch.arange(-radius, radius + 1, device=offset.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    dx = xx.reshape(1, -1, 1) - offset[:, 0].float().reshape(b, 1, -1)
    dy = yy.reshape(1, -1, 1) - offset[:, 1].float().reshape(b, 1, -1)
    sx, sy = scale[:, 0].float().reshape(b, 1, -1), scale[:, 1].float().reshape(b, 1, -1)
    kernel = torch.exp(-0.5 * ((dx / sx).square() + (dy / sy).square()))
    kernel = kernel / kernel.sum(1, keepdim=True).clamp_min(1e-8)
    patches = kernel * (amplitude.float() * probability.float()).reshape(b, 1, -1)
    return F.fold(patches, output_size, kernel_size=2 * radius + 1, padding=radius, stride=4)


class CompetitiveSceneNet(nn.Module):
    def __init__(self, channels=48, iterations=3, mode='competitive', detach_render_confidence=False, encoder='spire', disable_target_feedback=False,
                 disable_background_competition=False, disable_background_residual=False,
                 reset_background_state=False, detach_render_feedback=False):
        super().__init__()
        if mode not in ('competitive', 'independent', 'direct'):
            raise ValueError(mode)
        if iterations < 1:
            raise ValueError('iterations must be positive')
        if encoder not in ("compact", "spire"):
            raise ValueError(encoder)
        if disable_target_feedback and mode != 'competitive':
            raise ValueError('disable_target_feedback requires competitive mode')
        self.disable_target_feedback = disable_target_feedback
        interventions = dict(disable_background_competition=disable_background_competition,
                             disable_background_residual=disable_background_residual,
                             reset_background_state=reset_background_state,
                             detach_render_feedback=detach_render_feedback)
        if any(interventions.values()) and mode != 'competitive':
            raise ValueError('Path ablations require competitive mode')
        if detach_render_feedback and (disable_target_feedback or detach_render_confidence):
            raise ValueError('Full feedback detachment requires an active, otherwise unmodified feedback path')
        for name, value in interventions.items():
            setattr(self, name, value)
        self.config = dict(channels=channels, iterations=iterations, mode=mode, detach_render_confidence=detach_render_confidence, encoder=encoder, disable_target_feedback=disable_target_feedback)
        self.config.update(interventions)
        self.detach_render_confidence = detach_render_confidence
        self.mode, self.iterations = mode, iterations
        # Lossless rearrangement of input pixels followed by a learned projection.
        if encoder == "compact":
            self.encoder = nn.Sequential(nn.PixelUnshuffle(4), block(16, channels), block(channels, channels))
        else:
            from model.scene_spire_encoder import SceneSPIREEncoder
            self.encoder = SceneSPIREEncoder(channels)
        self.target_init = nn.Conv2d(channels, channels, 1)
        self.target_update = block(2 * channels + 16, channels)
        self.target_head = nn.Conv2d(channels, 6, 1)
        nn.init.normal_(self.target_head.weight, std=0.001)
        with torch.no_grad():
            self.target_head.bias.copy_(torch.tensor([-4.6, 0., 0., -1., -0.5, -0.5]))
        if mode != 'direct':
            self.background_init = nn.Conv2d(channels, channels, 1)
            self.background_update = block(2 * channels + 1, channels)
            self.background_head = nn.Conv2d(channels, 1, 1)
        if mode == 'competitive':
            self.background_competitor = nn.Conv2d(channels, 1, 1)
            nn.init.zeros_(self.background_competitor.weight)
            nn.init.zeros_(self.background_competitor.bias)
        self.step_size = 0.25

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or any(s % 8 for s in image.shape[-2:]):
            raise ValueError('Expected Bx1xHxW grayscale [0,1], H/W divisible by 8')
        features = self.encoder(image)
        state = self.target_init(features)
        h, w = features.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing='ij')
        grid = torch.stack((x, y), 0).float()[None] * 4
        target_image = torch.zeros_like(image)
        stages = []
        if self.mode != 'direct':
            context = F.avg_pool2d(features, 2)
            background_state = self.background_init(context)
            initial_background_state = background_state
        for _ in range(self.iterations):
            if self.mode == 'direct':
                background = torch.zeros_like(image)
                competitor = 0.
                observation = F.pixel_unshuffle(image, 4)
            else:
                feedback_target = target_image.detach() if self.detach_render_feedback else target_image
                background_input = image - feedback_target if self.mode == 'competitive' and not self.disable_target_feedback else image
                if self.reset_background_state:
                    background_state = initial_background_state
                background_state = background_state + self.step_size * self.background_update(
                    torch.cat((background_state, context, F.avg_pool2d(background_input, 8)), 1))
                coarse_background = self.background_head(background_state).sigmoid()
                background = F.interpolate(coarse_background, image.shape[-2:], mode='bilinear', align_corners=False)
                competitor = (F.interpolate(self.background_competitor(background_state), (h, w), mode='bilinear', align_corners=False)
                              if self.mode == 'competitive' and not self.disable_background_competition else 0.)
                observation = F.pixel_unshuffle(image - background if self.mode == 'competitive' and not self.disable_background_residual else image, 4)
            state = state + self.step_size * self.target_update(torch.cat((state, features, observation), 1))
            raw = self.target_head(state)
            # Two-way background/target competition: softmax target probability
            # equals sigmoid(target_logit - background_logit).
            logits = raw[:, :1] - competitor
            offset = 4 * raw[:, 1:3].tanh()
            coords = grid + offset
            amplitude = F.softplus(raw[:, 3:4])
            scale = 0.6 + 1.9 * raw[:, 4:6].sigmoid()
            render_probability = logits.sigmoid()
            if self.detach_render_confidence:
                # Block only the score-to-render path. Shared features and geometry
                # still receive reconstruction gradients; this is not full isolation.
                render_probability = render_probability.detach()
            target_image = (render_points(offset, scale, amplitude, render_probability, image.shape[-2:])
                            if self.mode != 'direct' else torch.zeros_like(image))
            stages.append(dict(logits=logits, coords=coords, background=background,
                               target_image=target_image, reconstruction=background + target_image,
                               scale=scale, amplitude=amplitude))
        return stages
