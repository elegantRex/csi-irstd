"""Point supervision, Hungarian assignment, and observed-image reconstruction."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def assign_points(coords, logits, truth):
    if len(truth) == 0:
        empty = torch.empty(0, dtype=torch.long, device=coords.device)
        return empty, empty
    if len(truth) > len(coords):
        raise ValueError('More GT points than available queries')
    # Pixel distance / grid stride keeps localization meaningful for tiny objects.
    with torch.no_grad():
        cost = torch.cdist(coords.float(), truth.float()) / 4 - logits.float().sigmoid()[:, None]
        row, col = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(row, device=coords.device), torch.as_tensor(col, device=coords.device)


def scene_loss(stages, image, targets, reconstruction_weight=0.1, background_weight=0.001, mode='competitive', location_weight=2., focal_alpha=.25, focal_gamma=2.):
    values = dict(existence=image.new_zeros(()), location=image.new_zeros(()),
                  reconstruction=image.new_zeros(()), background_tv=image.new_zeros(()))
    for stage in stages:
        for i, target in enumerate(targets):
            logits = stage['logits'][i].float().flatten()
            coords = stage['coords'][i].float().flatten(1).T
            truth = target['points'].to(image.device)
            valid = target['valid'].to(image.device)
            # Background outside the sensor footprint is not a negative example.
            query_valid = valid[:, ::4, ::4].flatten().bool()
            indices = torch.where(query_valid)[0]
            row, col = assign_points(coords[indices], logits[indices], truth)
            row = indices[row]
            labels = torch.zeros_like(logits)
            labels[row] = 1
            probability = logits.sigmoid()
            pt = probability * labels + (1 - probability) * (1 - labels)
            alpha = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
            focal = F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * (1 - pt).pow(focal_gamma) * alpha
            values['existence'] = values['existence'] + focal[query_valid].sum() / max(len(truth), 1)
            if len(row):
                values['location'] = values['location'] + F.smooth_l1_loss(coords[row] / 4, truth[col] / 4)
            if mode != 'direct':
                error = (stage['reconstruction'][i].float() - image[i]).abs()
                full = (error * valid).sum() / valid.sum().clamp_min(1)
                local = []
                height, width = image.shape[-2:]
                for x, y in truth.detach().round().long().tolist():
                    y0, y1, x0, x1 = max(0, y-4), min(height, y+5), max(0, x-4), min(width, x+5)
                    mask = valid[:, y0:y1, x0:x1]
                    local.append((error[:, y0:y1, x0:x1] * mask).sum() / mask.sum().clamp_min(1))
                values['reconstruction'] = values['reconstruction'] + full + (torch.stack(local).mean() if local else 0)
                bg = stage['background'][i].float()
                my, mx = valid[:, 1:] * valid[:, :-1], valid[:, :, 1:] * valid[:, :, :-1]
                tv = ((bg[:, 1:] - bg[:, :-1]).abs() * my).sum() / my.sum().clamp_min(1)
                tv = tv + ((bg[:, :, 1:] - bg[:, :, :-1]).abs() * mx).sum() / mx.sum().clamp_min(1)
                values['background_tv'] = values['background_tv'] + tv
    values = {k: v / (len(stages) * len(targets)) for k, v in values.items()}
    total = values['existence'] + location_weight * values['location'] + reconstruction_weight * values['reconstruction'] + background_weight * values['background_tv']
    return total, values
