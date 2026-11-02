from torch import nn
from model.spire_net import SPIRENet


class SceneSPIREEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = SPIRENet(base_channel=32)
        del self.backbone.final_layer
        self.projection = nn.Conv2d(32, channels, 1)

    def forward(self, image):
        b = self.backbone
        x = image.repeat(1, 3, 1, 1)
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.relu(b.bn2(b.conv2(x)))
        x = b.layer1(x)
        x = [trans(x) for trans in b.transition1]
        x = b.stage4(b.stage3(b.stage2(x)))
        return self.projection(x[0])
