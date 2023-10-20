import torch
import torch.nn as nn

# Code based on
# https://www.youtube.com/watch?v=4LktBHGCNfw&ab_channel=AladdinPersson

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
        torch.nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2))

  def forward(self, x):
    return self.conv(x)
  
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
          # https://www.youtube.com/watch?v=4LktBHGCNfw&ab_channel=AladdinPersson
          # why stride = 1 if feature == features[-1] else 2
          layers.append(Block(in_channels, feature, stride = 1 if feature == features[-1] else 2))
          in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
      x = self.initial(x)
      return torch.sigmoid(self.model(x))