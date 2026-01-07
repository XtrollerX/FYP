import torch
import torch.nn as nn

# DCGAN-style Generator + Discriminator for square images

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, feature_g=64, img_size=224):
        super().__init__()
        self.z_dim = z_dim
        self.img_size = img_size

        # We build a generator that outputs img_size x img_size
        # Start from 7x7 or 14x14 depending on img_size.
        # For 224: 7 -> 14 -> 28 -> 56 -> 112 -> 224 (x2 each time)
        self.net = nn.Sequential(
            # z -> 7x7
            nn.ConvTranspose2d(z_dim, feature_g * 16, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_g * 16),
            nn.ReLU(True),

            # 7 -> 14
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            # 14 -> 28
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            # 28 -> 56
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            # 56 -> 112
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            # 112 -> 224
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        # Input: 224x224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, feature_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 7x7 -> 1x1
            nn.Conv2d(feature_d * 16, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x):
        # output logits shape: (B, 1, 1, 1)
        return self.net(x).view(-1)
