import torch
import torch.nn as nn
import pytorch_lightning as pl
import ldm


class VAEAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.GroupNorm(32, channels),
            ldm.SelfAttention(channels=channels, num_heads=num_heads)
        )

    def forward(self, x: torch.Tensor):
        return x + self.attention(x)


class VAEResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res = nn.Identity() if (
            in_channels == out_channels
        ) else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        return self.layer(x) + self.res(x)


class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: float = 0.18215
    ):
        super().__init__()
        self.scale = scale_factor
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            VAEResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            VAEAttentionBlock(channels=256),
            VAEResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(channels=512),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.encoder(x)
        mean, log_var = out.chunk(2, dim=1)
        log_var = log_var.clamp(-20, 20)
        std = log_var.exp().sqrt()
        noise = torch.rand_like(std, device=std.device)
        x_enc = mean + std * noise
        return x_enc * self.scale


class VAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        scale_factor: float = 0.18215
    ):
        super().__init__()
        self.scale = scale_factor
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEAttentionBlock(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x /= self.scale
        return self.decoder(x)


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = VAEEncoder(in_channels)
        self.decoder = VAEDecoder(in_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(
        self,
        x: torch.Tensor
    ):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    encoder = VAEEncoder()
    decoder = VAEDecoder()
    x = torch.randn(2, 3, 256, 256)
    noise = torch.randn(2, 4, 32, 32)
    print(decoder(encoder(x, noise)).shape)
