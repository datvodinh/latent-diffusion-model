import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange, repeat


class VectorQuantizer(nn.Module):
    def __init__(
            self,
            num_embeds: int,
            embed_dim: int,
            beta: float = 0.25

    ):
        super().__init__()
        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeds, embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeds, 1/num_embeds)

    def quantize(self, x_latent: torch.Tensor):
        B, C, H, W = x_latent.shape
        x_latent_flat = rearrange(x_latent, "b c h w -> b (h w) c").contiguous()
        expand_embed = repeat(self.embedding.weight, "e d -> b e d", b=x_latent.shape[0])
        dist = torch.cdist(x_latent_flat, expand_embed, p=2.)**2  # [B,HW,E]
        encoding_inds = torch.argmin(dist, dim=-1)  # [B,HW]
        x_q = F.embedding(encoding_inds, self.embedding.weight)
        x_q = rearrange(
            x_q, "b (h w) d -> b d h w", b=B, h=H, w=W  # [B,D,H,W]
        ).contiguous()
        return x_q

    def forward(self, x_latent: torch.Tensor):
        x_q = self.quantize(x_latent)
        vq_loss = (x_q.detach() - x_latent)**2 * self.beta + (x_q - x_latent.detach())**2
        x_q = x_latent + (x_q - x_latent).detach()
        return x_q, vq_loss.mean()  # [B,D,H,W]


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
        latent_dim: int = 8
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            VAEResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 8
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 64),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 8,
                 num_embeds: int = 256,
                 ):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.quant_conv_in = nn.Conv2d(latent_dim, latent_dim, 1)
        self.vec_quant = VectorQuantizer(num_embeds=num_embeds, embed_dim=latent_dim)
        self.quant_conv_out = nn.Conv2d(latent_dim, latent_dim, 1)
        self.decoder = VAEDecoder(in_channels, latent_dim)

    def encode(self, x: torch.Tensor):
        x_latent = self.encoder(x)
        x_latent = self.quant_conv_in(x_latent)
        return x_latent

    def encode_quantize(self, x: torch.Tensor):
        x_latent = self.encode(x)
        x_quant, _ = self.vec_quant(x_latent)
        return x_quant

    def quantize(self, x: torch.Tensor):
        return self.vec_quant(x)[0]

    def decode(self, x_quant: torch.Tensor):
        x_quant = self.quant_conv_out(x_quant)
        return self.decoder(x_quant)

    def quantize_decode(self, x_latent: torch.Tensor):
        x_quant, vq_loss = self.vec_quant(x_latent)
        return self.decode(x_quant), vq_loss

    def forward(self, x: torch.Tensor):
        x_latent = self.encode(x)
        x_quant, vq_loss = self.vec_quant(x_latent)
        return self.decode(x_quant), vq_loss


if __name__ == "__main__":
    model = VariationalAutoEncoder()
    x = torch.randn(2, 3, 256, 256)
    print(model(x)[0].shape)
