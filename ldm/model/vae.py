import torch
import torch.nn as nn
import pytorch_lightning as pl
import ldm
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

    def forward(self, x_latent: torch.Tensor):
        B, C, H, W = x_latent.shape
        x_latent_flat = rearrange(x_latent, "b c h w -> b (h w) c").contiguous()
        expand_embed = repeat(self.embedding.weight, "e d -> b e d", b=B)
        dist = torch.cdist(x_latent_flat, expand_embed, p=2.)**2  # [B,HW,E]
        encoding_inds = torch.argmin(dist, dim=-1)  # [B,HW]
        quantized_latents = F.embedding(encoding_inds, self.embedding.weight)
        quantized_latents = rearrange(
            quantized_latents, "b (h w) d -> b d h w", b=B, h=H, w=W  # [B,D,H,W]
        ).contiguous()
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), x_latent)
        embedding_loss = F.mse_loss(quantized_latents, x_latent.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        quantized_latents = x_latent + (quantized_latents - x_latent).detach()
        return quantized_latents, vq_loss  # [B,D,H,W]


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
        latent_dim: int = 8,
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
            VAEResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1),
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
        latent_dim: int = 8,
        scale_factor: float = 0.18215
    ):
        super().__init__()
        self.scale = scale_factor
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim // 2, 512, kernel_size=3, padding=1),
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
        x /= self.scale
        return self.decoder(x)


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 8,
                 num_embeds: int = 256,
                 ):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.vec_quant = VectorQuantizer(num_embeds=num_embeds, embed_dim=latent_dim//2)
        self.decoder = VAEDecoder(in_channels, latent_dim)

    def forward(
        self,
        x: torch.Tensor
    ):
        x_latent = self.encoder(x)
        x_quant, vq_loss = self.vec_quant(x_latent)
        return self.decoder(x_quant), vq_loss


if __name__ == "__main__":
    model = VariationalAutoEncoder()
    x = torch.randn(2, 3, 256, 256)
    print(model(x)[0].shape)
