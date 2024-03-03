import ldm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange


class GeGLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(channels, 4*channels*2)
        self.linear_2 = nn.Linear(4*channels, channels)

    def forward(self, x):
        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_2(x)
        return x


class UNetAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int | None = None,
        num_heads: int = 4,

    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        )
        self.norm_1 = nn.LayerNorm([channels])
        self.self_attn = ldm.SelfAttention(channels, num_heads)
        if context_dim is not None:
            self.norm_2 = nn.LayerNorm([channels])
            self.cross_attn = ldm.CrossAttention(channels, context_dim, num_heads)
        self.norm_3 = nn.LayerNorm([channels])
        self.geglu = GeGLU(channels)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.context_dim = context_dim

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None
    ):
        B, C, H, W = x.shape
        res_out = x
        x = self.conv_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.self_attn(self.norm_1(x))
        if self.context_dim is not None:
            x = x + self.cross_attn(self.norm_2(x), context)
        x = x + self.geglu(self.norm_3(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        return x + res_out


class UNetResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 256
    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.linear_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res = nn.Identity() if (
            in_channels == out_channels
        ) else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor
    ):
        x_in = self.conv_in(x)
        time = self.linear_time(time)
        x_in = x_in + time[:, :, None, None]
        x_out = self.conv_out(x_in)
        return self.res(x) + x_out


class DownSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 256
    ):
        super().__init__()
        self.res_1 = UNetResidualBlock(
            in_channels, in_channels, time_dim
        )
        self.res_2 = UNetResidualBlock(
            in_channels, out_channels, time_dim
        )
        self.down_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x, t):
        x = self.res_1(x, t)
        x = self.res_2(x, t)
        x = self.down_conv(x)
        return x


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 256
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.res_1 = UNetResidualBlock(
            in_channels, in_channels, time_dim
        )
        self.res_2 = UNetResidualBlock(
            in_channels, out_channels, time_dim
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res_1(x, t)
        x = self.res_2(x, t)
        return x


class UNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        time_dim: int = 256,
        context_dim: int | None = None
    ):
        super().__init__()
        self.time_dim = time_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        self.down1 = DownSample(in_channels=64, out_channels=128)
        self.attn1 = UNetAttentionBlock(context_dim=context_dim, channels=128, num_heads=4)
        self.down2 = DownSample(in_channels=128, out_channels=256)
        self.attn2 = UNetAttentionBlock(context_dim=context_dim, channels=256, num_heads=8)
        self.down3 = DownSample(in_channels=256, out_channels=256)
        self.attn3 = UNetAttentionBlock(context_dim=context_dim, channels=256, num_heads=8)

        self.mid1 = UNetResidualBlock(in_channels=256, out_channels=256)
        self.attn4 = UNetAttentionBlock(context_dim=context_dim, channels=256, num_heads=8)
        self.mid2 = UNetResidualBlock(in_channels=256, out_channels=512)

        self.up1 = UpSample(in_channels=512, out_channels=256)
        self.attn5 = UNetAttentionBlock(context_dim=context_dim, channels=256, num_heads=8)
        self.up2 = UpSample(in_channels=256, out_channels=128)
        self.attn6 = UNetAttentionBlock(context_dim=context_dim, channels=128, num_heads=8)
        self.up3 = UpSample(in_channels=128, out_channels=64)
        self.attn7 = UNetAttentionBlock(context_dim=context_dim, channels=64, num_heads=4)
        self.outc = nn.Sequential(
            nn.GroupNorm(32, 64, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float().to(t.device) / channels)
        ) * t.repeat(1, channels // 2)

        pos_enc = torch.zeros((t.shape[0], channels), device=t.device)
        pos_enc[:, 0::2] = torch.sin(inv_freq)
        pos_enc[:, 1::2] = torch.cos(inv_freq)
        return pos_enc

    def forward_unet(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None
    ):
        x1 = self.conv_in(x)
        x2 = self.down1(x1, t)
        x2 = self.attn1(x2, context)
        x3 = self.down2(x2, t)
        x3 = self.attn2(x3, context)
        x4 = self.down3(x3, t)
        x4 = self.attn3(x4, context)

        x4 = self.mid1(x4, t)
        x4 = self.attn4(x4, context)
        x4 = self.mid2(x4, t)

        x = self.up1(x4, x3, t)
        x = self.attn5(x, context)
        x = self.up2(x, x2, t)
        x = self.attn6(x, context)
        x = self.up3(x, x1, t)
        x = self.attn7(x, context)
        output = self.outc(x)
        return output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None
    ):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = self.time_embed(t)
        return self.forward_unet(x, t, context)


if __name__ == '__main__':
    net = UNet(in_channels=8, out_channels=8)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(2, 8, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    print(t)
    print(net(x, t).shape)
