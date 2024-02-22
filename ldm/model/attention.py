import torch
import torch.nn as nn
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(
            self,
            channels: int,
            num_heads: int = 4
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.qkv_project = nn.Linear(channels, 3 * channels, bias=False)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.out_project = nn.Linear(channels, channels, bias=False)

    def _get_attention_score(self, x: torch.Tensor):
        q, k, v = self.qkv_project(x).chunk(3, dim=-1)
        attention_value, _ = self.mha(q, k, v)
        output = self.out_project(attention_value)
        return output

    def forward(
        self,
        x: torch.Tensor,
        reshape_out: bool = True
    ):
        if len(x.shape) == 4:  # image 4 dim: [B, C, H, W]
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
            output = self._get_attention_score(x)
            if reshape_out:
                return rearrange(output, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
            else:
                return output
        return self._get_attention_score(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            channels: int,
            cross_dim: int,
            num_heads: int = 4
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.q_project = nn.Linear(channels, channels, bias=False)
        self.kv_project = nn.Linear(cross_dim, 2 * channels, bias=False)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.out_project = nn.Linear(channels, channels, bias=False)

    def _get_attention_score(self, x: torch.Tensor, y: torch.Tensor):
        q = self.q_project(x)
        k, v = self.kv_project(y).chunk(2, dim=-1)
        attention_value, _ = self.mha(q, k, v)
        output = self.out_project(attention_value)
        return output

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if len(x.shape) == 4:  # image 4 dim: [B, C, H, W]
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        return self._get_attention_score(x, y)


if __name__ == "__main__":
    x = torch.rand(2, 32, 32, 32)
    model = SelfAttention(channels=32)
    print(model(x).shape)
