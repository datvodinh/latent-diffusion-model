from argparse import Namespace
from dataclasses import dataclass


def get_config(stage: str, args: Namespace = None):
    if stage == "stage1":
        return LDMConfigStage1
    else:
        return LDMConfigStage2


@dataclass
class LDMConfigStage1:
    stage: float = "stage1"

    # LR
    lr: float = 1e-4
    weight_decay: float = 0.001
    betas: tuple[float] = (0.5, 0.9)
    pct_start: float = 0.3

    # MODEL
    in_channels: int = 3
    latent_dim: int = 8
    num_embeds: int = 1024


@dataclass
class LDMConfigStage2:
    stage: float = "stage2"

    # LR
    lr: float = 1e-4
    weight_decay: float = 0.001
    betas: tuple[float] = (0.9, 0.999)
    pct_start: float = 0.3

    # MODEL
    in_channels: int = 3
    latent_dim: int = 8
    num_embeds: int = 1024

    # DIFFUSION
    max_timesteps: int = 1000
    beta_1: float = 0.00095
    beta_2: float = 0.0195
    mode: str = "ddim"
    sample_per_epochs: int = 5
    time_dim: int = 256
    dim: int = 32
    context_dim: int | None = None
