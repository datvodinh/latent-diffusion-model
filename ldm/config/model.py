from argparse import Namespace
from dataclasses import dataclass


def get_model_config(args: Namespace = None):
    config = LDMConfigStage1() if args.stage == "stage1" else LDMConfigStage2()
    config.update(args)
    print(config)
    return config


@dataclass
class BaseLDMConfig:
    def update(
        self,
        args: Namespace
    ):
        if isinstance(args, dict):
            args = Namespace(**args)
        for k, v in args.__dict__.items():
            if hasattr(self, k) and v is not None and k != "stage":
                setattr(self, k, v)
        print("LDM Config Updated!")


@dataclass
class LDMConfigStage1(BaseLDMConfig):
    stage: float = "stage1"

    # LR
    lr: float = 1e-4
    weight_decay: float = 0.001
    betas: tuple[float] = (0.9, 0.98)
    pct_start: float = 0.3

    # MODEL
    in_channels: int = 3
    latent_channels: int = 8
    num_embeds: int = 1024


@dataclass
class LDMConfigStage2(BaseLDMConfig):
    stage: float = "stage2"
    # LR
    lr: float = 1e-4
    weight_decay: float = 0.001
    betas: tuple[float] = (0.9, 0.98)
    pct_start: float = 0.3
    # MODEL
    in_channels: int = 3
    latent_channels: int = 8
    num_embeds: int = 1024
    time_dim: int = 256
    latent_dim: int = 64
    context_dim: int | None = None
    # DIFFUSION
    max_timesteps: int = 1000
    beta_1: float = 0.00095
    beta_2: float = 0.0195
    mode: str = "ddim"
    sample_per_epochs: int = 5
