from argparse import Namespace
from dataclasses import dataclass


def get_data_config(args: Namespace | None = None):
    config = CelebAHQConfigStage1() if args.stage == "stage1" else CelebAHQConfigStage2()
    config.update(args)
    print(config)
    return config


@dataclass
class BaseCelebAHQConfig:
    num_workers: int = 4
    seed: int = 42
    train_ratio: float = 0.95
    val_ratio: float = 0.05
    pin_memory: bool = True
    persistent_workers: bool = True

    def update(
        self,
        args: Namespace
    ):
        if isinstance(args, dict):
            args = Namespace(**args)
        for k, v in args.__dict__.items():
            if hasattr(self, k) and v is not None and k != "stage":
                setattr(self, k, v)
        print("CelebAHQ Config Updated!")


@dataclass
class CelebAHQConfigStage1(BaseCelebAHQConfig):
    # DATALOADER
    stage: float = "stage1"
    data_dir: str = "./"
    batch_size: int = 16
    max_batch_size: int = 32
    max_epochs: int = 50
    # DATASET
    in_channels: int = 3
    img_size: int = 256
    min_crop_len: float = 0.7


@dataclass
class CelebAHQConfigStage2(BaseCelebAHQConfig):
    # DATALOADER
    stage: float = "stage2"
    data_dir: str = "./"
    batch_size: int = 32
    max_batch_size: int = 64
    max_epochs: int = 150
    # DATASET
    in_channels: int = 3
    img_size: int = 256
    min_crop_len: float = 0.9
