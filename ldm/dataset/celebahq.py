import pytorch_lightning as pl
import ldm
import cv2
import os
import random
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from functools import partial


class CelebADataset(Dataset):
    def __init__(
        self,
        config: ldm.CelebAHQConfigStage1 | ldm.CelebAHQConfigStage2,
        list_path: str,
        stage: str = "train"
    ):
        self.config = config
        self.list_path = list_path
        self.stage = stage
        self.img_size = config.img_size
        self.resize = A.Resize(config.img_size, config.img_size, interpolation=cv2.INTER_AREA)
        if stage == "train":
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=config.img_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, index):
        img = Image.open(self.list_path[index])
        img = np.asarray(img)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = self.resize(image=img)['image']
        if self.stage == "train":
            crop_len = int(self.img_size * np.random.uniform(self.config.min_crop_len, 1))
            if self.config.stage == "stage1":
                if np.random.rand() < 0.5:
                    cropper = A.RandomCrop(height=crop_len, width=crop_len, p=0.5)
                else:
                    cropper = A.CenterCrop(height=crop_len, width=crop_len, p=0.5)
            else:
                cropper = A.CenterCrop(height=crop_len, width=crop_len, p=0.5)

            img = cropper(image=img)['image']

        return self.transform(image=img)['image']


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ldm.CelebAHQConfigStage1 | ldm.CelebAHQConfigStage2
    ):
        super().__init__()
        self.in_channels = config.in_channels
        self.config = config
        self.loader = partial(
            DataLoader,
            batch_size=config.batch_size,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers
        )

    def setup(self, stage: str):
        if stage == "fit":
            list_dir = os.listdir(self.config.data_dir)
            list_path = []
            for d in list_dir:
                if "DS_Store" not in d:
                    list_path.append(os.path.join(self.config.data_dir, d))
            random.shuffle(list_path)
            len_train = int(len(list_path) * self.config.train_ratio)
            len_val = int(len(list_path) * self.config.val_ratio)
            train_list = list_path[:len_train]
            val_list = list_path[len_train:min(len_train+len_val, len(list_path))]
            self.CelebA_train = CelebADataset(
                config=self.config, list_path=train_list, stage="train"
            )
            self.CelebA_val = CelebADataset(
                config=self.config, list_path=val_list, stage="val"
            )
        else:
            pass

    def train_dataloader(self):
        return self.loader(dataset=self.CelebA_train)

    def val_dataloader(self):
        return self.loader(dataset=self.CelebA_val)
