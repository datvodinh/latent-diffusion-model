import pytorch_lightning as pl
import torch
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
    def __init__(self, list_path: str, stage: str = "train"):
        self.list_path = list_path
        self.stage = stage
        if stage == "train":
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA),
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
        if self.stage == "train":
            crop_len = int(256 * np.random.uniform(0.7, 1))
            rand_num = np.random.rand()
            if rand_num < 1/3:
                cropper = A.RandomCrop(height=crop_len, width=crop_len)
                img = cropper(image=img)['image']
            elif rand_num > 1/3 and rand_num < 2/3:
                cropper = A.CenterCrop(height=crop_len, width=crop_len)
                img = cropper(image=img)['image']

        return self.transform(image=img)['image']


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        train_ratio: float = 0.95,
        val_ratio: float = 0.05
    ):
        super().__init__()
        self.in_channels = 3
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = min(train_ratio, 0.95)
        self.val_ratio = val_ratio
        self.seed = seed

        self.loader = partial(
            DataLoader,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def setup(self, stage: str):
        if stage == "fit":
            list_dir = os.listdir(self.data_dir)
            list_path = []
            for d in list_dir:
                list_path.append(os.path.join(self.data_dir, d))
            random.shuffle(list_path)
            len_train = int(len(list_path) * self.train_ratio)
            len_val = int(len(list_path) * self.val_ratio)
            train_list = list_path[:len_train]
            val_list = list_path[len_train:min(len_train+len_val, len(list_path))]
            self.CelebA_train = CelebADataset(list_path=train_list, stage="train")
            self.CelebA_val = CelebADataset(list_path=val_list, stage="val")
        else:
            pass

    def train_dataloader(self):
        return self.loader(dataset=self.CelebA_train)

    def val_dataloader(self):
        return self.loader(dataset=self.CelebA_val)
