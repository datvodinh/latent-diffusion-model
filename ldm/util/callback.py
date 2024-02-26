import ldm
import os
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)


class ModelCallback:
    def __init__(
        self,
        root_path: str,
        stage: str,
    ):
        ckpt_path = os.path.join(os.path.join(root_path, "model/"))
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt_monitor = "vae_loss_val" if (stage == "stage1") else "unet_loss_val"
        filename = "vae" if (stage == "stage1") else "model"
        self.ckpt_callback = ModelCheckpoint(
            monitor=ckpt_monitor,
            dirpath=ckpt_path,
            filename=filename,
            save_top_k=1,
            mode="min",
            save_weights_only=True
        )

        self.lr_callback = LearningRateMonitor("step")

        self.ema_callback = ldm.EMACallback(decay=0.999)

    def get_callback(self):
        return [
            self.ckpt_callback, self.lr_callback, self.ema_callback
        ]
