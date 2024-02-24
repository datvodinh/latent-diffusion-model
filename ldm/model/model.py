import ldm
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import OneCycleLR


class LatentDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        config: ldm.LDMConfigStage1 | ldm.LDMConfigStage2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config.stage == "stage1":
            self.vae = ldm.VariationalAutoEncoder(in_channels=config.in_channels)
            self.criterion = nn.MSELoss()

        elif config.stage == "stage2":
            self.vae = ldm.VariationalAutoEncoder(in_channels=config.in_channels)
            self.model = ldm.UNet(
                in_channels=4,
                out_channels=4,
                time_dim=config.time_dim,
                context_dim=config.context_dim
            )

        self.epoch_count = 0

    def load_vae_ckpt(self, ckpt_path: str):
        self.vae = ldm.VariationalAutoEncoder.load_from_checkpoint(
            checkpoint_path=ckpt_path, map_location=self.device
        )
        self.vae.requires_grad_(False)

    def _step_stage_1(self, batch):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        output, vq_loss = self.vae(x)
        loss = self.criterion(x, output) + vq_loss
        return loss

    def _step_stage_2(self, batch):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        n = x.shape[0]
        t = torch.randint(
            low=0, high=self.max_timesteps, size=(n//2+1,), device=x.device
        )
        t = torch.cat([t, self.max_timesteps - t - 1], dim=0)[:n]
        x_latent = self.vae.encode(x)
        x_noise, noise = self.noising(x_latent, t)
        noise_pred = self.model(x_noise, t)
        loss = self.criterion(noise, noise_pred)
        return loss

    def training_step(self, batch, idx):
        if self.config.stage == "stage1":
            loss = self._step_stage_1(batch)
            self.log_dict({"vae_loss_train": loss}, sync_dist=True, on_epoch=True)
        else:
            loss = self._step_stage_2(batch)
            self.log_dict({"unet_loss_train": loss}, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, idx):
        if self.config.stage == "stage1":
            loss = self._step_stage_1(batch)
            self.log_dict({"vae_loss_val": loss}, sync_dist=True, on_epoch=True)
        else:
            loss = self._step_stage_2(batch)
            self.log_dict({"unet_loss_val": loss}, sync_dist=True, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        if self.config.stage == "stage1":
            wandblog = self.logger.experiment
            x_org = next(iter(self.trainer.val_dataloaders))[0][:4].to(self.device)
            x_res, _ = self.vae(x_org)
            org_array = [x_org[i] for i in range(x_org.shape[0])]
            res_array = [x_res[i] for i in range(x_res.shape[0])]

            wandblog.log(
                {
                    "original": wandb.Image(
                        make_grid(org_array, nrow=4, normalize=True).permute(1, 2, 0).cpu().numpy(),
                        caption="Original Image!"
                    ),
                    "reconstructed": wandb.Image(
                        make_grid(res_array, nrow=4, normalize=True).permute(1, 2, 0).cpu().numpy(),
                        caption="Sampled Image!"
                    )
                }
            )

        self.epoch_count += 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,

        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.config.pct_start
        )
        return [optimizer], [scheduler]
