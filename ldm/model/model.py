import ldm
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
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
            self.vae = ldm.VariationalAutoEncoder(
                in_channels=config.in_channels,
                latent_dim=config.latent_dim,
                num_embeds=config.num_embeds
            )
            self.criterion = nn.MSELoss()

        elif config.stage == "stage2":
            self.vae = ldm.VariationalAutoEncoder(
                in_channels=config.in_channels,
                latent_dim=config.latent_dim,
                num_embeds=config.num_embeds
            )
            self.model = ldm.UNet(
                in_channels=config.latent_dim,
                out_channels=config.latent_dim,
                time_dim=config.time_dim,
                context_dim=config.context_dim
            )
            if config.mode == "ddpm":
                self.scheduler = ldm.DDPMScheduler(
                    config.max_timesteps, config.beta_1, config.beta_2
                )
            elif config.mode == "ddim":
                self.scheduler = ldm.DDIMScheduler(
                    config.max_timesteps, config.beta_1, config.beta_2
                )
            self.criterion = nn.MSELoss()
            self.sampling_kwargs = {
                'model': self.model,
                'in_channels': self.config.latent_dim,
                'dim': self.config.dim,
            }

            self.epoch_count = 0

    def load_vae_ckpt(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)['state_dict']
        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            if "vae." in k:
                new_state_dict[k.replace("vae.", "")] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        self.vae.load_state_dict(state_dict=new_state_dict)
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
            low=0, high=self.config.max_timesteps, size=(n//2+1,), device=x.device
        )
        t = torch.cat([t, self.config.max_timesteps - t - 1], dim=0)[:n]
        x_latent = self.vae.encode_quantize(x)
        x_noise, noise = self.scheduler.noising(x_latent, t)
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

    def _wandb_image(self, x: torch.Tensor, caption: str):
        return wandb.Image(
            make_grid(x, nrow=4, normalize=True).permute(1, 2, 0).cpu().numpy(),
            caption=caption
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,

        )
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.config.pct_start
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def sampling(
        self,
        labels=None,
        mode: int = "ddpm",
        demo: bool = True,
        n_samples: int = 16,
        timesteps: int = 1000,
    ):
        if mode == "ddpm":
            self.test_scheduler = ldm.DDPMScheduler(self.config.max_timesteps)
        elif mode == "ddim":
            self.test_scheduler = ldm.DDIMScheduler(self.config.max_timesteps)

        kwargs = {
            "n_samples": n_samples,
            "labels": labels,
            "timesteps": timesteps,
        } | self.sampling_kwargs
        if demo:
            return self.test_scheduler.sampling_demo(**kwargs)
        else:
            return self.test_scheduler.sampling(**kwargs)

    def on_train_epoch_end(self) -> None:
        if self.config.stage == "stage1":
            with torch.no_grad():
                wandblog = self.logger.experiment
                n = min(self.trainer.val_dataloaders.batch_size, 16)
                batch = next(iter(self.trainer.val_dataloaders))
                if isinstance(batch, (list, tuple)):
                    x_org = batch[0][:n].to(self.vae.device)
                else:
                    x_org = batch[:n].to(self.vae.device)
                x_res, _ = self.vae(x_org)
                org_array = [x_org[i] for i in range(x_org.shape[0])]
                res_array = [x_res[i] for i in range(x_res.shape[0])]

                wandblog.log(
                    {
                        "original": self._wandb_image(org_array, caption="Original Image!"),
                        "reconstructed": self._wandb_image(res_array, caption="Sampled Image!")
                    }
                )
        else:
            if self.config.sample_per_epochs > 0:
                if self.epoch_count % self.config.sample_per_epochs == 0:
                    with torch.no_grad():
                        wandblog = self.logger.experiment
                        n = min(self.trainer.val_dataloaders.batch_size, 16)
                        x_t = self.sampling(mode="ddim", n_samples=n, timesteps=100, demo=False)
                        img_array = [x_t[i] for i in range(x_t.shape[0])]

                        wandblog.log(
                            {
                                "sample": self._wandb_image(img_array, caption="Sampled Image!")
                            }
                        )
            self.epoch_count += 1
