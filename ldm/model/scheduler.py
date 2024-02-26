import torch


class DDPMScheduler:
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_1: int = 0.0001,
        beta_2: int = 0.02
    ) -> None:
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.max_timesteps = max_timesteps
        self._init_params()

    def _init_params(self, timesteps: int | None = None):
        self.beta = torch.linspace(self.beta_1, self.beta_2, timesteps or self.max_timesteps)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha = (1 - self.beta)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def noising(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        if t.device != x_0.device:
            t = t.to(x_0.device)
        noise = torch.randn_like(x_0, device=x_0.device)
        new_x = self.sqrt_alpha_hat.to(x_0.device)[t][:, None, None, None] * x_0
        new_noise = self.sqrt_one_minus_alpha_hat.to(x_0.device)[t][:, None, None, None] * noise
        return new_x + new_noise, noise

    @torch.no_grad()
    def sampling_t(
        self,
        x_t: torch.Tensor,
        model,
        labels: torch.Tensor,
        timesteps: int,
        t: int,
        n_samples: int = 16,
        cfg_scale: int = 3,
    ):
        time = torch.full((n_samples,), fill_value=t, device=model.device)
        pred_noise = model(x_t, time, labels)
        if cfg_scale > 0 and labels is not None:
            uncond_pred_noise = model(x_t, time, None)
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale)
        alpha = self.alpha.to(model.device)[time][:, None, None, None]
        sqrt_alpha = self.sqrt_alpha.to(model.device)[time][:, None, None, None]
        somah = self.sqrt_one_minus_alpha_hat.to(model.device)[time][:, None, None, None]
        sqrt_beta = self.sqrt_beta.to(model.device)[time][:, None, None, None]
        if t > 1:
            noise = torch.randn_like(x_t, device=model.device)
        else:
            noise = torch.zeros_like(x_t, device=model.device)

        x_t_new = 1 / sqrt_alpha * (x_t - (1-alpha) / somah * pred_noise) + sqrt_beta * noise
        return x_t_new.clamp(-1, 1)

    @torch.no_grad()
    def sampling(
        self,
        model,
        n_samples: int = 16,
        in_channels: int = 3,
        dim: int = 32,
        timesteps: int = 1000,
        cfg_scale: int = 3,
        labels=None,
        *args, **kwargs
    ):
        if labels is not None:
            n_samples = labels.shape[0]
        model.eval()
        x_t = torch.randn(
            n_samples, in_channels, dim, dim, device=model.device
        )
        step_ratios = self.max_timesteps // timesteps
        all_timesteps = torch.flip(torch.arange(0, timesteps) * step_ratios, dims=(0,))
        for t in all_timesteps:
            x_t = self.sampling_t(x_t=x_t, model=model, labels=labels, t=t, timesteps=timesteps,
                                  n_samples=n_samples, cfg_scale=cfg_scale)
        model.train()
        x_t = (x_t.clamp(-1, 1) + 1) / 2 * 255.  # range [0,255]
        return x_t.type(torch.uint8)

    @torch.no_grad()
    def sampling_demo(
        self,
        model,
        n_samples: int = 16,
        in_channels: int = 3,
        dim: int = 32,
        timesteps: int = 1000,
        cfg_scale: int = 3,
        labels=None,
        *args, **kwargs
    ):
        if labels is not None:
            n_samples = labels.shape[0]

        x_t = torch.randn(
            n_samples, in_channels, dim, dim, device=model.device
        )
        model.eval()
        step_ratios = self.max_timesteps // timesteps
        all_timesteps = torch.flip(torch.arange(0, timesteps) * step_ratios, dims=(0,))
        for t in all_timesteps:
            x_t = self.sampling_t(x_t=x_t, model=model, labels=labels, t=t, timesteps=timesteps,
                                  n_samples=n_samples, cfg_scale=cfg_scale)
            yield ((x_t.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)


class DDIMScheduler(DDPMScheduler):
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_1: int = 0.0001,
        beta_2: int = 0.02
    ) -> None:
        super().__init__(beta_1=beta_1, beta_2=beta_2, max_timesteps=max_timesteps)
        self._init_params()

    def _init_params(self, timesteps: int | None = None):
        self.beta = torch.linspace(self.beta_1, self.beta_2, timesteps or self.max_timesteps)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha = (1 - self.beta)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
        self.alpha_hat_prev = torch.cat([torch.tensor([1.]), self.alpha_hat], dim=0)[:-1]
        self.variance = (1 - self.alpha_hat_prev) / (1 - self.alpha_hat) * \
            (1 - self.alpha_hat / self.alpha_hat_prev)

    @torch.no_grad()
    def sampling_t(
        self,
        x_t: torch.Tensor, model, t: int,
        timesteps: int,
        labels: torch.Tensor | None = None,
        n_samples: int = 16,
        eta: float = 0.0,
        *args, **kwargs
    ):
        time = torch.full((n_samples,), fill_value=t, device=model.device)
        time_prev = time - self.max_timesteps // timesteps
        pred_noise = model(x_t, time, labels)

        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat.to(model.device)[time][:, None, None, None]
        sqrt_alpha_hat = self.sqrt_alpha_hat.to(model.device)[time][:, None, None, None]
        alpha_hat_prev = self.alpha_hat.to(model.device)[time_prev] if (
            time_prev[0] >= 0
        ) else torch.ones_like(time_prev, device=model.device)
        alpha_hat_prev = alpha_hat_prev[:, None, None, None]
        sqrt_alpha_hat_prev = torch.sqrt(alpha_hat_prev)
        posterior_std = torch.sqrt(self.variance)[time][:, None, None, None] * eta

        if t > 0:
            noise = torch.randn_like(x_t, device=model.device)
        else:
            noise = torch.zeros_like(x_t, device=model.device)

        x_0_pred = (x_t - sqrt_one_minus_alpha_hat * pred_noise) / sqrt_alpha_hat
        x_0_pred = x_0_pred.clamp(-1, 1)
        x_t_direction = torch.sqrt(1. - alpha_hat_prev - posterior_std**2) * pred_noise
        random_noise = posterior_std * noise
        x_t_1 = sqrt_alpha_hat_prev * x_0_pred + x_t_direction + random_noise

        return x_t_1


if __name__ == "__main__":
    dct = DDIMScheduler().__dict__
    for k in dct.keys():
        if isinstance(dct[k], torch.Tensor):
            print(k, dct[k].shape)
        else:
            print(k, dct[k])
