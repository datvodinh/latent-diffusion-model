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

    @torch.no_grad()
    def sampling_t(
        self,
        x_t: torch.Tensor,
        model,
        labels: torch.Tensor,
        time: torch.Tensor,
        time_prev: torch.Tensor,
        n_samples: int = 16,
        cfg_scale: int = 3,
        *args, **kwargs
    ):
        pred_noise = model(x_t, time, labels)
        if cfg_scale > 0 and labels is not None:
            uncond_pred_noise = model(x_t, time, None)
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale)
        alpha = self.alpha.to(model.device)[time][:, None, None, None]
        sqrt_alpha = self.sqrt_alpha.to(model.device)[time][:, None, None, None]
        somah = self.sqrt_one_minus_alpha_hat.to(model.device)[time][:, None, None, None]
        sqrt_beta = self.sqrt_beta.to(model.device)[time][:, None, None, None]
        if time[0] > 1:
            noise = torch.randn_like(x_t, device=model.device)
        else:
            noise = torch.zeros_like(x_t, device=model.device)

        x_t_new = 1 / sqrt_alpha * (x_t - (1-alpha) / somah * pred_noise) + sqrt_beta * noise
        return x_t_new

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
        x_t = torch.randn(
            n_samples, in_channels, dim, dim, device=model.device
        )
        step_ratios = self.max_timesteps // timesteps
        all_timesteps = torch.flip(torch.arange(0, timesteps) * step_ratios, dims=(0,))
        for t in all_timesteps:
            time = torch.full((n_samples,), fill_value=t, device=model.device)
            time_prev = time - self.max_timesteps // timesteps
            x_t = self.sampling_t(x_t=x_t, model=model, labels=labels, time=time, time_prev=time_prev,
                                  n_samples=n_samples, cfg_scale=cfg_scale, *args, **kwargs)
        return x_t

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
            time = torch.full((n_samples,), fill_value=t, device=model.device)
            time_prev = time - self.max_timesteps // timesteps
            x_t = self.sampling_t(x_t=x_t, model=model, labels=labels, time=time, time_prev=time_prev,
                                  n_samples=n_samples, cfg_scale=cfg_scale)
            yield x_t


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
        x_t: torch.Tensor,
        model,
        time: torch.Tensor,
        time_prev: torch.Tensor,
        pred_noise: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs
    ):
        if pred_noise is None:
            pred_noise = model(x_t, time, labels)

        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat.to(model.device)[time][:, None, None, None]
        sqrt_alpha_hat = self.sqrt_alpha_hat.to(model.device)[time][:, None, None, None]
        alpha_hat_prev = self.alpha_hat.to(model.device)[time_prev] if (
            time_prev[0] >= 0
        ) else torch.ones_like(time_prev, device=model.device)
        alpha_hat_prev = alpha_hat_prev[:, None, None, None]
        sqrt_alpha_hat_prev = torch.sqrt(alpha_hat_prev)
        eta = kwargs['eta'] if "eta" in kwargs.keys() else 1
        posterior_std = torch.sqrt(self.variance).to(model.device)[time][:, None, None, None] * eta

        if time[0] > 0:
            noise = torch.randn_like(x_t, device=model.device)
        else:
            noise = torch.zeros_like(x_t, device=model.device)

        x_0_pred = (x_t - sqrt_one_minus_alpha_hat * pred_noise) / sqrt_alpha_hat
        # if "quantize" in kwargs.keys():
        #     x_0_pred = kwargs['quantize'](x_0_pred)
        x_0_pred = x_0_pred.clamp(-1, 1)
        x_t_direction = 1. - alpha_hat_prev - posterior_std**2
        x_t_direction = torch.sqrt(torch.where(x_t_direction > 0, x_t_direction, 0)) * pred_noise

        random_noise = posterior_std * noise
        x_prev = sqrt_alpha_hat_prev * x_0_pred + x_t_direction + random_noise
        return x_prev


class PLMSScheduler(DDIMScheduler):
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_1: int = 0.0001,
        beta_2: int = 0.02
    ) -> None:
        super().__init__(beta_1=beta_1, beta_2=beta_2, max_timesteps=max_timesteps)
        self._init_params()

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
        x_t = torch.randn(
            n_samples, in_channels, dim, dim, device=model.device
        )
        step_ratios = self.max_timesteps // timesteps
        all_timesteps = torch.flip(torch.arange(0, timesteps) * step_ratios, dims=(0,))
        list_pred = []

        for t in all_timesteps:
            time = torch.full((n_samples,), fill_value=t, device=model.device)
            time_prev = time - self.max_timesteps // timesteps
            pred_noise = model(x_t, time)
            if len(list_pred) == 0:
                # Pseudo Improved Euler (2nd order)
                x_prev = self.sampling_t(
                    x_t=x_t, model=model, labels=labels, time=time, time_prev=time_prev,
                    cfg_scale=cfg_scale, **kwargs
                )
                pred_noise_next = model(x_prev, time_prev)
                pred_noise = (pred_noise+pred_noise_next) / 2
            elif len(list_pred) == 1:
                # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
                pred_noise = (3 * pred_noise - list_pred[-1]) / 2
            elif len(list_pred) == 2:
                # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
                pred_noise = (23 * pred_noise - 16 * list_pred[-1] + 5 * list_pred[-2]) / 12
            elif len(list_pred) >= 3:
                # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
                pred_noise = (55 * pred_noise - 59 * list_pred[-1] + 37 * list_pred[-2] - 9 * list_pred[-3]) / 24
            x_prev = self.sampling_t(
                x_t=x_t, model=model, labels=labels, pred_noise=pred_noise, time=time, time_prev=time_prev,
                cfg_scale=cfg_scale, **kwargs
            )
            list_pred.append(pred_noise)
        return x_prev


class Scheduler:
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_1: int = 0.0001,
        beta_2: int = 0.02
    ) -> None:
        self.ddpm = DDPMScheduler(max_timesteps, beta_1, beta_2)
        self.ddim = DDIMScheduler(max_timesteps, beta_1, beta_2)
        self.plms = PLMSScheduler(max_timesteps, beta_1, beta_2)

    def noising(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        if t.device != x_0.device:
            t = t.to(x_0.device)
        noise = torch.randn_like(x_0, device=x_0.device)
        new_x = self.ddpm.sqrt_alpha_hat.to(x_0.device)[t][:, None, None, None] * x_0
        new_noise = self.ddpm.sqrt_one_minus_alpha_hat.to(x_0.device)[t][:, None, None, None] * noise
        return new_x + new_noise, noise

    def sampling(
        self,
        mode: str = "ddim",
        **kwargs
    ):
        if mode == "ddpm":
            return self.ddpm.sampling(**kwargs)
        elif mode == "ddim":
            return self.ddim.sampling(**kwargs)
        elif mode == "plms":
            return self.plms.sampling(**kwargs)

    def sampling_demo(
        self,
        mode: str = "ddim",
        **kwargs
    ):
        if mode == "ddpm":
            return self.ddpm.sampling_demo(**kwargs)
        elif mode == "ddim":
            return self.ddim.sampling_demo(**kwargs)
        elif mode == "plms":
            return self.plms.sampling_demo(**kwargs)
