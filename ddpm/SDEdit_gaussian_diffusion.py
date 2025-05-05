import torch
from denoising_diffusion_pytorch import GaussianDiffusion
import torch.nn.functional as F
from random import random
from einops import rearrange
from tqdm import tqdm

class SDEeditGaussianDiffusion(GaussianDiffusion):
    def __init__(self,
                 *args,
                 num_edition_timesteps=50,
                 **kwargs
                 ):
        """
        Initialize the SDEeditGaussianDiffusion.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        # Checking that num of timestep of denoising for SDEdit
        self.num_edition_timesteps = num_edition_timesteps

    @torch.no_grad()
    def sample(self, batch_size, return_all_timesteps=False, condition=None):
        """
        Generate samples using SDEdit diffusion.
        Args:
            batch_size (int): Number of samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop
            if not self.is_ddim_sampling
            else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, *image_size),
            return_all_timesteps=return_all_timesteps,
            condition=condition,
        )

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, condition=None):
        """
        Sample from SDEdit diffusion using a loop over timesteps.
        Args:
            shape: Shape of the samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        batch = shape[0]

        # Noising condition
        t = torch.randint(0, self.num_edition_timesteps, (batch,), device=self.device).long()
        img = self.q_sample(x_start=condition, t=t).to(self.device)

        # Denoising image
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_edition_timesteps)), desc = 'sampling loop time step', total = self.num_edition_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)

        return ret
    
    def ddim_sample(
            self,
            shape,
            return_all_timesteps = False,
            condition=None
    ):
        """
        Sample from conditioned diffusion using ddim sampling.
        Args:
            shape: Shape of the samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        batch, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.num_edition_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        # Check value for this variables
        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Noising condition
        t = torch.randint(0, self.num_edition_timesteps, (batch,), device=self.device).long()
        img = self.q_sample(x_start=condition, t=t).to(self.device)
        
        # Denoising image
        imgs = [img]
        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
            leave=False,
        ):
            time_cond = torch.full(
                (batch,), time, device=self.device, dtype=torch.long
            )
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                condition,
                clip_x_start=True,
                rederive_pred_noise=True,
            )
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta
                * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)

        return ret
    
    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        offset_noise_strength=None,
        condition=None,
    ):
        raise NotImplementedError
        

    def forward(
            self, 
            img,
            *args,
            **kwargs
    ):
        raise NotImplementedError
