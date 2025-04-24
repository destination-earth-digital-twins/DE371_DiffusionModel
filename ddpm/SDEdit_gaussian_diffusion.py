import torch
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    default,
    extract,
)
import torch.nn.functional as F
from random import random
from einops import reduce, rearrange
from torch.nn.functional import mse_loss
from tqdm import tqdm

class SDEeditGaussianDiffusion(GaussianDiffusion):
    def __init__(self,
                 num_edition_timesteps=100,
                 *args,
                 **kwargs
                 ):
        """
        Initialize the SDEeditGaussianDiffusion.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Checking that num of timestep of denoising for SDEdit
        assert num_edition_timesteps < self.num_timesteps
        register_buffer('num_edition_timesteps', num_edition_timesteps)

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
        batch, device = shape[0], self.device

        # Noising condition
        noise = default(noise, lambda: torch.randn_like(condition))

        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(condition.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        t = torch.randint(0, self.num_timesteps-self.num_edition_timesteps, (batch,), device=device).long()
        img = self.q_sample(x_start=condition, t=t, noise=noise)

        # Denoising image
        imgs = [img]
        for t in tqdm(
            reversed(range(0, self.num_edition_timesteps)),
            desc="sampling loop time step",
            total=self.num_edition_timesteps,
            leave=False,
        ):
            img, x_start = self.p_sample(img, t, condition)
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
        """
        Calculate pixel-wise loss for conditioned diffusion.
        Args:
            x_start: Starting image tensor.
            t (int): Timestep.
            noise: Noise tensor.
            offset_noise_strength: Strength of offset noise.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Pixel-wise loss.
        """
        noise = default(noise, lambda: torch.randn_like(condition))

        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(condition.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        x = self.q_sample(x_start=condition, t=t, noise=noise)
        x_self_cond = None

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = condition
        elif self.objective == "pred_v":
            v = self.predict_v(condition, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for conditioned diffusion.
        Args:
            img: Input image tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            torch.Tensor: Forward pass result.
        """
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size[0] and w == img_size[1]
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps-self.num_edition_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
