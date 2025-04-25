import torch
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default
import torch.nn.functional as F
from random import random
from einops import rearrange
from tqdm import tqdm

class SDEeditGaussianDiffusion(GaussianDiffusion):
    def __init__(self,
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
        # Checking that num of timestep of denoising for SDEdit
        self.num_edition_timesteps = 1

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
        t = torch.randint(0, self.num_edition_timesteps, (batch,), device=device).long()
        img = self.q_sample(x_start=condition, t=t)

        # Denoising image
        # img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(self.num_timesteps-self.num_edition_timesteps, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_edition_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)

        return ret
    
    def ddim_sample(
            self,
            shape,
            return_all_timesteps = False
    ):
        raise NotImplementedError
    
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
