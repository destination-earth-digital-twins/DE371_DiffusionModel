from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from ddpm.elucidated_diffusion import ElucidatedDiffusion, exists, default, log, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one

# main class

class SDEditElucidatedDiffusion(ElucidatedDiffusion):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 100, # number of sampling steps
        sigma_min = 0.002,      # min noise level
        sigma_max = 0.80,       # max noise level
        sigma_data = 0.5,       # standard deviation of data distribution
        rho = 7,                # controls the sampling schedule
        P_mean = -1.2,          # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,            # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 0.0,
        num_edition_timesteps=5,
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            channels=channels,
            num_sample_steps=num_sample_steps,
            sigma_min=sigma_min,      # min noise level
            sigma_max=sigma_max,       # max noise level
            sigma_data=sigma_data,       # standard deviation of data distribution
            rho=rho,                # controls the sampling schedule
            P_mean=P_mean,          # mean of log-normal distribution from which noise is drawn for training
            P_std=P_std,            # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=S_churn,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
            S_tmin=S_tmin,
            S_tmax=S_tmax,
            S_noise=S_noise
        )
        
        assert num_edition_timesteps < num_sample_steps
        self.num_edition_timesteps=num_edition_timesteps

    

    # sampling

    @torch.no_grad()
    def sample(self,
               batch_size = 16, 
               num_sample_steps = None,
               condition=None,
               clamp = False
        ):
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        # sigmas = sigmas[:self.num_edition_timesteps]
        # gammas = gammas[:self.num_edition_timesteps]
        # Not operationnal ! 
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # Noising condition entirely
        _sigmas = self.noise_distribution(batch_size)
        # print('sigmas.shape',_sigmas.shape)
        padded_sigmas = rearrange(_sigmas, 'b -> b 1 1 1')
        # print('padded_sigmas.shape',padded_sigmas.shape)
        noise = torch.randn_like(condition)

        images = condition + padded_sigmas * noise  # alphas are 1. in the paper

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond=None, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond=None, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next

        return images


    # training

    def forward(self, img, *args, **kwargs):
        raise NotImplementedError
