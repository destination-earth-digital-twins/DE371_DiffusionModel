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
        
        assert num_edition_timesteps <= num_sample_steps
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


        sigmas_and_gammas = list(
            zip(
            sigmas[num_sample_steps - self.num_edition_timesteps:-1],
            sigmas[num_sample_steps - self.num_edition_timesteps+1:],
            gammas[num_sample_steps - self.num_edition_timesteps:-1]
            )
        )

        # Noising condition
        condition = normalize_to_neg_one_to_one(condition)
        init_sigma = sigmas[num_sample_steps - self.num_edition_timesteps]
        noise = torch.randn_like(condition, device = self.device)

        images = init_sigma * noise  + condition * bool(num_sample_steps - self.num_edition_timesteps)

        # Gradually denoising

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling # Algorithm 2 : line 4

            sigma_hat = sigma + gamma * sigma # Algorithm 2 : line 5
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps # Algorithm 2 : line 6

            self_cond = condition if self.self_condition else None

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat # Algorithm 2 : line 7

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = condition if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

        #images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)
    
    @torch.no_grad()
    def sample_using_dpmpp(self,
                batch_size = 16,
                num_sample_steps = None,
                condition=None,
                clamp = False
        ):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        # Noising condition
        condition = normalize_to_neg_one_to_one(condition)
        init_sigma = sigmas[num_sample_steps - self.num_edition_timesteps]
        noise = torch.randn_like(condition, device = self.device)

        images = init_sigma * noise  + condition * bool(num_sample_steps - self.num_edition_timesteps)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)


    # training

    def forward(self, img, *args, **kwargs):
        raise NotImplementedError
