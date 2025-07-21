from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
import os
from tqdm import tqdm
from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# main class

class ElucidatedDiffusion(nn.Module):
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
        n_leadtimes=14,
        fixed_forward_noise=False,
        fixed_sampling_noise=False
    ):
        super().__init__()
        #assert net.random_or_learned_sinusoidal_cond
        self.model = model
        self.spatial_conditions = model.spatial_conditions
        self.embedding_cond_dims = n_leadtimes

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.fixed_forward_noise = fixed_forward_noise
        self.fixed_sampling_noise = fixed_sampling_noise

        # SDEdit flag setting
        self.num_edition_timesteps=num_edition_timesteps
        self.sdedit_flag = False
        self.num_edition_timesteps = int(num_edition_timesteps)
        if num_edition_timesteps < num_sample_steps:
            self.sdedit_flag = True
            print(f'\n ############ Warning : num_edition_timesteps : {num_edition_timesteps} < num_sample_steps : {num_sample_steps} ; SDEdit mode activated')


        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        
    @property
    def device(self):
        return next(self.model.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, cond_2d = None, embedded_cond = None, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.model(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            cond_2d,
            embedded_cond=embedded_cond
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)
        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size = 16, num_sample_steps = None, condition=None, lt_cond=None, clamp = False, forward_noise=None, sampling_noise=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )
        if not self.sdedit_flag:
            sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
            
            # images is noise at the beginning
            init_sigma = sigmas[0]
            images = init_sigma * torch.randn(shape, device = self.device)

        
        else :
            sigmas_and_gammas = list(
            zip(
            sigmas[num_sample_steps - self.num_edition_timesteps:-1],
            sigmas[num_sample_steps - self.num_edition_timesteps+1:],
            gammas[num_sample_steps - self.num_edition_timesteps:-1]
            )
            )

            # Noising condition
            condition_sdedit = normalize_to_neg_one_to_one(condition)
            init_sigma = sigmas[num_sample_steps - self.num_edition_timesteps]
            if not self.fixed_forward_noise:
                noise = torch.randn_like(condition_sdedit, device = self.device)
            else :
                noise = forward_noise
                if forward_noise is None:
                    raise ValueError('Fixed forward noise mode but forward_noise is None!')

            images = init_sigma * noise  + condition_sdedit

        
        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            if not self.fixed_sampling_noise:
                eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling # Algorithm 2 : line 4
            else : 
                print('Using sampling noise', 'sampling_noise.shape', sampling_noise.shape, 'shape',shape)
                eps = self.S_noise * sampling_noise
                if sampling_noise is None:
                    raise ValueError('Fixed sampling noise mode but sampling_noise is None!')


            sigma_hat = sigma + gamma * sigma # Algorithm 2 : line 5
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps # Algorithm 2 : line 6

            cond_2d = condition if self.spatial_conditions else None
            cond_emb = lt_cond if self.embedding_cond_dims is not None else None
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, cond_2d, cond_emb, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat # Algorithm 2 : line 7

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                cond_2d = condition if self.spatial_conditions else None
                cond_emb = lt_cond if self.embedding_cond_dims is not None else None

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, cond_2d, cond_emb, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

        #images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size = 16, num_sample_steps = None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        if not self.sdedit_flag:
            images  = sigmas[0] * torch.randn(shape, device = device)
        
        else :
            # Noising condition
            condition = normalize_to_neg_one_to_one(condition)
            init_sigma = sigmas[num_sample_steps - self.num_edition_timesteps]
            noise = torch.randn_like(condition, device = self.device)

            images = init_sigma * noise  + condition

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        

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

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, img, *args, **kwargs):
        #TODO change terminology from cond_2d to cond
        batch_size, c, h, w, device, image_size, channels = *img.shape, img.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size} but they are (h={h},w={w})'
        assert c == channels, f'mismatch of image channels. It must be {channels} but it is {c}'

        img = normalize_to_neg_one_to_one(img)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(img)

        noised_images = img + padded_sigmas * noise  # alphas are 1. in the paper


        # Conditioned diffusion :
        cond_2d = None
        if self.spatial_conditions:
            cond_2d = kwargs.get('condition_tensor')
        
        cond_emb = None
        if self.embedding_cond_dims is not None:
            cond_emb = kwargs.get('leadtime')

        denoised = self.preconditioned_network_forward(noised_images, sigmas, cond_2d, cond_emb)

        losses = F.mse_loss(denoised, img, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
