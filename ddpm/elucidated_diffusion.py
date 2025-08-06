from math import sqrt
import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from utils import plotter_inconditionnal
from utils.utils import mirror_fill
import os
import time

rank = int(os.environ.get("LOCAL_RANK",0))
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
        config,
        *,
        image_size = (717,1121),
        channels = 3,
        image_pos = None, #Added to use patch diffusion
        patch_size= None, #Added to use patch diffusion
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
        n_leadtimes=14
    ):
        super().__init__()
        #assert net.random_or_learned_sinusoidal_cond
        self.model = model
        self.spatial_conditions = model.spatial_conditions
        self.embedding_cond_dims = n_leadtimes
        self.config = config
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        
        # SDEdit flag setting
        self.num_edition_timesteps=num_edition_timesteps
        self.sdedit_flag = False
        self.num_edition_timesteps = int(num_edition_timesteps)
        if num_edition_timesteps < num_sample_steps:
            self.sdedit_flag = True
            print(f'\n ############ Warning : num_edition_timesteps : {num_edition_timesteps} < num_sample_steps : {num_sample_steps} ; SDEdit mode activated')  
        self.patch_size= patch_size
        self.image_pos = image_pos
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

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.config = config
        if self.config.training_configuration == "mirror":
            # init data for mirroring   
            self.init_mirror_filling()  
        
    @property
    def device(self):
        return next(self.model.parameters()).device

    def init_mirror_filling(self):
        """ 
        That function defines variables that allow to fill the invalid datas of an image by valid datas, like a mirror
        """
        data_path = self.config.data_dir

        #choose a random file in data folder
        files = [f for f in os.listdir(data_path)]
        file_name = random.choice(files)
        file = os.path.join(data_path,file_name)

        img = np.load(file)
        img=torch.from_numpy(img).to("cuda")

        img = img.unsqueeze(0)
        img = img.permute((0,3,1,2))
        crop = self.config.crop
        img = img[:,:,crop[0]:crop[1],crop[2]:crop[3]]
        mask = (torch.abs(img) < 1000)


        self.valid_x_vert,self.invalid_x_vert,self.valid_y_vert,self.invalid_y_vert,self.valid_x_horiz,self.invalid_x_horiz,self.valid_y_horiz,self.invalid_y_horiz = mirror_fill(img,mask)

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return torch.div((self.sigma_data ** 2), (torch.add(sigma ** 2, self.sigma_data ** 2)))


    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, image_pos = None, patch_size = None, cond_2d = None, embedded_cond = None, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.model(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            x_self_cond = cond_2d,
            embedded_cond=embedded_cond,
            x_pos = image_pos,
            patch_size=patch_size,
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
    def sample(self, batch_size = 16, num_sample_steps = None, condition=None, lt_cond=None, clamp = False, image_pos=None):
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        
        shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
          
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
            noise = torch.randn_like(condition_sdedit, device = self.device)

            images = init_sigma * noise  + condition_sdedit

        
        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling # Algorithm 2 : line 4

            sigma_hat = sigma + gamma * sigma # Algorithm 2 : line 5
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps # Algorithm 2 : line 6

            cond_2d = condition if self.spatial_conditions else None
            
            if self.config.orography_conditioning and not self.spatial_conditions:
                self.spatial_conditions= True
                cond_2d = condition
            
            cond_emb = lt_cond if self.embedding_cond_dims is not None else None
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, image_pos, cond_2d = cond_2d, embedded_cond= cond_emb, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat # Algorithm 2 : line 7

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            
            if sigma_next != 0:
                cond_2d = condition if self.spatial_conditions else None
                cond_emb = lt_cond if self.embedding_cond_dims is not None else None
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, image_pos, cond_2d = cond_2d, embedded_cond= cond_emb, clamp = clamp)
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

        shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        

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

    def downscale(self, x, scaling_time, mode='bilinear'):
        for _ in range(scaling_time):
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode=mode)
        return x
    
    def forward(self, img, image_pos = None, patch_size = None, *args, **kwargs):
        #TODO change terminology from cond_2d to cond 
        
        mask = (torch.abs(img) < 1000) #need modification when adding variables
        
        if self.config.patch_diffusion:
            
            batch_size, c, h, w, device, image_size, channels = *img.shape, img.device, self.image_size, self.channels
            patch_coords = image_pos
            patch_size = patch_size
            assert h == patch_size, f"img's height must be patch size"
            assert w == patch_size, f"img's width must be patch size"
            # assert c == channels, 'mismatch of image channels'
            img = normalize_to_neg_one_to_one(img) 
            sigmas = self.noise_distribution(batch_size)
            padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

            noise = torch.randn_like(img)
            noised_images = img
            noised_images[:,:3,:,:] = img[:,:3,:,:] + padded_sigmas * noise[:,:3,:,:]  # alphas are 1. in the paper
            self_cond = None

            cond_2d = None
            if self.spatial_conditions:
                cond_2d = kwargs.get('condition_tensor')
            
            cond_emb = None
            if self.embedding_cond_dims is not None:
                    cond_emb = kwargs.get('leadtime')
            denoised = self.preconditioned_network_forward(noised_images, sigmas, patch_coords, patch_size, cond_2d = cond_2d, embedded_cond= cond_emb)
            
            denoised = denoised.masked_fill(~mask,0.)#filling outside with zeros to compute loss
            img = img.masked_fill(~mask,0.) #filling outside with zeros to compute loss
            
            # classic
            # losses = F.mse_loss(denoised[:,:3,:,:],img[:,:3,:,:],reduction='none')
            # losses = reduce(losses,'b ... -> b','mean')

            # multi scale
            loss = 0.
            for i in range(4):
                model_out_downscaled = self.downscale(denoised[:,:3,:,:], scaling_time=i)
                target_downscaled = self.downscale(img[:,:3,:,:], scaling_time=i)
                size = model_out_downscaled.shape[-1]*model_out_downscaled.shape[-2]
                loss_ms = F.mse_loss(model_out_downscaled, target_downscaled, reduction = 'none')
                loss += reduce(loss_ms, 'b ... -> b', 'mean')/size

            losses = losses * self.loss_weight(sigmas)

            return losses.mean()
        
        else :
            
            batch_size, c, h, w, device, image_size, channels = *img.shape, img.device, self.image_size, self.channels
            
            assert h == image_size[0] and w == image_size[1], f'height and width of image must be {image_size} but they are (h={h},w={w})'
            assert c == channels, f'mismatch of image channels. It must be {channels} but it is {c}'
            assert self.config.training_configuration in ["zero", "mirror", "rectangular"], f"training_configuration must be 'zero', 'mirror' or 'rectangular' and is {self.config.training_configuration}"
            #TODO : stock the mask in memory (self.mask) to use the condition with different variables
            
            if self.config.training_configuration == "zero": #filling invalid datas outside AROME with 0
                
                img_filled = img.masked_fill(~mask,0.5) 
                img = normalize_to_neg_one_to_one(img_filled) 
                sigmas = self.noise_distribution(batch_size)
                padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

                noise = torch.randn_like(img)
                noised_images = img + padded_sigmas * noise  # alphas are 1. in the paper

                cond_2d = None
                if self.spatial_conditions:
                    cond_2d = kwargs.get('condition_tensor')
                
                cond_emb = None
                if self.embedding_cond_dims is not None:
                    cond_emb = kwargs.get('leadtime')
                        
                denoised = self.preconditioned_network_forward(noised_images, sigmas, cond_2d = cond_2d, embedded_cond= cond_emb)
                
                denoised = denoised.masked_fill(~mask,0.)#filling outside with zeros to compute loss
                img = img.masked_fill(~mask,0.) #filling outside with zeros to compute loss
                
                losses = F.mse_loss(denoised,img,reduction='none')
                losses = reduce(losses,'b ... -> b','mean')
                losses = losses * self.loss_weight(sigmas)
                return losses.mean()
        
            elif self.config.training_configuration == "mirror": #filling datas outside AROME with mirrored datas
                
                img_filled = img.clone().to(img.device)
                
                for batch in range(self.config.batch_size):
                    
                    #filling datas outside AROME with mirrored datas, need to do vertical filling then horizontal filling 
                    img_filled[batch,:,self.invalid_y_vert,self.invalid_x_vert] = img_filled[batch,:,self.valid_y_vert,self.valid_x_vert] #vertical filling
                    img_filled[batch,:,self.invalid_y_horiz,self.invalid_x_horiz] = img_filled[batch,:,self.valid_y_horiz,self.valid_x_horiz] #horizontal filling
                    img = normalize_to_neg_one_to_one(img_filled) #filled img normalized
                    
                sigmas = self.noise_distribution(batch_size)
                padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')
                noise = torch.randn_like(img)
                noised_images = img + padded_sigmas * noise  # alphas are 1. in the paper
        
                cond_2d = None
                if self.spatial_conditions:
                    cond_2d = kwargs.get('condition_tensor')

                cond_emb = None
                if self.embedding_cond_dims is not None:
                    cond_emb = kwargs.get('leadtime')
                denoised = self.preconditioned_network_forward(noised_images, sigmas, cond_2d = cond_2d, embedded_cond= cond_emb)
                losses = F.mse_loss(denoised,img,reduction='none')
                losses = reduce(losses,'b ... -> b','mean')
                losses = losses * self.loss_weight(sigmas)
                return losses.mean()
            else :
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

                denoised = self.preconditioned_network_forward(noised_images, sigmas, cond_2d = cond_2d, embedded_cond= cond_emb)

                losses = F.mse_loss(denoised, img, reduction = 'none')
                losses = reduce(losses, 'b ... -> b', 'mean')

                losses = losses * self.loss_weight(sigmas)
                return losses.mean()



