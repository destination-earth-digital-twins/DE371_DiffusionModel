import os
import gc
import numpy as np
import torch
from tqdm import tqdm
import logging
from ddpm.ddpm_base import Ddpm_base
from utils.distributed import is_main_gpu
from utils.guided_loss import loss_dict

import time


class DDPM_Sampler(Ddpm_base):
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        dataloader=None,
        inversion_transforms=None,
    ) -> None:
        """
        Initialize the DDPM_Sampler class.
        Args:
            model (torch.nn.Module): The neural network model for sampling.
            config: Configuration settings for sampling.
            dataloader: The data loader for input data (optional).
        """
        super().__init__(model, config, dataloader, inversion_transforms=inversion_transforms)
        self.loss_func = loss_dict["L1Loss"]

    @torch.no_grad()
    def _guided_sample_batch(self, truth_sample_batch, guidance_loss_scale=100, random_noise=False):
        """
        Perform guided sampling of a batch of images.
        Args:
            truth_sample_batch (torch.Tensor): Ground truth image batch for guidance.
            guidance_loss_scale (float): Scaling factor for the guidance loss between [0 - 100].
            random_noise (bool): Whether to use random noise as the initial sample.
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        assert (
            0 <= guidance_loss_scale <= 100
        ), "Guidance loss scale must be between 0 and 100."
        noise = torch.randn_like(truth_sample_batch).to(self.gpu_id)
        t_l = torch.ones((truth_sample_batch.shape[0])).to(
            self.gpu_id
        ).long() * (self.timesteps - 1)

        if not random_noise:
            sample = self.model.q_sample(
                x_start=truth_sample_batch, t=t_l, noise=noise
            )
        else:
            sample = noise

        for t in reversed(range(0, self.timesteps)):
            sample, _ = self.model.p_sample(sample, t, None)
            sample = sample.detach().requires_grad_()
            loss = (
                self.loss_func(sample, truth_sample_batch)
                * guidance_loss_scale
            )
            # Compute the gradient of the loss and update the sample
            cond_grad = -torch.autograd.grad(loss, sample)[0]
            sample = sample.detach() + cond_grad
        sampled_images_unnorm = self.transforms_func(sample).cpu().numpy()
        return sampled_images_unnorm

    @torch.no_grad()
    def sample(self, filename_format="fake_sample_{i}.npy", Shape=(4, 256, 256)):
        """
        Generate and save sample images during training.
        Args:
            filename_format (str): Format of the filename to save the images.
        Returns:
            None
        """

        i = self.gpu_id if type(self.gpu_id) is int else 0

        if is_main_gpu():
            self.logger.info(
                f"Sampling {len(self.dataloader) * self.config.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")

        if self.config.v_i == 3:
        # Goes through every 16 members sample batches (= 1 whole AROME ensemble, as the sampler reads the dataset sequentially when sampling)
                zero_pad = torch.zeros(16, 1, 256, 256).to(self.gpu_id)
        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Sampling ", unit="batch"):
            # Get the list containing the n_ensemble sets of conditionning members -> array of shape [16, n_ensemble, n_condition*3, 256, 256]
            conditioning_sets = batch['condition_sample']
            # Transpose the array-> array of shape [n_ensemble, 16, 3, 256, 256]
            conditioning_sets = conditioning_sets.permute(1, 0, 2, 3, 4)
            lt = batch['leadtime'][0]
            d = batch['date'][0].split(" ")[0]

            if self.config.v_i == 3:
                    ensemble = torch.cat([
                        torch.cat((zero_pad, self._sample_batch(nb_img=len(set), condition=set.to(self.gpu_id))), dim=1).unsqueeze(0)
                        for set in conditioning_sets # Generates a member for all n_ensemble set from the conditioning_sets
                    ], dim=0).cpu().reshape(-1, 4, 256, 256) # reshape -> [n_ensemble*16, 4, 256, 256]
            else:
                    ensemble = torch.cat([
                        self._sample_batch(nb_img=len(set), condition=set.to(self.gpu_id)).unsqueeze(0)
                        for set in conditioning_sets # Generates a member for all n_ensemble set from the conditioning_sets
                    ], dim=0).cpu().reshape(-1, 4, 256, 256) # reshape -> [n_ensemble*16, 4, 256, 256]
                
            filename = filename_format.format(date = d, leadtime = lt + 1)
            save_path = os.path.join(self.config.output_dir, self.config.run_name, "samples", filename)
            np.save(save_path, ensemble.numpy())

            # print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
            del ensemble
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.distributed.barrier()

            # raise ValueError('STOP')

        self.logger.info(
            f"Sampling done. Images saved in {self.config.output_dir}/{self.config.run_name}/samples/")
