import os
import gc
import numpy as np
import torch
from tqdm import tqdm
import logging
from ddpm.ddpm_base import Ddpm_base
from utils.distributed import is_main_gpu
from utils.guided_loss import loss_dict
from utils.plotter import online_plot, online_plot_var, online_plot_mean, online_plot_quantiles
from datetime import datetime


class Sampler(Ddpm_base):
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        dataloader=None,
        inversion_transforms=None,
    ) -> None:
        """
        Initialize the Sampler class.
        Args:
            model (torch.nn.Module): The neural network model for sampling.
            config: Configuration settings for sampling.
            dataloader: The data loader for input data (optional).
        """
        super().__init__(model, config, dataloader, inversion_transforms=inversion_transforms)
        self.loss_func = loss_dict["L1Loss"]

    @torch.no_grad()
    def _guided_sample_batch(self, truth_sample_batch, guidance_loss_scale=100, random_noise=False):
        # OUTDATED FOR NOW. USED WITH DDIM
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
        sampled_images_unnorm = self.transforms_func(sample.cpu()).numpy()
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

        # shape of the images 
        x, y  = self.config.crop[1] - self.config.crop[0], self.config.crop[3] - self.config.crop[2]

        if self.config.sampling_mode == "simple":
            # To be removed

            if is_main_gpu():
                self.logger.info(
                    #f"Sampling {self.config.n_sample * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")
                    f"Sampling {self.config.n_sample} images...")
            with tqdm(total=self.config.n_sample // self.config.batch_size, desc="Sampling ", unit="batch",
                      disable= not is_main_gpu()) as pbar:
                b = 0
                while b < self.config.n_sample:
                    batch_size = min(self.config.n_sample - b, self.config.batch_size)
                    samples = super()._sample_batch(nb_img=batch_size)
                    for s in samples:
                        # Append the empty rr channel if only u v t2m
                        if len(s) == 3:
                            s = np.append(np.zeros(shape=(1, 256, 256)), s, axis=0)

                        filename = filename_format.format(sample_index=str(i))
                        save_path = os.path.join(self.config.output_dir ,self.config.run_name, "samples", filename)
                        np.save(save_path, s)
                        i += max(torch.cuda.device_count(), 1)
                    b += batch_size
                    pbar.update(1)
        elif "conditioned" in self.config.sampling_mode:
            if is_main_gpu():
                self.logger.info(
                    f"Sampling {len(self.dataloader) * self.config.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")

            # Build empty channels to extend the generated data with, in order to match the shape of the dataset (e.g. rr)
            if self.config.n_var != self.config.n_var_in_dataset:
                    zero_pad = torch.zeros(16, self.config.n_var_in_dataset - self.config.n_var, x, y ).to(self.gpu_id)

            # Goes through every 16 members sample batches (= 1 whole AROME ensemble, as the sampler reads the dataset sequentially when sampling)
            for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Sampling ", unit="batch"):
                
                # print('date, lt, member_id', (batch["date"],batch["leadtime"],batch["member_id"]))
                # Get the list containing the n_sampling_conditioning_sets sets of conditionning members (tensor of shape [n_members_dataset, n_sampling_conditioning_sets, n_condition*n_var, x, y])
                if self.config.sampling_mode == 'conditioned_input':
                    conditioning_sets = batch['condition_tensor']
                elif self.config.sampling_mode == 'conditioned_sdedit':
                    conditioning_sets = batch["img"]
                else :
                    raise NotImplementedError

                # Transpose the array-> array of shape [n_sampling_conditioning_sets, n_members_dataset, n_conditions*n_var, H, W]
                conditioning_sets = conditioning_sets.permute(1, 0, 2, 3, 4)
                if self.config.n_var != self.config.n_var_in_dataset:
                        # Generates a member for all n_sampling_conditioning_sets set from the conditioning_sets, u, v, t2m
                        ensemble = torch.cat([
                            torch.cat((zero_pad, self._sample_batch(nb_img=len(set), condition=set.to(self.gpu_id), lt_cond=batch['leadtime'].to(self.gpu_id), ensemble_mean=batch['ensemble_mean_tensor'].to(self.gpu_id))), dim=1).unsqueeze(0) # concatenate an empty rr channel
                            for set in conditioning_sets
                        ], dim=0).cpu().reshape(-1, self.config.n_var_in_dataset, x, y ) # reshape -> [n_sampling_conditioning_sets*16, 4, 256, 256]
                else:
                        # Generates a member for all n_sampling_conditioning_sets set from the conditioning_sets, rr, u, v, t2m
                        ensemble = torch.cat([
                            self._sample_batch(nb_img=len(set), condition=set.to(self.gpu_id), lt_cond=batch['leadtime'].to(self.gpu_id), ensemble_mean=batch['ensemble_mean_tensor'].to(self.gpu_id)).unsqueeze(0)
                            for set in conditioning_sets
                        ], dim=0).cpu().reshape(-1, self.config.n_var_in_dataset, x, y ) # reshape -> [n_sampling_conditioning_sets*16, 4, 256, 256]

                lt = batch['leadtime'][0]
                d = datetime.strptime(batch['date'][0], '%Y-%m-%d').date()
                filename = filename_format.format(date = d, leadtime = lt + 1) # lt + 1 to match MetScore's indicing
                save_path = os.path.join(self.config.output_dir, self.config.run_name, "samples", filename)
                np.save(save_path, ensemble.numpy())

                if self.config.plot:# and is_main_gpu():

                    arome_ensemble = conditioning_sets.detach().clone()[0]

                    
                    print('ensemble.shape',ensemble.shape)
                    print('arome_ensemble.shape',arome_ensemble.shape)

                    if self.config.predict_residue:
                        ensemble_mean = batch['ensemble_mean_tensor'].detach().cpu()
                        # print('ensemble_mean.shape',ensemble_mean.shape)
                        # ensemble = torch.sub(ensemble, torch.cat([torch.zeros(self.config.n_var_in_dataset - self.config.n_var, x, y ), ensemble_mean[0]], dim=0).unsqueeze(0).expand(ensemble.shape[0], -1, -1, -1))
                        arome_ensemble = torch.add(arome_ensemble[0], ensemble_mean)
                    if self.config.invert_norm == True:
                        detransform_func = self.transforms_func()
                        arome_ensemble = torch.stack([detransform_func(image) for image in arome_ensemble]).detach().clone()

                    
                        
                    
                    online_plot(
                        arome_ensemble.numpy(),
                        ensemble.numpy()[:,1:,:,:],
                        figname=os.path.join(self.config.output_dir, self.config.run_name, "samples", filename[:-4]+'.png'),
                        figtitle=f'Sample comparison for {batch["date"][0]}_{batch["leadtime"][0]}',
                        clim_global=None
                    )

        else:
            raise ValueError(f"Sampling mode {self.config.sampling_mode} not supported.")

        self.logger.info(
            f"Sampling done. Images saved in {self.config.output_dir}/{self.config.run_name}/samples/")