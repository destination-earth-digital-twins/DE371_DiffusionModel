import logging
import os
import warnings

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import matplotlib.colors as colors

from utils.distributed import get_rank, is_main_gpu, get_rank_num

cmapRR = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow", "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)

class Ddpm_base:
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        dataloader=None,
        val_dataloader=False,
        inversion_transforms=None,
    ) -> None:
        """
        Initialize the Trainer.
        Args:
            model (torch.nn.Module): The neural network model.
            config: Configuration settings.
            dataloader: DataLoader for training data.
        """
        self.optimizer = None
        self.scheduler = None
        self.config = config
        self.gpu_id = get_rank()
        if self.config.elucidated_diffusion_sampler == True:
            self.timesteps = model.num_sample_steps
        else:
            self.timesteps = model.num_timesteps
        self.dataloader = dataloader
        self.validation = config.validation
        self.val_dataloader = val_dataloader
        if self.validation and val_dataloader is None:
            raise ValueError("You set validation=True, but no val_dataloader found in Ddpm_base input.")
        self.snapshot_path = self.config.model_path
        self.model = model
        self.logger = logging.getLogger(f"logddp_{get_rank_num()}")

        # Load snapshot if available
        if self.snapshot_path is not None:
            if is_main_gpu():
                self.logger.info(f"Loading snapshot")
            self._load_snapshot(self.snapshot_path)
        else:
            self.epochs_run = 0

        # Move model to GPU
        model.to(torch.device(self.gpu_id))

        # Set training dataset information
        if config.invert_norm:
            if inversion_transforms is not None:
                self.transforms_func = inversion_transforms
        else:

            def transforms_func(x):
                return x

            self.transforms_func = transforms_func

        # if torch.__version__ >= "2.0.0":
        #     try:
        #         self.model = torch.compile(self.model)
        #     except:
        #         warnings.warn("Could not compile the model. Continuing without compilation.")

        # Convert model for multi-GPU training if available
        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
            )
            self.model = self.model.module

    def _load_snapshot(self, snapshot_path):
        """
        Load the snapshot of the training progress.
        Args:
            snapshot_path: Path to the snapshot file.
        """
        snapshot = torch.load(snapshot_path, map_location=get_rank())

        if "SCHEDULER_STATE" in snapshot and self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])

        if "WANDB_ID" in snapshot:
            self.wandb_id = snapshot["WANDB_ID"]


        if self.config.elucidated_diffusion_sampler == True:
            #Keys from classic Gaussian diffusion that are not used in the elucidated diffusion sampling
            unwanted_keys = ["betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod", "posterior_variance", "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2", "loss_weight"]
            filterd_state_dict = {k: v for k, v in snapshot["MODEL_STATE"].items() if k not in unwanted_keys}
            self.model.load_state_dict(filterd_state_dict)
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        
        self.epochs_run = snapshot["EPOCHS_RUN"]

        if self.optimizer is not None:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])

        # self.best_loss = snapshot["BEST_LOSS"]

        try:
            # Check if snapshot data configuration matches the current config
            data_config = snapshot["DATA"]
            # Load standard deviations and means from the snapshot
            self.stds = data_config["STDS"]
            self.means = data_config["MEANS"]
            if (
                data_config["V_IDX"] != self.config.var_indexes
                or data_config["CROP"] != self.config.crop
            ):
                raise ValueError(
                    "The variable indexes or crop of the snapshot do not match the current config"
                )
        except KeyError:
            # If data config is not available in the snapshot, issue a warning
            warnings.warn(
                "The snapshot does not contain data config, assuming it is the same as the current config"
            )

        if is_main_gpu():
            self.logger.info(
                f" Resuming model from {snapshot_path} at Epoch {self.epochs_run}"
            )

        self.epochs_run += 1

    def _sample_batch(self, nb_img=4, condition=None,  lt_cond=None, ensemble_mean=None):
        """
        Sample a batch of images.
        Args:
            nb_img (int): Number of images to sample.
            condition: Optional condition for conditional sampling.
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        if nb_img <= 0:
            return []  # No images to sample, return an empty list
        if condition is None:
            sampled_images = self.model.sample(batch_size=nb_img)
        else:
            sampled_images = self.model.sample(batch_size=nb_img, condition=condition, lt_cond=lt_cond)
        # member = residue + ensemble_mean when sampling. ensemble_mean is torch.zeros if the residue prediction is disabled
        # if not self.config.predict_residue:
        #     ensemble_mean = torch.zeros_like(ensemble_mean)
        # sampled_images = torch.add(sampled_images, ensemble_mean)
        
        if self.config.invert_norm == True:
            detransform_func = self.transforms_func()
            denorm_images = torch.stack([detransform_func(image) for image in sampled_images])
        else:
            denorm_images = self.transforms_func(sampled_images)
        return denorm_images
# def plot_grid(self, file_name, np_img):
#     """
#     Plot a grid of images with variable-specific colormaps and optional contours.
#     Args:
#         file_name (str): Name of the file to save the plot.
#         np_img (numpy.ndarray): Array of images to plot.
#     """
#     nb_image = len(np_img)

#     # Dictionnaire des colormaps personnalisées par variable
#     var_to_cmap = {
#         "rr": cmapRR,
#         "t2m": "bwr",
#         "x": "viridis",
#         "var_4": "plasma",  # Exemple, à adapter
#         # Ajoutez d'autres variables ici si besoin
#     }

#     fig, axes = plt.subplots(
#         nrows=min(6, nb_image),
#         ncols=len(self.config.var_indexes),
#         figsize=(10, 10),
#     )

#     for i in range(min(6, nb_image)):
#         for j, var in enumerate(self.config.var_indexes):
#             cmap = var_to_cmap.get(var, "gray")  # Valeur par défaut = "gray"
#             image = np_img[i, j]

#             if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
#                 ax = axes[i, j]
#             else:
#                 ax = axes[i]

#             im = ax.imshow(image, cmap=cmap, origin="lower")
#             ax.axis("off")
#             fig.colorbar(im, ax=ax)

#             # Ajout de contours pour var_4
#             if var == "var_4":
#                 ax.contour(image, colors='black', linewidths=0.5)

#     # Création du chemin de sauvegarde
#     save_path = os.path.join(
#         self.config.output_dir,
#         self.config.run_name,
#         "samples",
#         file_name,
#     )
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()
    def plot_grid(self, file_name, np_img):
        """
        Plot a grid of images.
        Args:
            file_name (str): Name of the file to save the plot.
            np_img (numpy.ndarray): Array of images to plot.
        """
        nb_image = len(np_img)
        var_to_cmap = {
        "rr": cmapRR,
        "t2m": "bwr",
        "u": "viridis",
        "v": "viridis",
        "t850": "bwr",
        "tpw850": "bwr",
        "z500": "viridis",
    }
        fig, axes = plt.subplots(
            nrows=min(6, nb_image),
            ncols=len(self.config.var_indexes),
            figsize=(10, 10),
        )
        for i in range(min(6, nb_image)):
            for j,var in enumerate(self.config.var_indexes):
                cmap = var_to_cmap.get(var, "gray")  # Valeur par défaut = "gray"

                # cmap = (
                #     "viridis" if self.config.var_indexes[j] != "t2m" else "bwr"
                # )
                image = np_img[i, j]
                
            
                if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[i]
                im = ax.imshow(image, cmap=cmap, origin="lower")
                ax.axis("off")
                fig.colorbar(im, ax=ax)
                # if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
                #     im = axes[i, j].imshow(image, cmap=cmap, origin="lower")
                #     axes[i, j].axis("off")
                #     fig.colorbar(im, ax=axes[i, j])
                # else:
                #     im = axes[i].imshow(image, cmap=cmap, origin="lower")
                #     axes[i].axis("off")
                #     fig.colorbar(im, ax=axes[i])
        # Save the plot to the specified file path
                if var == "z500":
                    ax.contour(image, colors='black', linewidths=0.5)
        plt.savefig(
            os.path.join(f"{self.config.output_dir}" , f"{self.config.run_name}", "samples", file_name),
            bbox_inches="tight",
        )
        plt.close()
