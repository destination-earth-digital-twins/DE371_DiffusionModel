import logging
import os
import warnings

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from utils.distributed import get_rank, is_main_gpu, get_rank_num


class Ddpm_base:
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        dataloader=None,
        val_dataloader=False,
        inversion_transforms=None,
    ) -> None:
        
        # Initialize the Trainer.
        # Args:
        #     model (torch.nn.Module): The neural network model.
        #     config: Configuration settings.
        #     dataloader: DataLoader for training data.
        
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

    def _sample_batch(self, nb_img=4, condition=None,  lt_cond=None, ensemble_mean=None, image_pos=None):
        """
        Sample a batch of images.
        Args:
            nb_img (int): Number of images to sample.
            condition: Optional condition for conditional sampling.
            image_pos : Optional image position encoding when using patch diffusion
            image_pos : Optional image position encoding when using patch diffusion
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        if nb_img <= 0:
            return []  # No images to sample, return an empty list
        if condition is None:
            sampled_images = self.model.sample(batch_size=nb_img)
        else:
            sampled_images = self.model.sample(batch_size=nb_img, condition=condition, image_pos=image_pos, lt_cond=lt_cond)
        # member = residue + ensemble_mean when sampling. ensemble_mean is torch.zeros if the residue prediction is disabled
        if not self.config.predict_residue:
            ensemble_mean = torch.zeros_like(ensemble_mean)
        sampled_images = torch.add(sampled_images, ensemble_mean)
        
        if self.config.invert_norm == True:
            detransform_func = self.transforms_func()
            denorm_images = torch.stack([detransform_func(image) for image in sampled_images])
        else:
            denorm_images = self.transforms_func(sampled_images)
        return denorm_images

    def plot_grid(self, file_name, np_img):
        """
        Plot a grid of images.
        Args:
            file_name (str): Name of the file to save the plot.
            np_img (numpy.ndarray): Array of images to plot.
        """
        nb_image = len(np_img)
        fig, axes = plt.subplots(
            nrows=min(6, nb_image),
            ncols=len(self.config.var_indexes),
            figsize=(10, 10),
        )
        for i in range(min(6, nb_image)):
            for j in range(len(self.config.var_indexes)):
                cmap = (
                    "viridis" if self.config.var_indexes[j] != "t2m" else "bwr"
                )
                image = np_img[i, j]
                if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
                    im = axes[i, j].imshow(image, cmap=cmap, origin="lower")
                    axes[i, j].axis("off")
                    fig.colorbar(im, ax=axes[i, j])
                else:
                    im = axes[i].imshow(image, cmap=cmap, origin="lower")
                    axes[i].axis("off")
                    fig.colorbar(im, ax=axes[i])
        # Save the plot to the specified file path
        plt.savefig(
            os.path.join(f"{self.config.output_dir}" , f"{self.config.run_name}", "samples", file_name),
            bbox_inches="tight",
        )
        plt.close()
        
    def plot_grid_big_domain(self, file_name, np_img):
        """
        Plot a grid of images.
        Args:
            file_name (str): Name of the file to save the plot.
            np_img (numpy.ndarray): Array of images to plot.
        """
        nb_image = len(np_img)
        
        var_names = ["u","v","t2m"]
        dict_var={'u':0,'v':1,'t2m':2}
        colormap=["viridis","viridis","coolwarm"]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle("model sample",y=0.7)
        img = np_img[0].detach()
        # img = img.masked_fill(~mask,float("nan"))
        # img = np.where(np.abs(img) > 1, np.nan,img)
        for id, var in enumerate(var_names):
            var_id=dict_var[var]
            ax = axes[id]
            im=ax.imshow(img[id],cmap = colormap[id],origin='lower')
            plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            ax.set_title(f'{var}',fontsize=12)
            ax.axis('off')
       
        plt.tight_layout()
                
        # Save the plot to the specified file path
        plt.savefig(
            os.path.join(f"{self.config.output_dir}" , f"{self.config.run_name}", "samples", file_name),
            bbox_inches="tight", dpi = 1500
        )
        plt.close()
        

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple
from torch import Size
import torch

from mfai.torch.padding import pad_batch, undo_padding

class ModelType(Enum):
    """
    Enum to classify the models depending on their architecture family.
    Having the model expose this as an attributee facilitates top level code:
    reshaping input tensors, iterating only on a subset of models, etc.
    """

    GRAPH = 1
    CONVOLUTIONAL = 2
    VISION_TRANSFORMER = 3
    LLM = 4
    MULTIMODAL_LLM = 5


class ModelABC(ABC):
    # concrete subclasses shoudl set register to True
    # to be included in the registry of available models.
    register: bool = False

    @property
    @abstractmethod
    def onnx_supported(self) -> bool:
        """
        Indicates if our model supports onnx export.
        """

    @property
    @abstractmethod
    def settings_kls(self):
        """
        Returns the settings class for this model.
        """

    @property
    @abstractmethod
    def supported_num_spatial_dims(self) -> Tuple[int, ...]:
        """
        Returns the number of input spatial dimensions supported by the model.
        A 2d vision model supporting (H, W) should return (2,).
        A model supporting both 2d and 3d inputs (by settings) should return (2, 3).
        Once instanciated the model will be in 2d OR 3d mode.
        """

    @property
    @abstractmethod
    def settings(self) -> Any:
        """
        Returns the settings instance used to configure for this model.
        """

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """
        Returns the model type.
        """

    @property
    @abstractmethod
    def num_spatial_dims(self) -> int:
        """
        Returns the number of spatial dimensions of the instanciated model.
        """

    @property
    @abstractmethod
    def features_last(self) -> bool:
        """
        Indicates if the features are the last dimension in the input/output tensors.
        Conv and ViT typically have features as the second dimension (Batch, Features, ...)
        versus GNNs for which features are the last dimension (Batch, ..., Features)
        """

    @property
    def features_second(self) -> bool:
        return not self.features_last

    def check_required_attributes(self):
        # we check that the model has defined the following attributes.
        # this must be called at the end of the __init__ of each subclass.
        required_attrs = ["in_channels", "out_channels", "input_shape"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute : {attr}")


class AutoPaddingModel(ABC):
    @abstractmethod
    def validate_input_shape(self, input_shape: Size) -> Tuple[bool | Size]:
        """ Given an input shape, verifies whether the inputs fit with the 
            calling model's specifications. 

        Args:
            input_shape (Size): The shape of the input data, excluding any batch dimension and channel dimension.  
                                For example, for a batch of 2D tensors of shape [B,C,W,H], [W,H] should be passed.
                                For 3D data instead of shape [B,C,W,H,D], instead, [W,H,D] should be passed. 

        Returns:
            Tuple[bool, Size]: Returns a tuple where the first element is a boolean signaling whether the given input shape 
                                already fits the model's requirements. If that value is False, the second element contains the closest 
                                shape that fits the model, otherwise it will be None.
        """
        
    def _maybe_padding(self, data_tensor: torch.Tensor)-> Tuple[torch.Tensor, Optional[torch.Size]]:
        """ Performs an optional padding to ensure that the data tensor can be fed 
            to the underlying model. Padding will happen if if 
            autopadding was enabled via the settings.

        Args:
            data_tensor (torch.Tensor): the input data to be potentially padded. 

        Returns:
            Tuple[torch.Tensor, Optional[torch.Size]]: the padded tensor, where the original data is found in the center, 
            and the old size if padding was possible. If not possible or the shape is already fine, 
            the data is returned untouched and the second return value will be none. 
        """
        if not self._settings.autopad_enabled:
            return data_tensor, None
        
        old_shape = data_tensor.shape[-len(self.input_shape):]
        valid_shape, new_shape = self.validate_input_shape(data_tensor.shape[-len(self.input_shape):])
        if not valid_shape:
            return pad_batch(batch=data_tensor, new_shape=new_shape, pad_value=0), old_shape
        return data_tensor, None
    
    def _maybe_unpadding(self, data_tensor: torch.Tensor, old_shape: torch.Size)-> torch.Tensor:
        """Potentially removes the padding previously added to the given tensor. This action 
           is only carried out if autopadding was enabled via the settings.

        Args:
            data_tensor (torch.Tensor): The data tensor from which padding is to be removed. 
            old_shape (torch.Size): The previous shape of the data tensor. It can either be 
            [W,H] or [W,H,D] for 2D and 3D data respectively. old_shape is returned by self._maybe_padding.

        Returns:
            torch.Tensor: The data tensor with the padding removed, if possible.
        """
        if self._settings.autopad_enabled and old_shape is not None:
            return undo_padding(data_tensor, old_shape=old_shape)
        return data_tensor