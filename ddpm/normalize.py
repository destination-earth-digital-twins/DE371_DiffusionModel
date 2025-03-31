import torch
import numpy as np
import os
from tabulate import tabulate

import ddpm.special_transforms as special_transforms
from utils.config import DataTransformConfig
from utils.distributed import is_main_gpu
################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing

var_dict = {
    "rr": 0,
    "u": 1,
    "v": 2,
    "t2m": 3,
    "orog": 4,
    "z500": 5,
    "t850": 6,
    "tpw850": 7,
}

class SpecialNormalize(object):
    def __init__(self, config):
        self.config = config
        
        self.data_transforms = {}
        for var in self.config.var_indexes:
            tfconfig = getattr(self.config,var,"default")
            self.data_transforms[var] = DataTransformConfig(
                tfconfig
            )
            # if data must undergo special transform (e.g log), we retrieve it there
            special_transform = self.data_transforms[var].special_transform
            self.data_transforms[var]._update_from_dict(
                {"special_transform": getattr(special_transforms, str(special_transform), None)})
        
        ### shaping normalization constants
        offset = np.load(os.path.join(self.config.stat_folder,self.config.offset_file))[self.config.VI].astype(np.float32)
        scale = np.load(os.path.join(self.config.stat_folder,self.config.scale_file))[self.config.VI].astype(np.float32)
        
        self.offset = torch.from_numpy(offset).view(-1, 1, 1)
        self.scale = (1.0 / 0.95) * torch.from_numpy(scale).view(-1, 1, 1)
        
        if is_main_gpu():
            to_table = zip(self.config.var_indexes,self.offset,self.scale)
            
            table = tabulate(
                to_table,
                headers=[
                    "Variable",
                    "Offset Constants",
                    "Scale Constants"
                ],
                tablefmt="simple_outline",
            )
            
            print(table)
        
    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Input sample should be a torch tensor. Got {type(sample)}."
            )
        if sample.ndim < 3:
            raise ValueError(
                f"Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}."
            )

        for var in self.data_transforms:
            if self.data_transforms[var].special_transform is not None:
                sample[var_dict[var]] = self.data_transforms[var].special_transform.direct(sample[var_dict[var]])
        
        sample = (sample - self.offset) / self.scale
        
        return sample

    def denorm(self, sample):
        """
        revert the __call__ function to produce "physical space" samples
        sample can be batched, and should be either of shape N x C X H x W or C x H x W
        """
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Input sample should be a torch tensor. Got {type(sample)}."
            )
        if sample.ndim < 3:
            raise ValueError(
                f"Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}."
            )

        ### non-batched ops
        elif sample.ndim == 3:
            for var in self.data_transforms:
                if self.data_transforms[var]["special_transform"] is not None:
                    sample[var_dict[var]] = self.data_transforms[var]["special_transform"].direct(sample[var_dict[var]])
            
        ### batched ops
        else:
            for var in self.data_transforms:
                if self.data_transforms[var]["special_transform"] is not None:
                    sample[:,var_dict[var]] = self.data_transforms[var]["special_transform"].reverse(sample[:,var_dict[var]])

        ### reverting normalizations
        sample = sample * self.value_sup + self.value_inf
        
        return sample