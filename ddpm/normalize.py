import torch
import numpy as np
import os
from tabulate import tabulate

import ddpm.special_transforms as special_transforms
from utils.distributed import is_main_gpu, get_rank

################ reference dictionary to know how to index variables in base numpy arrays
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
    """
    Class to Compose special transformations applied to some specific variables and classical normalization.
    Draws indication from entries of config which are named after variables.
    """
    def __init__(self, config):
        self.config = config
        
        # gathering variable-specific special transformations
        self.data_transforms = {}
        for var in self.config.var_indexes:
            special_transform = getattr(self.config,f"{var}_transform",None)
            self.data_transforms[var] = getattr(special_transforms, str(special_transform), None)
        # loading shaping normalization constants
        # the transformation involves data <- 0.95 * (data - offset) / scale and is broadcasted to all selected variables
        offset = np.load(os.path.join(self.config.stat_folder,self.config.offset_file))[self.config.VI].astype(np.float32)
        scale = np.load(os.path.join(self.config.stat_folder,self.config.scale_file))[self.config.VI].astype(np.float32)
        print("la scale est scale",scale)
        
        self.offset = torch.from_numpy(offset).view(-1, 1, 1)
        self.scale = (1.0 / 0.95) * torch.from_numpy(scale).view(-1, 1, 1)
        
        # logging used constants
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
        """
        Casts all variables to normalized space, by first applying special transforms and then normalizing.
        
        sample: torch.tensor, of shape (C,H,W)
        """
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Input sample should be a torch tensor. Got {type(sample)}."
            )
        if sample.ndim != 3 and sample.ndim != 4:
            raise ValueError(
                f"Expected sample to be a tensor image of size (C, H, W) or (C, H, W, M). Got tensor.size() = {sample.size()}."
            )

        if sample.ndim == 4:
            samples = []
            if sample.size()[-1] !=16:
                raise ValueError(
                    f"Expected last dim of sample.size() to be 16 but is {sample.size()[-1]}"
                )
            for var in self.data_transforms:
                if self.data_transforms[var] is not None:
                    sample[var_dict[var]] = self.data_transforms[var].direct(sample[var_dict[var]])
                    
            for member in range(sample.size()[-1]):
                
                sample_member = sample[:,:,:,member]
                sample_member[sample_member==9999]=torch.nan 
                
                sample_member = (sample_member - self.offset) / self.scale
                sample_member = torch.nan_to_num(sample_member,nan = 9999.)
                samples.append(sample_member)
            
            sample = torch.stack(samples,axis=-1)
            # print("sample final shape",sample.shape)
            return sample
                    
        else :    
            for var in self.data_transforms:
                if self.data_transforms[var] is not None:
                    sample[var_dict[var]] = self.data_transforms[var].direct(sample[var_dict[var]])
            
            sample[sample==9999]=torch.nan 

            sample = (sample - self.offset) / self.scale
            sample = torch.nan_to_num(sample,nan = 9999.)

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
                if self.data_transforms[var] is not None:
                    sample[var_dict[var]] = self.data_transforms[var].reverse(sample[var_dict[var]])
            
        ### batched ops
        else:
            for var in self.data_transforms:
                if self.data_transforms[var] is not None:
                    sample[:,var_dict[var]] = self.data_transforms[var].reverse(sample[:,var_dict[var]])

        ### reverting normalizations
        sample = sample * self.scale.to(sample.device) + self.offset.to(sample.device)
        
        return sample