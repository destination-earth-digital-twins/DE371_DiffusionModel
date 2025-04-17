#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@authors: gandonb, rabaultj, brochetc


DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples

"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ddpm.normalize import var_dict, SpecialNormalize
from utils.utils import filter_dates, filter_lead_times

from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import math

from mfai.torch.namedtensor import NamedTensor
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


################
class ISDataset(Dataset):
    def __init__(self, config, path, csv_file):
        """
        Initialize the ISDataset.
        Args:
            config: Configuration settings.
            path (str): Directory path containing data.
            csv_file (str): CSV file containing sample labels (file names, dates, lead times, members id) 
        """
        self.data_dir = path
        self.labels = pd.read_csv( csv_file, index_col=False)
        self.config = config
        self.labels = filter_dates(self.labels, self.config.date_start, self.config.date_stop)
        self.labels = filter_lead_times(self.labels, self.config.leadtimes)
        if "Unnamed: 0" in self.labels:
            self.labels = self.labels.drop("Unnamed: 0", axis=1)


        self.CI = config.crop
        self.config.VI = [var_dict[var] for var in config.var_indexes]
        self.ensembles = None
        # if sampling : Proceed sampling n_sampling_conditioning_sets times, -> the final ensemble contains 16*n_sampling_conditioning_sets members
        # if training : Prepare only 1 conditioning set
        self.n_conditioning_sets = self.config.n_sampling_conditioning_sets if self.config.mode == "Sample" else self.config.n_training_conditioning_sets

        # shape of the images 
        self.height_dim, self.width_dim = self.config.crop[1] - self.config.crop[0], self.config.crop[3] - self.config.crop[2]

        # Group labels by guiding column if specified
        if self.config.guiding_col is not None:
            self.ensembles = self.labels.groupby([self.config.guiding_col])
        # Add positional encoding
        self.transform = transforms.Compose(

            [
                transforms.ToTensor(),
                SpecialNormalize(self.config),
            ]
        )
        
        self.labels = self.labels.reset_index(drop=True)

    def inversion_transforms(self):
        """
        Returns function to revert normalisation and special transforms for generated samples.
        """
        return SpecialNormalize(self.config).denorm

    def __len__(self):
        """
        Get the length of the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary containing 'img' (sample), 'img_id' (sample ID), and 'condition' (conditional used for training), and 'condition_sample' (condition used for sampling).
        """
        file_name = self.labels.iloc[idx, 0] # Name of the current sample in the dataset
        sample = self.file_to_torch(file_name) # target
        mean_cond = self.config.mean_conditionning # Use the mean as a condition ?
        var_cond = self.config.var_conditionning # Use the var as a condition ?
        mean_var_dir = self.config.mean_var_dir # Dir containing the pre-computed mean and var values

        # Get the ensemble df
        ensemble_id = self.labels.at[idx, self.config.guiding_col] # Get the ensemble id of the current member
        ensemble_df = self.ensembles.get_group((ensemble_id,)) # Group every membre from this ensemble in a df

        # Build the tensors for the sampling and the training
        condition_tensor = self.get_conditioning_members(ensemble_df, idx)

        # Get the date, lt, and member id of the current member
        row = ensemble_df.iloc[0] if not ensemble_df.empty else {"Date": "", "LeadTime": 0, "Member": ""}
        date = str(pd.to_datetime(row["Date"]).strftime('%Y-%m-%d'))
        lt = row["LeadTime"]
        member = row["Member"]

        # Using the mean and/or the var of the ensemble as additionnal conditions
        if mean_cond or var_cond:
            mean_var_file = torch.from_numpy(np.load(os.path.join(mean_var_dir, date + "_" + str(lt) + ".npy")))
            if self.config.n_var == 3:
                mean_var_file = mean_var_file[:, 1:, :, :] # Pop the rr channel
            if mean_cond:
                mean = mean_var_file[0].unsqueeze(0).expand(self.n_conditioning_sets, -1, -1, -1)
                condition_tensor = torch.cat([condition_tensor, mean], dim=1)
            if var_cond:
                var = mean_var_file[1].unsqueeze(0).expand(self.n_conditioning_sets, -1, -1, -1)
                condition_tensor = torch.cat([condition_tensor, var], dim=1)


        sample_id = re.search(r"\d+", file_name).group()
        return {"id_in_csv": idx, "img": sample, "img_id": sample_id, "condition_tensor": condition_tensor, "member_id": member, "date": date, "leadtime": lt}

    def get_conditioning_members(self, ensemble_df, idx):
        """
            Loads the conditioning members for the training and the sampling, stacks them
            and returns the result

            Args:
                ensemble_df (df): the sub df contatining the considered ensemble
                idx (int): ID in the __getitem__.

            Returns:
                torch.Tensor: The resulting tensor of shape [n_sampling_conditioning_sets*n_conditions*n_var, self.height_dim, self.width_dim] 
        """
        if self.config.n_conditions > self.config.n_members_dataset:
            raise ValueError(
                f"The number of conditioning members must not exceed the number of members in the dataset. Got {self.config.n_conditions} conditioning members and {self.config.n_members_dataset} members in the dataset."
            )

        # Remove the target from the possible conditions used for training
        ensemble_df_without_target = ensemble_df[ensemble_df['Name'] != self.labels.iloc[idx, 0]]


        # Enables the bootstrap_conditions sampling : the same set of condtionning members is used to generate the n_sampling_conditioning_sets samples
        if self.config.bootstrap_conditions:
            condition = torch.stack([
                self.df_to_torch(ensemble_df_without_target, self.config.n_conditions) for _ in range(self.n_conditioning_sets)
            ])
            return condition

        condition = self.df_to_torch(ensemble_df_without_target, self.config.n_conditions)
        condition = condition.unsqueeze(0).expand_as(torch.zeros(self.n_conditioning_sets, self.config.n_var*self.config.n_conditions, self.height_dim, self.width_dim))
        return condition
    
    def df_to_torch(self, ens_df, n_cond):
        """
            sample members from the ensemble df.
            loads the members
            returns the concatenated members
            Args:
                ens_df (df): df containing an ensemble caracteristics.
                n_cond (int): number of member to sample
            Returns:
                torch.Tensor: Torch tensor of shape [n_conditions*n_var, self.height_dim, self.width_dim] containing the concatenated members.
        """
        selected_members = ens_df.sample(n=n_cond)['Name'].values
        condition_tensor = torch.cat(
            [self.file_to_torch(name) for name in selected_members] + [torch.empty((0, self.height_dim, self.width_dim))], dim=0 # torch.empty in case of n_condition = 0
        )
        return condition_tensor
        
    def file_to_torch(self, file_name):
        """
        Convert a file to a torch tensor.
        Args:
            file_name (str or list): Name of the file or list of file names.
        Returns:
            torch.Tensor: Torch tensor representing the sample.
        """
        if type(file_name) == list:
            file_name = file_name[0]
        sample_path = os.path.join(self.data_dir, file_name)
        sample = np.float32(np.load(sample_path + ".npy"))[
            self.config.VI, self.CI[0] : self.CI[1], self.CI[2] : self.CI[3]
        ]
        sample = sample.transpose((1, 2, 0))
        sample = self.transform(sample)
        return sample


class MultiOptionNormalize(object):
    def __init__(self, value_sup, value_inf, dataset_config, config):
        self.value_sup = value_sup
        self.value_inf = value_inf
        self.dataset_config = dataset_config
        self.config = config
        self.gaussian_std = self.dataset_config.rr_transform["gaussian_std"]
        ### setting gaussian noise conditions
        if self.gaussian_std:
            for _ in range( self.dataset_config.rr_transform["log_transform_iteration"]):
                self.gaussian_std = np.log(1 + self.gaussian_std)

        if np.ndim(self.value_sup) > 1:
            ### preparing blur of constant normalization fields (for rr exclusively)torch.abs(sample[:,var_dict['rr']].sub_(sample[:,var_dict['rr']] * mask_no_rr))
            if (
                self.dataset_config.normalization["for_rr"]["blur_iteration"]
                > 0
            ):
                gaussian_filter = (
                    np.float32(
                        [
                            [1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1],
                        ]
                    )
                    / 256.0
                )
                for _ in range(
                    self.dataset_config.normalization["for_rr"][
                        "blur_iteration"
                    ]
                ):
                    self.value_sup[var_dict["rr"]] = scipy.ndimage.convolve(
                        self.value_sup[var_dict["rr"]],
                        gaussian_filter,
                        mode="mirror",
                    )
            self.value_inf = torch.from_numpy(self.value_inf)
            self.value_sup = torch.from_numpy(self.value_sup)
        else:
            ### shaping normalization constants
            self.value_inf = torch.from_numpy(self.value_inf).view(-1, 1, 1)
            self.value_sup = torch.from_numpy(self.value_sup).view(-1, 1, 1)

    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Input sample should be a torch tensor. Got {type(sample)}."
            )
        if sample.ndim < 3:
            raise ValueError(
                f"Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}."
            )
        ### transforming rain rates to logits (iterative transforms)
        for _ in range(self.dataset_config.rr_transform["log_transform_iteration"]):
            sample[var_dict["rr"]] = torch.log(1 + sample[var_dict["rr"]])
        ### randomly symmetrizing rain rates around 0 (50% of rain rates are negative)
        if (self.dataset_config.rr_transform["symetrization"] and np.random.random() <= 0.5):
            sample[var_dict["rr"]] = -sample[var_dict["rr"]]
        ### adding random noise (AT RUNTIME) to rain rates below a certain threshold
        if self.gaussian_std != 0:
            gaussian_std_map = (
                np.random.choice(
                    [-1, 1],
                    size=(self.config.image_size, self.config.image_size),
                )
                * self.gaussian_std
            )
            gaussian_noise = np.mod(
                np.random.normal(
                    0,
                    self.gaussian_std,
                    size=(self.config.image_size, self.config.image_size),
                ),
                self.gaussian_std,
            )
            mask_no_rr = sample[var_dict["rr"]].numpy() <= self.gaussian_std
            sample[var_dict["rr"]] = sample[var_dict["rr"]].add_(
                from_numpy(gaussian_noise * mask_no_rr)
            )
        ### performing different types of normalization (centering around mean or capping min-max/quantiles)
        if self.dataset_config.normalization["func"] == "mean":
            sample = (sample - self.value_inf) / self.value_sup
        elif self.dataset_config.normalization["func"] in ["minmax", "quant"]:
            sample = -1 + 2 * ((sample - self.value_inf) / (self.value_sup - self.value_inf))
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
            # reverting log transforms
            for _ in range(
                self.dataset_config.rr_transform["log_transform_iteration"]
            ):
                sample[var_dict["rr"]] = torch.exp(sample[var_dict["rr"]]) - 1
            # reverting symmetrization
            if self.dataset_config.rr_transform["symetrization"]:
                sample[var_dict["rr"]] = -sample[var_dict["rr"]]
            # reverting gaussian noise by setting below threshold to 0
            if self.gaussian_std != 0:
                mask_no_rr = (
                    sample[var_dict["rr"]].numpy() <= self.gaussian_std
                )
                sample[var_dict["rr"]] = torch.abs(
                    sample[var_dict["rr"]].sub_(
                        sample[var_dict["rr"]] * mask_no_rr
                    )
                )
        ### batched ops
        else:
            # reverting log transforms
            for _ in range(
                self.dataset_config.rr_transform["log_transform_iteration"]
            ):
                sample[:, var_dict["rr"]] = (
                    torch.exp(sample[:, var_dict["rr"]]) - 1.0
                )
            # reverting symmetrization
            if self.dataset_config.rr_transform["symetrization"]:
                sample[:, var_dict["rr"]] = torch.abs(
                    sample[:, var_dict["rr"]]
                )
            # reverting gaussian noise by setting below threshold to 0
            if self.gaussian_std != 0:
                mask_no_rr = (
                    sample[:, var_dict["rr"]].numpy() <= self.gaussian_std
                )
                sample[:, var_dict["rr"]] = torch.abs(
                    sample[:, var_dict["rr"]].sub_(
                        sample[:, var_dict["rr"]] * mask_no_rr
                    )
                )
        ### reverting normalizations
        if self.dataset_config.normalization["func"] == "mean":
            sample = sample * self.value_sup + self.value_inf
        elif self.dataset_config.normalization["func"] in ["minmax", "quant"]:
            sample = self.value_inf + 0.5 * (
                self.value_sup - self.value_inf
            ) * ((sample + 1.0))
        return sample


class rrISDataset(ISDataset):
    def __init__(self, config, path, csv_file, add_coords=False):
        """
        Initialize the rrISDataset.
            This subclasses ISDataset and overwrites prepare_transformations / init_normalization methods
            because there are many ways we can desire to calibrate the rain
        Args:
            config: Configuration settings.
            path (str): Directory path containing data.
            csv_file (str): CSV file containing labels and information.
            add_coords (bool): Whether to add positional encoding.

        """
        super().__init__(config, path, csv_file, add_coords=False)

    def prepare_tranformations(self):
        # transformations = []
        normalization = self.dataset_config.normalization["func"]
        if normalization != "None":
            if self.dataset_config.rr_transform["symetrization"]:
                if normalization == "means":
                    # mean of rr is 0
                    self.value_inf[var_dict["rr"]] = np.zeros_like(
                        self.value_inf[var_dict["rr"]]
                    )
                elif normalization == "minmax":
                    # min of 'negative rain' is -max
                    self.value_inf[var_dict["rr"]] = -self.value_sup[
                        var_dict["rr"]
                    ]
                    
        transformations = transforms.Compose(
                [
                    transforms.ToTensor(),
                    MultiOptionNormalize(
                        self.value_sup,
                        self.value_inf,
                        self.dataset_config,
                        self.config,
                    ),
                ]
        )
        return transformations

    def inversion_transforms(self):
        detransform_func = MultiOptionNormalize(
            self.value_sup, self.value_inf, self.dataset_config, self.config
        ).denorm
        return detransform_func

    def init_normalization(self):
        normalization_func = self.dataset_config.normalization["func"]
        if normalization_func == "mean":
            stds, means = self.load_stat_files(
                normalization_func, "std", "mean"
            )
            return stds[self.VI] * 1.0 / 0.95, means[self.VI]

        if normalization_func == "minmax":
            maxs, mins = self.load_stat_files(normalization_func, "max", "min")
            return maxs[self.VI], mins[self.VI]

        if normalization_func == "quant":
            q99, q01 = self.load_stat_files(normalization_func, "q99", "q01")
            return q99[self.VI], q01[self.VI]

        print("No normalization set")
        return None, None

    def load_stat_files(self, normalization_func, str_sup, str_inf):
        # Your normalization files should be name "[var]_[stat_version]_log_log_..._[ppx].npy" with:
        #   var: 'min', 'max' or 'mean', 'std' or 'Q01', 'Q99' or 'Q10', Q90'
        #   stat_version: an identifier for the stat file
        #   log_log...: 'log_' will be repeated log_transform_iteration times
        #   ppx: if the stats are per pixel, _ppx must be added at the end of the file
        print(f"Normalization set to {normalization_func}")
        norm_vars = []
        for name in (str_sup, str_inf):
            filename = f"{name}_{self.dataset_config.stat_version}"
            filename += (
                "_log"
                * self.dataset_config.rr_transform["log_transform_iteration"]
            )

            if self.dataset_config.normalization["per_pixel"]:
                filename += "_ppx"

            filename += ".npy"

            try:
                path = os.path.join(
                    self.data_dir, self.dataset_config.stat_folder, filename
                )
                norm_var = np.load(path).astype("float32")
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"{name} file was not found at this location: {path}"
                )
            norm_vars.append(norm_var)
        return norm_vars

class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        self.num_samples = len(self.dataset) // self.num_replicas
        if not drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples += 1

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        start = self.rank * self.num_samples
        end = min(start + self.num_samples, len(self.dataset))
        indices = list(range(start, end))
        return iter(indices)

    def __len__(self):
        return self.num_samples
def convert_to_namedtensor(batch):
    
    n_conditions = int(batch['condition_tensor'].shape[2]/batch['img'].shape[1])
    batch['img'] = NamedTensor(
                            batch['img'],
                            names=["members","features", "lat", "lon"],
                            feature_names=["u", "v", "t2m"]
    )

    
    batch['condition_tensor'] = NamedTensor(
                            batch['condition_tensor'],
                            names=["members","time","features", "lat", "lon"],
                            feature_names=n_conditions*["u", "v", "t2m"]
    )
    
    return batch

class NamedTensorDataLoader:
    def __init__(self,config, dataloader):
        self.dataloader = dataloader
        self.config = config

    def __iter__(self):
        for batch in self.dataloader:
            yield convert_to_namedtensor(batch)  

    def __len__(self):
        return len(self.dataloader)
    
    def __getattr__(self,name):
        return getattr(self.dataloader,name)    

