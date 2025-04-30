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

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from ddpm.normalize import var_dict, SpecialNormalize
from utils.utils import filter_dates, filter_lead_times

from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

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
        condition_tensor, condition_tensor_denorm = self.get_conditioning_members(ensemble_df, idx, return_denorm=True)

        # Get the date, lt, and member id of the current member
        row = ensemble_df.iloc[0] if not ensemble_df.empty else {"Date": "", "LeadTime": 0, "Member": ""}
        date = str(pd.to_datetime(row["Date"]).strftime('%Y-%m-%d'))
        lt = row["LeadTime"]
        
        if self.config.guiding_col is not None:
            member = row[self.config.guiding_col]
        else:
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
        return {"id_in_csv": idx, "img": sample, "img_id": sample_id, "condition_tensor": condition_tensor, "condition_tensor_denorm" : condition_tensor_denorm,"member_id": member, "date": date, "leadtime": lt}

    def get_conditioning_members(self, ensemble_df, idx,return_denorm=False):
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

        if not return_denorm:
            # Enables the bootstrap_conditions sampling : the same set of condtionning members is used to generate the n_sampling_conditioning_sets samples
            if self.config.bootstrap_conditions:
                condition = torch.stack([
                    self.df_to_torch(ensemble_df_without_target, self.config.n_conditions) for _ in range(self.n_conditioning_sets)
                ])
                return condition

            condition = self.df_to_torch(ensemble_df_without_target, self.config.n_conditions)
            condition = condition.unsqueeze(0).expand_as(torch.zeros(self.n_conditioning_sets, self.config.n_var*self.config.n_conditions, self.height_dim, self.width_dim))
            return condition
        else :
            # Enables the bootstrap_conditions sampling : the same set of condtionning members is used to generate the n_sampling_conditioning_sets samples
            if self.config.bootstrap_conditions:
                norm_condition = torch.stack([
                    self.df_to_torch(ensemble_df_without_target, self.config.n_conditions,return_denorm=return_denorm)[0] for _ in range(self.n_conditioning_sets)
                ])
                condition = torch.stack([
                    self.df_to_torch(ensemble_df_without_target, self.config.n_conditions,return_denorm=return_denorm)[1] for _ in range(self.n_conditioning_sets)
                ])
                return norm_condition, condition

            norm_condition, condition = self.df_to_torch(ensemble_df_without_target, self.config.n_conditions, return_denorm=return_denorm)
            condition = condition.unsqueeze(0).expand_as(torch.zeros(self.n_conditioning_sets, self.config.n_var*self.config.n_conditions, self.height_dim, self.width_dim))
            norm_condition = norm_condition.unsqueeze(0).expand_as(torch.zeros(self.n_conditioning_sets, self.config.n_var*self.config.n_conditions, self.height_dim, self.width_dim))
            return norm_condition, condition
    
    def df_to_torch(self, ens_df, n_cond, return_denorm=False):
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
        if not return_denorm:
            condition_tensor = torch.cat(
                [self.file_to_torch(name, return_denorm=return_denorm) for name in selected_members] + [torch.empty((0, self.height_dim, self.width_dim))], dim=0 # torch.empty in case of n_condition = 0
            )
            return condition_tensor
        else :
            norm_condition_tensor = torch.cat(
                [self.file_to_torch(name, return_denorm=return_denorm)[0] for name in selected_members] + [torch.empty((0, self.height_dim, self.width_dim))], dim=0 # torch.empty in case of n_condition = 0
            )
            condition_tensor = torch.cat(
                [self.file_to_torch(name, return_denorm=return_denorm)[1] for name in selected_members] + [torch.empty((0, self.height_dim, self.width_dim))], dim=0 # torch.empty in case of n_condition = 0
            )
                
            return norm_condition_tensor, condition_tensor
        
    def file_to_torch(self, file_name, return_denorm=False):
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
        norm_sample = sample.transpose((1, 2, 0))
        norm_sample = self.transform(norm_sample)
        if return_denorm:
            return norm_sample, torch.tensor(sample)
        else :
            return norm_sample

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
