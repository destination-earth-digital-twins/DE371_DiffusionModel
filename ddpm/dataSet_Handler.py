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
        file_name = self.labels.iloc[idx, 0]
        sample = self.file_to_torch(file_name)
        # Proceed sampling n_ensemble times, -> the final ensemble contains 16*n_ensemble members
        n_sampling = self.config.n_ensemble
        # Number of conditionning members
        n_conditions = self.config.n_conditions
        # Number of channels. e.g. to sample u, v, t2m, n_var=3
        n_var = self.config.v_i
        mean_cond = self.config.mean_conditionning
        var_cond = self.config.var_conditionning
        mean_var_dir = self.config.mean_var_dir

        # Get the ensemble df
        ensemble_id = self.labels.at[idx, self.config.guiding_col]
        group = self.ensembles.get_group((ensemble_id,))

        # Prepare n_sampling sets of random conditionning members
        seeds_list = []
        for i in range(n_sampling):
            # Get conditional sample if ensembles are specified
            if self.ensembles is not None and n_conditions > 0:
                group_ensemble = group[group['Name'] != self.labels.iloc[idx, 0]]

                # Random conditionning members for the training
                rows = group_ensemble.sample(n=n_conditions)['Name'].values
                conditions = [self.file_to_torch(name) for name in rows]
                condition_train = torch.stack(conditions, dim=0).reshape(n_conditions * n_var, 256, 256)

                # Random conditionning members for the sampling
                rows_sampling = group_ensemble.sample(n=n_conditions - 1)['Name'].values if n_conditions > 1 else []
                conditions_sample = [self.file_to_torch(file_name)] + [self.file_to_torch(name) for name in rows_sampling]
                condition_sample = torch.stack(conditions_sample, dim=0).reshape(n_conditions * n_var, 256, 256)
            
            # Allow the sampling with 0 conditionning member when using the mean and/or the var of the ensemble as conditions
            elif self.ensembles is not None and n_conditions == 0:
                condition_train = torch.empty((0, 256, 256))
                condition_sample = torch.empty((0, 256, 256))

            else:
                condition_train = torch.empty(0)
                condition_sample = torch.empty(0)

            seeds_list.append(condition_sample)
        seeds_tensor = torch.stack(seeds_list, dim=0)

        # Enables the "StyleGAN-like sampling" : the same sets of condtionning members are used to generate the n_ensemble samples
        if self.config.stylegan_like_sampling:
            seeds_tensor = seeds_tensor[0].expand_as(seeds_tensor)

        row = group.iloc[0] if not group.empty else {"Date": "", "LeadTime": 0, "Member": ""}
        date = str(pd.to_datetime(row["Date"]).strftime('%Y-%m-%d'))
        lt = row["LeadTime"]
        member = row["Member"]

        # print("####### shape seeds before mean var ", seeds_tensor.shape)
        
        # Using the mean and/or the var of the ensemble as additionnal conditions
        if mean_cond or var_cond:
            mean_var_file = torch.from_numpy(np.load(os.path.join(mean_var_dir, date + "_" + str(lt) + ".npy")))
            if self.config.v_i == 3:
                mean_var_file = mean_var_file[:, 1:, :, :] # Pop the rr channel
            if mean_cond:
                mean = mean_var_file[0]
                condition_train = torch.cat([condition_train, mean], dim=0)
                seeds_tensor = torch.cat([seeds_tensor, mean.unsqueeze(0).expand(seeds_tensor.shape[0], -1, -1, -1)], dim=1)
            if var_cond:
                var = mean_var_file[1]
                condition_train = torch.cat([condition_train, var], dim=0)
                seeds_tensor = torch.cat([seeds_tensor, var.unsqueeze(0).expand(seeds_tensor.shape[0], -1, -1, -1)], dim=1)

        sample_id = re.search(r"\d+", file_name).group()
        return {"id_in_csv": idx, "img": sample, "img_id": sample_id, "condition": condition_train, "condition_sample": seeds_tensor, "member_id": member, "date": date, "leadtime": lt}

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
