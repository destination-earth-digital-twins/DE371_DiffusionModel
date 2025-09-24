#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import torch
import argparse
import os
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
import utils.utils as utils
from ast import literal_eval as make_tuple
import glob

torch.manual_seed(42) #reproducibility of runs
def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--gen_data_dir', type = str,default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit/sampling_sdedit_ddpm/sampling_10steps/samples/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit/sampling_sdedit_ddpm/sampling_10steps/unbiased_samples/')

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default='/project/home/p200177/DE_371/datasets/big_domain_stats_and_csv/big_domain_optim_u_v_t2m/big_domain_optim_val_u_v_t2m.csv')
    parser.add_argument("--date_start", type=str, default = "2020-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45])
    parser.add_argument("--var_indices", type=str2intlist, default=[0,1,2,3])
    parser.add_argument("--crop", type=str2intlist, default=[7,711,0,1120])
    params = parser.parse_args()

    # create output and pack directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    ################## loading dates and file names ##
    df = pd.read_csv(params.dates_file) # use directly full csv path
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()

    #################### main loop ##################
    for date_ in list_dates:
        datename = date_.strftime('%Y-%m-%d')
        for lt in params.leadtimes:
            # save_path = f'{params.output_dir}4var_fake_ensemble_{datename}_{lt}.npy'
            save_path = f'{params.output_dir}genFsemble_{datename}_{lt}_1000_16.npy'
            if not os.path.isfile(save_path):
                # print(f'Unbiasing 4var_fake_ensemble_{datename}_{lt}.npy.')
                print(f'Unbiasing genFsemble_{datename}_{lt}.npy.')
                # Loading AROME ensemble   
                df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
                Nb = len(df0)
                Ens_AROME = np.zeros((Nb,) + tuple((3,crop[1] - crop[0],crop[3] - crop[2])))
                print("Ens AROME shape", Ens_AROME.shape)
                for i,s in enumerate(df0['Name']):
                    file_path = os.path.join(params.real_data_dir,s)
                    sn = np.load(file_path)[:,:,params.var_indices].astype(np.float32)
                    crop = params.crop
                    if np.sum(crop)>512: #means we are using big domain datas
                        sn = sn.transpose(2,0,1)
                        sn = sn[:,crop[0]:crop[1],crop[2]:crop[3]]
                    Ens_AROME[i] = sn
                Ens_AROME = Ens_AROME[:,:,:,:]
                # Loading Generated ensemble
                # Ens_Gen = np.load(f'{params.gen_data_dir}4var_fake_ensemble_{datename}_{lt}.npy')
                Ens_Gen = np.load(f'{params.gen_data_dir}/4var_fake_ensemble_{datename}_{lt}.npy')[:,1:,:,:] #not debiaising rr because i am only using u v t
                
                # Creating unbiased new ensemble
                Ens_Gen_unbiased = Ens_Gen + np.expand_dims(Ens_AROME.mean(axis=0),0) - np.expand_dims(Ens_Gen.mean(axis=0),0)
                np.save(save_path, Ens_Gen_unbiased)
            else :
                # print(f'4var_fake_ensemble_{datename}_{lt}.npy already exists, Switching to next sample.')
                print(f'Unbiasing genFsemble_{datename}_{lt}_1000_16.npy already exists, Switching to next sample.')

    print('Unbiasing done.')










