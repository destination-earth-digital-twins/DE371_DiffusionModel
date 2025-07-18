# Function to plot the perturbated samples from the non temporal gan

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch 
import argparse
import utils.utils as utils
import os 
import pandas as pd
import scipy
from math import ceil

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    
    ########################### Directories ###########################
    
    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,  default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default ='/project/scratch/p200177/DE_371/victorsanchez/results/perturbation/coherence_temporelle_gan_classique/')
    parser.add_argument('--gen_sample_dir',type = str, default ="/project/home/p200177/DE_371/experiments_WP1/inversion_process_analysis/final_inversion_on_test_set/perceptual_exp45/perturbation/112_stochastic_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_-1_16_/samples/")
    parser.add_argument("--Shape", type=tuple, default=(3,256,256), help='size of the samples')
    # Dataset information
    parser.add_argument("--normalization", type=str, default="", choices=["minmax", "meanmax", ""])
    parser.add_argument('--max_file', type=str, default='MaxNew_4_var.npy') # use 'MaxNew_4_var.npy' if AROME data # max_rr_log.npy
    parser.add_argument('--mean_file', type=str, default='Mean_4_var.npy') # not used if minmax normalization
    parser.add_argument('--min_file', type=str, default='min_rr_log.npy')  # not used if meanmax normalization
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1,2,3])
    parser.add_argument("--var_names", type=utils.str2intlist, default=['u','v','t2m'])
    parser.add_argument('--device', type=str, default='cuda')

    ############################ SEQUENCE PARAMETERS #################    
    parser.add_argument('--multi_timestep_mode', action='store_true')
    parser.add_argument('--nb_timesteps', type=int, default=44)
    parser.add_argument('--timestep_period', type=int, default=1)
    parser.add_argument('--stack_sample_along_time_and_variable', action='store_true')

    ############################ INVERSION PARAMETERS ################
    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default = 'Large_lt_test_labels.csv')
    parser.add_argument("--date_start", type=str, default = "2021-07-01")
    parser.add_argument("--date_stop", type=str, default = "2021-07-02")
    parser.add_argument("--leadtimes", type=utils.str2intlist,
                        default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
    # [3,6,9,12,15,18,21,24,27,30,33,36,39,42]
    # [6,12,18,24,30,36,42,45]

    parser.add_argument("--seed", type=int, default=42)
    
    params = parser.parse_args()

    # fix some of the inputs
    params.Shape = tuple(params.Shape)
    
    # create output directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    output_dir_temporal_exp = params.output_dir + 'Temporal_Experiments/'
    if not os.path.exists(output_dir_temporal_exp):
        os.makedirs(output_dir_temporal_exp)
    output_dir_plots = params.output_dir + 'plots/'
    if not os.path.exists(output_dir_plots):
        os.makedirs(output_dir_plots)

    # set the seed for reproduciibility of runs
    seed = params.seed
    torch.manual_seed(seed)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()
    Means=None
    Maxs=None
    Mins=None
    if params.normalization=="meanmax":
        Means = np.load(f'{params.real_data_dir}{params.mean_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="minmax":
        Mins = np.load(f'{params.real_data_dir}/stat_files/{params.min_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
        Maxs = np.load(f'{params.real_data_dir}/stat_files/{params.max_file}')[params.var_indices].reshape(1,params.Shape[0],1,1)
    elif params.normalization=="":
        pass
    else:
        raise ValueError(f"Unknown normalization: {params.normalization}")

    #################### main loop ##################

    # Setting coordinates of special points on map
    alpes_mountain_coord = [125,240]
    pyrenean_mountain_coord = [25,20]
    massif_central_mountain_coord = [110,128]
    sea_coord = [30,140]
    toulouse_coord = [63,38]

    pixel_coordinate_dict = {'toulouse' : toulouse_coord,
                        'alpes' : alpes_mountain_coord,
                        'pyrenee' : pyrenean_mountain_coord,
                        'massif_central': massif_central_mountain_coord,
                        'sea' : sea_coord
                        }

    print('pixel_coordinate:', pixel_coordinate_dict)
    nb_sample_total = len(list_dates)*16
    AROME_diurnal_cycle = np.zeros((len(pixel_coordinate_dict), 2, params.nb_timesteps-2, nb_sample_total))
    AROME_pearsons_first_to_each_leadtime_img = np.zeros((3, params.nb_timesteps-2, nb_sample_total))
    AROME_pearsons_sliding_img = np.zeros((3, params.nb_timesteps-2, nb_sample_total))
    AROME_temporal_difference = np.zeros((3, params.nb_timesteps-2, nb_sample_total))
    AROME_absolute_temporal_difference = np.zeros((3, params.nb_timesteps-2, nb_sample_total))
    
    nb_child_member = 7
    if 112//16 < nb_child_member:
        nb_child_member = 112//16
    nb_sample_total_gen = len(list_dates)*16*nb_child_member
    gen_diurnal_cycle = np.zeros((len(pixel_coordinate_dict), 2, params.nb_timesteps-2, nb_sample_total_gen))
    gen_pearsons_first_to_each_leadtime_img = np.zeros((3, params.nb_timesteps-2, nb_sample_total_gen))
    gen_pearsons_sliding_img = np.zeros((3, params.nb_timesteps-2, nb_sample_total_gen))
    gen_temporal_difference = np.zeros((3, params.nb_timesteps-2, nb_sample_total_gen))
    gen_absolute_temporal_difference = np.zeros((3, params.nb_timesteps-2, nb_sample_total_gen))

    print(f'AROME : {nb_sample_total}, GEN : {nb_sample_total_gen}')

    for id_date, date_ in enumerate(list_dates):
        print(date_)
        # Importing True - Sequence for the whole day

        Ens_r = utils.load_batch_sequence_from_date(
                    df_extract,
                    date_,
                    params.real_data_dir,
                    concatenate_variable_and_time=params.stack_sample_along_time_and_variable,
                    dt=params.timestep_period,
                    Shape=params.Shape,
                    var_indices=params.var_indices,
                    normalization=params.normalization,
                    Means=Means,
                    Mins=Mins,
                    Maxs=Maxs
                )

        Ens_r = Ens_r[:,1:,:,:].numpy() # Ignore first member
        # Importing Generated Samples
        Ens_gen = []
        date_=str(date_)[:10]
        for lt in params.leadtimes :
            try :
                path_to_sample = params.gen_sample_dir + f"genFsemble_{date_}_{lt}_{params.invstep}_16.npy"    
                Ens_gen.append(np.load(path_to_sample))
            except :
                print(f"File 'genFsemble_{date_}_{lt}_{params.invstep}_16.npy' Not Found")
        
        # Ens_gen = utils.rescale(Ens_gen, Means, Maxs, 1/0.95)
        Ens_gen = torch.tensor(np.array(Ens_gen), dtype=torch.float32).transpose(1,0)
        print('Ens_r shape :', Ens_r.shape)
        print('Ens_gen shape :', Ens_gen.shape)
        for t in range(0, params.nb_timesteps-2):
            for member_id, Ens_r_member in enumerate(Ens_r):
                # print('member_id :', member_id)
                # print('Ens_r_member shape :', Ens_r_member.shape)
                for key_id, key in enumerate(pixel_coordinate_dict):
                    pixel_coordinate=pixel_coordinate_dict[key]
                    # Diurnal Cycle for real sample
                    u = Ens_r_member[t][0][pixel_coordinate[0]][pixel_coordinate[1]]
                    v = Ens_r_member[t][1][pixel_coordinate[0]][pixel_coordinate[1]]
                    t2m = Ens_r_member[t][2][pixel_coordinate[0]][pixel_coordinate[1]]
                    AROME_diurnal_cycle[key_id, 0, t, member_id+16*id_date] = np.sqrt(u**2+v**2)
                    AROME_diurnal_cycle[key_id, 1, t, member_id+16*id_date] = t2m
                     
                for var_id in range(3):
                    # Pearson Correlation on real samples
                    AROME_pearsons_first_to_each_leadtime_img[var_id, t, member_id+16*id_date] = scipy.stats.pearsonr(
                                                                        Ens_r_member[0][var_id].flatten(),
                                                                        Ens_r_member[t][var_id].flatten()
                    ).statistic
                    if t==0 :
                        AROME_pearsons_sliding_img[var_id, t, member_id+16*id_date]=np.nan
                    elif t < params.nb_timesteps-2:
                        AROME_pearsons_sliding_img[var_id, t, member_id+16*id_date] = scipy.stats.pearsonr(
                                                                        Ens_r_member[t][var_id].flatten(),
                                                                        Ens_r_member[t+1][var_id].flatten()
                        ).statistic


                    AROME_temporal_difference[var_id, t, member_id+16*id_date] = np.mean(Ens_r_member[t+1][var_id] - Ens_r_member[t][var_id])
                    AROME_absolute_temporal_difference[var_id, t, member_id+16*id_date] = np.mean(np.abs(Ens_r_member[t+1][var_id] - Ens_r_member[t][var_id]))


                for offset_child_member in range(nb_child_member):
                    gen_member_id = offset_child_member + member_id*(len(Ens_gen)//len(Ens_r))
                    # print(id_date,member_id , member_id+(16*(1+offset_child_member))*id_date)
                    for key_id, key in enumerate(pixel_coordinate_dict):
                        pixel_coordinate=pixel_coordinate_dict[key]
                        # Diurnal Cycle for perturbated sample
                        u = Ens_gen[gen_member_id][t][0][pixel_coordinate[0]][pixel_coordinate[1]]
                        v = Ens_gen[gen_member_id][t][1][pixel_coordinate[0]][pixel_coordinate[1]]
                        t2m = Ens_gen[gen_member_id][t][2][pixel_coordinate[0]][pixel_coordinate[1]]

                        gen_diurnal_cycle[key_id, 0, t, gen_member_id+16*id_date*nb_child_member] = np.sqrt(u**2+v**2)
                        gen_diurnal_cycle[key_id, 1, t, gen_member_id+16*id_date*nb_child_member] = t2m

                    for var_id in range(3):
                        # Pearson Correlation on perturbated samples
                        gen_pearsons_first_to_each_leadtime_img[var_id, t, gen_member_id+16*id_date*nb_child_member] = scipy.stats.pearsonr(
                                                                            Ens_gen[gen_member_id][1][var_id].flatten(),
                                                                            Ens_gen[gen_member_id][t+1][var_id].flatten()
                        ).statistic

                        if t < params.nb_timesteps-2:

                            
                            gen_pearsons_sliding_img[var_id, t, gen_member_id+16*id_date*nb_child_member] = scipy.stats.pearsonr(
                                                                            Ens_gen[gen_member_id][t][var_id].flatten(),
                                                                            Ens_gen[gen_member_id][t+1][var_id].flatten()
                            ).statistic
                        
                        if t < params.nb_timesteps-2:
                            gen_temporal_difference[var_id, t, gen_member_id+16*id_date*nb_child_member] = torch.mean(Ens_gen[gen_member_id][t+1][var_id] - Ens_gen[gen_member_id][t][var_id])
                            gen_absolute_temporal_difference[var_id, t, gen_member_id+16*id_date*nb_child_member] = torch.mean(np.abs(Ens_gen[gen_member_id][t+1][var_id] - Ens_gen[gen_member_id][t][var_id]))
                            
    
    list_ticks = np.array(params.leadtimes)
    
    
    AROME_diurnal_cycle = np.mean(AROME_diurnal_cycle, -1)
    gen_diurnal_cycle = np.mean(gen_diurnal_cycle, -1)
    print('Saving Diurnal Cycle')
    np.save(output_dir_temporal_exp+f'AROME_Diurnal_Cycle_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', AROME_diurnal_cycle)
    np.save(output_dir_temporal_exp+f'gen_Diurnal_Cycle_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', gen_diurnal_cycle)
    print('Plotting Diurnal Cycle')
    for key_id, key in enumerate(pixel_coordinate_dict):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
        ax[0].plot(range(len(AROME_diurnal_cycle[key_id,0])),  AROME_diurnal_cycle[key_id,0], linewidth=6, color='k', label='AROME')
        ax[0].plot(range(len(gen_diurnal_cycle[key_id,0])),  gen_diurnal_cycle[key_id,0], linewidth=6, label='Generated')
        ax[0].set_ylabel('Wind speed (m/s)', size = 30)
        ax[0].set_xticks(range(len(list_ticks)), labels=list_ticks)

        ax[1].plot(range(len(AROME_diurnal_cycle[key_id,1])), AROME_diurnal_cycle[key_id,1], linewidth=6, color='k', label='AROME')
        ax[1].plot(range(len(gen_diurnal_cycle[key_id,1])),  gen_diurnal_cycle[key_id,1], linewidth=6, label='Generated')
        ax[1].set_ylabel('Temperature at 2m (K)', size = 30)
        ax[1].set_xticks(range(len(list_ticks)), labels=list_ticks)

        for i in ax :
            i.legend()

        fig.suptitle(f'Diurnal Cycle on pixel {key}', size=30)
        output_dir_diurnal_cycle = output_dir_plots + 'Diurnal_Cycle/'
        if not os.path.exists(output_dir_diurnal_cycle):
            os.makedirs(output_dir_diurnal_cycle)
        fig.savefig(output_dir_diurnal_cycle+f'Diurnal_Cycle_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}_{key}.pdf') 
    
    
    AROME_pearsons_sliding_img = np.mean(AROME_pearsons_sliding_img, -1)
    gen_pearsons_sliding_img = np.mean(gen_pearsons_sliding_img, -1)
    AROME_pearsons_first_to_each_leadtime_img = np.mean(AROME_pearsons_first_to_each_leadtime_img, -1)
    gen_pearsons_first_to_each_leadtime_img = np.mean(gen_pearsons_first_to_each_leadtime_img, -1)

    
    print('Saving Pearson Correlation')
    np.save(output_dir_temporal_exp+f'AROME_Pearson_Correlation_first_to_each_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', AROME_pearsons_first_to_each_leadtime_img)
    np.save(output_dir_temporal_exp+f'Gen_Pearson_Correlation_first_to_each_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', gen_pearsons_first_to_each_leadtime_img)
    np.save(output_dir_temporal_exp+f'AROME_Pearson_Correlation_sliding_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', AROME_pearsons_sliding_img)
    np.save(output_dir_temporal_exp+f'Gen_Pearson_Correlation_sliding_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', gen_pearsons_sliding_img)

    print('Plotting Pearson Correlation')
    
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(len(AROME_pearsons_first_to_each_leadtime_img[0])),  AROME_pearsons_first_to_each_leadtime_img[0], linewidth=6, color='k', label='AROME')
    ax[0].plot(range(len(gen_pearsons_first_to_each_leadtime_img[0])),  gen_pearsons_first_to_each_leadtime_img[0], linewidth=6, label='Generated')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(len(list_ticks)), labels=list_ticks)
    
    ax[1].plot(range(len(AROME_pearsons_first_to_each_leadtime_img[1])),  AROME_pearsons_first_to_each_leadtime_img[1], linewidth=6, color='k', label='AROME')
    ax[1].plot(range(len(gen_pearsons_first_to_each_leadtime_img[1])),  gen_pearsons_first_to_each_leadtime_img[1], linewidth=6, label='Generated')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[2].plot(range(len(AROME_pearsons_first_to_each_leadtime_img[2])), AROME_pearsons_first_to_each_leadtime_img[2], linewidth=6, color='k', label='AROME')
    ax[2].plot(range(len(gen_pearsons_first_to_each_leadtime_img[2])), gen_pearsons_first_to_each_leadtime_img[2], linewidth=6, label='Generated')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(len(list_ticks)), labels=list_ticks)

    for i in ax :
        i.legend()

    fig.suptitle('Pearson Correlation First to Each Leadtime', size=30)
    output_dir_pearson_correlation = output_dir_plots + 'Pearson_Correlation/'
    if not os.path.exists(output_dir_pearson_correlation):
        os.makedirs(output_dir_pearson_correlation)
    fig.savefig(output_dir_pearson_correlation+f'Pearson_Correlation_first_to_each_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.pdf') 

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(len(AROME_pearsons_sliding_img[0])),  AROME_pearsons_sliding_img[0], linewidth=6, color='k', label='AROME')
    ax[0].plot(range(len(gen_pearsons_sliding_img[0])),  gen_pearsons_sliding_img[0], linewidth=6, label='Generated')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[1].plot(range(len(AROME_pearsons_sliding_img[1])),  AROME_pearsons_sliding_img[1], linewidth=6, color='k', label='AROME')
    ax[1].plot(range(len(gen_pearsons_sliding_img[1])),  gen_pearsons_sliding_img[1], linewidth=6, label='Generated')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(len(list_ticks)), labels=list_ticks)
    
    ax[2].plot(range(len(AROME_pearsons_sliding_img[2])), AROME_pearsons_sliding_img[2], linewidth=6, color='k', label='AROME')
    ax[2].plot(range(len(gen_pearsons_sliding_img[2])), gen_pearsons_sliding_img[2], linewidth=6, label='Generated')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(len(list_ticks)), labels=list_ticks)

    for i in ax :
        i.legend()

    fig.suptitle('Pearson Correlation between X(t) and X(t+1)', size=30)
    fig.savefig(output_dir_pearson_correlation+f'Pearson_Correlation_sliding_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.pdf') 


    AROME_temporal_difference = np.mean(AROME_temporal_difference, -1)
    gen_temporal_difference = np.mean(gen_temporal_difference, -1)
    AROME_absolute_temporal_difference  = np.mean(AROME_absolute_temporal_difference, -1)
    gen_absolute_temporal_difference  = np.mean(gen_absolute_temporal_difference, -1)
    print('Saving Temporal Difference')
    np.save(output_dir_temporal_exp+f'Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', AROME_temporal_difference)
    np.save(output_dir_temporal_exp+f'Absolute_Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', AROME_absolute_temporal_difference)
    np.save(output_dir_temporal_exp+f'Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', gen_temporal_difference)
    np.save(output_dir_temporal_exp+f'Absolute_Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.npy', gen_absolute_temporal_difference)
    print('Plotting Temporal Difference')

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(len(AROME_temporal_difference[0])),  AROME_temporal_difference[0], linewidth=6, color='k', label='AROME')
    ax[0].plot(range(len(gen_temporal_difference[0])),  gen_temporal_difference[0], linewidth=6, label='Generated')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[1].plot(range(len(AROME_temporal_difference[1])),  AROME_temporal_difference[1], linewidth=6, color='k', label='AROME')
    ax[1].plot(range(len(gen_temporal_difference[1])),  gen_temporal_difference[1], linewidth=6, label='Generated')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[2].plot(range(len(AROME_temporal_difference[2])), AROME_temporal_difference[2], linewidth=6, color='k', label='AROME')
    ax[2].plot(range(len(gen_temporal_difference[2])),  gen_temporal_difference[2], linewidth=6, label='Generated')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(len(list_ticks)), labels=list_ticks)

    for i in ax :
        i.legend()

    fig.suptitle('Absolute Temporal Difference for Each Leadtime : ∆X = X(t+1) - X(t)', size=30)
    output_dir_temporal_difference = output_dir_plots + 'Temporal_Difference/'
    if not os.path.exists(output_dir_temporal_difference):
        os.makedirs(output_dir_temporal_difference)
    fig.savefig(output_dir_temporal_difference+f'Absolute_Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.pdf') 

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16,16))
    ax[0].plot(range(len(AROME_absolute_temporal_difference[0])),  AROME_absolute_temporal_difference[0], linewidth=6, color='k', label='AROME')
    ax[0].plot(range(len(gen_absolute_temporal_difference[0])),  gen_absolute_temporal_difference[0], linewidth=6, label='Generated')
    ax[0].set_ylabel('Wind speed U (m/s)', size = 30)
    ax[0].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[1].plot(range(len(AROME_absolute_temporal_difference[1])),  AROME_absolute_temporal_difference[1], linewidth=6, color='k', label='AROME')
    ax[1].plot(range(len(gen_absolute_temporal_difference[1])),  gen_absolute_temporal_difference[1], linewidth=6, label='Generated')
    ax[1].set_ylabel('Wind speed V (m/s)', size = 30)
    ax[1].set_xticks(range(len(list_ticks)), labels=list_ticks)

    ax[2].plot(range(len(AROME_absolute_temporal_difference[2])), AROME_absolute_temporal_difference[2], linewidth=6, color='k', label='AROME')
    ax[2].plot(range(len(gen_absolute_temporal_difference[2])),  gen_absolute_temporal_difference[2], linewidth=6, label='Generated')
    ax[2].set_ylabel('Temperature at 2m (K)', size = 30)
    ax[2].set_xticks(range(len(list_ticks)), labels=list_ticks)

    for i in ax :
        i.legend()

    fig.suptitle('Temporal Difference for Each Leadtime : ∆X = |X(t+1) - X(t)|', size=30)
    fig.savefig(output_dir_temporal_difference+f'Temporal_Difference_over_{nb_sample_total}_samples_{params.nb_timesteps}_nb_var_{len(params.var_names)}.pdf') 
      
            