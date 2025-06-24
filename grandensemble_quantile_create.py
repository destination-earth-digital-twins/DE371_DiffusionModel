#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
import os
from tqdm import trange

def load_tab_var(tab, param):
    return {'rr' : tab[:,0],
                        'ff': np.sqrt(tab[:,1]**2+tab[:,2]),
                        't2m': tab[:,3]
        }[param]

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

font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 25,
}

def plot_quantile(
    data_list,
    output_dir,
    name_quantile,
    param,
    leadtime,
    clim,
    id_quantile_to_plot = [0,1,6,9,12],
    denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
    ):

    fig, axs = plt.subplots(
            nrows=1, ncols=len(id_quantile_to_plot), figsize=(15, 5)
        )
    cmap = plt.get_cmap("PiYG", 7)
    os.makedirs(output_dir+f'/{name_quantile}/', exist_ok=True)
    for idx, id_quantile in enumerate(id_quantile_to_plot):
        name = denom[id_quantile]
        quantile = data_list[id_quantile]

        im = axs[idx].imshow(
            quantile,
            origin="lower",
            cmap=cmap,
            clim=clim[id_quantile]
        )
        fig.colorbar(im, ax=axs[idx], shrink=0.5)
        axs[idx].set_title(F'{name}', fontdict=font)

        fig.suptitle(f"Quantiles of {name_quantile} for {param} variable", fontdict=font)
        fig.tight_layout()
        fig.savefig(
            output_dir+f'/{name_quantile}/plot_{name_quantile}_{param}_{leadtime}.pdf'
        )
        plt.close()

if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('--param', type=str2intlist, default=['ff', 't2m'])
    parser.add_argument('--base_dir', type=str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/grandEnsemble/AROME/')
    parser.add_argument('--generated_sample_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/sdedit_ED/sampling_4steps/')
    parser.add_argument('--output_dir', type=str, default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/sdedit_ED/sampling_4steps/quantiles_data/')
    parser.add_argument('--exp_name', type=str, default='Sdedit_4steps')
    parser.add_argument('--leadtimes', type=str2intlist, default=[6,12,18,24,30,36,42]) # echeance de la prevision, n'importe quelle valeur entre 0 et 45h est disponible (par pas de 1h)
    parser.add_argument('--inv_step', type=int, default=1000)

    parser.add_argument('--unbias', action="store_true")
    args = parser.parse_args()


    ###SetUp 
    members = list(range(1,2,1))   # les 16 membres PEARO  <-- le "2" pour le 2e run audn AllMB a été chargé     
    quantile_to_compute = [0,0.5,1,5,10,25,50,75,90,95,99,99.5,100]
    Nsmall = 16
    Nlbc = 25
    nbrandinit = 50
    Generatednameout = [args.exp_name]
    plot_id_nbrandinit = 0
    os.makedirs(args.output_dir, exist_ok=True)

    AROME_large_ensemble_file_format = "_grand_ensemble_{leadtime}_875.npy"
    AROME_small_ensemble_file_format = "true_grand_ensemble_{leadtime}_draw_{draw_idx}.npy"
    Generated_large_ensemble_file_format = "fake_grand_ensemble_{leadtime}_draw_{draw_idx}.npy"

    # Initialisation projection
    lplotq=True
    lrandinit=False

    def initsmall(lstbc,lstic,Ns,Nlbc):
        r''' Tirage aléatoire des membres conditionneurs '''
        yic = random.sample(lstic, Ns)
        ybc = random.sample(lstbc, Ns)
        mb=np.zeros((Ns))
        # Find members corresponding to (yic,ybc) pairs
        for k in range(Ns):
            loc_bc=np.where(np.asarray(lstbc)==ybc[k])
            #index member of the PEARO experiment start from 1
            #if python storage of members start at 0 remove '+1'
            mb[k]=(yic[k]-1)*Nlbc + loc_bc[0][0]
        return mb

    lstlbc = [2,20,9,5,32,15,19,21,13,1,34,12,10,31,23,11,8,24,29,22,28,25,6,33,14,7,30,27,0,18,4,26,3,16,17]
    lstic  = list(range(1,26))
    for param in args.param:
        for leadtime in args.leadtimes :
            # Tirage nrandinit ensembles conditionneurs
            if lrandinit:
                print("tirage aléatoire des membres conditionneurs")
                mb = np.zeros((Nsmall,nbrandinit))
                for r in range(nbrandinit):
                    mb[:,r] = initsmall(lstlbc,lstic,Nsmall,Nlbc)
                np.save(args.output_dir + '/' + 'nbrandinit_MBs',mb,allow_pickle=True)

            reseau = '2021-10-01T21:00:00Z'
            clim_quantiles = list()

            ################################################
            ############# Large AROME Ensemble #############
            ################################################

            print("Loading large real ensemble")
            filename = AROME_large_ensemble_file_format.format(leadtime = leadtime)
            tabref = load_tab_var(tab = np.load(args.base_dir + filename, allow_pickle=True), param=param)

            print(f"Computing quantiles and stdev of large real ensemble for param {param}")
            Qrefs = np.percentile(tabref, quantile_to_compute, interpolation='nearest', axis=0)
            data_ref_list = [Qrefs[i] for i in range(13)]
            sdev_ref = np.std(tabref,axis=0,ddof=1)
            data_ref_list.append(sdev_ref)

            large_AROME_dir = args.output_dir+'/large_AROME'
            os.makedirs(large_AROME_dir, exist_ok=True)
            np.save(f"{large_AROME_dir}/large_AROME_{leadtime}_{param}.npy", np.concatenate([Qrefs, sdev_ref[np.newaxis,:]]))

            for quantile in Qrefs:
                clim_quantiles.append((quantile.min(), quantile.max()))

            print("Plotting AROME Quantile of large real ensemble")
            plot_quantile(
                data_list=Qrefs,
                output_dir=args.output_dir,
                name_quantile='large_AROME',
                param=param,
                leadtime=leadtime,
                clim=clim_quantiles,
                id_quantile_to_plot = [0,1,6,9,12],
                denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
            )

            # Keeping track of quantiles spatial means
            qref_avg = np.zeros((np.size(quantile_to_compute)))
            for q in range(len(quantile_to_compute)):
                qref_avg[q] = Qrefs[q].mean()

            ################################################
            ############# Small AROME Ensemble #############
            ################################################

            print("Loading small real ensemble")
            q0small = []
            q05small = []
            q1small = []
            q5small = []
            q10small = []
            q25small = []
            q50small = []
            q75small = []
            q90small = []
            q95small = []
            q99small = []
            q995small = []
            q100small = []
            sdev_small = []
            qavg_small=pd.DataFrame(columns=['leadtime','Quantiles','Init','DiffSmall', 'DiffRelSmall'])
            print("Computing quantiles and stdev of small real ensemble")
            for i in trange(nbrandinit):
                filename = AROME_small_ensemble_file_format.format(leadtime = leadtime, draw_idx=i)
                tabs = load_tab_var(tab = np.load(args.generated_sample_dir + 'samples/' + filename, allow_pickle=True), param=param)

                Qsmall = np.percentile(tabs, quantile_to_compute, interpolation='nearest',axis=0)
                q0small.append(Qsmall[0])
                q05small.append(Qsmall[1]) 
                q1small.append(Qsmall[2])
                q5small.append(Qsmall[3])
                q10small.append(Qsmall[4])
                q25small.append(Qsmall[5])
                q50small.append(Qsmall[6])
                q75small.append(Qsmall[7])
                q90small.append(Qsmall[8])
                q95small.append(Qsmall[9])
                q99small.append(Qsmall[10])
                q995small.append(Qsmall[11])
                q100small.append(Qsmall[12])
                sdev_small.append(np.std(tabs,axis=0,ddof=1))

                for q in range(len(quantile_to_compute)):
                    qsmall_avg = np.mean(Qsmall[q])
                    newq=pd.DataFrame([[leadtime,quantile_to_compute[q],i,qsmall_avg-qref_avg[q],(qsmall_avg-qref_avg[q])/qref_avg[q]]],columns=['leadtime','Quantiles','Init','DiffSmall','DiffRelSmall'])
                    qavg_small=qavg_small._append(newq,ignore_index=True)
                
                Quantiles_Xtremes_avg_dir = args.output_dir+'/Quantiles_Xtremes_avg'
                os.makedirs(Quantiles_Xtremes_avg_dir, exist_ok=True)
                qavg_small.to_pickle(Quantiles_Xtremes_avg_dir + "/" + "Quantiles_Xtremes_avg_AROME_" + param + "_" + "Small" +"_"+ str(reseau) + "+" + str(leadtime) + ".pkl")
                
                small_AROME_dir = args.output_dir+'/small_AROME'
                os.makedirs(small_AROME_dir, exist_ok=True)
                np.save(f"{small_AROME_dir}/small_AROME_{leadtime}_{param}.npy",
                        np.array([np.array(q0small),np.array(q05small), np.array(q1small),
                                np.array(q5small), np.array(q10small), np.array(q25small),
                                np.array(q50small), np.array(q75small), np.array(q90small),
                                np.array(q95small), np.array(q99small), np.array(q995small),
                                np.array(q100small), np.array(sdev_small)]))
                
                
                data_list = [q0small,q05small, q1small,
                                q5small, q10small, q25small,
                                q50small, q75small, q90small,
                                q95small, q99small, q995small,
                                q100small, sdev_small]
                data_list = [np.percentile(np.array(q),50,method='nearest',axis=0) for q in data_list]

                print("Plotting AROME Quantile Small")
                
                plot_quantile(
                    data_list=Qsmall,
                    output_dir=args.output_dir,
                    name_quantile='small_AROME',
                    param=param,
                    leadtime=leadtime,
                    clim=clim_quantiles,
                    id_quantile_to_plot = [0,1,6,9,12],
                    denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
                )

                print("Computing diff wrt to large real ensemble")

                data_diff_list = [q - qref for (q,qref) in zip(data_list, data_ref_list)]
                median_quantiles_diffsmall_dir = args.output_dir+'/median_quantiles_diffsmall'
                os.makedirs(median_quantiles_diffsmall_dir, exist_ok=True)
                np.save(f"{median_quantiles_diffsmall_dir}/median_quantiles_diffsmall_{leadtime}_{param}.npy",np.array(data_list))

            ################################################
            ############# Large Generated Ensemble #############
            ################################################

            qavg=pd.DataFrame(columns=['leadtime','Quantiles','Init','Diff'+Generatednameout[k], 'DiffRel'+Generatednameout[k]])
            Qs = []
            for i in trange(nbrandinit):
                print("Loading files containing generated members")
                gen_filename = Generated_large_ensemble_file_format.format(leadtime = leadtime, draw_idx=i)
                gen_tabs = load_tab_var(tab = np.load(args.generated_sample_dir + 'samples/' + gen_filename, allow_pickle=True), param=param)

                print("Computing percentiles on Generated")
                percentile = np.percentile(gen_filename,quantile_to_compute,interpolation='nearest',axis=0)
                Qs.append(np.concatenate([percentile, np.std(gen_tabs,axis=0,ddof=1)[np.newaxis,:]]))
                print("Keeping track of spatial means of quantiles for Generated")
                for q in range(np.size(quantile_to_compute)):
                    qm = np.mean(Qs[-1][q])
                    newq = pd.DataFrame([[leadtime,quantile_to_compute[q],i,qm-qref_avg[q],(qm-qref_avg[q])/qref_avg[q]]],columns=['leadtime','Quantiles','Init','Diff'+Generatednameout[k],'DiffRel'+Generatednameout[k]])
                    qavg = qavg._append(newq,ignore_index=True)
                    
                if plot_id_nbrandinit == i:
                    print("Plotting Generated Quantile")
                    
                    plot_quantile(
                            data_list=percentile,
                            output_dir=args.output_dir,
                            name_quantile='Generated',
                            param=param,
                            leadtime=leadtime,
                            clim=clim_quantiles,
                            id_quantile_to_plot = [0,1,6,9,12],
                            denom = ["Q0","Q05", "Q1", "Q5", "Q10","Q25","Q50","Q75","Q90","Q95", "Q99", "Q995", "Q100", "Sdev"]
                    )

            median_quantiles = np.percentile(np.array(Qs),50,interpolation='nearest',axis=0)
            median_generated_dir = args.output_dir+'/Generated'
            os.makedirs(median_generated_dir, exist_ok=True)
            np.save(f"{median_generated_dir}/Generated_{Generatednameout[k]}_{leadtime}_{param}.npy", np.array(Qs))
            np.save(f"{median_generated_dir}/median_Generated_{Generatednameout[k]}_{leadtime}_{param}.npy",median_quantiles)

            
            print("saving Quantiles averages")
            print(qavg.head())
            Quantiles_Xtremes_avg_dir = args.output_dir+'/Quantiles_Xtremes_avg'
            os.makedirs(Quantiles_Xtremes_avg_dir, exist_ok=True)
            qavg.to_pickle(Quantiles_Xtremes_avg_dir + "/" + "Quantiles_Xtremes_avg_Generated_" + param + "_" + Generatednameout[k]  +"_"+ str(reseau) + "+" + str(leadtime) + ".pkl")
