#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple plot with epygram of several runs

Files are retrieved with Vortex ; they are stored in a cache defined by environment variable $MTOOLDIR

Link to epygram documentation on CNRM PC : 
file:///home/common/epygram/public/EPyGrAM/1.4.7/epygram/doc_sphinx/html/index.html

Created Created sometimes in 2021
@author: raynaud
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

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

    parser.add_argument("--ech", type=int, default=6)
    parser.add_argument('--param', type=str2intlist, default=['ff', 't2m'])
    parser.add_argument('--leadtimes', type=str2intlist, default=[6,12,18,24,30,36,42])
    parser.add_argument("--base_dir", type=str, default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/')
    args = parser.parse_args()

    PATH_exp = {#'Sdedit_ED_4steps' : "/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/sdedit_ED/sampling_GE_4steps/quantile_data/",
    #'Sdedit_ED_3steps' : "/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/sdedit_ED/sampling_GE_3steps/quantile_data/",
    'Sdedit_ED_3steps' : "/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/GrandEnsemble/sdedit_ED/test4_sampling_GE_3steps_900_members_5draws/quantile_data/"
    }
    base_dir = args.base_dir
    output_dir = base_dir + 'quantiles_plots/'
    os.makedirs(output_dir, exist_ok=True)
    for param in args.param:
        for leadtime in args.leadtimes :

            list_expe = ['Sdedit_ED_3steps']
            list_data_xtremes = []
            for exp in list_expe:
                list_data_xtremes.append(pd.read_pickle(PATH_exp[exp]+ 'Quantiles_Xtremes_avg/' + 'Quantiles_Xtremes_avg_Generated_' + param + f'_{exp}_2021-10-01T21:00:00Z+'+ str(leadtime) + '.pkl'))
            list_data_xtremes.append(pd.read_pickle(PATH_exp[exp]+ 'Quantiles_Xtremes_avg/' + 'Quantiles_Xtremes_avg_AROME_' + param + f'_Small_2021-10-01T21:00:00Z+'+ str(leadtime) + '.pkl'))

            merge = pd.concat(list_data_xtremes)
            value_vars = ["DiffSdedit_ED_3steps" , "DiffSmall"]
            merge = merge.loc[(merge['Quantiles'] != 0.5) & (merge['Quantiles'] != 99.5)]

            dd=pd.melt(merge,
                        id_vars=['Quantiles'],
                        value_vars=value_vars,
                        var_name=''
            )

            sns.boxplot(x='Quantiles', y='value', data=dd, hue='')
            if param=='ff':
                plt.ylim(-5,5)
            elif param=='t2m':
                plt.ylim(-2,3)
            plt.suptitle(f'variable : {param} for leadtime +{leadtime}')
            plt.axhline(0.0,0,1)
            plt.xlabel(f"Quantiles", fontweight='bold', fontsize=18)
            plt.xticks(size=14)
            plt.yticks(size=14)
            plt.savefig(output_dir + 'Diffqavg_overinit_allQ_Xtremes_' + param + '_' + str(leadtime)+'.pdf')
            plt.clf()
            plt.close()


