import numpy as np
import matplotlib.pyplot as plt
import os
rank = int(os.environ.get("LOCAL_RANK",0))
# var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}

def online_plot_mean(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          figtitle=" ", 
          figname=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          clim_global=[(-5,5),(-5,5),(270,300)],
          axis_title_global=''
          ):
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]
            if not clim_global :
                vmin = np.min([np.min(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                vmax = np.min([np.max(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                clim_mean = (vmin, vmax)
            else :
                clim_mean = clim_global[id]
            ax[0][id].set_title(f"mean {axis_title_global}{var} real")
            im = ax[0][id].imshow(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0), origin="lower", cmap=colormap_var[id], clim=clim_mean)
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"mean {axis_title_global}{var} generated")
            im = ax[1][id].imshow(np.mean(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0),  origin="lower", cmap=colormap_var[id], clim=clim_mean)
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            diff = np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0) - np.mean(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0)
            ax[2][id].set_title(f"diff of mean {axis_title_global}{var}")
            im = ax[2][id].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def online_plot_var(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          figtitle=" ", 
          figname=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          clim_global=[(0,5),(0,5),(0,5)],
          axis_title_global=''
          ):
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]
            if not clim_global :
                vmin = np.min([np.min(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                vmax = np.min([np.max(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                clim_var = (vmin, vmax)
            else :
                clim_var = clim_global[id]
            ax[0][id].set_title(f"var {axis_title_global}{var} real")
            im = ax[0][id].imshow(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0), origin="lower", cmap=colormap_var[id], clim=clim_var)
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"var {axis_title_global}{var} generated")
            im = ax[1][id].imshow(np.var(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0),  origin="lower", cmap=colormap_var[id], clim=clim_var)
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            diff = np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0) - np.var(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0)
            ax[2][id].set_title(f"diff var {axis_title_global}{var}")
            im = ax[2][id].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def online_plot(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          figtitle=" ", 
          figname=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          clim_global=[(-5,5),(-5,5),(270,300)],
          axis_title_global=''
          ):

        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]
            if not clim_global :
                vmin = np.min([np.min(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                vmax = np.min([np.max(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                clim = (vmin, vmax)
            else :
                clim = clim_global[id]
            ax[0][id].set_title(f"{axis_title_global}{var} real")
            im = ax[0][id].imshow(packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], origin="lower", cmap=colormap_var[id], clim=clim)
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"{axis_title_global}{var} generated")
            im = ax[1][id].imshow(pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]],  origin="lower", cmap=colormap_var[id], clim=clim)
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            diff = packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]] - pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]]
            ax[2][id].set_title(f"diff {axis_title_global}{var}")
            im = ax[2][id].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname}")
        plt.close()
        return

def online_plot_quantiles(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          axis_title_global='',
          quantiles_list=[0.01,0.1,0.9,0.99]
          ):

        cmap = plt.get_cmap("PiYG", 8)
        quantiles_arome = np.quantile(packsample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_gen = np.quantile(pert_sample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        vmins = np.zeros((len(quantiles_list), len(var_names)))
        vmaxs = np.zeros((len(quantiles_list), len(var_names)))

        # Quantiles of AROME
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=3, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                vmins[quantile_idx][var_idx] = np.min(
                    [np.min(quantiles_arome[quantile_idx][var_idx])]
                )
                vmaxs[quantile_idx][var_idx] = np.min(
                    [np.max(quantiles_arome[quantile_idx][var_idx])]
                )

                clim = (vmins[quantile_idx][var_idx],vmaxs[quantile_idx][var_idx])

                ax[var_idx][quantile_idx].set_title(f"{axis_title_global}{var} real - Q{quantile}")
                im = ax[var_idx][quantile_idx].imshow(quantiles_arome[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                fig.colorbar(im, ax=ax[var_idx][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of AROME ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_AROME.png', dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname_info+'_AROME.png'}")
            print('reason: ', e)
        plt.close()
        
        # Quantiles of Generated samples
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=3, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                clim = (vmins[quantile_idx][var_idx],vmaxs[quantile_idx][var_idx])

                ax[var_idx][quantile_idx].set_title(f"{axis_title_global}{var} GEN - Q{quantile}")
                im = ax[var_idx][quantile_idx].imshow(quantiles_gen[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                fig.colorbar(im, ax=ax[var_idx][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of Generated ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_GEN.png', dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname_info+'_GEN.png'}")
            print('reason: ', e)
        plt.close()
        return