import numpy as np
import matplotlib.pyplot as plt
# var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}

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
          clim_global=[],
          axis_title_global=''
          ):
        print('packsample.shape', packsample.shape)
        print('pert_sample.shape', pert_sample.shape)
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)), nrows=3, ncols=len(var_names))
        for id, var in enumerate(var_names):
            var_id = dict_var[var]
            if not clim_global :
                vmin = np.min([np.min(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                vmax = np.min([np.max(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                clim = (vmin, vmax)
            else :
                clim = clim_global
            ax[0][id].set_title(f"{axis_title_global}{var} real")
            im = ax[0][id].imshow(packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[0][id], shrink=0.5)

            ax[1][id].set_title(f"{axis_title_global}{var} generated")
            im = ax[1][id].imshow(pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]],  origin="lower", cmap=colormap_var[id])
            fig.colorbar(im, ax=ax[1][id], shrink=0.5)

            diff = packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]] - pert_sample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]]
            ax[2][id].set_title(f"{axis_title_global}{var} generated")
            im = ax[2][id].imshow(diff, origin="lower", cmap="RdYlGn")
            fig.colorbar(im, ax=ax[2][id], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception:
            print(f"unable to save figure: {figname}")
        plt.close()
        return