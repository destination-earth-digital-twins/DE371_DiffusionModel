import numpy as np
import matplotlib.pyplot as plt


def plotter3D_3var(img,save_path,title):
    """
    Plots 3 (for each variable) images, from img of shape(batch_size,variables,latitude,longitude) and saves it 

    Args:
        img (tensor): img of shape (batch_size,variables,latitude,longitude)
        save_path (string): path to save the plotted images
    """
    
    if len(img.shape)!=3 and len(img.shape)!=4:
        raise ValueError(f"Length of img.shape must be 3 or 4 and is{len(img.shape)}")

    if len(img.shape) == 4:
        img_copy = img.squeeze(0)
    elif len(img.shape) ==3:
        img_copy = img   
    fig, axes = plt.subplots(1, 3, figsize=(12, 10))
    fig.suptitle(title,y=0.7)
    axes = axes.flatten()

    # Plot des deux premiers canaux en haut
    for i in range(3):
        ax = axes[i]
        if i==0:
            im = ax.imshow(img_copy[i].detach().cpu().numpy(), cmap='viridis', origin='lower')
            ax.set_title(f"u : vent zonal", fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            
        elif i==1 : 
            
            im = ax.imshow(img_copy[i].detach().cpu().numpy(), cmap='viridis', origin='lower')
            ax.set_title('v : vent méridional',fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
        else :
            im = ax.imshow(img_copy[2].detach().cpu().numpy(), cmap='coolwarm', origin='lower')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("t2m", fontsize=12)    
            ax.axis("off")


    plt.tight_layout()
    plt.savefig(save_path, dpi=1500)
    plt.close()
    
    

def plotter2D_3var(img,save_path,lat,title):
    """
    Plots 3 (for each variable) images on a latitude, from img of shape(batch_size,variables,latitude,longitude) and saves it 

    Args:
        img (tensor): img of shape (batch_size,variables,latitude,longitude)
        save_path (string): path to save the plotted images
        lat (int) : latitude to plot (from 0 (south) to 711(north))
    """

    assert(lat < 712 and lat >=0), f'lat must be between 0 and 711, and is {lat}'
    
    if len(img.shape)!=4:
        raise ValueError(f"Length of img.shape must be 4 and is{len(img.shape)}")
    
    x = np.arange(0, img.shape[3]) 

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    axes = axes.flatten()

    for i in range(3):
        ax = axes[i]
        y = img[0, i, 300, :].detach().cpu().numpy()
        ax.plot(x, y, lw=0.5)
        if i==0:
            ax.set_title(f"u : vent zonal", fontsize=12)
        elif i==1: 
            ax.set_title('v : vent méridional')
        else :
            ax.set_title('t2m')

        ax.set_xlabel("Colonne")
        ax.set_ylabel("Valeur")
        ax.grid(True)

    # masquer la 4ᵉ sous-figure
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    
    