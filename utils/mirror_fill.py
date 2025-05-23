import torch
import numpy as np
import os 
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title)
    axes = axes.flatten()

    # Plot des deux premiers canaux en haut
    for i in range(2):
        ax = axes[i]
        im = ax.imshow(img_copy[i].detach().cpu().numpy(), cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if i==0:
            ax.set_title(f"u : vent zonal", fontsize=12)
        else : 
            ax.set_title('v : vent méridional')
        ax.axis("off")

    # Plot du 3ᵉ canal en bas à gauche
    ax = axes[2]
    im = ax.imshow(img_copy[2].detach().cpu().numpy(), cmap='coolwarm', origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("t2m", fontsize=12)
    ax.axis("off")

    # Case vide en bas à droite
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def mirror_fill(img, mask):
    """
    Fills an img with invalid datas with valid datas, keeping a continuous physical aspect
    Args:
        img (torch.tensor) : an image of shape (batch_size,variables,latitude,longitude)
        mask (torch.tensor) : a tensor of img's shape containing True (valid datas of img) and False (invalid datas of img)
    """
    device, dtype = img.device, img.dtype
    img_np = img[0].cpu().numpy()      
    mask_no_batch = mask[0].cpu().numpy() 
    var, H, W = img_np.shape
    filled = img_np.copy()
    valid_x_v = []
    invalid_x_v=[]
    valid_y_v = []
    invalid_y_v=[]
    
    valid_x_h = []
    invalid_x_h=[]
    valid_y_h = []
    invalid_y_h=[]
    # vertical filling
    for x in range(W):
        rows = np.where(mask_no_batch[0,:,x])[0]
        if rows.size == 0:
            continue
        r_min, r_max = rows.min(), rows.max()
        
        for i in range(r_min):
            i_ref = min(r_min + (r_min - i), H-1)
            valid_x_v.append(x)
            invalid_x_v.append(x)
            # filled[0,i,x] = filled[0,i_ref,x]
            valid_y_v.append(i_ref)
            invalid_y_v.append(i)
        for i in range(r_max+1, H):

            i_ref = max(r_max - (i - r_max), 0)
            
            # filled[0,i,x] = filled[0,i_ref,x]
            
            valid_x_v.append(x)
            invalid_x_v.append(x)
            # filled[0,i,x] = filled[0,i_ref,x]
            valid_y_v.append(i_ref)
            invalid_y_v.append(i)
            
    # horizontal pass
    for y in range(H):
        cols = np.where(mask_no_batch[0,y,:])[0]
        if cols.size == 0:
            continue
        c_min, c_max = cols.min(), cols.max()

        for j in range(c_min):

            j_ref = min(c_min + (c_min - j), W-1)
            # filled[0,y,j] = filled[0,y,j_ref]
            
            valid_x_h.append(j_ref)
            invalid_x_h.append(j)
            # filled[0,i,x] = filled[0,i_ref,x]
            valid_y_h.append(y)
            invalid_y_h.append(y)

        for j in range(c_max+1, W):
            j_ref = max(c_max - (j - c_max), 0)
            # filled[0,y,j] = filled[0,y,j_ref]
            
            valid_x_h.append(j_ref)
            invalid_x_h.append(j)
            # filled[0,i,x] = filled[0,i_ref,x]
            valid_y_h.append(y)
            invalid_y_h.append(y)

    return torch.IntTensor(valid_x_v), torch.IntTensor(invalid_x_v), torch.IntTensor(valid_y_v), torch.IntTensor(invalid_y_v), torch.IntTensor(valid_x_h), torch.IntTensor(invalid_x_h), torch.IntTensor(valid_y_h), torch.IntTensor(invalid_y_h)


data_path = "/project/home/p200177/DE_371/datasets/dataset_Meteo_France_big_domain_optim"
file_name = "2021-06-17T21:00:00Z_u_v_t2m_0_0.npy"
file = os.path.join(data_path,file_name)

img = np.load(file)
img=torch.from_numpy(img).to("cuda")

img = img.unsqueeze(0)
img = img.permute((0,3,1,2))

img = img[:,:,3:715,:1120]
mask = (torch.abs(img) < 1000)

valid_x_vert,invalid_x_vert,valid_y_vert,invalid_y_vert,valid_x_horiz,invalid_x_horiz,valid_y_horiz,invalid_y_horiz = mirror_fill(img,mask)
