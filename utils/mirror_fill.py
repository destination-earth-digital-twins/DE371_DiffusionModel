import torch
import numpy as np

import numpy as np
import torch

def mirror_fill(img, mask):
    """
    Fills an img with invalid datas with valid datas, keeping a continuous physical aspect
    Args:
        img (torch.tensor) : an image of shape (batch_size,variables,latitude,longitude)
        mask (torch.tensor) : a tensor of img's shape containing True (valid datas of img) and False (invalid datas of img)
    """
    device, dtype = img.device, img.dtype
    img_np = img[0].cpu().numpy()      
    orig_valid = mask[0].cpu().numpy() 
    var, H, W = img_np.shape
    filled = img_np.copy()

    # Passe verticale (toujours à partir de orig_valid)
    for v in range(var):
        for x in range(W):
            rows = np.where(orig_valid[v,:,x])[0]
            if rows.size == 0:
                continue
            r_min, r_max = rows.min(), rows.max()
            # au-dessus
            for i in range(r_min):
                i_ref = min(r_min + (r_min - i), H-1)
                filled[v,i,x] = filled[v,i_ref,x]
            # en-dessous
            for i in range(r_max+1, H):
                i_ref = max(r_max - (i - r_max), 0)
                filled[v,i,x] = filled[v,i_ref,x]

    # Passe horizontale (toujours à partir de orig_valid)
    for v in range(var):
        for y in range(H):
            cols = np.where(orig_valid[v,y,:])[0]
            if cols.size == 0:
                continue
            c_min, c_max = cols.min(), cols.max()
            # à gauche
            for j in range(c_min):
                j_ref = min(c_min + (c_min - j), W-1)
                filled[v,y,j] = filled[v,y,j_ref]
            # à droite
            for j in range(c_max+1, W):
                j_ref = max(c_max - (j - c_max), 0)
                filled[v,y,j] = filled[v,y,j_ref]

    # Reconversion en tensor
    out = torch.from_numpy(filled).to(device=device, dtype=dtype)
    return out.unsqueeze(0)
