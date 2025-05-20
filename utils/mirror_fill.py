import torch
import numpy as np


def mirror_fill(img, mask):
    """
    Fills an img with invalid datas with valid datas, keeping a continuous physical aspect
    Args:
        img (torch.tensor) : an image of shape (batch_size,variables,latitude,longitude)
        mask (torch.tensor) : a tensor of img's shape containing True (valid datas of img) and False (invalid datas of img)
    """
    device, dtype = img.device, img.dtype
    img_np = img[0].cpu().numpy()      
    mask = mask[0].cpu().numpy() 
    var, H, W = img_np.shape
    filled = img_np.copy()
    valid_index_row = []
    invalid_index_row=[]
    valid_index_row_r = []
    invalid_index_row_r=[]
    # vertical filling
    for x in range(W):
        rows = np.where(mask[0,:,0])[0]
        if rows.size == 0:
            continue
        r_min, r_max = rows.min(), rows.max()
        
        for j in range(r_min):
            j_ref = min(r_min + (r_min - j), H-1)
            invalid_index_row.append(j)
            valid_index_row.append(j_ref)
            filled[0,j,x] = filled[0,j_ref,x]

        for j in range(r_max+1, H):
            invalid_index_row_r.append(j)
            valid_index_row_r.append(j_ref)
            j_ref = max(r_max - (j - r_max), 0)
            filled[0,j,x] = filled[0,j_ref,x]

    # horizontal pass
    for y in range(H):
        cols = np.where(mask[0,y,:])[0]
        if cols.size == 0:
            continue
        c_min, c_max = cols.min(), cols.max()

        for j in range(c_min):

            j_ref = min(c_min + (c_min - j), W-1)
            filled[0,y,j] = filled[0,y,j_ref]
            # valid_index_col.append(j_ref)
            # invalid_index_col.append(j)

        for j in range(c_max+1, W):
            j_ref = max(c_max - (j - c_max), 0)
            filled[0,y,j] = filled[0,y,j_ref]
            # valid_index_col.append(j_ref)
            # invalid_index_col.append(j)

    out = torch.from_numpy(filled).to(device=device, dtype=dtype)
    return out.unsqueeze(0)



def compute_mirror_indices(mask):
    """
    mask : torch.Tensor de shape (1, var, H, W), dtype=bool
    renvoie :
      invalid_idx, source_idx :
      deux tuples de LongTensor (v_idx, i_idx, j_idx) de même longueur N
      tels que pour tout 0 <= k < N :
          filled[0, v_idx[k], i_idx[k], j_idx[k]] =
              img[0, v_idx[k], source_i[k], j_idx[k]]
    """
    mask_np = mask[0].cpu().numpy()      # (var, H, W)
    var, H, W = mask_np.shape

    inv_i, inv_j = [], []
    src_i, src_j = [], []

    # ——— vertical mirror ———
    for v in range(var):
        for x in range(W):
            rows = np.where(mask_np[v,:,x])[0]
            if rows.size == 0:
                continue
            r_min, r_max = rows.min(), rows.max()

            # au-dessus de r_min
            above = np.arange(r_min)
            i_ref = np.minimum(r_min + (r_min - above), H-1)
            for i, ir in zip(above, i_ref):
                inv_i.append(i) 
                inv_j.append(x)
                src_i.append(ir) 
                src_j.append(x)

            # en dessous de r_max
            below = np.arange(r_max+1, H)
            i_ref = np.maximum(r_max - (below - r_max), 0)
            for i, ir in zip(below, i_ref):
                inv_i.append(i)
                inv_j.append(x)
                src_i.append(ir)
                src_j.append(x)

    # ——— horizontal mirror ———
    for v in range(var):
        for y in range(H):
            cols = np.where(mask_np[v,y,:])[0]
            if cols.size == 0:
                continue
            c_min, c_max = cols.min(), cols.max()

            # à gauche de c_min
            left = np.arange(c_min)
            j_ref = np.minimum(c_min + (c_min - left), W-1)
            for j, jr in zip(left, j_ref):
                inv_i.append(y)
                inv_j.append(j)
                src_i.append(y)
                src_j.append(jr)

            # à droite de c_max
            right = np.arange(c_max+1, W)
            j_ref = np.maximum(c_max - (right - c_max), 0)
            for j, jr in zip(right, j_ref):
                inv_i.append(y)
                inv_j.append(j)
                src_i.append(y)
                src_j.append(jr)

    # convertir en LongTensor
    device = mask.device
    invalid_idx = (
        torch.IntTensor(inv_i).to(device),
        torch.IntTensor(inv_j).to(device),
    )
    source_idx = (
        torch.IntTensor(src_i).to(device),
        torch.IntTensor(src_j).to(device),
    )

    return invalid_idx, source_idx


