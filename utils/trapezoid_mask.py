import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trapezoid_mask(H, W, top_width, bottom_width, height):
    
    ys = np.arange(H)[:, None]        
    xs = np.arange(W)[None, :]        
    center_x = W / 2

    y0 = (H - height) // 2           
    y1 = y0 + height - 1             
    
    mask = np.zeros((H, W), dtype=bool)


    for y in range(y0, y1 + 1):
        t = (y - y0) / (height - 1) 
        half_w = (1 - t) * (top_width / 2) + t * (bottom_width / 2)
        mask[y, :] = np.abs(xs - center_x) <= half_w

    return mask

