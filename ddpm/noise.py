import numpy as np
import torch
from scipy.fftpack import dct 
import torch.fft as fft
import matplotlib.pyplot as plt


def make_base_noise(base: str='white', dims: tuple=(1,)) -> torch.tensor:
    """
    Create base noise tensor with torch primitives
    """
    
    assert len(dims)>0, "Invalid dims given for noise generation"
    
    match base with:
        case 'white':
            return torch.randn(dims)
        
        case _:
            raise ValueError(f"Base noise {base} is not implemented")
        
def filter_noise(noise: torch.tensor, target: str='pink', dims: tuple=(0,1))-> torch.tensor:
    """
    Transform given noise tensor into target colored noise
    The transformation appends on dimension pairs (2D filtering)
    """
    
    assert len(dims)==2, f"Cannot operate on Non-2D fields, got dims={dims}"
    assert len(noise.shape)>=len(dims), f"Noise tensor has shape {noise.shape}, incompatible with dims {dims}"
    
    match target with:
        case 'pink':
            ft_arr = fft.fftshift(fft.fft2(noise,dim=dims,norm="ortho"))

            # Pink noise filtering
            beta = 1.5
            eps=1e-8
            pink_ft_arr = ft_arr / np.maximum(f ** beta, eps)

            # Inverse FFT to get pink noise
            noise = fft.ifft2(fft.ifftshift(pink_ft_arr), dim=dims, norm="ortho").real

        case _:
            raise ValueError(f"Target noise {target} is not implemented")
    
    return noise
    

def make_noise(base: str='white', target: str='white', dims: tuple=(1,)) -> torch.tensor
    """
    Create noise with desired color from base noise
    """
    
    noise = make_base_noise(base, dims=dims)
    
    if base==target:
        return base_noise
    
    noise = filter_noise(noise, target, dims=(-1,-2))
    
    return noise