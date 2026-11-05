import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', img_size=256, in_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        if to_pixel == 'linear':
            self.model = nn.Linear(in_dim, in_channels * patch_size * patch_size)
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1], print(h, w, x.shape[1])
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        
        return x