import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union
from absl import logging
import torch.nn.functional as F
import numpy as np

def downsample(x: torch.FloatTensor, include_t_dim: bool = True, factor: int = 2):
    if include_t_dim:
        return F.avg_pool3d(x, kernel_size=factor, stride=factor)
    else:
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

def downsample_module(include_t_dim: bool = True, factor: int = 2):
    if include_t_dim:
        return nn.AvgPool3d(kernel_size=factor, stride=factor)
    else:
        return nn.AvgPool2d(kernel_size=factor, stride=factor)
    


def cmp_output_label(output_tensor, label_tensor, atol = 0.001, rtol =  0.016):
    output_tensor = output_tensor.contiguous().view(-1).float()
    label_tensor = label_tensor.contiguous().view(-1).float()
    diff = torch.abs(output_tensor-label_tensor).float()
    print("diff>0.04 ratio:", (diff > 0.04).float().mean())
    print(f"max_abs_diff: {torch.max(diff).item()}, mean_abs_diff: {torch.mean(diff).item()}")
    idx = torch.argmax(diff)
    print(f"max_abs_diff_idx: {idx}, a[{idx}] = {output_tensor[idx]}, b[{idx}] = {label_tensor[idx]}")
    close = torch.isclose(output_tensor, label_tensor, atol=atol, 
                          rtol=rtol, equal_nan=False)
    # 找出不近似相等的元素
    not_close_indices = torch.where(~close)
    
    # 打印出不近似相等的元素
    cnt = 0
    for index in not_close_indices[0]:
        print(f"a[{index}] = {output_tensor[index]}, b[{index}] = {label_tensor[index]}, abs_err = {torch.abs(output_tensor[index]-label_tensor[index])}")
        cnt += 1
        if (cnt == 10):
            break
        


def cmp_output_label_numpy(output_tensor, label_tensor, atol = 0.001, rtol =  0.016):
    output_tensor = output_tensor.flatten()
    label_tensor = label_tensor.flatten()
    diff = np.abs(output_tensor-label_tensor)
    print("diff>0.04 ratio:", np.mean((diff > 0.04).astype(np.float32)))
    print(f"max_abs_diff: {np.max(diff)}, mean_abs_diff: {np.mean(diff)}")
    idx = np.argmax(diff)
    print(f"max_abs_diff_idx: {idx}, a[{idx}] = {output_tensor[idx]}, b[{idx}] = {label_tensor[idx]}")
    close = np.isclose(output_tensor, label_tensor, atol=atol, 
                          rtol=rtol, equal_nan=False)
    # 找出不近似相等的元素
    not_close_indices = np.where(~close)
    
    # 打印出不近似相等的元素
    cnt = 0
    for index in not_close_indices[0]:
        print(f"a[{index}] = {output_tensor[index]}, b[{index}] = {label_tensor[index]}, abs_err = {np.abs(output_tensor[index]-label_tensor[index])}")
        cnt += 1
        if (cnt == 10):
            break


class CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 'valid',
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        custom_padding=None,
    ) -> None:
        super(CausalConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 
                                         dilation, groups, bias, padding_mode, device, dtype)
        assert self.padding == 'valid', 'Must use VALID padding for raw Conv.'
        self.custom_padding = custom_padding
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        def pad_list2tuple(pad_list):
            tmp_pad = []
            for pad in pad_list:
                tmp_pad.extend(pad)
            return tuple(tmp_pad)
        
        if self.stride[0] > 1:
            tt = 0
        if self.custom_padding:
            if self.custom_padding == "edge_constant":
                pads = [(0,0) for k in self.kernel_size[::-1]]
                pads[len(self.kernel_size)-1] = (self.kernel_size[0] - 1, 0)
                pads = pads
                pads = pad_list2tuple(pads)

                input = F.pad(input, pads, mode="replicate")
                pads = [((k-1) // 2, k // 2) for k in self.kernel_size[::-1]]
                pads[len(self.kernel_size)-1] = (0, 0)
                pads = pads
                pads = pad_list2tuple(pads)
                input = F.pad(input, pads, mode="constant")
            else:
                pads = [((k-1) // 2, k // 2) for k in self.kernel_size[::-1]]
                pads[len(self.kernel_size)-1] = (self.kernel_size[0] - 1, 0)
                pads =  pads + [(0, 0)] + [(0, 0)]
                pads = pad_list2tuple(pads)
                input = F.pad(input, pads, mode=self.custom_padding)
        output = F.conv3d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return output
    

import torch
import torch.nn as nn

class CustomGroupNorm(nn.Module):
    """Group normalization (arxiv.org/abs/1803.08494) implemented in PyTorch.

    This op is similar to batch normalization, but statistics are shared across
    equally-sized groups of channels and not shared across batch dimension.
    Thus, group normalization does not depend on the batch composition and does
    not require maintaining internal state for storing statistics.
    The user should either specify the total number of channel groups or the
    number of channels per group.
    """
    def __init__(self, num_channels, num_groups=32, group_size=None, epsilon=1e-6, use_bias=True, use_scale=True):
        super(CustomGroupNorm, self).__init__()
        if (num_groups is None and group_size is None) or (num_groups is not None and group_size is not None):
            raise ValueError('Either `num_groups` or `group_size` should be specified. If `group_size` is to be specified, pass `num_groups=None`.')
        
        if group_size is not None:
            if num_channels % group_size != 0:
                raise ValueError('Number of channels ({}) is not multiple of the group size ({}).'.format(num_channels, group_size))
            num_groups = num_channels // group_size
        elif num_groups is not None:
            if num_channels % num_groups != 0:
                raise ValueError('Number of groups ({}) does not divide the number of channels ({}).'.format(num_groups, num_channels))
        
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale
        
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones(num_channels))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        N, C, *dims = x.shape
        x = x.view(N, self.num_groups, C // self.num_groups, *dims)
        # The dimension D does not participate in the calculation of mean and variance. 
        mean = x.mean(dim=(2, *range(x.ndim-2, x.ndim)), keepdim=True)
        var = x.var(dim=(2, *range(x.ndim-2, x.ndim)), keepdim=True, unbiased=False)

        x = (x - mean) / (var + self.epsilon).sqrt()
        x = x.view(N, C, *dims)

        if self.use_scale:
            x = self.scale.view(1, C, *([1] * (x.ndim - 2))) * x
        if self.use_bias:
            x = self.bias.view(1, C, *([1] * (x.ndim - 2))) + x

        return x

class CustomResBlock(nn.Module):
    """Basic Residual Block."""
    def __init__(self, in_channels, out_channels, activation_fn=nn.ReLU(), use_conv_shortcut=False):
        super(CustomResBlock, self).__init__()

        self.use_conv_shortcut = use_conv_shortcut

        self.CausalConv_0 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), bias=False, padding="valid", custom_padding="constant")
        self.GroupNorm_debug_0 = CustomGroupNorm(in_channels)
        self.CausalConv_1 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), bias=False, padding="valid", custom_padding="constant")
        self.GroupNorm_debug_1 = CustomGroupNorm(out_channels)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.CausalConv_2 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), bias=False, padding="valid", custom_padding="constant")
            else:
                self.CausalConv_2 = CausalConv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False, padding="valid", custom_padding="constant")
        self.activation_fn = activation_fn

    def forward(self, x):
        residual = x
        x = self.GroupNorm_debug_0(x)
        x = self.activation_fn(x)
        x = self.CausalConv_0(x)
        x = self.GroupNorm_debug_1(x)
        x = self.activation_fn(x)
        x = self.CausalConv_1(x)

        if x.shape[1] != residual.shape[1]:
            residual = self.CausalConv_2(residual)
        return x + residual

