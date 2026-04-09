from .model_utils import CustomResBlock, downsample_module, CustomGroupNorm, CausalConv3d, cmp_output_label
import torch.nn as nn
import torch.nn.functional as f
import torch
from omegaconf import OmegaConf
from .LFQ import LFQ
import numpy as np
import pickle

class NamedModuleList(nn.Module):
    def __init__(self, named_modules):
        super(NamedModuleList, self).__init__()
        self._modules = {}
        for name, module in named_modules:
            self._modules[name] = module

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        else:
            raise TypeError("Index must be a string")

    def __setitem__(self, idx, module):
        if isinstance(idx, str):
            self._modules[idx] = module
        else:
            raise TypeError("Index must be a string")

    def __delitem__(self, idx):
        if isinstance(idx, str):
            del self._modules[idx]
        else:
            raise TypeError("Index must be a string")

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.filters = config['vqvae']['filters']
        self.num_res_blocks = config['vqvae']['num_enc_res_blocks']
        self.channel_multipliers = config['vqvae']['channel_multipliers']
        self.temporal_downsample = config['vqvae']['temporal_downsample']
        self.embedding_dim = config['vqvae']['embedding_dim']
        self.conv_downsample = config['vqvae']['conv_downsample']
        self.custom_conv_padding = config['vqvae'].get('custom_conv_padding', None)
        self.norm_type = config['vqvae']['norm_type']
        self.num_remat_block = config['vqvae'].get('num_enc_remat_blocks', 0)
        activation_fn = config['vqvae']['activation_fn']
        if activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn == 'swish':
            self.activation_fn = nn.SiLU()
        else:
            raise NotImplementedError
        self.blocks = []
        self.build_blocks()
        # only for weight named
        self.encoder = NamedModuleList(self.blocks)

    def build_blocks(self):
        filters = self.filters
        casual_conv_name = "CausalConv_"
        casual_conv_idx = 0
        downsample_name = "downsample_"
        downsample_idx = 0
        resnet_name = "ResBlock_"
        resnet_idx = 0
        self.blocks.append((casual_conv_name+str(casual_conv_idx),
                            CausalConv3d(in_channels=3, out_channels=filters, kernel_size=(3, 3, 3), 
                                         padding="valid", custom_padding="constant")))
        casual_conv_idx += 1
        num_blocks = len(self.channel_multipliers)
        pre_filters = filters
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            # repeat computing for training don't realize
            for _ in range(self.num_res_blocks):
                self.blocks.append((resnet_name+str(resnet_idx),
                                    CustomResBlock(pre_filters, filters, self.activation_fn)))
                resnet_idx += 1
                pre_filters = filters
            if i < num_blocks - 1:
                if self.conv_downsample:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    self.blocks.append((casual_conv_name+str(casual_conv_idx),
                                        CausalConv3d(in_channels=pre_filters, out_channels=filters, 
                                                     kernel_size=(3, 3, 3), stride=(t_stride, 2, 2),
                                                     bias=True, padding="valid", 
                                                     custom_padding="constant")))
                    pre_filters = filters
                    casual_conv_idx += 1
                else:
                    self.blocks.append((downsample_name+str(downsample_idx),
                                        downsample_module(self.temporal_downsample[i])))
                    downsample_idx += 1
            
        for _ in range(self.num_res_blocks):
            self.blocks.append((resnet_name+str(resnet_idx), 
                                CustomResBlock(filters, filters, self.activation_fn)))
            resnet_idx += 1
        self.blocks.append(("GroupNorm_debug_0", CustomGroupNorm(filters)))
        self.blocks.append(("activation_fn", self.activation_fn))
        self.blocks.append((casual_conv_name+str(casual_conv_idx),
                            CausalConv3d(in_channels=filters, out_channels=self.embedding_dim, 
                                         kernel_size=(1, 1, 1), bias=True, padding="valid", 
                                         custom_padding="constant")))

    def forward(self, x, is_train=False):
        for _, block in self.blocks:
            x = block(x)
        return x

class LFQFIXVQVAE(nn.Module):
    
    def __init__(self, config) -> None:
        super(LFQFIXVQVAE, self).__init__()
        self.encoder = Encoder(config)
        self.quantizer = LFQ(config)
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x, indices = self.quantizer(x, False)
        return x, indices
    

def lfq_weight_load(model, weight_path):
    with open(weight_path, "rb") as f:
        data = pickle.load(f)
    params_dict = data["torch_params"]
    for name, param in model.named_parameters():
        if name in params_dict:
            param_v = params_dict[name]
            # 3,3,3, 3, 128->128, 3, 3, 3, 3
            if "Conv" in name and "bias" not in name:
                param_v = np.transpose(param_v, (4, 3, 0, 1, 2))
            if "project" in name and "weight" in name:
                param_v = np.transpose(param_v, (1, 0))
            param.data.copy_(torch.from_numpy(param_v).cuda())
    
    
    
    
    
    
