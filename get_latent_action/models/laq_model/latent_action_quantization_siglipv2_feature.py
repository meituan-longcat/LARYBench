from pathlib import Path
import math
import os
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from get_latent_action.models.laq_model.attention import Transformer, ContinuousPositionBias
from get_latent_action.models.laq_model.nsvq import NSVQ
from transformers import AutoConfig, AutoModel
import omegaconf
from omegaconf import OmegaConf
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import yaml
from functools import partial
from timm.models.layers import trunc_normal_

import sys

def exists(val):
    return val is not None

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

class SigLip2VisionEncoder(nn.Module):
    def __init__(
            self, 
            model_path=None):
        super().__init__()

        self.select_layer = -1

        if model_path is None:
            model_path = os.environ.get("SIGLIP2_PATH")
            if model_path is None:
                raise ValueError("SIGLIP2_PATH environment variable is not set.")
        model_path = model_path.replace("hdfs:///", "/mnt/hdfs/")
        self.vision_model = AutoModel.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True).vision_model

    def forward(self, pixel_values: torch.FloatTensor):
        vit_embeds = self.extract_feature(pixel_values) # (B<vit_batch_size>, N, C)
        return vit_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        # if self.ps_version == 'v1':
        #     warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
        #                   'which results in a transposed image.')
        # else:
        x = x.permute(0, 2, 1, 3).contiguous() # back to (N, W, H, C)
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        #SigLIP2 没有CLS Token
        # vit_embeds = vit_embeds[:, 1:, :]

        return vit_embeds

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_backbone(backbone):
    for p in backbone.parameters():
        if hasattr(p, "requires_grad") and p.requires_grad is not None:
            p.requires_grad = False
    backbone = backbone.eval()
    backbone.train = disabled_train

class LatentActionQuantizationSiglipv2Feature(nn.Module):
    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head = 64,
        heads = 8,
        channels = 3,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.encoder = SigLip2VisionEncoder()
 
        freeze_backbone(self.encoder)

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)

        self.vq = NSVQ(
            dim=dim, # 1024 不用改
            num_embeddings=codebook_size, # 8 不用改
            embedding_dim=quant_dim, # 32 不用改
            device='cuda',
            code_seq_len=code_seq_len, # 4 可以改
            patch_size=patch_size, # 32 可能需要改，可能需要匹配
            image_size=image_size, # 224， 
        )
            
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict = False)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]

        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        
        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]
        
        return first_tokens, last_tokens

    def decode(
        self,
        tokens,
        actions,
    ):
        # tokens: [64, 1, 7, 7, 1024]
        # actions: [64, 1, 4, 4, 1024]
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        rest_frames_tokens = tokens # 16, 1, 16, 16, 1024

        return rest_frames_tokens
    
    def forward(
        self,
        video,
        step = 0,
        mask = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:] # 64, 3, 1, 224, 224 | 64, 3, 1, 224, 224
        
        with torch.no_grad():
            first_frame_tokens = self.encoder.extract_feature(first_frame.squeeze(2)).detach() # outputs.last_hidden_state: [bs, 257, 1024]
            first_frame_tokens = first_frame_tokens.unsqueeze(1)  # [bs, 1, 1024, dim] dim = 1152
            batch_size, _, _, dim = first_frame_tokens.shape
            first_frame_tokens = first_frame_tokens.reshape(batch_size, 1, 14, 14, dim)

            rest_frames_tokens = self.encoder.extract_feature(rest_frames.squeeze(2)).detach() # outputs.last_hidden_state: [bs, 257, 1024]
            rest_frames_tokens = rest_frames_tokens.unsqueeze(1)  # [bs, 1, 1024, dim] dim = 1152
            batch_size, _, _, dim = rest_frames_tokens.shape
            rest_frames_tokens = rest_frames_tokens.reshape(batch_size, 1, 14, 14, dim)

        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1) # 64, 2, 16, 16, 1024

        shape = tokens.shape
        *_, h, w, _ = shape # h w 为7

        # 对前后帧进行时序、空间上的建模
        first_tokens, last_tokens = self.encode(tokens) # 64, 1, 16, 16, 1024 | 64, 1, 16, 16, 1024

        '''
        first_tokens: 64, 49, 1024 
        first_packed_fhw_shape: 1, 7, 7
        last_tokens: 64, 49, 1024 
        last_packed_fhw_shape: 1, 7, 7
        下面的操作相当于是将二维进行一维化
        '''
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')
        
        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        self.lookup_free_quantization = False
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        '''
        tokens: 64, 4, 1024
        perplexity: scalar
        codebook_usage: 8
        indices: 64, 4
        '''
        tokens, perplexity, codebook_usage, indices = self.vq(first_tokens, last_tokens, codebook_training_only = False)
        num_unique_indices = indices.unique().size(0)
        
        if ((step % 10 == 0 and step < 100)  or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
            print(f"update codebook {step}")
            self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return tokens, indices
        
        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return
        
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w) # [64, 1, 2, 2, 1024]
        concat_tokens = first_frame_tokens.detach() # + tokens [64, 1, 7, 7, 1024]
        recon_features = self.decode(concat_tokens, tokens)

        returned_recon = rearrange(recon_features, 'b 1 h w c -> b c h w')
        video_features = rest_frames_tokens 

        if return_recons_only:
            returned_first = rearrange(first_frame_tokens.detach(), 'b 1 h w c -> b c h w')
            returned_second = rearrange(rest_frames_tokens.detach(), 'b 1 h w c -> b c h w')
            return returned_first, returned_second, returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video_features, recon_features, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video_features, recon_features)

        return recon_loss, num_unique_indices
        

    def inference(
        self,
        video,
        step = 0,
        mask = None,
        return_only_codebook_ids=False,
        user_action_token_num=None
    ):
        
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        # rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)
        with torch.no_grad():
            first_frame_tokens = self.encoder.extract_feature(first_frame.squeeze(2)).detach() # outputs.last_hidden_state: [bs, 257, 1024]
            first_frame_tokens = first_frame_tokens.unsqueeze(1)  # [bs, 1, 1024, dim] dim = 1152
            batch_size, _, _, dim = first_frame_tokens.shape
            first_frame_tokens = first_frame_tokens.reshape(batch_size, 1, 14, 14, dim)

            rest_frames_tokens = self.encoder.extract_feature(rest_frames.squeeze(2)).detach() # outputs.last_hidden_state: [bs, 257, 1024]
            rest_frames_tokens = rest_frames_tokens.unsqueeze(1)  # [bs, 1, 1024, dim] dim = 1152
            batch_size, _, _, dim = rest_frames_tokens.shape
            rest_frames_tokens = rest_frames_tokens.reshape(batch_size, 1, 14, 14, dim)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)


        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, user_action_token_num=user_action_token_num)
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        
    
        if return_only_codebook_ids:
            return tokens, indices

        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return
        

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens #.detach() #+ tokens
        recon_video = self.decode(concat_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 
        
        return returned_recon