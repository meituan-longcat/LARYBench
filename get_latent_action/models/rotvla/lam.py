import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from .blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from .to_pixel import ToPixel

from torchvision import transforms
import copy 
from torch.cuda.amp import autocast
import numpy as np
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):

    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity = flat_affinity / temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

class SoftVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_latents,
        latent_dim,
        entropy_loss_ratio=0.01,
        tau=0.07,
        l2_norm=True,
        code_restart: bool = True,
        num_codes: int =4 ,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.tau = tau
        self.embedding = nn.Parameter(torch.randn(num_codes,num_latents, latent_dim))
        self.embedding.data.uniform_(-1.0 / self.num_latents, 1.0 / self.num_latents)
        self.register_buffer("usage", torch.zeros(num_codes,num_latents), persistent=True)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)
        
        
        self.code_restart = code_restart


    def update_usage(self, min_enc) -> None:
        
        for i , idx in enumerate(min_enc.transpose(0,1)):
            self.usage[i][idx] = self.usage[i][idx] + 1  # Add used code


    def random_restart(self) -> None:

        return 


    def reset_usage(self) -> None:

        #return 
        if self.code_restart:
            # Reset usage between epochs
            self.usage.zero_()

            if hasattr(self, "inner_vq"):
                self.inner_vq.reset_usage()

    

    def forward(self, z,ind):
        # Handle different input shapes
        batch_size, seq_length, _ = z.shape
        

        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z, p=2, dim=-1)
        else :
            z_segments = z
        logits = torch.einsum('ble, lke -> blk', z_segments, embedding.detach())
        probs = F.softmax(logits / self.tau, dim=-1)  
        z_q = torch.einsum('btk, tke -> bte', probs, embedding)


        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (batch*segment_length, num_codebooks)
        if not self.training or self.code_restart:
            self.update_usage(indices)

        
        zq_z_cos = torch.tensor(0)
        
        return z_q,z,logits,indices,zq_z_cos
        


class SoftDINOLatentActionModel(nn.Module):
    """
    Latent action Soft-VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            num_codes: int=4,
            dropout: float = 0.0,
            use_identity: int = 0,
    ) -> None:
        super(SoftDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # DINOv2 visual encoder. Fetched from torch.hub by default; set
        # DINOV2_REPO_PATH to a local clone of facebookresearch/dinov2 to load offline.
        dinov2_local = os.environ.get("DINOV2_REPO_PATH")
        if dinov2_local:
            self.dino_encoder = torch.hub.load(dinov2_local, 'dinov2_vitb14_reg', source='local')
        else:
            self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        dino_dim = 768

        self.num_codes = num_codes

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)


        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)

        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.to_pixel = ToPixel(to_pixel='linear', img_size=224, in_channels=3,
                                in_dim=dino_dim, patch_size=self.patch_size)
        self.vq_action = SoftVectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
            num_codes=num_codes,
        )
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.use_identity = use_identity
        if self.use_identity :
            self.register_buffer("identity", torch.eye(num_codes).unsqueeze(0).unsqueeze(0))

    def vq_encode(self, videos: Tensor) -> Dict:
        B, T = videos.shape[:2]
        
        
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)

        

        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=T)
    
        def vq_encode_(ind=[0,1]):
            t = len(ind)
            padded_patches = dion_features[:,ind]
            action_pad = self.action_latent.expand(B, t, -1, -1)
            padded_patches = torch.cat([action_pad, padded_patches], dim=2)

            # Encode
            z = self.encoder(padded_patches) 


            z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

            # Vector quantize
            z_action = z_action.reshape(B * (T - 2), self.num_codes, self.latent_dim)
            z_q, z, emb, indices, zq_z_cos = self.vq_action(z_action,ind)
            z_m = z_q.clone()
            dtype = z_q.dtype
            
            with autocast(enabled=False):
                z_q_fp32 = z_q.float()
                U, V, Vh = torch.linalg.svd(z_q_fp32, full_matrices=False)

                det_uv = torch.det(U @ Vh)
                D = torch.eye(U.shape[-1], device=z_q.device).unsqueeze(0).repeat(B, 1, 1)
                D[:, -1, -1] = det_uv
                # Rotation matrix in SO(n)
                z_q = (U @ D @ Vh).to(dtype)

            z_q = z_q.reshape(B, T - 2, self.num_codes, self.latent_dim)
            return {
                #"patches": dion_features,
                "z_q": z_q,
                "z": z,
                "z_m": z_m,
                "emb": emb,
                "indices": indices,
                "zq_z_cos": zq_z_cos,
            }

        out_one = vq_encode_([0,1])
        out_two = vq_encode_([1,2])
            
        z_q = out_one["z_q"]@ self.identity.repeat(B,1,1,1).transpose(2,3) @ out_two["z_q"]


        z = out_one["z"]@out_two["z"]

        out_three = {
            "z_q": z_q,
            "z": z,
            "emb": out_two["emb"],
            "indices": out_two["indices"],
        }



        return {
            "patches": dion_features,
            "z_q":[out_one["z_q"],out_two["z_q"],out_three["z_q"]],
            "z": [out_one["z"],out_two["z"],out_three["z"]],
            "z_m":[out_one["z_m"],out_two["z_m"]],
            "emb": [out_one["emb"],out_two["emb"],out_three["emb"]],
            "indices": [out_one["indices"],out_two["indices"],out_three["indices"]],
            "zq_z_cos": [out_one["zq_z_cos"],out_two["zq_z_cos"] ],
        }
    
    def forward(self, batch: Dict,order=None,epoch_len=0) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2] 
        assert T==3, 'T should be 3'
        H, W = batch["videos"].shape[3:5]

        if (self.use_identity>0 and order is not None):
            batch = copy.deepcopy(batch)
            videos_copy = copy.deepcopy(batch["videos"])
            batch["videos"][:,0] = videos_copy[:,int(order[0])]
            batch["videos"][:,1] = videos_copy[:,int(order[1])]
            batch["videos"][:,2] = videos_copy[:,int(order[2])]
        
    
        


        outputs = self.vq_encode(batch["videos"]) 


        def decode(ind):

            # Decode
            video_patches = self.patch_up(outputs["patches"][:, [ind]])
            video_action_patches = torch.cat([self.action_up(outputs["z_q"][ind]), 
                                                video_patches],
                                                dim=2)

            video_recon = self.decoder(video_action_patches)



            video_recon = video_recon[:, :, -video_patches.shape[2]:] 
            assert video_recon.shape[1]==1
            
            video_recon = self.to_pixel(video_recon[:,0])
            return video_recon
        
        video_recon_one =  decode(0)
        video_recon_two =  decode(1)

        video_patches = self.patch_up(outputs["patches"][:, [0]])

        video_action_patches = torch.cat([self.action_up(outputs["z_q"][2]), 
                                            video_patches],
                                            dim=2)

        video_recon_three = self.decoder(video_action_patches)

        video_recon_three = video_recon_three[:, :, -video_patches.shape[2]:]   

        video_recon_three = self.to_pixel(video_recon_three[:,0])
        outputs.update(
            {
                "recon": [video_recon_one,video_recon_two,video_recon_three],
                "target": batch["videos"][:,1:].detach(),#[:, [-1]]
                "first": batch["videos"][:,[0]].detach()
            }
        )

        return outputs

    def get_latent_actions(self,videos):

        B, T = videos.shape[:2]
        
        
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)

        

        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=T)
    
        def vq_encode_(ind=[0,1]):
            t = len(ind)
            padded_patches = dion_features[:,ind]
            action_pad = self.action_latent.expand(B, t, -1, -1)
            padded_patches = torch.cat([action_pad, padded_patches], dim=2)

            # Encode
            z = self.encoder(padded_patches) 


            z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

            # Vector quantize
            z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
            z_m, z, emb, indices, zq_z_cos = self.vq_action(z_action,ind)

            dtype = z_m.dtype
            
            with autocast(enabled=False):
                z_q_fp32 = z_m.float()
                U, V, Vh = torch.linalg.svd(z_q_fp32, full_matrices=False)

                det_uv = torch.det(U @ Vh)
                D = torch.eye(U.shape[-1], device=z_q_fp32.device).unsqueeze(0).repeat(B, 1, 1)
                D[:, -1, -1] = det_uv
                # Rotation matrix in SO(n)
                z_q = (U @ D @ Vh).to(dtype)


            z_q = z_q.reshape(B,(T - 1), self.num_codes, self.latent_dim)

            return {
                #"patches": dion_features,
                "z_q": z_q,
                "z_m": z_m,
                "z": z,
                #"emb": emb,
                "indices": indices,
                #"zq_z_cos": zq_z_cos,
            }
        out_one = vq_encode_([0,1])
        
        return out_one
    
    def decode_latent(self,videos,zq_1,zq_2,zq_identity):
        B, T = videos.shape[:2]
        
        
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)

        

        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=T)
        video_patches = self.patch_up(dion_features)
       
        def decode(zq):

            # Decode
            
            video_action_patches = torch.cat([self.action_up(zq), 
                                                video_patches[:,[0]]],
                                                dim=2)

            video_recon = self.decoder(video_action_patches)



            video_recon = video_recon[:, :, -video_patches.shape[2]:] 
            assert video_recon.shape[1]==1
            video_recon = self.to_pixel(video_recon[:,0])
            return video_recon
        

        dtype = zq_2.dtype
        
        with autocast(enabled=False):
            z_q_fp32 = zq_2.float()
            U, V, Vh = torch.linalg.svd(z_q_fp32, full_matrices=False)

            det_uv = torch.det(U @ Vh)
            D = torch.eye(U.shape[-1], device=zq_2.device).unsqueeze(0).repeat(zq_2.shape[0], 1, 1)
            D[:, -1, -1] = det_uv
            # Rotation matrix in SO(n)
            zq_2 = (U @ D @ Vh).to(dtype)


        zq_2 = zq_2.reshape(B,(T - 1), self.num_codes, self.latent_dim)
        video_recon = decode(zq_1)
        video_recon2 = decode(zq_2)
        video_recon3 = decode(zq_identity)
        return {"video_recon":video_recon,"video_recon2":video_recon2,"video_identity":video_recon3,"zq_1":zq_1,"zq_2":zq_2,"zq_identity":zq_identity}
    
    @property
    def device(self):
        return next(self.parameters()).device

