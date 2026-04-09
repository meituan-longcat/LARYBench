from einops import rearrange, reduce, pack, unpack
from transformers import Dinov2Model
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from omegaconf import OmegaConf
import os
from utils.model_utils import print_model_params

env_model = os.environ.get("USE_MODEL")
model_dir = os.environ.get("MODEL_DIR")
if env_model == 'villa-x':
    from get_latent_action.models.villa_x.lam.model import IgorModel #只适配villa-x
elif env_model == 'flux2':
    from get_latent_action.models.flux2.src.flux2.util import load_ae
    from get_latent_action.models.flux2.src.flux2.sampling import encode_video_batch_refs_final 
elif env_model == 'wan2-2':
    from get_latent_action.models.wan2_2.wan.modules.vae2_2 import Wan2_2_VAE # 只是适配wan
elif env_model == 'vjepa2':
    from get_latent_action.models.vjepa2.evals.video_classification_frozen.models import init_module
else:
    from get_latent_action.tokenizer import get_dinov3_tokenizer, get_dinov3_reps
    from get_latent_action.models.laq_model import LatentActionQuantization, LatentActionQuantizationDinov2Feature, LatentActionQuantizationDinov3Feature, LatentActionQuantizationSiglipv2Feature, LatentActionQuantizationMagvit2
    from get_latent_action.models.univla.genie.model import ControllableDINOLatentActionModel



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

def get_dynamic_tokenizer(model):
    if model == 'lapa':
        # load lapa dynamics 
        dynamics = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=224,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        ).cuda()
        dynamics.load(f"{model_dir}/laq_openx.pt")
    elif model == 'magvit2':
        dynamics = LatentActionQuantizationMagvit2(
        dim = 18,
        quant_dim=32,
        codebook_size = 8,
        image_size = 224,
        patch_size = 16,
        spatial_depth = 4, #8
        temporal_depth = 4, #8
        dim_head = 64,
        heads = 16,
        code_seq_len=16,
        model_type='en18',
        ).cuda()
        dynamics.load(f"{model_dir}/magvit2.pt")
    elif model == 'dinov2':
        # load dino dynamics 
        dynamics = LatentActionQuantizationDinov2Feature(
            dim = 1024,
            quant_dim=32,
            codebook_size = 8,
            image_size = 224,
            patch_size = 14,
            spatial_depth = 4, #8
            temporal_depth = 4, #8
            dim_head = 64,
            heads = 16,
            code_seq_len=16,
        ).cuda()
        dynamics.load(f"{model_dir}/laq_dinov2.pt")
    elif model == 'dinov3':
        # load dinov3 dynamics 
        dynamics = LatentActionQuantizationDinov3Feature(
        dim = 1024,
        quant_dim=32,
        codebook_size = 8,
        image_size = 224,
        patch_size = 16,
        spatial_depth = 4, #8
        temporal_depth = 4, #8
        dim_head = 64,
        heads = 16,
        code_seq_len=16,
        ).cuda()
        pkg = torch.load(f"{model_dir}/laq_dinov3.pt")
        dynamics.load_state_dict(pkg['model'])
    elif model == 'siglip2':
        # load siglip2 dynamics
        dynamics = LatentActionQuantizationSiglipv2Feature(
        dim = 768,
        quant_dim=32,
        codebook_size = 8,
        image_size = 224, # 512
        patch_size = 16,
        spatial_depth = 4, #8
        temporal_depth = 4, #8
        dim_head = 64,
        heads = 16,
        code_seq_len=16,
        ).cuda()
        dynamics.load(f"{model_dir}/siglip2.pt")
    elif model == 'univla':
        # load univla dynamics
        dynamics = ControllableDINOLatentActionModel(
            in_dim=3,
            model_dim=768,
            latent_dim=128,
            num_latents=16,
            patch_size=14,
            enc_blocks=12,
            dec_blocks=12,
            num_heads=12,
            dropout=0.,
        ).cuda()
        univla_ckpt_path = os.environ.get("UNIVLA_CKPT_PATH", os.path.join(model_dir or "", "univla-latent-action-model", "lam-stage-2.ckpt"))
        ckpt = torch.load(univla_ckpt_path)["state_dict"]
        dynamics.load_state_dict(
            {k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True
        )
    elif model == 'villa-x':
        villa_x_lam_path = os.environ.get("VILLA_X_CKPT_PATH")
        dynamics = IgorModel.from_pretrained(villa_x_lam_path).cuda()
    elif model == 'flux2':
        dynamics = load_ae("flux.2-dev", device="cuda")
    elif model == 'wan2-2':
        # 注意：这里路径建议改为 args 传入或保持你脚本中的硬编码
        dynamics = Wan2_2_VAE(
            vae_pth=os.environ.get("WAN22_VAE_PATH"),
            device="cuda"
        )
    elif model == 'vjepa2':
        args_model = {
            "encoder": {
                "checkpoint_key": "target_encoder",
                "img_temporal_dim_size": None,
                "model_name": "vit_large",
                "patch_size": 16,
                "tubelet_size": 2,
                "uniform_power": True,
                "use_rope": True
            }
        }

        args_wrapper = {
            "max_frames": 128,
            "use_pos_embed": False
        }

        dynamics = init_module(
        module_name='get_latent_action.models.vjepa2.evals.video_classification_frozen.modelcustom.vit_encoder_multiclip',
        frames_per_clip=16,
        resolution=224,
        checkpoint=os.environ.get("VJEPA2_CKPT_PATH"),
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device="cuda",
        )
    elif model == 'dinov3-origin':
        dynamics = get_dinov3_tokenizer()
    elif 'dinov3-cs' in model:
        import re
        cs_match = re.search(r'cs(\d+)', model)
        sl_match = re.search(r'sl(\d+)', model)
        dim_match = re.search(r'dim(\d+)', model)
        lr_match = re.search(r'lr([0-9.]+[eE][-+]?[0-9]+)', model)

        # 转换为整数 (加一个判空保护是个好习惯)
        cs = int(cs_match.group(1)) if cs_match else None
        sl = int(sl_match.group(1)) if sl_match else None
        dim = int(dim_match.group(1)) if dim_match else None
        lr = lr_match.group(1)

        dynamics = LatentActionQuantizationDinov3Feature(
        dim = 1024,
        quant_dim=dim,
        codebook_size = cs,
        image_size = 224,
        patch_size = 16,
        spatial_depth = 4, #8
        temporal_depth = 4, #8
        dim_head = 64,
        heads = 16,
        code_seq_len=sl,
        ).cuda()
        pkg = torch.load(f"{model_dir}/dinov3_cs{str(cs)}_sl{str(sl)}_dim{str(dim)}_lr{lr}.pt")
        dynamics.load_state_dict(pkg['model'])
    if model == 'wan2-2':
        print_model_params(dynamics.model)
    else:
        print_model_params(dynamics)
        
    # freeze 
    if model == 'wan2-2':
        freeze_backbone(dynamics.model)
    elif model == 'dinov3-origin':
        pass
    else:
        freeze_backbone(dynamics)

    return dynamics



def get_lavit_dino_magvit_dynamics_indices(x, lavit_tokenizer, dino_tokenizer, magvit_tokenizer, dinov3_tokenizer, siglipv2_tokenizer):
    '''
    expect input_tensor has the same property as the input of magvit:
        输入是0-1的tensor图像就可以了, 输入形状应该为类似：64, 3, 2, 224, 224
        至于一些特殊的变换，在dynamic tokenizer内部forward的时候会自动进行合适的处理，这里只要保证x是0-1的tensor 图像就可以了
    '''
    with torch.no_grad():
        # lavit_index_batch = lavit_tokenizer(x, return_only_codebook_ids=True) # [bs,16]
        # dino_index_batch = dino_tokenizer(x, return_only_codebook_ids=True) # [bs,16]
        # magvit_index_batch = magvit_tokenizer(x, return_only_codebook_ids=True) # [bs,16]
        # dinov3_index_batch = dinov3_tokenizer(x, return_only_codebook_ids=True) # [bs,16]
        siglipv2_index_batch = siglipv2_tokenizer(x, return_only_codebook_ids=True) # [bs,16]



    # _image_indices = torch.cat((lavit_index_batch, dino_index_batch+8, magvit_index_batch+8+8), dim=1) # [bs, 16*3]
    # return _image_indices.cpu().numpy()
    return siglipv2_index_batch.cpu().numpy()

def get_latent_action(x, tokenizer, model_name):
    with torch.no_grad():
        if model_name == 'univla':
            outputs = tokenizer.vq_encode(x.permute(0, 2, 1, 3, 4))
            indices = outputs['indices']
            tokens = outputs['z_q'].squeeze(1)
        elif model_name == 'flux2':
            tokens = encode_video_batch_refs_final(tokenizer, x) # x Image列表 tokens: 1, num_frames*N, D
            return tokens.cpu().numpy()
        elif model_name == 'dinov3-origin':
            b, c, f, h, w = x.shape
            x = x.reshape(b*f, c, 1, h, w)
            x = get_dinov3_reps(x, tokenizer)
            dim = x.shape[-1]
            tokens = x.reshape(b, f, -1, dim)
            return tokens.cpu().numpy()
        else:
            tokens, indices = tokenizer(x, return_only_codebook_ids=True) # [bs,16]

    return tokens.cpu().numpy(), indices.cpu().numpy()