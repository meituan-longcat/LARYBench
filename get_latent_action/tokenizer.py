from einops import rearrange, reduce, pack, unpack
from transformers import Dinov2Model, DINOv3ViTModel
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn.functional as F
from get_latent_action.models.magvit_torch.LFQ import LFQ
import numpy as np
import pickle
from omegaconf import OmegaConf

# 从环境变量读取模型路径根目录
# DINOv2 模型路径
_DINO_V2_PATH = os.environ.get("DINO_V2_PATH")
# DINOv3 模型路径
_DINO_V3_PATH = os.environ.get("DINO_V3_PATH")
# SigLIP2 模型路径
_SIGLIP2_PATH = os.environ.get("SIGLIP2_PATH")
# Open-MAGVIT2 模型路径
_MAGVIT2_CONFIG_PATH = os.environ.get("MAGVIT2_CONFIG_PATH")
_MAGVIT2_TOKENIZER_PATH = os.environ.get("MAGVIT2_TOKENIZER_PATH")

from get_latent_action.models.magvit2.lfqgan_pretrain import VQModel

# 作用：给定视频帧，返回帧离散之后的token id
def get_image_from_path(img_path, img_size=[128, 128], zero_centering=False):
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    frames_array=Image.open(img_path).convert('RGB')
    plt.imshow(frames_array)
    plt.title(f"origin pic {frames_array.size}" )
    plt.show()
   
    target_size = (img_size[0], img_size[1])

    # Define the transformation
    resize_transform = transforms.Resize(target_size)
    
    frames_array=np.expand_dims(np.array(frames_array),axis=0)
    frames_array = torch.tensor(frames_array, dtype=torch.float32)
    # print(frames_array.shape)
    frames_array = frames_array.permute(0, 3, 1, 2)   
    resized_images = []

    # Apply transformation to each image in the batch
    for img in frames_array:
        resized_img = resize_transform(img)
        resized_images.append(resized_img)

    # Stack all resized images into a single tensor
    frames_array_resized = torch.stack(resized_images)
    # If necessary, you can permute back to channels_last format
    
    frames_array_resized = frames_array_resized.permute(1,0, 2, 3)

    # If you need a numpy array at the end
    # frames_array_resized = frames_array_resized.numpy()
    
    frames_array = frames_array_resized/255    
    frames_array=frames_array.unsqueeze(0)
    return frames_array

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    return config

def load_dino(cpu=True):
    dino_path = _DINO_V2_PATH
    encoder = Dinov2Model.from_pretrained(pretrained_model_name_or_path=dino_path)
    if cpu:
        return encoder.eval()
    else:
        return encoder.cuda()
    
def load_dinov3(cpu=True):
    dino_path = _DINO_V3_PATH
    encoder = DINOv3ViTModel.from_pretrained(pretrained_model_name_or_path=dino_path)
    if cpu:
        return encoder.eval()
    else:
        return encoder.cuda()

def get_siglip2_encoder(cpu=True):
    """Load SigLIP2 vision encoder from SIGLIP2_PATH env variable."""
    from transformers import AutoModel
    model_path = _SIGLIP2_PATH
    if model_path is None:
        raise ValueError("SIGLIP2_PATH environment variable is not set.")
    encoder = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).vision_model
    if cpu:
        return encoder.eval()
    else:
        return encoder.cuda().eval()

def get_dino_indices(input_tensor, encoder, num_codebooks=128,multi_token_loss=False):
    if not multi_token_loss:
        pos_start=0
        pos_end=1
    else:
        pos_start=0
        pos_end=257

    outputs = encoder(input_tensor)
    last_hidden_states = outputs.last_hidden_state[:,pos_start:pos_end,:]
    x=last_hidden_states.detach()
    
    x = rearrange(x, 'b n (c d) ->b n c d', c = num_codebooks)
    codebook_dim=x.shape[-1]
    mask = 2 ** torch.arange(codebook_dim-1, -1, -1, device=x.device, dtype=torch.long)
    codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
    quantized = torch.where(x > 0, codebook_value, -codebook_value) # higher than 0 filled 

    indices = reduce((quantized > 0).int() * mask.int(), 'b n c d -> b n c', 'sum')

    bs, a,b = indices.shape

    return indices.view(bs,a*b)

def get_dinov3_reps(input_tensor, encoder):
    '''
    expect input_tensor has the same property as the input of magvit:
        输入是0-1的tensor图像就可以了, 输入形状应该为类似：64, 3, 1, 224, 224
    '''
    # transform
    input_tensor = input_tensor.squeeze(2)
    input_tensor = input_tensor*2-1
    
    with torch.no_grad():
        outputs = encoder(input_tensor) # outputs.last_hidden_state: [bs, 257, 1024]
    last_hidden_states = outputs.last_hidden_state[:,5:,:] #  [bs, 256, 1024]
    x = last_hidden_states.detach()
    x = x.unsqueeze(1) # [bs, 1, 256, 1024]
    
    batch_size, _, _, dim = x.shape

    x = x.reshape(batch_size, 1, 14, 14, dim)
    return x # [bs, 1, 16, 16, 1024]

def get_dino_reps(input_tensor, encoder):
    '''
    expect input_tensor has the same property as the input of magvit:
        输入是0-1的tensor图像就可以了, 输入形状应该为类似：64, 3, 1, 224, 224
    '''
    # transform
    input_tensor = input_tensor.squeeze(2)
    input_tensor = input_tensor*2-1
    
    with torch.no_grad():
        outputs = encoder(input_tensor) # outputs.last_hidden_state: [bs, 257, 1024]
    last_hidden_states = outputs.last_hidden_state[:,1:,:] #  [bs, 256, 1024]
    x = last_hidden_states.detach()
    x = x.unsqueeze(1) # [bs, 1, 256, 1024]
    
    batch_size, _, _, dim = x.shape

    x = x.reshape(batch_size, 1, 16, 16, dim)
    return x # [bs, 1, 16, 16, 1024]

def get_siglip2_reps(input_tensor, encoder):
    '''
    expect input_tensor has the same property as the input of magvit:
        输入是0-1的tensor图像就可以了, 输入形状应该为类似：64, 3, 1, 224, 224
    '''
    # transform
    input_tensor = input_tensor.squeeze(2)
    
    with torch.no_grad():
        outputs = encoder(
                pixel_values=input_tensor,
                output_hidden_states=False,
                return_dict=True)
    last_hidden_states = outputs.last_hidden_state #  [bs, 196, 1024]
    x = last_hidden_states.detach()
    x = x.unsqueeze(1) # [bs, 1, 196, 1024]
    
    batch_size, _, _, dim = x.shape

    x = x.reshape(batch_size, 1, 14, 14, dim)
    return x # [bs, 1, 16, 16, 1024]

def get_reps_magvit2(input_tensor, model):
    # 输入是0-1的tensor图像就可以了, 输入形状应该为类似：64, 3, 1, 224, 224
    with torch.no_grad():
        x = model.encoder(input_tensor.squeeze(2))  # [8, 3, 1, 224, 224]
    return x.detach()
    
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
    
def get_dino_tokenizer():
    dino_tokenizer = load_dino(cpu=False)
    freeze_backbone(dino_tokenizer)
    return dino_tokenizer

def get_dinov3_tokenizer():
    dino_tokenizer = load_dinov3(cpu=False)
    freeze_backbone(dino_tokenizer)
    return dino_tokenizer

def get_siglip2_tokenizer():
    siglip2_tokenizer = get_siglip2_encoder(cpu=False)
    freeze_backbone(siglip2_tokenizer)
    return siglip2_tokenizer

def load_vqgan_new(config, model_type, ckpt_path=None):
    model = VQModel(**config.model.init_args, model_type=model_type)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.cuda()

def get_magvit2_tokenizer(model_type):
    magvit_config_path=_MAGVIT2_CONFIG_PATH
    magvit_tokenizer_path=_MAGVIT2_TOKENIZER_PATH
    config_model = load_config(magvit_config_path, display=False)
    magvit_tokenizer = load_vqgan_new(config_model, model_type, ckpt_path=magvit_tokenizer_path)

    freeze_backbone(magvit_tokenizer)
    return magvit_tokenizer
