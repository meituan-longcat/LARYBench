import os
from abc import ABC, abstractmethod
from typing import List
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.transforms.functional as F
from torchvision.io import read_video

from registry import MODEL

from utils.model_utils import print_model_params, freeze_backbone
from get_latent_action.models.laq_model import (LatentActionQuantization,
                                                LatentActionQuantizationDinov2Feature,
                                                LatentActionQuantizationDinov3Feature,
                                                LatentActionQuantizationSiglipv2Feature,
                                                LatentActionQuantizationMagvit2)
from get_latent_action.models.univla.genie.model import ControllableDINOLatentActionModel
from get_latent_action.models.villa_x.lam.model import IgorModel
from get_latent_action.models.flux2.src.flux2.util import load_ae
from get_latent_action.models.flux2.src.flux2.sampling import encode_video_batch_refs_final
from get_latent_action.models.wan2_2.wan.modules.vae2_2 import Wan2_2_VAE # 只是适配wan
from get_latent_action.tokenizer import (get_dinov3_tokenizer, get_dino_tokenizer, get_siglip2_tokenizer,
                                         get_dinov3_reps, get_dino_reps, get_siglip2_reps)
from get_latent_action.models.vjepa2.evals.video_classification_frozen.models import init_module


class LaryBaseModel(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.get('name', 'LaryBaseModel')

    @abstractmethod
    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        pass

    @abstractmethod
    def prepare_model_for_extraction(self):
        pass

    @abstractmethod
    def process_image(self, src_img_path, tgt_img_path, image_size):
        pass

    @abstractmethod
    def process_video(self, data: pd.DataFrame, idx, image_size):
        pass


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class LaqBaseModel(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        freeze_backbone(self.model)
        self.model.to("cuda").eval()

    def _get_latent_action(self, batch_input):
        with torch.no_grad():
            tokens, indices = self.model(batch_input, return_only_codebook_ids=True)
        return tokens.cpu().numpy(), indices.cpu().numpy()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        if batch_data.ndim == 6:
            B, P, C, T, H, W = batch_data.shape
            flat_input = batch_data.view(B * P, C, T, H, W).to("cuda")
            flat_tokens, flat_ids = self._get_latent_action(flat_input)
            # flat_tokens: (B*P, N, D)  flat_ids: (B*P, N)
            batch_tokens = flat_tokens.reshape(B, P, flat_tokens.shape[-2], flat_tokens.shape[-1])
            batch_ids = flat_ids.reshape(B, P, -1)
        else:
            batch_input = batch_data.to("cuda")
            batch_tokens, batch_ids = self._get_latent_action(batch_input)
        return batch_tokens, batch_ids

    @staticmethod
    def _load_image(path, image_size):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        return img / 255.0

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path, image_size)
        tgt_img = self._load_image(tgt_img_path, image_size)
        pair = np.stack([src_img, tgt_img])
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        return tensor

    def process_video(self, data: pd.DataFrame, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)
        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (self.image_size, self.image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        pairs_list = []
        for i in range(len(selected_frames) - 1):
            pair = np.stack([selected_frames[i], selected_frames[i + 1]])
            pairs_list.append(pair)
        tensor = torch.tensor(np.array(pairs_list), dtype=torch.float32).permute(0, 4, 1, 2, 3)

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class LAPALaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "lapa")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = LatentActionQuantization(*args, **kwargs)
        self.model.to('cuda')
        self.model.load(ckpt_path)
        self.prepare_model_for_extraction()


@MODEL.register_module()
class Magvit2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "magvit2")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = LatentActionQuantizationMagvit2(*args, **kwargs)
        self.model.to('cuda')
        self.model.load(ckpt_path)
        self.prepare_model_for_extraction()


@MODEL.register_module()
class Dinov2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "dinov2")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = LatentActionQuantizationDinov2Feature(*args, **kwargs)
        self.model.to('cuda')
        self.model.load(ckpt_path)
        self.prepare_model_for_extraction()


@MODEL.register_module()
class Dinov3LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "dinov3")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = LatentActionQuantizationDinov3Feature(*args, **kwargs)
        self.model.to('cuda')
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        self.prepare_model_for_extraction()


@MODEL.register_module()
class Siglip2LaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "siglip2")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = LatentActionQuantizationSiglipv2Feature(*args, **kwargs)
        self.model.to('cuda')
        self.model.load(ckpt_path)
        self.prepare_model_for_extraction()


@MODEL.register_module()
class UnivlaLaryWrap(LaqBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "univla")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = ControllableDINOLatentActionModel(*args, **kwargs)
        self.model.to('cuda')
        ckpt = torch.load(ckpt_path)["state_dict"]
        self.model.load_state_dict(
            {k.replace("lam.", ""): v for k, v in ckpt.items()}, strict=True
        )
        self.prepare_model_for_extraction()


def read_video_tensor(fp: str, resize_h: int = None, resize_w: int = None):
    """
    Read video and return as tensor.

    Args:
        fp: Video file path
        resize_h, resize_w: Target dimensions (optional)

    Returns:
        video: Tensor of shape (T, H, W, C)
        fps: Frames per second
    """
    video, _, info = read_video(fp, pts_unit="sec")
    fps = int(info.get("video_fps", 25.0))

    if resize_h is not None and resize_w is not None:
        video = video.permute(0, 3, 1, 2)
        video = F.resize(video, [resize_h, resize_w], antialias=True)
        video = video.permute(0, 2, 3, 1)

    return video, fps


#villa-x
@MODEL.register_module()
class VillaXLaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "villa-x")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = IgorModel.from_pretrained(ckpt_path).cuda()
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            batch_output = self.model.idm(batch_data.to("cuda"), return_dict=True)
            batch_tokens = batch_output['vq_tokens'].cpu().numpy()
            if config.mode == 'image':
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], -1, self.model.config.action_latent_dim)
            else:
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 8, -1, self.model.config.action_latent_dim)
            batch_ids = batch_output['indices'].cpu().numpy().reshape(batch_tokens.shape[0], batch_tokens.shape[1] if len(
                batch_tokens.shape) > 2 else -1, -1)
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        freeze_backbone(self.model)
        self.model.to('cuda').eval()

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return np.array(img.resize((self.image_size, self.image_size), Image.LANCZOS))

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path)
        tgt_img = self._load_image(tgt_img_path)
        tensor = torch.tensor(np.array([src_img, tgt_img]), dtype=torch.uint8)
        return tensor

    def process_video(self, data, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (image_size, image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        tensor, _ = read_video_tensor(video_path, resize_h=image_size, resize_w=image_size)
        tensor = tensor[indices]

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class Flux2LaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "flux2")
        self.model = load_ae("flux.2-dev", device="cuda")
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            batch_tokens = encode_video_batch_refs_final(self.model, batch_data)  # x Image列表 tokens: 1, num_frames*N, D
            batch_tokens = batch_tokens.cpu().numpy()
            if config.mode == 'image':
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 2, -1, batch_tokens.shape[-3])
            else:
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 9, -1, batch_tokens.shape[-3])
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        freeze_backbone(self.model)
        self.model.to("cuda").eval()

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return np.array(img.resize((self.image_size, self.image_size), Image.LANCZOS))

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path)
        tgt_img = self._load_image(tgt_img_path)
        tensor = torch.tensor(np.stack([src_img, tgt_img]), dtype=torch.float32) * 2 - 1
        return tensor

    def process_video(self, data, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (self.image_size, self.image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        pair = np.stack(selected_frames)
        tensor = torch.tensor(pair, dtype=torch.float32) * 2 - 1

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class Wan2_2LaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "wan2-2")
        ckpt_path = kwargs.pop("ckpt_path", None)
        self.model = Wan2_2_VAE(vae_pth=ckpt_path, device="cuda")
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            video_list = [t.to("cuda") for t in list(batch_data)]
            with torch.no_grad():
                latents = self.model.encode(video_list)
            batch_tokens = []
            for l in latents:
                la = np.transpose(l.cpu().numpy(), (1, 2, 3, 0))
                la = la.reshape(la.shape[0], -1, la.shape[-1])
                batch_tokens.append(la)
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model.model)
        freeze_backbone(self.model.model)
        self.model.model.to("cuda").eval()

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return F.to_tensor(img).sub_(0.5).div_(0.5)

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path)
        tgt_img = self._load_image(tgt_img_path)
        tensor = torch.stack([src_img, tgt_img], dim=1)
        return tensor

    def process_video(self, data, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (self.image_size, self.image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        tensors = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = Image.fromarray(all_frames[safe_idx])
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            tensor = F.to_tensor(img).sub_(0.5).div_(0.5)
            tensors.append(tensor)
        tensor = torch.stack(tensors, dim=1)

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class Vjepa2LaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "vjepa2")
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

        self.model = init_module(
            module_name='get_latent_action.models.vjepa2.evals.video_classification_frozen.modelcustom.vit_encoder_multiclip',
            frames_per_clip=16,
            resolution=224,
            checkpoint=os.environ.get("VJEPA2_CKPT_PATH"),
            model_kwargs=args_model,
            wrapper_kwargs=args_wrapper,
            device="cuda",
        )

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            if config.mode == "video":
                transformed_batch = []
                for video_frames in batch_data:
                    t_frames = torch.stack(self.model.transform(video_frames), dim=0)
                    transformed_batch.append(t_frames.to("cuda", non_blocking=True))
                clips_batch = [[torch.stack(transformed_batch, dim=0).squeeze(1)]]
                clip_indices_batch = batch_rel_indices.to("cuda")
            else:
                clips_batch = [[torch.stack([self.model.transform(clip)[0] for clip in batch_data], dim=0).to("cuda",
                                                                                                              non_blocking=True)]]
                clip_indices_batch = torch.tensor([0, config.stride], device="cuda").unsqueeze(0).repeat(
                    len(batch_data), 1)

            batch_tokens = self.model(clips_batch, clip_indices_batch)[0].cpu().numpy()
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        freeze_backbone(self.model)
        self.model.to('cuda').eval()

    @staticmethod
    def _load_image(path):
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path)
        tgt_img = self._load_image(tgt_img_path)
        return [src_img, tgt_img]

    def process_video(self, data: pd.DataFrame, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            selected_frames.append(all_frames[safe_idx])
        return global_idx, selected_frames, rel_indices


@MODEL.register_module()
class Dinov2OriginLaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop("name", "dinov2-origin")
        self.model = get_dino_tokenizer()
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            batch_input = batch_data.to("cuda")
            b, c, f, h, w = batch_input.shape
            batch_input = batch_input.reshape(b * f, c, 1, h, w)
            x = get_dino_reps(batch_input, self.model)
            dim = x.shape[-1]
            tokens = x.reshape(b, f, -1, dim)
            batch_tokens = tokens.cpu().numpy()
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        self.model.to("cuda").eval()

    @staticmethod
    def _load_image(path, image_size):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        return img / 255.0

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path, image_size)
        tgt_img = self._load_image(tgt_img_path, image_size)
        pair = np.stack([src_img, tgt_img])
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        return tensor

    def process_video(self, data: pd.DataFrame, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (image_size, image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        pair = np.stack(selected_frames)
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class Dinov3OriginLaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name', 'dinov3-origin')
        self.model = get_dinov3_tokenizer()
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            batch_input = batch_data.to("cuda")
            b, c, f, h, w = batch_input.shape
            batch_input = batch_input.reshape(b * f, c, 1, h, w)
            x = get_dinov3_reps(batch_input, self.model)
            dim = x.shape[-1]
            tokens = x.reshape(b, f, -1, dim)
            batch_tokens = tokens.cpu().numpy()
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        self.model.to("cuda").eval()

    @staticmethod
    def _load_image(path, image_size):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        return img / 255.0

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path, image_size)
        tgt_img = self._load_image(tgt_img_path, image_size)
        pair = np.stack([src_img, tgt_img])
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        return tensor

    def process_video(self, data: pd.DataFrame, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (image_size, image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        pair = np.stack(selected_frames)
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)

        return global_idx, tensor, rel_indices


@MODEL.register_module()
class Siglip2OriginLaryWrap(LaryBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name', 'siglip2-origin')
        self.model = get_siglip2_tokenizer()
        self.prepare_model_for_extraction()

    def get_latent_action(self, batch_data, batch_rel_indices=None, config=None):
        with torch.no_grad():
            batch_input = batch_data.to("cuda")
            b, c, f, h, w = batch_input.shape
            batch_input = batch_input.reshape(b * f, c, 1, h, w)
            x = get_siglip2_reps(batch_input, self.model)
            dim = x.shape[-1]
            tokens = x.reshape(b, f, -1, dim)
            batch_tokens = tokens.cpu().numpy()
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
        return batch_tokens, batch_ids

    def prepare_model_for_extraction(self):
        print_model_params(self.model)
        self.model.to("cuda").eval()

    @staticmethod
    def _load_image(path, image_size):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        return img / 255.0

    def process_image(self, src_img_path, tgt_img_path, image_size):
        src_img = self._load_image(src_img_path, image_size)
        tgt_img = self._load_image(tgt_img_path, image_size)
        pair = np.stack([src_img, tgt_img])
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        return tensor

    def process_video(self, data: pd.DataFrame, idx, image_size):
        row = data.iloc[idx]
        global_idx = data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (image_size, image_size),
                             interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        pair = np.stack(selected_frames)
        tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)

        return global_idx, tensor, rel_indices
