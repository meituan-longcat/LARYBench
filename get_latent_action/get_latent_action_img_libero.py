# -*- coding: utf-8 -*-
import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torchvision.transforms.functional as F
from get_latent_action.dynamics import get_latent_action, get_dynamic_tokenizer
env_model = os.environ.get("USE_MODEL")
if env_model == 'vjepa2':
    from get_latent_action.models.vjepa2.evals.video_classification_frozen.utils import make_transforms
    DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    vjepa2_transform = make_transforms(
            training=False,
            num_views_per_clip=1,
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0,
            auto_augment=False,
            motion_shift=False,
            crop_size=224,
            normalize=DEFAULT_NORMALIZATION,
        )
    
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class ImagePairDataset(Dataset):
    """
    针对 src_img 和 tgt_img 图像对的 Dataset
    """
    def __init__(self, df: pd.DataFrame, args):
        self.data = df
        self.args = args

    def __len__(self):
        return len(self.data)

    def load_image(self, path):
        if self.args.model == 'wan2-2':
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            tensor = F.to_tensor(img).sub_(0.5).div_(0.5)
            return tensor
        elif self.args.model == 'flux2':
            # Flux2 通常需要 PIL 格式
            img = Image.open(path).convert('RGB')
            return img.resize((224, 224), Image.LANCZOS)
        elif self.args.model == 'villa-x':
            img = Image.open(path).convert('RGB')
            imgs_np = np.array(img.resize((224, 224), Image.LANCZOS))
            return imgs_np
        elif self.args.model == 'vjepa2':
            img = Image.open(path).convert('RGB')
            imgs_np = np.array(img)
            return imgs_np
        else:
            # 其他模型通常需要 Normalized Tensor
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            return img / 255.0

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        global_idx = self.data.index[idx]
        
        try:
            src_img = self.load_image(row['src_img'])
            tgt_img = self.load_image(row['tgt_img'])
            
            if self.args.model == 'wan2-2':
                tensor = torch.stack([src_img, tgt_img], dim=1)
                return global_idx, tensor
            elif self.args.model == 'flux2':
                # 返回 PIL List 供后续处理
                return global_idx, [src_img, tgt_img]
            elif self.args.model == 'villa-x':
                tensor = torch.tensor(np.array([src_img, tgt_img]), dtype=torch.uint8)
                return global_idx, tensor
            elif self.args.model == 'vjepa2':
                return global_idx, [src_img, tgt_img]
            else:
                # 构造 (C, 2, H, W) 格式的 Pair
                # pair 形状: (2, H, W, C) -> permute -> (C, 2, H, W)
                pair = np.stack([src_img, tgt_img])
                tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
                return global_idx, tensor
        except Exception as e:
            print(f"Error loading images at index {global_idx}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    indices, data = zip(*batch)

    if isinstance(data[0], list): # Flux2 (PIL images)
        return indices, list(data)
    else: # Default (Tensors)
        return indices, torch.stack(data) # (B, C, 2, H, W)

class ActionProcessor:
    def __init__(self, args):
        self.args = args
        self.model = get_dynamic_tokenizer(args.model)
        if self.args.model == 'wan2-2':
            self.model.model.to("cuda").eval()
        else:
            self.model.to("cuda").eval()

    def process(self, df: pd.DataFrame, output_dir: str):
        dataset = ImagePairDataset(df, self.args)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, 
                            num_workers=self.args.num_workers, collate_fn=collate_fn)

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Partition {self.args.partition}"):
                if batch is None: continue
                batch_indices, batch_data = batch

                if self.args.model == 'wan2-2':
                    # 被 collate 后的 (B, C, T, H, W)
                    video_list = [t.to("cuda") for t in list(batch_data)]
                    with torch.no_grad():
                        latents = self.model.encode(video_list) 

                    batch_tokens = []
                    for l in latents:
                        la = np.transpose(l.cpu().numpy(), (1, 2, 3, 0))
                        la = la.reshape(la.shape[0], -1, la.shape[-1])
                        batch_tokens.append(la)
                    # Wan 是连续 VAE，没有离散索引， indices 传空数组
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                elif self.args.model == 'flux2':
                    # Flux2 逐个 Pair 处理 (或根据 get_latent_action 实现支持 batch)
                    batch_tokens, batch_ids = get_latent_action(batch_data, self.model, self.args.model)
                    batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 2, -1, batch_tokens.shape[-1])
                    batch_ids = batch_ids.reshape(batch_ids.shape[0], 2, -1, batch_ids.shape[-1])
                elif self.args.model == 'villa-x':
                    batch_output = self.model.idm(batch_data.to("cuda"), return_dict=True)  # B, T, H, W, C uint8
                    batch_tokens = batch_output['vq_tokens'].cpu().numpy()
                    batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], -1, self.model.config.action_latent_dim)
                    batch_ids = batch_output['indices'].cpu().numpy().reshape(batch_tokens.shape[0], -1)
                elif self.args.model == 'vjepa2':
                    batch_size = len(batch_data)
                    clips_batch = [[torch.stack([vjepa2_transform(clip)[0] for clip in batch_data], dim=0).to("cuda", non_blocking=True)]]

                    clip_indices_batch = torch.tensor([0, 1], device="cuda").unsqueeze(0).repeat(batch_size, 1)
                    batch_tokens = self.model(clips_batch, clip_indices_batch)[0].cpu().numpy() # B 196 1024
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                elif self.args.model == 'dinov3-origin' or self.args.model == 'dinov2-origin' or self.args.model == 'siglip2-origin':
                    batch_input = batch_data.to("cuda")

                    batch_tokens = get_latent_action(batch_input, self.model, self.args.model)
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                else: 
                    # 通用模型：Batch 处理 (B, C, 2, H, W)
                    batch_input = batch_data.to("cuda")
                    batch_tokens, batch_ids = get_latent_action(batch_input, self.model, self.args.model)
                    
                # 此时 batch_tokens 形状通常为 (B, Tokens, Dim)
                for i, g_idx in enumerate(batch_indices):
                    save_path = os.path.join(output_dir, f"latent_action_{g_idx:08d}.npz")
                    np.savez_compressed(save_path, tokens=batch_tokens[i],
                                        indices=batch_ids[i])
                    df.at[g_idx, 'la_path'] = save_path

        return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument('--model', type=str, default='dinov3_cs8_sl16_bs128_n')
    parser.add_argument('--dataset', type=str, default='calvin')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--perspective', type=str, default='1st')
    parser.add_argument('--num_partitions', type=int, default=1)
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256) # 图像对处理可以调大 batch
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    setup_seed(args.seed)
    
    save_dir = os.path.join(args.output_root, args.dataset, args.split, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # 读取 CSV
    full_df = pd.read_csv(args.input_file)
    if 'la_path' not in full_df.columns:
        full_df['la_path'] = ""

    # 分区
    partitions = np.array_split(full_df, args.num_partitions)
    current_df = partitions[args.partition].copy()

    # 处理
    processor = ActionProcessor(args)
    processed_df = processor.process(current_df, save_dir)

    # 保存
    csv_out = os.path.join(args.output_root, f"{args.split}_la_{args.dataset}_{args.model}_{args.partition}.csv")
    processed_df.to_csv(csv_out, index=False)
    print(f"Done. Results saved to {csv_out}")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()