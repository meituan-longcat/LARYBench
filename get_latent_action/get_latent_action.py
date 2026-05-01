import os
import gc
import argparse
import ast
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random
from PIL import Image
from torchvision.io import read_video
import torchvision.transforms.functional as F

# 假设这些是从你的外部模块导入的
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

def load_video_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# def villa_x_read_video(fp: str):
#     video, _, info = read_video(fp, pts_unit="sec")
#     return video, int(info.get("video_fps", 25.0))
def villa_x_read_video(fp: str, resize_h: int = None, resize_w: int = None):
    """
    fp: 视频路径
    resize_h, resize_w: 目标高度和宽度。如果为 None 则保持原样。
    """
    # 1. 读取视频: video 形状为 (T, H, W, C)
    video, _, info = read_video(fp, pts_unit="sec")
    fps = int(info.get("video_fps", 25.0))

    if resize_h is not None and resize_w is not None:
        # 2. 调整维度以适配 torchvision: (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2)
        
        # 3. 执行 Resize
        # antialias=True 可以获得更好的抗锯齿效果
        video = F.resize(video, [resize_h, resize_w], antialias=True)
        
        # 4. 还原维度: (T, C, H, W) -> (T, H, W, C)
        video = video.permute(0, 2, 3, 1)

    return video, fps

def read_video_to_pil(video_path, width=None, height=None):
    """
    读取视频并将每一帧转换为 PIL Image 对象
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # 如果指定了尺寸，进行 Resize (与 main 函数中的 match_image_size 逻辑类似)
        if width is not None and height is not None:
            img = img.resize((width, height), Image.LANCZOS)
            
        frames.append(img)
    
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    return frames

class ActionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args):
        self.data = df
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        global_idx = self.data.index[idx]
        video_path = row['video_path']
        
        # 解析 sample_indices: "3,8,13..." -> [3, 8, 13...]
        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)
        if self.args.model == 'vjepa2':
            selected_frames = []
            for f_idx in indices:
                safe_idx = max(0, min(int(f_idx), total_frames - 1))
                img = all_frames[safe_idx] # 保持原始 numpy 数组 (RGB)
                selected_frames.append(img)
            return global_idx, selected_frames, rel_indices


        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (224, 224), interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)
        if self.args.model == 'wan2-2':
            selected_frames = []
            for f_idx in indices:
                safe_idx = max(0, min(int(f_idx), total_frames - 1))
                # 转换 NumPy 为 PIL 对象
                img = Image.fromarray(all_frames[safe_idx])
                # Resize 到 224x224
                img = img.resize((224, 224), Image.LANCZOS)
                # 归一化到 [-1, 1] 范围
                tensor = F.to_tensor(img).sub_(0.5).div_(0.5)
                selected_frames.append(tensor)
            
            # Wan VAE 期望形状为 (C, T, H, W)
            tensor = torch.stack(selected_frames, dim=1)
        elif self.args.model == 'villa-x':
            tensor, _ = villa_x_read_video(video_path, resize_h=224, resize_w=224)
            if tensor.shape[1] != 224 or tensor.shape[2] != 224:
                print("\nbig mistake!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            tensor = tensor[indices]
        elif self.args.model == 'dinov3-origin' or self.args.model == 'dinov2-origin' or self.model == 'siglip2-origin':
            pair = np.stack(selected_frames)
            tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        elif self.args.model == 'flux2':
            pair = np.stack(selected_frames)
            tensor = torch.tensor(pair, dtype=torch.float32) * 2 - 1
        else:
            # 构造 Frame Pairs: (Num_Pairs, 2, H, W, C)
            pairs_list = []
            for i in range(len(selected_frames) - 1):
                pair = np.stack([selected_frames[i], selected_frames[i+1]])
                pairs_list.append(pair)
            
            # (Num_Pairs, C, 2, H, W)
            tensor = torch.tensor(np.array(pairs_list), dtype=torch.float32).permute(0, 4, 1, 2, 3)
        return global_idx, tensor, rel_indices

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    indices, tensors, rel_indices = zip(*batch)
    if isinstance(tensors[0], list):  # vjepa2
        return indices, list(tensors), torch.stack(rel_indices)
    return indices, torch.stack(tensors), torch.stack(rel_indices)

class ActionProcessor:
    def __init__(self, args):
        self.args = args
        self.model = get_dynamic_tokenizer(args.model)
        if self.args.model == 'wan2-2':
            self.model.model.to("cuda").eval()
        else:
            self.model.to("cuda").eval()

    def process(self, df: pd.DataFrame, output_dir: str):
        dataset = ActionDataset(df, self.args)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, 
                            num_workers=self.args.num_workers, collate_fn=collate_fn)


        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Partition {self.args.partition}"):
                if batch is None: continue
                batch_indices, batch_tensors, batch_rel_indices = batch

                if self.args.model == 'wan2-2':
                    # 被 collate 后的 (B, C, T, H, W)
                    video_list = [t.to("cuda") for t in list(batch_tensors)]
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
                    batch_tokens = get_latent_action(batch_tensors, self.model, self.args.model)
                    batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 9, -1, batch_tokens.shape[-3])
                    # batch_ids = batch_ids.reshape(batch_ids.shape[0], 9, -1, batch_ids.shape[-1])
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                elif self.args.model == 'villa-x':
                    batch_output = self.model.idm(batch_tensors.to("cuda"), return_dict=True)  # B, T, H, W, C uint8
                    batch_tokens = batch_output['vq_tokens'].cpu().numpy()
                    batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 8, -1, self.model.config.action_latent_dim)
                    batch_ids = batch_output['indices'].cpu().numpy().reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)
                elif self.args.model == 'vjepa2':
                    batch_size = len(batch_tensors)
                    transformed_batch = []
                    for video_frames in batch_tensors:
                        t_frames = torch.stack(vjepa2_transform(video_frames), dim=0)
                        transformed_batch.append(t_frames.to("cuda", non_blocking=True))
                    
                    clips_batch = [[torch.stack(transformed_batch, dim=0).squeeze(1)]]
                    
                    # 直接使用从 Dataset 传过来的相对索引，并转到 GPU
                    clip_indices_batch = batch_rel_indices.to("cuda")
                
                    batch_tokens = self.model(clips_batch, clip_indices_batch)[0].cpu().numpy()
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                elif self.args.model == 'dinov3-origin' or self.args.model == 'dinov2-origin' or self.args.model == 'siglip2-origin':
                    batch_input = batch_tensors.to("cuda")
                    
                    batch_tokens = get_latent_action(batch_input, self.model, self.args.model)
                    batch_ids = [np.array([]) for _ in range(len(batch_tokens))]
                else:
                    B, P, C, T, H, W = batch_tensors.shape
                    
                    # Flatten for inference
                    flat_input = batch_tensors.view(B * P, C, T, H, W).to("cuda")
                    flat_tokens, flat_idx = get_latent_action(flat_input, self.model, self.args.model)
                    
                    # Reshape back
                    batch_tokens = flat_tokens.reshape(B, P, flat_tokens.shape[-2], flat_tokens.shape[-1])
                    batch_ids = flat_idx.reshape(B, P, -1)

                for i, global_idx in enumerate(batch_indices):

                    save_name = f"latent_action_{global_idx:08d}.npz"
                    save_path = os.path.join(output_dir, save_name)
                    
                    np.savez_compressed(save_path, tokens=batch_tokens[i], indices=batch_ids[i])
                    df.at[global_idx, 'la_path'] = save_path

        return df

def main():
    parser = argparse.ArgumentParser()
    _LAB_DIR = os.environ.get("LATENT_ACTION_BENCH_DIR")
    parser.add_argument("--input_file", type=str, default=os.path.join(_LAB_DIR, 'select_action/sample_robot_1st.csv'))
    parser.add_argument("--output_root", type=str, default=os.path.join(_LAB_DIR, 'latent_action'))
    parser.add_argument('--model', type=str, default='dinov3-origin')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--dataset', type=str, default='robot_1st')
    parser.add_argument('--num_partitions', type=int, default=1)
    parser.add_argument('--partition', type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    
    # 建立输出目录结构
    save_dir = os.path.join(args.output_root, args.dataset, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # 读取并切分数据
    full_df = pd.read_csv(args.input_file)
    partitions = np.array_split(full_df, args.num_partitions)
    current_df = partitions[args.partition].copy()

    # 处理
    processor = ActionProcessor(args)
    processed_df = processor.process(current_df, save_dir)

    # 保存当前分区的 CSV 结果
    csv_out = os.path.join(args.output_root, f"{args.split}_la_{args.dataset}_{args.model}_{args.partition}.csv")
    processed_df.to_csv(csv_out, index=False)
    print(f"Partition {args.partition} finished. CSV saved to {csv_out}")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()