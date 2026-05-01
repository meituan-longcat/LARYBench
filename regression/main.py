import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from regression.dit import DiT
from utils.model_utils import print_model_params
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import random

# ==========================================
# 0. Dimension Labels Configuration
# ==========================================
DIM_CALVIN = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
DIM_AGIBOT = ['l_x', 'l_y', 'l_z', 'r_x', 'r_y', 'r_z', 'l_qx', 'l_qy', 'l_qz', 'l_qw', 'r_qx', 'r_qy', 'r_qz', 'r_qw', 'r_g', 'l_g']
DIM_ROBOCOIN = ['l_x', 'l_y', 'l_z', 'l_roll', 'l_pitch', 'l_yaw', 'r_x', 'r_y', 'r_z', 'r_roll', 'r_pitch', 'r_yaw']

# 各数据集按语义分组的维度索引：position / orientation / gripper
# CALVIN:   x y z | roll pitch yaw | gripper
# AGIBOT:   l_x l_y l_z r_x r_y r_z | l_qx..l_qw r_qx..r_qw | r_g l_g
# ROBOCOIN: l_x l_y l_z r_x r_y r_z | l_roll..l_yaw r_roll..r_yaw | l_g r_g
GROUP_INDICES = {
    'calvin':    {'position': [0, 1, 2], 'orientation': [3, 4, 5], 'gripper': [6]},
    'agibotbeta':{'position': [0, 1, 2, 3, 4, 5], 'orientation': [6, 7, 8, 9, 10, 11, 12, 13], 'gripper': [14, 15]},
    'robocoin':  {'position': [0, 1, 2, 6, 7, 8], 'orientation': [3, 4, 5, 9, 10, 11]},
}

REGRESSION_DATA_SUBDIR = {
    'calvin': 'calvin',
    'vlabench': 'vlabench',
    'vlabench_15': 'vlabench',
    'vlabench_30': 'vlabench',
    'agibotbeta': 'agibot_45',
    'robocoin': 'robocoin_10',
}

REGRESSION_SPLIT_DIR = {
    ('calvin', 'train'): 'train_stride5',
    ('calvin', 'val'): 'val_stride5',
}

def get_dim_labels(dataset_name):
    if dataset_name == 'agibotbeta': return DIM_AGIBOT
    elif dataset_name == 'robocoin': return DIM_ROBOCOIN
    return DIM_CALVIN

def get_action_dim(dataset_name):
    if dataset_name == 'agibotbeta': return 16
    elif dataset_name == 'robocoin': return 12
    return 7

def get_group_indices(dataset_name):
    """返回当前数据集的 position / orientation / gripper 分组索引，不存在则返回 None。"""
    return GROUP_INDICES.get(dataset_name, None)

def get_action_steps(chunk_size, action_mode):
    return 1 if action_mode == 'relative' else chunk_size

def get_regression_root(action_data_root, action_mode='absolute'):
    root = action_data_root or os.environ.get('DATA_DIR')
    if not root:
        return None
    root = os.path.normpath(root)
    expected_leaf = 'regression_relative' if action_mode == 'relative' else 'regression'
    if os.path.basename(root) == expected_leaf:
        return root
    return os.path.join(root, expected_leaf)

def get_regression_data_subdir(dataset_name):
    dataset_key = dataset_name.lower()
    return REGRESSION_DATA_SUBDIR.get(dataset_key, dataset_key)

def get_relative_stats_path(action_data_root, dataset_name):
    regression_root = get_regression_root(action_data_root, 'relative')
    if not regression_root:
        return None
    subdir = get_regression_data_subdir(dataset_name)
    stats_dir = os.path.join(regression_root, subdir)
    dataset_key = dataset_name.lower()
    dataset_specific = os.path.join(stats_dir, f"relative_action_stats_{dataset_key}.json")
    default = os.path.join(stats_dir, "relative_action_stats.json")
    if os.path.exists(dataset_specific):
        return dataset_specific
    if os.path.exists(default):
        return default
    return dataset_specific

def get_action_data_root(action_data_root, dataset_name, split, action_mode='absolute'):
    if action_mode != 'relative' and not action_data_root:
        return None
    regression_root = get_regression_root(action_data_root, action_mode)
    if not regression_root:
        return None
    dataset_key = dataset_name.lower()
    subdir = get_regression_data_subdir(dataset_name)
    root = os.path.join(regression_root, subdir)
    split_dir = REGRESSION_SPLIT_DIR.get((dataset_key, split.lower()))
    if split_dir:
        root = os.path.join(root, split_dir)
    return root

def resolve_under_root(root, subdir, value):
    if value is None or pd.isna(value):
        return value
    path = str(value)
    if os.path.isabs(path):
        return path
    if subdir and subdir in Path(path).parts:
        return os.path.join(root, path)
    effective_root = os.path.join(root, subdir) if subdir else root
    return os.path.join(effective_root, path)

def to_action_target(action, action_dim, action_mode):
    action = np.asarray(action)
    flat = action.reshape(-1)
    if action_mode == 'relative':
        if flat.size == action_dim:
            return flat.astype(np.float32, copy=False)
        if flat.size % action_dim != 0:
            raise ValueError(f"Action size {flat.size} is not divisible by action_dim={action_dim}.")
        seq = flat.reshape(-1, action_dim)
        return (seq[-1] - seq[0]).astype(np.float32, copy=False)
    return flat.astype(np.float32, copy=False)

# ==========================================
# Utils
# ==========================================
def save_separate_eval_samples(
    pred_norm: np.ndarray, gt_norm: np.ndarray, mean: np.ndarray, std: np.ndarray,
    src_img_path: str, tgt_img_path: str, chunk_size: int, epoch: int, save_dir: str,
    dim_labels: list, sample_idx: int = 0, dt: float = 0.05
):
    if 'vlabench' in save_dir:
        dt = 0.1
    else:
        dt = 0.03
    vis_dir = Path(save_dir) / "eval_vis" / f"epoch_{epoch}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"sample_{sample_idx}"

    try:
        if src_img_path and os.path.exists(src_img_path):
            plt.imsave(vis_dir / f"{file_prefix}_src.png", mpimg.imread(src_img_path))
        if tgt_img_path and os.path.exists(tgt_img_path):
            plt.imsave(vis_dir / f"{file_prefix}_tgt.png", mpimg.imread(tgt_img_path))
    except Exception as e:
        pass

    pred_phys = (pred_norm * std) + mean
    gt_phys = (gt_norm * std) + mean
    
    pred_seq = pred_phys.reshape(chunk_size, -1)
    gt_seq = gt_phys.reshape(chunk_size, -1)
    T, D = gt_seq.shape

    plt.style.use('default')
    cols = 3
    rows = (D + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows), constrained_layout=True)
    if D > 1: axes = axes.flatten()
    else: axes = [axes]

    time_steps = np.arange(T) * dt
    mse = np.mean((pred_seq - gt_seq)**2)

    for i in range(D):
        ax = axes[i]

        ax.plot(time_steps, gt_seq[:, i], '-', color='#1f77b4', label='GT', alpha=0.8, linewidth=2)      # 经典蓝 实线
        ax.plot(time_steps, pred_seq[:, i], '-', color='#d62728', label='Pred', alpha=0.8, linewidth=2)   # 经典红 实线
        abs_error = np.abs(pred_seq[:, i] - gt_seq[:, i])
        ax.plot(time_steps, abs_error, '--', color='#ffc107', label='Error', alpha=0.8, linewidth=1.5)    # 琥珀黄 虚线
        
        title = dim_labels[i] if i < len(dim_labels) else f"Dim {i}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel("Time (s)", fontsize=10)
        if i == 0: ax.legend(fontsize=9, loc='best', framealpha=0.8)
            
    for i in range(D, len(axes)): axes[i].axis('off')

    plt.suptitle(f"Action Trajectory (MSE: {mse:.4f})", fontsize=14, y=1.02)
    plt.savefig(vis_dir / f"{file_prefix}_action.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(vis_dir / f"{file_prefix}_action.png", format='png', dpi=150, bbox_inches='tight')
    plt.close()

def compute_per_dim_mse(pred, target, action_steps, action_dim_labels):
    num_dims = len(action_dim_labels)
    pred_reshaped = pred.view(-1, action_steps, num_dims)
    target_reshaped = target.view(-1, action_steps, num_dims)
    squared_diff = (pred_reshaped - target_reshaped) ** 2
    mse_per_dim = squared_diff.mean(dim=(0, 1))
    return {label: mse_per_dim[i].item() for i, label in enumerate(action_dim_labels)}

def compute_group_mse(pred, target, action_steps, dataset_name):
    """计算 position / orientation / gripper 三个语义分组的平均 MSE。"""
    group_indices = get_group_indices(dataset_name)
    if group_indices is None:
        return {}
    action_dim = get_action_dim(dataset_name)
    pred_reshaped = pred.view(-1, action_steps, action_dim)
    target_reshaped = target.view(-1, action_steps, action_dim)
    squared_diff = (pred_reshaped - target_reshaped) ** 2  # (B, T, D)
    result = {}
    for group_name, indices in group_indices.items():
        result[f"mse_{group_name}"] = squared_diff[:, :, indices].mean().item()
    return result

# ==========================================
# 1. Dataset Definition
# ==========================================
class ActionExpertDataset(Dataset):
    def __init__(
        self,
        csv_path,
        dataset_name,
        chunk_size,
        global_stats_json=None,
        action_mean=None,
        action_std=None,
        action_mode='absolute',
        action_data_root=None,
    ):
        self.data = pd.read_csv(csv_path)
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.action_dim = get_action_dim(dataset_name)
        self.action_mode = action_mode
        self.action_steps = get_action_steps(chunk_size, action_mode)
        self.action_data_root = action_data_root
        
        # Detect split from CSV path or filename
        self.split = self._detect_split(csv_path)

        # Resolve relative paths
        self._resolve_paths()

        # Calvin 的全局统计信息
        self.action_mean = action_mean
        self.action_std = action_std
        
        # Agibot / Robocoin 的 Robot Type 全局统计信息
        self.robot_stats = {}
        if global_stats_json and os.path.exists(global_stats_json):
            with open(global_stats_json, 'r') as f:
                raw_stats = json.load(f)
                robot_raw_stats = raw_stats.get('robot_stats', raw_stats)
                for r_type, stats in robot_raw_stats.items():
                    if not isinstance(stats, dict) or 'mean' not in stats or 'std' not in stats:
                        continue
                    m_arr = np.array(stats['mean'])
                    s_arr = np.array(stats['std'])
                    s_arr = np.where(s_arr < 1e-6, 1.0, s_arr)
                    self.robot_stats[r_type] = (m_arr, s_arr)

        if 'la_path' not in self.data.columns:
            raise ValueError("CSV must contain 'la_path' for latent actions.")

    def _detect_split(self, csv_path):
        """Detect split (train/val) from CSV path."""
        path_lower = str(csv_path).lower()
        if '_train' in path_lower or 'train_' in path_lower:
            return 'train'
        elif '_val' in path_lower or 'val_' in path_lower:
            return 'val'
        return 'train'  # Default

    def _resolve_paths(self):
        """Resolve relative paths in the DataFrame.

        Avoids any os.path.exists() calls on HDFS/FUSE paths (which cause
        fuse_lock_inode hangs).  Sub-directory layout per dataset is declared
        statically; fall back to a flat layout when not listed.
        """
        from lary.path_resolver import resolve_la_path, get_data_root

        data_root = get_data_root(self.dataset_name, self.split)
        action_root = get_action_data_root(self.action_data_root, self.dataset_name, self.split, self.action_mode)

        # Static column→subdirectory mapping per dataset.
        # 'None' means the column does not exist on disk and should be skipped.
        DATASET_COL_SUBDIR = {
            'calvin': {
                'src_img':   'images',
                'tgt_img':   'images',
                'action':    'actions',
            },
            'agibotbeta': {
                'src_img':   'images',
                'tgt_img':   'images',
                'action':    'actions',
            },
            'robocoin': {
                'src_img':   'images',
                'tgt_img':   'images',
                'action':    'actions',
            },
        }

        col_subdir = DATASET_COL_SUBDIR.get(self.dataset_name.lower(), {})

        # Resolve data paths — pure string operations, zero HDFS stat calls
        data_path_cols = ['src_img', 'tgt_img', 'action']
        for col in data_path_cols:
            if col not in self.data.columns:
                continue
            subdir = col_subdir.get(col, '')   # '' means flat layout
            if subdir is None:
                # Column declared absent for this dataset; zero-out so __getitem__
                # won't try to load it.
                self.data[col] = None
                continue
            if col == 'action' and action_root:
                self.data[col] = self.data[col].apply(
                    lambda x: resolve_under_root(action_root, subdir, x)
                )
                continue
            effective_root = os.path.join(data_root, subdir) if (data_root and subdir) else data_root
            if effective_root:
                self.data[col] = self.data[col].apply(
                    lambda x: resolve_under_root(data_root, subdir, x)
                    if pd.notna(x) and not os.path.isabs(str(x)) else x
                )

        # Resolve la_path: probe ONCE using the first valid row, then batch-apply prefix.
        # This avoids calling os.path.exists() on HDFS for every single row (which causes
        # fuse_lock_inode hangs when the CSV has hundreds-of-thousands of entries).
        if 'la_path' in self.data.columns:
            first_valid = self.data['la_path'].dropna()
            first_valid = first_valid[~first_valid.apply(lambda x: os.path.isabs(str(x)))]
            if not first_valid.empty:
                resolved_first = resolve_la_path(str(first_valid.iloc[0]), self.dataset_name, self.split)
                fname = os.path.basename(str(first_valid.iloc[0]))
                la_prefix = resolved_first[: resolved_first.rfind(fname)]
                if la_prefix:
                    self.data['la_path'] = self.data['la_path'].apply(
                        lambda x: os.path.join(la_prefix, os.path.basename(str(x)))
                        if pd.notna(x) and not os.path.isabs(str(x)) else x
                    )
                else:
                    self.data['la_path'] = self.data['la_path'].apply(
                        lambda x: resolve_la_path(str(x), self.dataset_name, self.split) if pd.notna(x) else x
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        la_data = np.load(row['la_path'])
        latent_action = la_data['tokens'].flatten() if 'tokens' in la_data else la_data.flatten()
        action = to_action_target(np.load(row['action']), self.action_dim, self.action_mode)
            
        # === 核心修改：根据 robot_type 获取全局 Mean 和 Std ===
        if self.dataset_name in ['agibotbeta', 'robocoin']:
            robot_type = str(row['robot_type'])
            if robot_type in self.robot_stats:
                a_mean, a_std = self.robot_stats[robot_type]
            else:
                # Fallback: 如果找不到对应的 robot_type，使用 0 和 1
                a_mean, a_std = np.zeros(self.action_dim), np.ones(self.action_dim)
        else:
            a_mean, a_std = self.action_mean, self.action_std

        # 扩展到输出时间步数：absolute 预测完整 chunk；relative 只预测 last - first。
        a_mean_tiled = np.tile(a_mean, self.action_steps)
        a_std_tiled = np.tile(a_std, self.action_steps)

        action_norm = (action - a_mean_tiled) / a_std_tiled
        
        src_img_path = str(row['src_img']) if 'src_img' in row else ""
        tgt_img_path = str(row['tgt_img']) if 'tgt_img' in row else ""

        return (
            torch.from_numpy(latent_action).float(),
            torch.from_numpy(action_norm).float(),
            src_img_path,
            tgt_img_path,
            torch.from_numpy(a_mean_tiled).float(), 
            torch.from_numpy(a_std_tiled).float()
        )

# ==========================================
# 2. Model Definitions (保持不变)
# ==========================================
class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
    def forward(self, x): return self.ffn(x) + x

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks: x = block(x)
        return self.fc2(self.layer_norm2(x))

class ActionExpertMLP(nn.Module):
    def __init__(self, input_dim, action_dim=7, hidden_dim=4096, num_blocks=2, action_steps=5):
        super().__init__()
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.model = MLPResNet(num_blocks, input_dim, hidden_dim, action_dim * action_steps)
    def forward(self, latent_action): return self.model(latent_action)
    def loss(self, pred, target): return F.huber_loss(pred, target, delta=1.0)

class ActionExpertDiT(nn.Module):
    def __init__(self, latent_dim, action_dim=7, action_steps=5, hidden_size=512, depth=6, num_heads=8):
        super().__init__()
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.dit = DiT(in_channels=action_dim, hidden_size=hidden_size, depth=depth, num_heads=num_heads, token_size=latent_dim, future_action_window_size=action_steps, num_latent_action=1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')

    def forward(self, latent_action, noisy_actions, timesteps):
        return self.dit(noisy_actions, timesteps, latent_action.unsqueeze(1))

    def loss(self, latent_action, action_target):
        batch_size = latent_action.shape[0]
        gt_action_seq = action_target.view(batch_size, self.action_steps, self.action_dim)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latent_action.device).long()
        noise = torch.randn_like(gt_action_seq)
        noisy_actions = self.noise_scheduler.add_noise(gt_action_seq, noise, timesteps)
        noise_pred = self.forward(latent_action, noisy_actions, timesteps)
        return F.huber_loss(noise_pred, noise, delta=1.0)

    @torch.no_grad()
    def sample(self, latent_action):
        batch_size = latent_action.shape[0]
        device = latent_action.device
        action_seq = torch.randn((batch_size, self.action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(500) 
        for t in self.noise_scheduler.timesteps:
            model_output = self.dit(action_seq, t.unsqueeze(0).to(device), latent_action.unsqueeze(1))
            action_seq = self.noise_scheduler.step(model_output, t, action_seq).prev_sample
        return action_seq.view(batch_size, -1)

# ==========================================
# 3. Training & Evaluation Engine (保持不变)
# ==========================================
def train_one_epoch(model, loader, optimizer, scheduler, accelerator, epoch, model_type, action_steps, dim_labels):
    model.train()
    total_loss, total_samples = 0.0, 0
    unwrapped_model = accelerator.unwrap_model(model)
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
    for step, batch in enumerate(pbar):
        latent_action, action = batch[0], batch[1]
        optimizer.zero_grad()
        
        if model_type == 'dit': loss = unwrapped_model.loss(latent_action, action)
        else: loss = unwrapped_model.loss(model(latent_action), action)
        
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        
        avg_loss = accelerator.gather(loss).mean().item()
        total_loss += avg_loss
        total_samples += 1

        if accelerator.is_local_main_process:
            metrics = {"train/step_loss": avg_loss, "train/lr": optimizer.param_groups[0]['lr']}
            if step % 20 == 0:
                with torch.no_grad():
                    if model_type == 'dit':
                        sample_size = min(4, latent_action.shape[0])
                        sample_pred = unwrapped_model.sample(latent_action[:sample_size]) 
                        sample_target = action[:sample_size]
                    else:
                        sample_pred, sample_target = model(latent_action), action
                    step_mse = F.mse_loss(sample_pred, sample_target).item()
                    metrics["train/mse"] = step_mse
                    for k, v in compute_per_dim_mse(sample_pred, sample_target, action_steps, dim_labels).items(): metrics[f"train/mse_{k}"] = v
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'mse': f"{step_mse:.4f}"})
            else:
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            wandb.log(metrics)

    return total_loss / total_samples

def evaluate(model, loader, accelerator, model_type, epoch, save_dir, action_steps, dim_labels, dataset_name, prefix="val"):
    model.eval()
    all_preds, all_targets = [], []
    total_huber_loss, total_samples = 0.0, 0
    unwrapped_model = accelerator.unwrap_model(model)
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc=f"Evaluating {prefix}", disable=not accelerator.is_local_main_process)):
            latent_action, action = batch[0], batch[1]
            src_paths, tgt_paths, a_mean, a_std = batch[2], batch[3], batch[4], batch[5]

            if model_type == 'dit':
                h_loss = unwrapped_model.loss(latent_action, action)
                pred_action = unwrapped_model.sample(latent_action)
            else:
                pred_direct = model(latent_action)
                h_loss = unwrapped_model.loss(pred_direct, action)
                pred_action = pred_direct 
            
            total_huber_loss += accelerator.gather(h_loss).mean().item()
            total_samples += 1

            all_preds.append(accelerator.gather_for_metrics(pred_action))
            all_targets.append(accelerator.gather_for_metrics(action))

            if step == 0 and accelerator.is_local_main_process:
                current_batch_size = pred_action.shape[0]
                plot_limit = min(5, current_batch_size) 
                for i in range(plot_limit):
                    save_separate_eval_samples(
                        pred_norm=pred_action[i].cpu().numpy(), 
                        gt_norm=action[i].cpu().numpy(),
                        mean=a_mean[i].cpu().numpy(), 
                        std=a_std[i].cpu().numpy(),
                        src_img_path=src_paths[i], 
                        tgt_img_path=tgt_paths[i],
                        chunk_size=action_steps, 
                        epoch=epoch, 
                        save_dir=os.path.join(save_dir, prefix),
                        dim_labels=dim_labels, 
                        sample_idx=i  # 使用循环索引 i 作为 sample_idx，确保保存的文件名不冲突
                    )
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    val_loss_final = total_huber_loss / total_samples
    val_total_mse = F.mse_loss(all_preds, all_targets).item()
    val_dim_metrics = compute_per_dim_mse(all_preds, all_targets, action_steps, dim_labels)
    val_group_metrics = compute_group_mse(all_preds, all_targets, action_steps, dataset_name)
    
    return val_loss_final, val_total_mse, val_dim_metrics, val_group_metrics

# ==========================================
# 4. Main Execution
# ==========================================
def make_reproducible(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--val_unseen_csv', type=str, default=None)
    parser.add_argument('--global_stats_json', type=str, default=None, help="Path to global_robot_stats.json")
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb_project', type=str, default="vla-action-expert")
    parser.add_argument('--wandb_name', type=str, default="mlp_norm_run")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='calvin')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'dit'])
    parser.add_argument('--action_mode', type=str, default='absolute', choices=['absolute', 'relative'],
                        help="absolute predicts the full action chunk; relative predicts last action minus first action.")
    parser.add_argument('--action_data_root', type=str, default=None,
                        help="Optional LARYBench data root. Relative mode reads regression/<dataset>_relative/ under this root.")
    parser.add_argument('--dit_hidden_size', type=int, default=512)
    parser.add_argument('--dit_depth', type=int, default=6)
    args = parser.parse_args()
    global_stats_json = args.global_stats_json
    if args.action_mode == 'relative' and global_stats_json is None:
        global_stats_json = get_relative_stats_path(args.action_data_root, args.dataset)
    args.global_stats_json = global_stats_json

    make_reproducible(args.seed)
    # Do NOT use accelerate's log_with="wandb" — it calls wandb.init() in every
    # worker process, causing all 8 GPUs to hit the network simultaneously and hang.
    # Instead, only the main process initialises wandb directly below.
    accelerator = Accelerator(project_dir=args.save_dir)
    if accelerator.is_local_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    action_dim = get_action_dim(args.dataset)
    action_steps = get_action_steps(args.stride, args.action_mode)
    dim_labels = get_dim_labels(args.dataset)
    
    # Calvin 的全局统计信息
    action_mean, action_std = None, None
    if args.action_mode == 'relative' and global_stats_json and os.path.exists(global_stats_json):
        with open(global_stats_json, 'r') as f:
            stats = json.load(f)
        if args.dataset in ['agibotbeta', 'robocoin']:
            pass
        else:
            action_mean = np.array(stats['mean'])
            action_std = np.array(stats['std'])
            action_std = np.where(action_std < 1e-6, 1.0, action_std)
    elif args.action_mode == 'relative':
        raise FileNotFoundError(
            "Relative action mode requires relative-action statistics. "
            f"Expected: {global_stats_json}. Generate them with utils/prepare_relative_actions.py "
            "or pass --global_stats_json explicitly."
        )
    elif args.dataset == 'calvin':
        action_mean = np.array([0.03993005, -0.1113833 ,  0.50033228,  1.04580053, -0.08165425, 1.58390577, -0.08441296]) 
        action_std = np.array([0.14403107, 0.09919957, 0.05518382, 2.89455128, 0.13053949, 0.57474015, 0.99643086]) 
    elif args.dataset not in ['agibotbeta', 'robocoin']:
        action_mean = np.array([ 0.02959172,  0.39965126,  0.25788001,  1.0136418 , -0.38189239, -0.13961033,  0.54842541]) 
        action_std = np.array([0.15966601, 0.13813421, 0.1362726 , 2.70449661, 0.63210694, 1.72885403, 0.49764945]) 


    train_dataset = ActionExpertDataset(
        args.train_csv, args.dataset, args.stride, global_stats_json, action_mean, action_std,
        action_mode=args.action_mode, action_data_root=args.action_data_root
    )
    val_dataset = ActionExpertDataset(
        args.val_csv, args.dataset, args.stride, global_stats_json, action_mean, action_std,
        action_mode=args.action_mode, action_data_root=args.action_data_root
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    val_unseen_loader = None
    if args.val_unseen_csv and os.path.exists(args.val_unseen_csv):
        val_unseen_dataset = ActionExpertDataset(
            args.val_unseen_csv, args.dataset, args.stride, global_stats_json, action_mean, action_std,
            action_mode=args.action_mode, action_data_root=args.action_data_root
        )
        val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    sample_la = train_dataset[0][0]
    la_dim = sample_la.shape[0]
    if accelerator.is_local_main_process:
        print(f"Regression dimensions: latent_action_dim={la_dim}, action_output_dim={action_dim * action_steps}")

    if args.model_type == 'dit':
        model = ActionExpertDiT(latent_dim=la_dim, action_dim=action_dim, action_steps=action_steps, hidden_size=args.dit_hidden_size, depth=args.dit_depth)
    else:
        model = ActionExpertMLP(input_dim=la_dim, action_dim=action_dim, hidden_dim=4096, num_blocks=2, action_steps=action_steps)
    
    if accelerator.is_local_main_process: print_model_params(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(args.epochs * len(train_loader) * 0.1))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    if val_unseen_loader: val_unseen_loader = accelerator.prepare(val_unseen_loader)

    best_val_mse = float('inf')
    best_result = {}  # 用于记录最佳结果
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, args.model_type, action_steps, dim_labels)
        val_loss, val_mse, val_dim_metrics, val_group_metrics = evaluate(model, val_loader, accelerator, args.model_type, epoch, args.save_dir, action_steps, dim_labels, args.dataset, prefix="val_seen")
        
        if val_unseen_loader:
            u_val_loss, u_val_mse, u_val_dim_metrics, u_val_group_metrics = evaluate(model, val_unseen_loader, accelerator, args.model_type, epoch, args.save_dir, action_steps, dim_labels, args.dataset, prefix="val_unseen")
    
        if accelerator.is_local_main_process:
            log_dict = {"loss/train": train_loss, "loss/val_seen": val_loss, "val_seen/mse": val_mse, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']}
            for k, v in val_dim_metrics.items(): log_dict[f"val_seen/mse_{k}"] = v
            for k, v in val_group_metrics.items(): log_dict[f"val_seen/{k}"] = v
            
            if val_unseen_loader:
                log_dict.update({"loss/val_unseen": u_val_loss, "val_unseen/mse": u_val_mse})
                for k, v in u_val_dim_metrics.items(): log_dict[f"val_unseen/mse_{k}"] = v
                for k, v in u_val_group_metrics.items(): log_dict[f"val_unseen/{k}"] = v

            wandb.log(log_dict)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Seen Loss: {val_loss:.4f} | Val Seen MSE: {val_mse:.4f}")
            if val_group_metrics:
                group_str = "  ".join([f"{k}: {v:.4f}" for k, v in val_group_metrics.items()])
                print(f"          | Val Seen Groups -> {group_str}")
            if val_unseen_loader:
                print(f"          | Val Unseen Loss: {u_val_loss:.4f} | Val Unseen MSE: {u_val_mse:.4f}")
                if u_val_group_metrics:
                    u_group_str = "  ".join([f"{k}: {v:.4f}" for k, v in u_val_group_metrics.items()])
                    print(f"          | Val Unseen Groups -> {u_group_str}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save({'model_state_dict': accelerator.unwrap_model(model).state_dict()}, os.path.join(args.save_dir, "best_model.pth"))
                # 记录最佳结果
                best_result = {
                    "best_epoch": epoch,
                    "train_loss": train_loss,
                    "val_seen_loss": val_loss,
                    "val_seen_mse": val_mse,
                }
                for k, v in val_dim_metrics.items():
                    best_result[f"val_seen_mse_{k}"] = v
                for k, v in val_group_metrics.items():
                    best_result[f"val_seen_{k}"] = v
                if val_unseen_loader:
                    best_result["val_unseen_loss"] = u_val_loss
                    best_result["val_unseen_mse"] = u_val_mse
                    for k, v in u_val_dim_metrics.items():
                        best_result[f"val_unseen_mse_{k}"] = v
                    for k, v in u_val_group_metrics.items():
                        best_result[f"val_unseen_{k}"] = v
    
    # 训练结束后将最佳结果保存到 CSV
    if accelerator.is_local_main_process and best_result:
        best_csv_path = os.path.join(args.save_dir, "best_result.csv")
        pd.DataFrame([best_result]).to_csv(best_csv_path, index=False)
        print(f"\nBest result saved to {best_csv_path}")
        print(f"Best Epoch: {best_result['best_epoch']} | Val Seen Loss: {best_result['val_seen_loss']:.4f} | Val Seen MSE: {best_result['val_seen_mse']:.4f}")

    if accelerator.is_local_main_process:
        wandb.finish()
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
