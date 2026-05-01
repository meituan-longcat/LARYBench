"""
Latent action extraction module.

Provides unified interface for extracting latent actions from videos or image pairs.
"""

import os
import gc
import random
from pathlib import Path
from typing import Optional, List, Union, Dict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F

from lary.config import get_config
from lary.path_resolver import resolve_data_path


# =============================================================================
# Core Functions
# =============================================================================

def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    from torchvision.io import read_video

    video, _, info = read_video(fp, pts_unit="sec")
    fps = int(info.get("video_fps", 25.0))

    if resize_h is not None and resize_w is not None:
        video = video.permute(0, 3, 1, 2)
        video = F.resize(video, [resize_h, resize_w], antialias=True)
        video = video.permute(0, 2, 3, 1)

    return video, fps


class VideoActionDataset(Dataset):
    """Dataset for video-based latent action extraction."""

    def __init__(self, df: pd.DataFrame, model: str, image_size: int = 224,
                 dataset: str = None, split: str = None):
        self.data = df.copy()
        self.model = model
        self.image_size = image_size
        self.dataset = dataset
        self.split = split
        # Resolve relative video paths
        if 'video_path' in self.data.columns:
            self.data['video_path'] = self.data['video_path'].apply(
                lambda x: resolve_data_path(str(x), dataset, split) if pd.notna(x) else x
            )

        # Setup vjepa2 transforms if needed
        if model == 'vjepa2':
            from get_latent_action.models.vjepa2.evals.video_classification_frozen.utils import make_transforms
            self.transform = make_transforms(
                training=False,
                num_views_per_clip=1,
                random_horizontal_flip=False,
                random_resize_aspect_ratio=(1.0, 1.0),
                random_resize_scale=(1.0, 1.0),
                reprob=0,
                auto_augment=False,
                motion_shift=False,
                crop_size=224,
                normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        global_idx = self.data.index[idx]
        video_path = row['video_path']

        indices = [int(i) for i in str(row['sample_indices']).split(',')]
        rel_indices = torch.tensor([i - indices[0] for i in indices], dtype=torch.long)

        all_frames = load_video_frames(video_path)
        if not all_frames:
            return None

        total_frames = len(all_frames)

        if self.model == 'vjepa2':
            selected_frames = []
            for f_idx in indices:
                safe_idx = max(0, min(int(f_idx), total_frames - 1))
                selected_frames.append(all_frames[safe_idx])
            return global_idx, selected_frames, rel_indices

        # Default processing for other models
        selected_frames = []
        for f_idx in indices:
            safe_idx = max(0, min(int(f_idx), total_frames - 1))
            img = cv2.resize(all_frames[safe_idx], (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
            selected_frames.append(img / 255.0)

        if self.model == 'wan2-2':
            tensors = []
            for f_idx in indices:
                safe_idx = max(0, min(int(f_idx), total_frames - 1))
                img = Image.fromarray(all_frames[safe_idx])
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                tensor = F.to_tensor(img).sub_(0.5).div_(0.5)
                tensors.append(tensor)
            tensor = torch.stack(tensors, dim=1)
        elif self.model == 'villa-x':
            tensor, _ = read_video_tensor(video_path, resize_h=self.image_size, resize_w=self.image_size)
            tensor = tensor[indices]
        elif self.model == 'dinov3-origin' or self.model == 'dinov2-origin' or self.model == 'siglip2-origin':
            pair = np.stack(selected_frames)
            tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
        elif self.model == 'flux2':
            pair = np.stack(selected_frames)
            tensor = torch.tensor(pair, dtype=torch.float32) * 2 - 1
        else:
            pairs_list = []
            for i in range(len(selected_frames) - 1):
                pair = np.stack([selected_frames[i], selected_frames[i+1]])
                pairs_list.append(pair)
            tensor = torch.tensor(np.array(pairs_list), dtype=torch.float32).permute(0, 4, 1, 2, 3)

        return global_idx, tensor, rel_indices


class ImagePairDataset(Dataset):
    """Dataset for image-pair based latent action extraction."""

    def __init__(self, df: pd.DataFrame, model: str, image_size: int = 224,
                 stride: int = 5, dataset: str = None, split: str = None):
        self.data = df.copy()
        self.model = model
        self.image_size = image_size
        self.stride = stride
        self.dataset = dataset
        self.split = split

        # Resolve relative paths if needed
        self._resolve_paths()

        if model == 'vjepa2':
            from get_latent_action.models.vjepa2.evals.video_classification_frozen.utils import make_transforms
            self.transform = make_transforms(
                training=False,
                num_views_per_clip=1,
                random_horizontal_flip=False,
                random_resize_aspect_ratio=(1.0, 1.0),
                random_resize_scale=(1.0, 1.0),
                reprob=0,
                auto_augment=False,
                motion_shift=False,
                crop_size=224,
                normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def _resolve_paths(self):
        """Resolve relative paths in the DataFrame using path_resolver (split-aware)."""
        path_columns = ['src_img', 'tgt_img', 'src_state', 'tgt_state', 'action']
        for col in path_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(
                    lambda x: resolve_data_path(str(x), self.dataset, self.split) if pd.notna(x) else x
                )

    def __len__(self):
        return len(self.data)

    def load_image(self, path: str):
        if self.model == 'wan2-2':
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            return F.to_tensor(img).sub_(0.5).div_(0.5)
        elif self.model == 'flux2':
            img = Image.open(path).convert('RGB')
            return np.array(img.resize((self.image_size, self.image_size), Image.LANCZOS))
        elif self.model == 'villa-x':
            img = Image.open(path).convert('RGB')
            return np.array(img.resize((self.image_size, self.image_size), Image.LANCZOS))
        elif self.model == 'vjepa2':
            img = Image.open(path).convert('RGB')
            return np.array(img)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            return img / 255.0

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        global_idx = self.data.index[idx]

        try:
            src_img = self.load_image(row['src_img'])
            tgt_img = self.load_image(row['tgt_img'])

            if self.model == 'wan2-2':
                tensor = torch.stack([src_img, tgt_img], dim=1)
                return global_idx, tensor
            elif self.model == 'flux2':
                tensor = torch.tensor(np.stack([src_img, tgt_img]), dtype=torch.float32) * 2 - 1
                return global_idx, tensor
            elif self.model == 'villa-x':
                tensor = torch.tensor(np.array([src_img, tgt_img]), dtype=torch.uint8)
                return global_idx, tensor
            elif self.model == 'vjepa2':
                return global_idx, [src_img, tgt_img]
            else:
                pair = np.stack([src_img, tgt_img])
                tensor = torch.tensor(pair, dtype=torch.float32).permute(3, 0, 1, 2)
                return global_idx, tensor
        except Exception as e:
            print(f"Error loading images at index {global_idx}: {e}")
            return None


def video_collate_fn(batch):
    """Collate function for video dataset."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    indices, tensors, rel_indices = zip(*batch)
    if isinstance(tensors[0], list):
        return indices, list(tensors), torch.stack(rel_indices)
    return indices, torch.stack(tensors), torch.stack(rel_indices)


def image_collate_fn(batch):
    """Collate function for image pair dataset."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    indices, data = zip(*batch)

    if isinstance(data[0], list):
        return indices, list(data)
    else:
        return indices, torch.stack(data)


@dataclass
class ExtractionConfig:
    """Configuration for extraction."""
    model: str
    dataset: str
    input_file: Optional[str] = None
    output_dir: Optional[str] = None
    split: str = "all"
    batch_size: int = 16
    num_workers: int = 8
    num_partitions: int = 1
    partition: int = 0
    gpus: List[int] = None
    mode: str = "video"
    stride: int = 5
    perspective: str = "1st"
    seed: int = 42


class LatentActionExtractor:
    """Main class for extracting latent actions."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._setup_model()

    def _setup_model(self):
        """Initialize the model."""
        from get_latent_action.dynamics import get_dynamic_tokenizer

        self.model = get_dynamic_tokenizer(self.config.model)
        if self.config.model == 'wan2-2':
            self.model.model.to("cuda").eval()
        else:
            self.model.to("cuda").eval()

    def extract(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """Extract latent actions from dataset."""
        if self.config.mode == "image":
            dataset = ImagePairDataset(df, self.config.model, stride=self.config.stride,
                                       dataset=self.config.dataset, split=self.config.split)
            collate_fn = image_collate_fn
        else:
            dataset = VideoActionDataset(df, self.config.model,
                                         dataset=self.config.dataset, split=self.config.split)
            collate_fn = video_collate_fn

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Partition {self.config.partition}"):
                if batch is None:
                    continue

                batch_indices, batch_data = batch[:2]
                if self.config.mode == "video":
                    batch_rel_indices = batch[2]

                batch_tokens, batch_ids = self._process_batch(batch_data, batch_rel_indices if self.config.mode == "video" else None)

                for i, global_idx in enumerate(batch_indices):
                    save_name = f"latent_action_{global_idx:08d}.npz"
                    save_path = os.path.join(output_dir, save_name)
                    np.savez_compressed(save_path, tokens=batch_tokens[i], indices=batch_ids[i])
                    # Store relative path for portability
                    relative_path = os.path.join(self.config.dataset, self.config.split, self.config.model, save_name)
                    df.at[global_idx, 'la_path'] = relative_path

        return df

    def _process_batch(self, batch_data, batch_rel_indices=None):
        """Process a batch of data through the model."""
        from get_latent_action.dynamics import get_latent_action

        if self.config.model == 'wan2-2':
            video_list = [t.to("cuda") for t in list(batch_data)]
            with torch.no_grad():
                latents = self.model.encode(video_list)
            batch_tokens = []
            for l in latents:
                la = np.transpose(l.cpu().numpy(), (1, 2, 3, 0))
                la = la.reshape(la.shape[0], -1, la.shape[-1])
                batch_tokens.append(la)
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]

        elif self.config.model == 'flux2':
            batch_tokens = get_latent_action(batch_data, self.model, self.config.model)
            if self.config.mode == "image":
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 2, -1, batch_tokens.shape[-3])
            else:
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 9, -1, batch_tokens.shape[-3])
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]

        elif self.config.model == 'villa-x':
            batch_output = self.model.idm(batch_data.to("cuda"), return_dict=True)
            batch_tokens = batch_output['vq_tokens'].cpu().numpy()
            if self.config.mode == "image":
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], -1, self.model.config.action_latent_dim)
            else:
                batch_tokens = batch_tokens.reshape(batch_tokens.shape[0], 8, -1, self.model.config.action_latent_dim)
            batch_ids = batch_output['indices'].cpu().numpy().reshape(batch_tokens.shape[0], batch_tokens.shape[1] if len(batch_tokens.shape) > 2 else -1, -1)

        elif self.config.model == 'vjepa2':
            # VJEPA2 specific processing
            if self.config.mode == "video":
                transformed_batch = []
                for video_frames in batch_data:
                    t_frames = torch.stack(self.model.transform(video_frames), dim=0)
                    transformed_batch.append(t_frames.to("cuda", non_blocking=True))
                clips_batch = [[torch.stack(transformed_batch, dim=0).squeeze(1)]]
                clip_indices_batch = batch_rel_indices.to("cuda")
            else:
                clips_batch = [[torch.stack([self.model.transform(clip)[0] for clip in batch_data], dim=0).to("cuda", non_blocking=True)]]
                clip_indices_batch = torch.tensor([0, self.config.stride], device="cuda").unsqueeze(0).repeat(len(batch_data), 1)

            batch_tokens = self.model(clips_batch, clip_indices_batch)[0].cpu().numpy()
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]

        elif self.config.model == 'dinov3-origin' or self.config.model == 'dinov2-origin' or self.config.model == 'siglip2-origin':
            batch_input = batch_data.to("cuda")
            batch_tokens = get_latent_action(batch_input, self.model, self.config.model)
            batch_ids = [np.array([]) for _ in range(len(batch_tokens))]

        else:
            # batch_data shape: (B, P, C, T, H, W)  — P frame-pairs per video
            # The LAQ model forward() expects 5-D input (B*P, C, T, H, W).
            # Flatten the batch and pair dimensions, run inference, then restore.
            if batch_data.ndim == 6:
                B, P, C, T, H, W = batch_data.shape
                flat_input = batch_data.view(B * P, C, T, H, W).to("cuda")
                flat_tokens, flat_ids = get_latent_action(flat_input, self.model, self.config.model)
                # flat_tokens: (B*P, N, D)  flat_ids: (B*P, N)
                batch_tokens = flat_tokens.reshape(B, P, flat_tokens.shape[-2], flat_tokens.shape[-1])
                batch_ids = flat_ids.reshape(B, P, -1)
            else:
                batch_input = batch_data.to("cuda")
                batch_tokens, batch_ids = get_latent_action(batch_input, self.model, self.config.model)

        return batch_tokens, batch_ids


def extract_latent_actions(
    model: str,
    dataset: str,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    split: str = "all",
    batch_size: int = 16,
    num_workers: int = 8,
    gpus: Optional[List[int]] = None,
    mode: str = "video",
    stride: int = 5,
    perspective: str = "1st",
    num_partitions: int = 1,
    partition: int = 0,
) -> None:
    """
    Main function to extract latent actions.

    Args:
        model: Model name (villa-x, vjepa2, wan2-2, etc.)
        dataset: Dataset name
        input_file: Input CSV file path
        output_dir: Output directory
        split: Data split (train, val, all)
        batch_size: Batch size
        num_workers: Number of data loading workers
        gpus: List of GPU IDs
        mode: Processing mode (video or image)
        stride: Frame stride for image mode
        perspective: Camera perspective
        num_partitions: Number of partitions for distributed processing
        partition: Current partition index
    """
    config = get_config()

    # Setup paths
    if input_file is None:
        # All metadata CSVs live in $LARY_ROOT/data/ with canonical naming:
        #   {dataset}_metadata_{split}.csv
        lary_root = os.environ.get("LARY_ROOT", str(config.paths.project_root))
        data_dir = Path(lary_root) / "data"

        possible_paths = [
            data_dir / f"{dataset}_metadata_{split}.csv",
            data_dir / f"{dataset}_{split}.csv",
        ]

        for p in possible_paths:
            if p.exists():
                input_file = str(p)
                print(f"Found input file: {input_file}")
                break
        if input_file is None:
            raise ValueError(
                f"Input file not found for dataset '{dataset}' / split '{split}'.\n"
                f"Expected one of:\n" + "\n".join(f"  {p}" for p in possible_paths) + "\n"
                f"Run `python prepare_metadata.py` first, or use --input to specify the file directly."
            )
    if output_dir is None:
        output_dir = os.environ.get("LARY_LA_DIR")
        if output_dir is None:
            output_dir = str(config.paths.log_dir / "latent_action")

    # Setup GPU
    if gpus:
        gpu_id = gpus[partition % len(gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup seed
    setup_seed(42)

    # Create output directory (include split to separate train/val)
    save_dir = Path(output_dir) / dataset / split / model
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    full_df = pd.read_csv(input_file)
    if 'la_path' not in full_df.columns:
        full_df['la_path'] = ""

    # Partition data if needed
    if num_partitions > 1:
        partitions = np.array_split(full_df, num_partitions)
        current_df = partitions[partition].copy()
    else:
        current_df = full_df.copy()

    # Create extractor and run
    extraction_config = ExtractionConfig(
        model=model,
        dataset=dataset,
        input_file=input_file,
        output_dir=output_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        num_partitions=num_partitions,
        partition=partition,
        gpus=gpus,
        mode=mode,
        stride=stride,
        perspective=perspective,
    )

    extractor = LatentActionExtractor(extraction_config)
    processed_df = extractor.extract(current_df, str(save_dir))

    # Save results to $LARY_ROOT/data/ with canonical naming:
    #   video mode : {split}_la_{dataset}_{model}.csv
    #   image mode : {split}_la_{dataset}_{stride}_{model}.csv
    #   partitioned: append _{partition} before .csv
    lary_root = os.environ.get("LARY_ROOT", str(config.paths.project_root))
    data_out_dir = Path(lary_root) / "data"
    data_out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "image":
        base_name = f"{split}_la_{dataset}_{stride}_{model}"
    else:
        base_name = f"{split}_la_{dataset}_{model}"

    if num_partitions > 1:
        csv_name = f"{base_name}_{partition}.csv"
    else:
        csv_name = f"{base_name}.csv"

    csv_out = data_out_dir / csv_name
    processed_df.to_csv(csv_out, index=False)
    print(f"Partition {partition} finished. CSV saved to {csv_out}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
