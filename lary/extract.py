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
from torch import nn
import torchvision.transforms.functional as F

from lary.config import get_config
from lary.path_resolver import resolve_data_path
from registry.registry import MODEL


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


class VideoActionDataset(Dataset):
    """Dataset for video-based latent action extraction."""

    def __init__(self, df: pd.DataFrame, model: nn.Module, image_size: int = 224,
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
        if model.name == 'vjepa2':
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
        return self.model.process_video(self.data, idx, self.image_size)


class ImagePairDataset(Dataset):
    """Dataset for image-pair based latent action extraction."""

    def __init__(self, df: pd.DataFrame, model: nn.Module, image_size: int = 224,
                 stride: int = 5, dataset: str = None, split: str = None):
        self.data = df.copy()
        self.model = model
        self.image_size = image_size
        self.stride = stride
        self.dataset = dataset
        self.split = split

        # Resolve relative paths if needed
        self._resolve_paths()

        if model.name == 'vjepa2':
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

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        global_idx = self.data.index[idx]

        try:
            tensor = self.model.process_image(row['src_img'], row['tgt_img'], self.image_size)
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
        self.model = MODEL.build(self.config.model)

    def extract(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """Extract latent actions from dataset."""
        if self.config.mode == "image":
            dataset = ImagePairDataset(df, self.model, stride=self.config.stride,
                                       dataset=self.config.dataset, split=self.config.split)
            collate_fn = image_collate_fn
        else:
            dataset = VideoActionDataset(df, self.model, dataset=self.config.dataset,
                                         split=self.config.split)
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

                batch_tokens, batch_ids = self.model.get_latent_action(batch_data, batch_rel_indices if self.config.mode == "video" else None, self.config.mode)

                for i, global_idx in enumerate(batch_indices):
                    save_name = f"latent_action_{global_idx:08d}.npz"
                    save_path = os.path.join(output_dir, save_name)
                    np.savez_compressed(save_path, tokens=batch_tokens[i], indices=batch_ids[i])
                    # Store relative path for portability
                    relative_path = os.path.join(self.config.dataset, self.config.split, self.model.name, save_name)
                    df.at[global_idx, 'la_path'] = relative_path

        return df


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

    # Partition data if needed. Avoid np.array_split on DataFrames here because
    # some NumPy/Pandas combinations can yield ndarray partitions.
    if num_partitions > 1:
        partition_indices = np.array_split(np.arange(len(full_df)), num_partitions)
        current_df = full_df.iloc[partition_indices[partition]].copy()
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
