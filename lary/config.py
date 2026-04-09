"""
Unified configuration management for LARY.

Supports configuration from:
1. Environment variables (highest priority)
2. Config file (YAML)
3. Default values
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import yaml


def _get_env_or_default(env_name: str, default: str = "") -> str:
    """Get value from environment variable or return default."""
    return os.environ.get(env_name, default)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Try environment variable first
    env_root = os.environ.get("LARY_ROOT")
    if env_root:
        return Path(env_root)
    # Otherwise, find by looking for pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent


@dataclass
class PathConfig:
    """Path configuration."""
    # Project paths
    project_root: Path = field(default_factory=_get_project_root)
    log_dir: Path = field(default_factory=lambda: Path(_get_env_or_default("LARY_LOG_DIR", "~/logs/LARY")).expanduser())

    # Data paths (data files are located via DATA_DIR env var, see path_resolver.py)

    # Model paths
    model_dir: Path = field(default_factory=lambda: Path(_get_env_or_default("MODEL_DIR", "~/models/la_models")).expanduser())

    # External model paths
    dino_v2_path: Optional[Path] = field(default_factory=lambda: _optional_path("DINO_V2_PATH"))
    dino_v3_path: Optional[Path] = field(default_factory=lambda: _optional_path("DINO_V3_PATH"))
    lavit_path: Optional[Path] = field(default_factory=lambda: _optional_path("LAVIT_PATH"))
    siglip_path: Optional[Path] = field(default_factory=lambda: _optional_path("SIGLIP_PATH"))
    magvit_config_path: Optional[Path] = field(default_factory=lambda: _optional_path("MAGVIT_CONFIG_PATH"))
    magvit_tokenizer_path: Optional[Path] = field(default_factory=lambda: _optional_path("MAGVIT_TOKENIZER_PATH"))
    magvit2_config_path: Optional[Path] = field(default_factory=lambda: _optional_path("MAGVIT2_CONFIG_PATH"))
    magvit2_tokenizer_path: Optional[Path] = field(default_factory=lambda: _optional_path("MAGVIT2_TOKENIZER_PATH"))

    # External project paths

    # Conda
    conda_sh_path: Optional[Path] = field(default_factory=lambda: _optional_path("CONDA_SH_PATH"))

    def __post_init__(self):
        """Ensure directories exist."""
        for attr_name in ['log_dir', 'model_dir']:
            attr_value = getattr(self, attr_name)
            if attr_value and not attr_value.exists():
                attr_value.mkdir(parents=True, exist_ok=True)


def _optional_path(env_name: str) -> Optional[Path]:
    """Get optional path from environment variable."""
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else None


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str = "dinov3-origin"
    device: str = "cuda"

    # LAQ model parameters
    dim: int = 1024
    quant_dim: int = 32
    codebook_size: int = 8
    image_size: int = 224
    patch_size: int = 16
    spatial_depth: int = 4
    temporal_depth: int = 4
    dim_head: int = 64
    heads: int = 16
    code_seq_len: int = 16


@dataclass
class DataConfig:
    """Data processing configuration."""
    batch_size: int = 256
    num_workers: int = 8
    num_partitions: int = 8
    seed: int = 42
    stride: int = 5


@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # WandB
    wandb_project: str = "lary"
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("WANDB_API_KEY"))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        paths = PathConfig(**data.get("paths", {}))
        model = ModelConfig(**data.get("model", {}))
        data_cfg = DataConfig(**data.get("data", {}))

        return cls(paths=paths, model=model, data=data_cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        data = {
            "paths": {
                "project_root": str(self.paths.project_root),
                "log_dir": str(self.paths.log_dir),
                "data_dir": str(self.paths.data_dir),
                "model_dir": str(self.paths.model_dir),
            },
            "model": {
                "name": self.model.name,
                "device": self.model.device,
            },
            "data": {
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
            }
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
