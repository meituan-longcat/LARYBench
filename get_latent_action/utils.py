from typing import List
import cv2
import numpy as np
from torchvision.io import read_video
import torchvision.transforms.functional as F


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
    video, _, info = read_video(fp, pts_unit="sec")
    fps = int(info.get("video_fps", 25.0))

    if resize_h is not None and resize_w is not None:
        video = video.permute(0, 3, 1, 2)
        video = F.resize(video, [resize_h, resize_w], antialias=True)
        video = video.permute(0, 2, 3, 1)

    return video, fps
