# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license.

import os
import gc
import json
import math
import logging
import pprint
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local imports (Assumed to be available in the environment)
from classification.evals.video_classification_frozen.utils import make_transforms
from classification.src.datasets.data_manager import init_data
from classification.src.models.attentive_pooler import AttentiveClassifier, TemporalAttentiveClassifier, FeatureEvaluator
from classification.src.utils.checkpoint_loader import robust_checkpoint_loader
from classification.src.utils.distributed import AllReduce, init_distributed
from classification.src.utils.logging import AverageMeter, CSVLogger
import wandb 
from utils.model_utils import print_model_params

# --- Configuration & Setup ---
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
pp = pprint.PrettyPrinter(indent=4)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def setup_env():
    """Handle distributed environment setup."""
    try:
        # For SLURM environments
        if "SLURM_LOCALID" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

# --- Utility Functions ---

def clear_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def gather_distributed_data(predictions, labels, samples, world_size, rank, device):
    """
    Gather predictions, labels, and sample paths from all GPUs.
    Returns aggregated lists only on rank 0.
    """
    # Convert to tensors
    pred_tensor = torch.tensor(predictions, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    # Gather sizes
    local_size = torch.tensor([len(predictions)], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    sizes = [int(s.item()) for s in all_sizes]
    max_size = max(sizes)

    # Pad and Gather Tensors
    def pad_and_gather(tensor, max_len, pad_val=-1):
        padded = torch.full((max_len,), pad_val, dtype=torch.long, device=device)
        padded[:len(tensor)] = tensor
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return gathered

    all_preds_padded = pad_and_gather(pred_tensor, max_size)
    all_labels_padded = pad_and_gather(label_tensor, max_size)

    # Gather Objects (Strings/Paths)
    all_samples_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_samples_list, samples)

    if rank == 0:
        gathered_preds, gathered_labels, gathered_samples = [], [], []
        for i in range(world_size):
            valid_len = sizes[i]
            gathered_preds.extend(all_preds_padded[i][:valid_len].cpu().numpy().tolist())
            gathered_labels.extend(all_labels_padded[i][:valid_len].cpu().numpy().tolist())
            gathered_samples.extend(all_samples_list[i])
        return gathered_preds, gathered_labels, gathered_samples
    
    return None, None, None

def save_evaluation_metrics(all_preds, all_labels, all_samples, num_classes, save_dir, class_names=None):
    """Calculate, save and visualize confusion matrix and stats."""
    if class_names is None or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    # 1. Confusion Matrix
    cm = confusion_matrix(all_preds, all_labels, labels=range(num_classes))
    
    # Save JSON
    with open(os.path.join(save_dir, "confusion_matrix.json"), 'w') as f:
        json.dump({
            "confusion_matrix": cm.tolist(),
            "class_names": class_names
        }, f, indent=2)

    # Visualize (Percentage)
    plt.figure(figsize=(12, 10))
    cm_percent = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    sns.heatmap(cm_percent, annot=False, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (%)')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Classification Stats
    precision = np.diag(cm) / (np.sum(cm, axis=1) + 1e-9)
    recall = np.diag(cm) / (np.sum(cm, axis=0) + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    stats = []
    for i in range(num_classes):
        stats.append({
            "class": class_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i])
        })
        logger.info(f"Class {class_names[i]} Precision: {precision[i]*100:.2f}%")

    with open(os.path.join(save_dir, "classification_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    # 3. Save Predictions Case File
    # Note: Mapping logic assumes specific classes, adjust if generic
    mapping = {i: name for i, name in enumerate(class_names)}
    with open(os.path.join(save_dir, "case.txt"), "w") as f:
        for path, pred, true in zip(all_samples, all_preds, all_labels):
            f.write(f"{path}, {mapping.get(pred, pred)}, {mapping.get(true, true)}\n")

# --- Core Logic ---

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 权重参数. 
                   - 如果是 float (如 1.0), 则作为全局标量.
                   - 如果是 list 或 tensor (如 [0.1, 0.9]), 则对应每个类别的权重.
            gamma: 聚焦参数, 越大越关注难分样本.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 处理 alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: [N, C] (Logits, 未经过 Softmax)
        # targets: [N] (类别索引)
        
        # 1. 计算标准 Cross Entropy Loss (不进行归约，保持 shape 为 [N])
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. 计算概率 pt
        pt = torch.exp(-ce_loss)
        
        # 3. 计算 Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # 4. 计算 Alpha Term
        if self.alpha is not None:
            # 确保 alpha 在正确的设备上 (CPU/GPU)
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 如果 alpha 是向量，根据 targets 取出对应的权重
            if self.alpha.numel() > 1:
                # alpha_t 形状为 [N]
                alpha_t = self.alpha.gather(0, targets)
            else:
                alpha_t = self.alpha
                
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def run_epoch(
    device,
    model_list,
    data_loader,
    training=False,
    optimizer=None,
    scaler=None,
    schedulers=None,
    use_bfloat16=False,
    collect_preds=False
):
    """Unified training and validation loop."""
    for m in model_list:
        m.train(mode=training)

    # criterion = FocalLoss(alpha=[0.55, 0.69, 0.70, 1.10, 1.18, 1.26, 1.89, 4.84], gamma=2.0, reduction='mean') # 类别不平衡所需要做的
    # class_weights = torch.tensor([4.38, 5.49, 5.58, 8.78, 9.48, 10.11, 15.12, 38.68]).cuda()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # 标签平滑0.1
    # criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean') # 类别不平衡所需要做的


    top1_meters = [AverageMeter() for _ in model_list]
    loss_meters = [AverageMeter() for _ in model_list]
    
    # Storage for evaluation
    all_preds_per_model = [[] for _ in model_list]
    all_labels = []
    all_samples = []

    # Schedulers step (per epoch) is handled outside, but if per-step is needed, add here.
    
    for i, batch in enumerate(data_loader):
        # Batch: [labels, deltas (features), paths]
        labels = batch[0].to(device, non_blocking=True)
        features = batch[1].to(device, non_blocking=True)
        indices = batch[2] # 样本序号
        batch_size = len(labels)

        # Step schedulers if per-iteration
        if training and schedulers:
            for s_group in schedulers:
                for s in s_group: s.step()

        # Forward
        # context = torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16)
        context = torch.cuda.amp.autocast(dtype=torch.float16, enabled=(use_bfloat16 and training))

        # if not training:
        #     context = torch.inference_mode() # Combine with autocast if needed, or separate

        with context:
            # model_list is a list of classifiers (ensembling or multi-head training)
            outputs = [[m(features)] for m in model_list]
            
            # if training:
            losses = [[criterion(o, labels) for o in out_group] for out_group in outputs]

        # Metrics & Logging
        with torch.no_grad():
            # Average softmax outputs across heads if multiple exist within one model wrapper
            outputs_softmax = [sum([F.softmax(o, dim=1) for o in out_group]) / len(out_group) for out_group in outputs]
            
            for idx, (out_sm, meter, loss_group, l_meter) in enumerate(zip(outputs_softmax, top1_meters, losses, loss_meters)):
                acc = 100.0 * out_sm.max(dim=1).indices.eq(labels).sum() / batch_size
                meter.update(float(AllReduce.apply(acc)))
                
                avg_loss_val = sum([l.item() for l in loss_group]) / len(loss_group)
                # 使用 AllReduce 确保分布式环境下 Loss 记录准确
                l_meter.update(float(AllReduce.apply(torch.tensor(avg_loss_val, device=device))))

                if collect_preds:
                    all_preds_per_model[idx].extend(out_sm.max(dim=1).indices.cpu().numpy().tolist())

            if collect_preds:
                if len(all_labels) < (i + 1) * batch_size: # Avoid duplicating labels for each model
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_samples.extend(indices)

        # Backward
        if training:
            if use_bfloat16 and scaler:
                for s, loss_group, opt in zip(scaler, losses, optimizer):
                    for loss in loss_group:
                        s.scale(loss).backward()
                    s.step(opt)
                    s.update()
            else:
                for loss_group, opt in zip(losses, optimizer):
                    for loss in loss_group:
                        loss.backward()
                    opt.step()
            
            for opt in optimizer:
                opt.zero_grad()

        # Periodic Logging
        if i % 10 == 0:
            agg_acc = np.array([m.avg for m in top1_meters])
            agg_loss = np.array([m.avg for m in loss_meters])
            logger.info(f"[{i:5d}] Max Acc: {agg_acc.max():.3f}% | Min Loss: {agg_loss.min():.4f} | Mem: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
    
    # Return results
    best_idx = np.argmax([m.avg for m in top1_meters])
    max_acc = top1_meters[best_idx].avg
    best_loss = loss_meters[best_idx].avg # 获取对应最好模型的 Loss
    if not training:
        best_lr = 0
    else:
        best_lr = optimizer[best_idx].param_groups[0]["lr"] 
    
    if collect_preds:
        return max_acc, best_loss, all_preds_per_model[best_idx], all_labels, all_samples, best_idx, best_lr
    
    return max_acc, best_loss, None, None, None, best_idx, best_lr 


# --- Schedulers ---

class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            # Unpack config
            ref, final, start = group["mc_ref_lr"], group["mc_final_lr"], group["mc_start_lr"]
            warmup = group["mc_warmup_steps"]
            
            if self._step < warmup:
                prog = float(self._step) / max(1, warmup)
                lr = start + prog * (ref - start)
            else:
                prog = float(self._step - warmup) / max(1, self.T_max - warmup)
                lr = max(final, final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * prog)))
            group["lr"] = lr

class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        prog = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref, final = group["mc_ref_wd"], group["mc_final_wd"]
            wd = final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * prog))
            # Clamp logic from original code
            wd = max(final, wd) if final <= ref else min(final, wd)
            group["weight_decay"] = wd

# --- Dataloader ---

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    persistent_workers=True,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        persistent_workers=persistent_workers,
    )
    return data_loader, data_sampler

# --- Main Execution ---

def main(args_eval, resume_preempt=False):
    setup_env()
    
    # 1. Parse Configuration
    exp_cfg = args_eval.get("experiment", {})
    data_cfg = exp_cfg.get("data", {})
    opt_cfg = exp_cfg.get("optimization", {})
    
    # Paths & Params
    folder = args_eval.get("folder", "")
    os.makedirs(folder, exist_ok=True)
    
    num_classes = data_cfg.get("num_classes")
    batch_size = opt_cfg.get("batch_size")
    num_epochs = opt_cfg.get("num_epochs")
    use_bf16 = opt_cfg.get("use_bfloat16", False)
    
    # Device Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()
    logger.info(f"Rank {rank}/{world_size} initialized.")

    # Logger
    if rank == 0:
        wandb.init(
            project="action-classification", # 你可以修改项目名称
            name=args_eval.get("tag", "default_run"),
            config=args_eval,
            resume="allow"
        )
    csv_logger = CSVLogger(os.path.join(folder, f"log_r{rank}.csv"), 
                          ("%d", "epoch"), ("%.5f", "train_acc"), ("%.5f", "val_acc")) if rank == 0 else None

    # 2. Model Initialization (Classifier Only)
    # Note: Encoder is frozen and not loaded here based on original logic (features passed as deltas)
    classifiers = []
    opt_configs = opt_cfg.get("multihead_kwargs", [])
    
    for _ in opt_configs:
        clf = FeatureEvaluator(
            input_dim=exp_cfg.get("classifier", {}).get("dim", 1024), 
            num_heads=exp_cfg.get("classifier", {}).get("num_heads", 2),
            depth=exp_cfg.get("classifier", {}).get("num_probe_blocks", 1),
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        classifiers.append(DistributedDataParallel(clf, static_graph=True))
    print_model_params(clf)
    # 3. Data Loaders
    common_loader_args = dict(
        dataset_type=data_cfg.get("dataset_type", "VideoDataset"),
        img_size=data_cfg.get("resolution", 224),
        frames_per_clip=data_cfg.get("frames_per_clip", 16),
        frame_step=data_cfg.get("frame_step", 4),
        num_segments=data_cfg.get("num_segments", 1),
        eval_duration=data_cfg.get("clip_duration"),
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=args_eval.get("num_workers", 8),
        normalization=data_cfg.get("normalization", ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
        persistent_workers=True
    )

    train_loader, train_sampler = make_dataloader(
        root_path=[data_cfg.get("dataset_train")], training=True, 
        num_views_per_segment=1, **common_loader_args
    )
    
    val_loader, _ = make_dataloader(
        root_path=[data_cfg.get("dataset_val")], training=False,
        num_views_per_segment=data_cfg.get("num_views_per_segment", 1), **common_loader_args
    )
    
    ipe = len(train_loader) # Iterations per epoch

    dataset_mapping = train_loader.dataset.action_to_id 
    # 将 ID 排序，生成对应的 class_names 列表
    # 例如：{ "run": 1, "walk": 0 } -> ["walk", "run"]
    class_names = [k for k, v in sorted(dataset_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)

    # 4. Optimizers & Schedulers
    optimizers, scalers, lr_schedulers, wd_schedulers = [], [], [], []
    
    for clf, kwargs in zip(classifiers, opt_configs):
        p_groups = [{
            "params": clf.parameters(),
            "mc_warmup_steps": int(kwargs.get("warmup") * ipe),
            "mc_start_lr": kwargs.get("start_lr"),
            "mc_ref_lr": kwargs.get("lr"),
            "mc_final_lr": kwargs.get("final_lr"),
            "mc_ref_wd": kwargs.get("weight_decay"),
            "mc_final_wd": kwargs.get("final_weight_decay"),
        }]
        opt = torch.optim.AdamW(p_groups)
        optimizers.append(opt)
        scalers.append(torch.cuda.amp.GradScaler() if use_bf16 else None)
        lr_schedulers.append(WarmupCosineLRSchedule(opt, T_max=int(num_epochs * ipe)))
        wd_schedulers.append(CosineWDSchedule(opt, T_max=int(num_epochs * ipe)))

    # 5. Resume Checkpoint
    start_epoch = 0
    ckpt_path = os.path.join(folder, "latest.pt")
    if (args_eval.get("resume_checkpoint") or resume_preempt) and os.path.exists(ckpt_path):
        ckpt = robust_checkpoint_loader(ckpt_path, map_location="cpu")
        start_epoch = ckpt["epoch"]
        for c, sd in zip(classifiers, ckpt["classifiers"]): c.load_state_dict(sd)
        for o, sd in zip(optimizers, ckpt["opt"]): o.load_state_dict(sd)
        if ckpt["scaler"]: 
            for s, sd in zip(scalers, ckpt["scaler"]): s.load_state_dict(sd)
        
        # Fast-forward schedulers
        for _ in range(start_epoch * ipe):
            for s in lr_schedulers + wd_schedulers: s.step()
        logger.info(f"Resumed from epoch {start_epoch}")

    val_only = args_eval.get("val_only", False)

    # num_epochs = 101
    # 6. Training Loop
    for epoch in range(start_epoch, num_epochs): 
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_sampler.set_epoch(epoch)
        clear_memory()

        if val_only:
            train_acc = -1
            train_loss = -1
            best_train_idx = 0
            best_train_lr = 0
        else:
            # Train
            train_acc, train_loss, _, _, _, best_train_idx, best_train_lr = run_epoch(
                device, classifiers, train_loader, training=True,
                optimizer=optimizers, scaler=scalers, schedulers=[lr_schedulers, wd_schedulers],
                use_bfloat16=use_bf16
            )

        # Validate
        val_acc, val_loss, val_preds, val_labels, val_samples, best_clf_idx, best_val_lr = run_epoch(
            device, classifiers, val_loader, training=False,
            use_bfloat16=use_bf16, collect_preds=True
        )


        # Logging & Checkpointing
        logger.info(f"[{epoch+1}] Train Acc: {train_acc:.3f}% Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}% Loss: {val_loss:.4f}")

        # 1. 先初始化变量，防止后面引用报错
        all_preds, all_labels, all_samples = None, None, None

        # 2. 所有 Rank 必须一起参与数据收集 (Collective Communication)
        if world_size > 1:
            all_preds, all_labels, all_samples = gather_distributed_data(
                val_preds, val_labels, val_samples, world_size, rank, device
            )
        else:
            # 单卡情况
            all_preds, all_labels, all_samples = val_preds, val_labels, val_samples

        # 3. 只有 Rank 0 负责写文件和保存
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)

            
            wandb.log({
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "best_classifier_index": best_train_idx,
                "best_classifier_lr": best_train_lr,  
            })

            # Save Checkpoint
            torch.save({
                "classifiers": [c.state_dict() for c in classifiers],
                "opt": [o.state_dict() for o in optimizers],
                "scaler": [s.state_dict() for s in scalers] if scalers[0] else None,
                "epoch": epoch + 1,
            }, ckpt_path)

            # Save Metrics (此时 all_preds 在 Rank 0 上已经有数据了)
            if all_preds:
                logger.info(f"Saving metrics using best classifier index: {best_clf_idx}")
                save_evaluation_metrics(all_preds, all_labels, all_samples, num_classes, folder, class_names)

        if world_size > 1:
            dist.barrier()

        if val_only:
            if rank == 0: wandb.finish() # <--- 修改：结束 wandb
            return

    if rank == 0: wandb.finish() # <--- 修改：结束 wandb

