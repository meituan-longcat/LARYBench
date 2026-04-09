# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint
import gc

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def clear_memory():
    """清理显存和内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_memory_usage(prefix=""):
    """打印显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"{prefix} GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# def gather_predictions_across_gpus(predictions, labels, samples, world_size, rank, device):
#     """
#     在多卡训练中聚合所有GPU的预测结果和标签
    
#     Args:
#         predictions: 当前GPU的预测结果列表
#         labels: 当前GPU的标签列表
#         world_size: GPU总数
#         rank: 当前GPU的rank
#         device: 设备
    
#     Returns:
#         tuple: (聚合后的预测结果, 聚合后的标签) 仅在rank 0上返回有效数据
#     """
#     import torch.distributed as dist
    
#     # 将预测结果和标签转换为tensor
#     pred_tensor = torch.tensor(predictions, dtype=torch.long, device=device)
#     label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
#     # 获取每个GPU上的数据长度
#     local_pred_size = torch.tensor([len(predictions)], dtype=torch.long, device=device)
#     local_label_size = torch.tensor([len(labels)], dtype=torch.long, device=device)
    
#     # 收集所有GPU的数据长度
#     all_pred_sizes = [torch.zeros_like(local_pred_size) for _ in range(world_size)]
#     all_label_sizes = [torch.zeros_like(local_label_size) for _ in range(world_size)]
    
#     dist.all_gather(all_pred_sizes, local_pred_size)
#     dist.all_gather(all_label_sizes, local_label_size)
    
#     # 转换为Python列表
#     pred_sizes = [int(size.item()) for size in all_pred_sizes]
#     label_sizes = [int(size.item()) for size in all_label_sizes]
    
#     # 准备接收缓冲区
#     max_pred_size = max(pred_sizes)
#     max_label_size = max(label_sizes)
    
#     # 填充到相同长度（用-1填充，后续会过滤掉）
#     padded_pred_tensor = torch.full((max_pred_size,), -1, dtype=torch.long, device=device)
#     padded_label_tensor = torch.full((max_label_size,), -1, dtype=torch.long, device=device)
    
#     padded_pred_tensor[:len(predictions)] = pred_tensor
#     padded_label_tensor[:len(labels)] = label_tensor
    
#     # 收集所有GPU的数据
#     all_preds_padded = [torch.zeros_like(padded_pred_tensor) for _ in range(world_size)]
#     all_labels_padded = [torch.zeros_like(padded_label_tensor) for _ in range(world_size)]
    
#     dist.all_gather(all_preds_padded, padded_pred_tensor)
#     dist.all_gather(all_labels_padded, padded_label_tensor)
    
#     if rank == 0:
#         # 只在rank 0上处理聚合后的数据
#         gathered_preds = []
#         gathered_labels = []
        
#         for i in range(world_size):
#             # 去除填充的-1值
#             valid_preds = all_preds_padded[i][:pred_sizes[i]]
#             valid_labels = all_labels_padded[i][:label_sizes[i]]
            
#             gathered_preds.extend(valid_preds.cpu().numpy().tolist())
#             gathered_labels.extend(valid_labels.cpu().numpy().tolist())
        
#         return gathered_preds, gathered_labels
#     else:
#         return None, None
def gather_predictions_across_gpus(predictions, labels, samples, world_size, rank, device):
    """
    在多卡训练中聚合所有GPU的预测结果、标签和样本路径
    
    Args:
        predictions: 当前GPU的预测结果列表
        labels: 当前GPU的标签列表
        samples: 当前GPU的样本路径列表
        world_size: GPU总数
        rank: 当前GPU的rank
        device: 设备
    
    Returns:
        tuple: (聚合后的预测结果, 聚合后的标签, 聚合后的样本路径) 仅在rank 0上返回有效数据
    """
    import torch.distributed as dist
    import torch
    
    # 将预测结果和标签转换为tensor
    pred_tensor = torch.tensor(predictions, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # 获取每个GPU上的数据长度
    local_pred_size = torch.tensor([len(predictions)], dtype=torch.long, device=device)
    local_label_size = torch.tensor([len(labels)], dtype=torch.long, device=device)
    
    # 收集所有GPU的数据长度
    all_pred_sizes = [torch.zeros_like(local_pred_size) for _ in range(world_size)]
    all_label_sizes = [torch.zeros_like(local_label_size) for _ in range(world_size)]
    
    dist.all_gather(all_pred_sizes, local_pred_size)
    dist.all_gather(all_label_sizes, local_label_size)
    
    # 转换为Python列表
    pred_sizes = [int(size.item()) for size in all_pred_sizes]
    label_sizes = [int(size.item()) for size in all_label_sizes]

    # 准备接收缓冲区
    max_pred_size = max(pred_sizes)
    max_label_size = max(label_sizes)

    # 填充到相同长度（用-1填充，后续会过滤掉）
    padded_pred_tensor = torch.full((max_pred_size,), -1, dtype=torch.long, device=device)
    padded_label_tensor = torch.full((max_label_size,), -1, dtype=torch.long, device=device)

    padded_pred_tensor[:len(predictions)] = pred_tensor
    padded_label_tensor[:len(labels)] = label_tensor
    
    # 收集所有GPU的数据
    all_preds_padded = [torch.zeros_like(padded_pred_tensor) for _ in range(world_size)]
    all_labels_padded = [torch.zeros_like(padded_label_tensor) for _ in range(world_size)]
    
    dist.all_gather(all_preds_padded, padded_pred_tensor)
    dist.all_gather(all_labels_padded, padded_label_tensor)
    
    # 收集所有样本路径字符串列表
    all_samples = [[] for _ in range(world_size)]
    dist.all_gather_object(all_samples, samples)
    
    if rank == 0:
        # 只在rank 0上处理聚合后的数据
        gathered_preds = []
        gathered_labels = []
        gathered_samples = []
        
        for i in range(world_size):
            # 去除填充的-1值
            valid_preds = all_preds_padded[i][:pred_sizes[i]]
            valid_labels = all_labels_padded[i][:label_sizes[i]]
            
            gathered_preds.extend(valid_preds.cpu().numpy().tolist())
            gathered_labels.extend(valid_labels.cpu().numpy().tolist())
            gathered_samples.extend(all_samples[i])  # 直接扩展无须解码
        
        return gathered_preds, gathered_labels, gathered_samples
    else:
        return None, None, None


def save_and_visualize_confusion_matrix(all_preds, all_labels, num_classes, save_dir, class_names=None):
    """
    计算、保存和可视化混淆矩阵（行：预测标签，列：真实标签）

    Args:
        all_preds: 所有预测结果的列表
        all_labels: 所有真实标签的列表
        num_classes: 类别数量
        save_dir: 保存目录
        epoch: 当前epoch（可选）
        class_names: 类别名称列表，如果为None则使用序号（可选）
    """
    import os
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵（行：预测标签，列：真实标签）
    cm = confusion_matrix(all_preds, all_labels, labels=range(num_classes))

    # 处理类别名称
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    elif len(class_names) != num_classes:
        logger.warning(f"类别名称数量({len(class_names)})与类别数量({num_classes})不匹配，使用序号")
        class_names = [str(i) for i in range(num_classes)]

    # 保存混淆矩阵数据
    cm_file = os.path.join(save_dir, f"confusion_matrix.json")

    # 将numpy数组转换为列表以便JSON序列化
    cm_data = {
        "confusion_matrix": cm.tolist(),
        "num_classes": num_classes,
        "total_samples": len(all_labels),
        "class_names": class_names
    }

    with open(cm_file, 'w') as f:
        json.dump(cm_data, f, indent=2)

    logger.info(f"Confusion matrix saved to {cm_file}")

    # 可视化混淆矩阵（百分比）
    plt.figure(figsize=(12, 10))

    # 计算百分比矩阵用于显示（按预测标签归一化）
    # 对于每个预测类别，显示其中有多少比例实际属于各个真实类别
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # 创建热力图
    sns.heatmap(cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=class_names,    # 真实标签
                yticklabels=class_names,    # 预测标签
                cbar_kws={'label': 'Percentage (%)'})

    plt.title(f'Confusion Matrix', fontsize=16)
    plt.xlabel('True Label', fontsize=14)     # x轴为真实标签
    plt.ylabel('Predicted Label', fontsize=14) # y轴为预测标签
    plt.tight_layout()

    # 保存可视化图片
    cm_img_file = os.path.join(save_dir, f"confusion_matrix.png")
    plt.savefig(cm_img_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix visualization saved to {cm_img_file}")

    # 同时保存原始计数矩阵的可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix\n(Counts)', fontsize=16)
    plt.xlabel('True Label', fontsize=14)
    plt.ylabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    # 保存计数矩阵图片
    cm_count_img_file = os.path.join(save_dir, f"confusion_matrix_counts.png")
    plt.savefig(cm_count_img_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix counts visualization saved to {cm_count_img_file}")


    # 计算每个类别的精确率、召回率和F1分数（注意axis变化）
    # 交换后：precision = TP / (TP + FP) = cm[i, i] / sum_row_i
    #        recall    = TP / (TP + FN) = cm[i, i] / sum_col_i
    precision = np.diag(cm) / np.sum(cm, axis=1)  # 行归一化
    recall = np.diag(cm) / np.sum(cm, axis=0)     # 列归一化
    f1 = 2 * (precision * recall) / (precision + recall)

    # 处理除零情况
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    # 保存详细统计信息
    stats = {
        "per_class_stats": []
    }

    for i in range(num_classes):
        stats["per_class_stats"].append({
            "class": i,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(np.sum(cm, axis=1)[i])  # 支持度为预测为该类别的样本数
        })

    stats_file = os.path.join(save_dir, f"classification_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Classification statistics saved to {stats_file}")

    # 打印每类精确率（与混淆矩阵对角线百分比一致）
    logger.info("Per-class precision (from confusion matrix):")
    for i in range(num_classes):
        logger.info(f"Class {i}: {precision[i]*100:.2f}%")

    return cm


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 4)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 2)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    class_names = args_data.get("class_names", None)  # 添加类别名称支持
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "train_acc"), ("%.5f", "acc"))

    # Initialize model

    # -- init models
    encoder = None
    # encoder = init_module(
    #     module_name=module_name,
    #     frames_per_clip=frames_per_clip,
    #     resolution=resolution,
    #     checkpoint=checkpoint,
    #     model_kwargs=args_model,
    #     wrapper_kwargs=args_wrapper,
    #     device=device,
    # )
    # -- init classifier
    classifiers = [
        AttentiveClassifier(
            embed_dim=1024,  # ****
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        persistent_workers=True
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,  
        normalization=normalization,
        persistent_workers=True
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)

        # 每个epoch开始前清理显存
        clear_memory()
        if val_only:
            train_acc = -1.0
        else:
            train_acc, _, _, _, _ = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                num_classes=num_classes,
            )

        val_acc, val_preds, val_labels, best_classifier_idx, val_samples = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            num_classes=num_classes,
            collect_predictions=True,
        )

        # 在验证后生成混淆矩阵
        if val_preds is not None and val_labels is not None:
            # 多卡情况下需要聚合所有GPU的预测结果
            if world_size > 1:
                # 将预测结果和标签转换为tensor并聚合
                all_preds_gathered, all_labels_gathered, all_samples_gathered = gather_predictions_across_gpus(
                    val_preds, val_labels, val_samples, world_size, rank, device
                )

                # 只在rank 0上生成混淆矩阵
                if rank == 0:
                    logger.info(f"Using predictions from classifier {best_classifier_idx} (best performing) for confusion matrix")
                    logger.info(f"Gathered {len(all_preds_gathered)} predictions from {world_size} GPUs")
                    save_and_visualize_confusion_matrix(
                        all_preds=all_preds_gathered,
                        all_labels=all_labels_gathered,
                        num_classes=num_classes,
                        save_dir=folder,
                        class_names=class_names
                    )

                    # 保存预测结果到文件
                    mapping = {0: 'grab', 1: 'place', 2: 'insert', 3: 'withdraw', 4: 'unscrew', 5: 'screw'}
                    with open(os.path.join(folder, "case.txt"), "w") as f:
                        for sample_path, pred_class, true_class in zip(all_samples_gathered, all_preds_gathered, all_labels_gathered):
                            f.write(f"{os.path.basename(sample_path)}, {mapping[pred_class]}, {mapping[true_class]}\n")

            else:
                # 单卡情况
                logger.info(f"Using predictions from classifier {best_classifier_idx} (best performing) for confusion matrix")
                save_and_visualize_confusion_matrix(
                    all_preds=val_preds,
                    all_labels=val_labels,
                    num_classes=num_classes,
                    save_dir=folder,
                    class_names=class_names
                )
                with open(os.path.join(folder, "case.txt"), "w") as f:
                    for sample_path, pred_class, true_class in zip(all_samples_gathered, all_preds_gathered, all_labels_gathered):
                        f.write(f"{os.path.basename(sample_path)}, {mapping[pred_class]}, {mapping[true_class]}\n")


        logger.info("[%5d] train: %.3f%% test: %.3f%%" % (epoch + 1, train_acc, val_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)

        if val_only:
            return

        save_checkpoint(epoch + 1)

def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_classes,
    collect_predictions=False,
):
    # 获取全局rank变量
    world_size, rank = init_distributed()

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    top1_meters = [AverageMeter() for _ in classifiers]

    # 收集所有分类器的预测和标签用于混淆矩阵
    all_predictions_per_classifier = [[] for _ in classifiers]
    all_labels = []
    all_samples = []  # 新增保存样本路径

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            labels = data[0].to(device, non_blocking=True)
            deltas = data[1].to(device, non_blocking=True)
            samples = data[2]
            batch_size = len(labels)

            # Forward and prediction
            if training:
                outputs = [[c(deltas)] for c in classifiers]
            else:
                with torch.no_grad():
                    outputs = [[c(deltas)] for c in classifiers]

        # Compute loss
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]

        # 计算准确率
        with torch.no_grad():
            outputs_softmax = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
            top1_accs = [100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size for coutputs in outputs_softmax]
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs):
                t1m.update(t1a)

            # 收集混淆矩阵数据（仅验证时）
            if not training and collect_predictions:
                # 收集所有分类器的预测结果
                for idx, coutputs in enumerate(outputs_softmax):
                    preds = coutputs.max(dim=1).indices
                    all_predictions_per_classifier[idx].extend(preds.cpu().numpy().tolist())

                # 标签只需要收集一次
                if len(all_labels) == 0 or len(all_labels) < (itr + 1) * batch_size:
                    all_labels.extend(labels.cpu().numpy().tolist())

                    all_samples.extend(samples)  # 保存样本文件路径

        if training:
            if use_bfloat16:
                [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [[lij.backward() for lij in li] for li in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]

        # 删除不需要的变量以释放显存
        del labels, deltas, outputs, losses
        if 'outputs_softmax' in locals():
            del outputs_softmax

        # 定期清理显存
        if itr % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        if itr % 10 == 0:
            logger.info(
                "[%5d] %.3f%% [%.3f%% %.3f%%] [mem: %.2e]"
                % (
                    itr,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    # 每类精确率将从混淆矩阵中计算，不再在这里统计

    # 最终清理显存
    clear_memory()

    # 返回预测和标签（如果需要收集）
    if collect_predictions and not training:
        # 找到最佳分类器的索引
        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        best_classifier_idx = np.argmax(_agg_top1)
        best_predictions = all_predictions_per_classifier[best_classifier_idx]
        return _agg_top1.max(), best_predictions, all_labels, best_classifier_idx, all_samples
    else:
        return _agg_top1.max(), None, None, None, None


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


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


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
