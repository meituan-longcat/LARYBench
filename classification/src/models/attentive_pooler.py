# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from classification.src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from classification.src.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


class TemporalAttentiveClassifier(nn.Module):
    """
    核心分类器：
    1. 负责处理时间维度 (Temporal Embedding)。
    2. 包含 Pooler 和 最终分类 Linear。
    注意：此类输入的 embed_dim 是固定的（模型内部维度）。
    """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
        max_temporal_len=128, # 支持的最大帧数，超过则插值
    ):
        super().__init__()
        
        # 1. Pooler 模块
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        
        # 2. 分类头
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

        # 3. 时序位置编码 (Learnable)
        # Shape: (1, MaxT, 1, D) -> 方便广播到 (B, T, N, D)
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_temporal_len, 1, embed_dim))
        trunc_normal_(self.temporal_embed, std=init_std)

    def forward(self, x):
        # x: (B, N, D) or (B, T, N, D)
        
        # 兼容性处理：如果是 4D 输入 (视频/时序)
        if x.dim() == 4:
            B, T, N, D = x.shape
            
            # --- 时间编码逻辑 Start ---
            if T <= self.temporal_embed.shape[1]:
                # 情况 A: 输入帧数 <= 预设最大帧数，直接切片
                # self.temporal_embed[:, :T, :, :] 会广播到 (B, T, N, D)
                x = x + self.temporal_embed[:, :T, :, :]
            else:
                # 情况 B: 输入帧数 > 预设最大帧数，进行插值 (Interpolation)
                # permute 为 (1, D, MaxT, 1) 以适配 interpolate
                t_emb = self.temporal_embed.permute(0, 3, 1, 2) 
                # 插值到 (T, 1)
                t_emb = F.interpolate(t_emb, size=(T, 1), mode='bilinear', align_corners=False)
                # permute 回 (1, T, 1, D)
                t_emb = t_emb.permute(0, 2, 3, 1)
                x = x + t_emb
            # --- 时间编码逻辑 End ---

            # 融合 T 和 N 维度: (B, T, N, D) -> (B, T*N, D)
            x = x.flatten(1, 2)

        # 此时 x 必定是 (B, L, D)，无论是纯空间还是时空混合
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x


class FeatureEvaluator(nn.Module):
    """
    评测专用封装器 (Wrapper)：
    1. 负责将不同特征维度的输入 (input_dim) 映射到统一的模型维度 (model_dim)。
    2. 保证公平对比：除了 Projector 不同，后端的 AttentiveClassifier 是完全一样的。
    """
    def __init__(
        self,
        input_dim,          # 输入特征的维度 (例如 ResNet=2048, ViT=768)
        model_dim=768,      # 统一的内部维度 (你自己设定，例如 512 或 768)
        num_classes=1000,   # 类别数
        # 以下参数用于构建 Classifier，保持默认或按需调整
        num_heads=12,
        depth=1,
        max_temporal_len=128,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()

        # --- Projector (Adapter) ---
        # 作用：公平的“翻译器”，把不同来源的特征映射到相同的维度空间
        self.projector = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim), # 归一化，至关重要，消除不同特征幅度的差异
            nn.GELU()                # 激活函数，增加非线性适配能力
        )

        # --- Shared Core Classifier ---
        # 作用：统一的评测标准
        self.classifier = TemporalAttentiveClassifier(
            embed_dim=model_dim,      # 这里必须用统一的 model_dim
            num_heads=num_heads,
            depth=depth,
            num_classes=num_classes,
            max_temporal_len=max_temporal_len,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
            # 其他参数使用默认值即可，或者透传
        )

        self._init_projector()

    def _init_projector(self):
        # 专门初始化 Projector，Classifier 已经在其内部初始化了
        for m in self.projector:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input x: 
        #   (B, N, input_dim) -> 图像特征
        #   (B, T, N, input_dim) -> 视频特征
        
        # Step 1: 维度对齐
        # Linear 层只作用在最后一个维度，所以不关心前面是 N 还是 T, N
        x = self.projector(x) 
        
        # Step 2: 统一分类
        # 此时 x 的最后一维已经是 model_dim
        logits = self.classifier(x)
        
        return logits