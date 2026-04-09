# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn

from src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from src.utils.tensors import trunc_normal_


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


class MultiAttributeAttentiveClassifier(nn.Module):
    """Multi-Attribute Attentive Classifier

    A classifier that predicts multiple attributes simultaneously.
    Each attribute can have multiple classes.
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
        num_attributes=4,
        num_classes_per_attribute=3,
        complete_block=True,
        use_activation_checkpointing=False,
        shared_features=True,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio for transformer blocks
            depth: Depth of the pooler
            norm_layer: Normalization layer
            init_std: Initialization standard deviation
            qkv_bias: Whether to use bias in QKV projections
            num_attributes: Number of attributes to predict
            num_classes_per_attribute: Number of classes for each attribute
            complete_block: Whether to use complete cross-attention block
            use_activation_checkpointing: Whether to use activation checkpointing
            shared_features: Whether to share feature extraction across attributes
        """
        super().__init__()
        self.num_attributes = num_attributes
        self.num_classes_per_attribute = num_classes_per_attribute
        self.shared_features = shared_features

        if shared_features:
            # Shared feature extractor for all attributes
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
            # Separate classifier heads for each attribute
            self.attribute_heads = nn.ModuleList([
                nn.Linear(embed_dim, num_classes_per_attribute, bias=True)
                for _ in range(num_attributes)
            ])
        else:
            # Separate pooler and classifier for each attribute
            self.poolers = nn.ModuleList([
                AttentivePooler(
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
                for _ in range(num_attributes)
            ])
            self.attribute_heads = nn.ModuleList([
                nn.Linear(embed_dim, num_classes_per_attribute, bias=True)
                for _ in range(num_attributes)
            ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Dictionary with predictions for each attribute:
            {
                'attribute_0': tensor of shape (batch_size, num_classes_per_attribute),
                'attribute_1': tensor of shape (batch_size, num_classes_per_attribute),
                ...
            }
        """
        attribute_names = ['x_translation', 'y_translation', 'z_translation', 'gripper']
        if self.shared_features:
            # Extract shared features
            features = self.pooler(x).squeeze(1)  # (batch_size, embed_dim)

            # Predict each attribute using shared features
            predictions = {}
            for i, head in enumerate(self.attribute_heads):
                predictions[attribute_names[i]] = head(features)
        else:
            # Extract features and predict each attribute separately
            predictions = {}
            for i, (pooler, head) in enumerate(zip(self.poolers, self.attribute_heads)):
                features = pooler(x).squeeze(1)  # (batch_size, embed_dim)
                predictions[attribute_names[i]] = head(features)

        return predictions

    def get_attribute_predictions(self, x, attribute_names=None):
        """
        Get predictions with custom attribute names

        Args:
            x: Input tensor
            attribute_names: List of attribute names (optional)

        Returns:
            Dictionary with named predictions
        """
        predictions = self.forward(x)

        if attribute_names is not None:
            assert len(attribute_names) == self.num_attributes, \
                f"Expected {self.num_attributes} attribute names, got {len(attribute_names)}"

            named_predictions = {}
            for i, name in enumerate(attribute_names):
                named_predictions[name] = predictions[f'attribute_{i}']
            return named_predictions

        return predictions

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)"""
    
    def __init__(self, dim, max_seq_len=32, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转矩阵
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """预计算并缓存旋转矩阵"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :], persistent=False)
    
    def rotate_half(self, x):
        """旋转输入张量的一半维度"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, seq_len):
        """应用旋转位置编码"""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, num_tokens, dim)
        Returns:
            x with rotary positional encoding applied
        """
        seq_len = x.shape[1]
        return self.apply_rotary_pos_emb(x, seq_len)
    
class TemporalAttentivePooler(nn.Module):
    """Attentive Pooler with Temporal Support"""

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
        max_temporal_len=32,  # 最大时序长度
        use_rope=True,  # 是否使用RoPE
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_rope = use_rope
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        # RoPE位置编码
        if use_rope:
            self.rope = RotaryPositionalEncoding(
                dim=embed_dim,
                max_seq_len=max_temporal_len
            )

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias
            )

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
        """
        Args:
            x: shape (B, T, N, D) or (B, N, D)
               B: batch size
               T: temporal length (optional)
               N: number of tokens
               D: embedding dimension
        Returns:
            Pooled features with shape (B, T, num_queries, D) or (B, num_queries, D)
        """
        has_temporal = x.dim() == 4
        
        if has_temporal:
            B, T, N, D = x.shape
            
            # 应用RoPE位置编码
            if self.use_rope:
                x = self.rope(x)  # (B, T, N, D)
            
            # 重塑为 (B*T, N, D) 以便处理
            x = x.reshape(B * T, N, D)
        else:
            B, N, D = x.shape
            T = 1

        # 通过self-attention blocks
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        blk, x, False, None, use_reentrant=False
                    )
                else:
                    x = blk(x)

        # Cross attention pooling
        q = self.query_tokens.repeat(B * T, 1, 1)
        q = self.cross_attention_block(q, x)  # (B*T, num_queries, D)

        # 恢复时序维度
        if has_temporal:
            q = q.reshape(B, T, -1, D)  # (B, T, num_queries, D)
        
        return q

class TemporalAttentiveClassifier(nn.Module):
    """Attentive Classifier with Temporal Support"""

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
        max_temporal_len=32,
        use_rope=True,
        temporal_pooling='mean',  # 'mean', 'max', 'last', 'attention'
    ):
        super().__init__()
        self.temporal_pooling = temporal_pooling
        
        self.pooler = TemporalAttentivePooler(
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
            max_temporal_len=max_temporal_len,
            use_rope=use_rope,
        )
        
        # 时序聚合attention（可选）
        if temporal_pooling == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.temporal_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.temporal_query, std=init_std)
        
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        """
        Args:
            x: shape (B, T, N, D) or (B, N, D)
        Returns:
            logits: shape (B, num_classes)
        """
        x = self.pooler(x)  
        
        if x.dim() == 4:  # Has temporal dimension
            B, T, _, D = x.shape
            x = x.squeeze(2)  # (B, T, D)
            
            # 时序聚合
            if self.temporal_pooling == 'mean':
                x = x.mean(dim=1)  # (B, D)
            elif self.temporal_pooling == 'max':
                x = x.max(dim=1)[0]  # (B, D)
            elif self.temporal_pooling == 'last':
                x = x[:, -1]  # (B, D)
            elif self.temporal_pooling == 'attention':
                q = self.temporal_query.repeat(B, 1, 1)  # (B, 1, D)
                x, _ = self.temporal_attention(q, x, x)  # (B, 1, D)
                x = x.squeeze(1)  # (B, D)
        else:
            x = x.squeeze(1)  # (B, D)
        
        x = self.linear(x)
        return x