# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.nn.functional as F


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, gram_mode=False, training_mode=False):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list] List of Tensors (index: masks for given seq length)
        """
        if masks is None:
            outputs = []
            for x_fpc in x:
                if gram_mode:
                    # First we make the image bigger
                    B, C, T, H, W = x_fpc.shape
                    x_2d = x_fpc.permute(0, 2, 1, 3, 4).reshape(
                        B * T, C, H, W
                    )  # (B*T, C, H, W)
                    x_up = F.interpolate(
                        x_2d, scale_factor=2, mode="bicubic", align_corners=False
                    )
                    _, _, H_up, W_up = x_up.shape
                    x_up = x_up.view(B, T, C, H_up, W_up).permute(
                        0, 2, 1, 3, 4
                    )  # (B,C,T,H_up,W_up)

                    # Then we make the pass through the backbone
                    out = self.backbone(x_up)
                    B, N, D = out.shape
                    H_up_patches = W_up_patches = int(
                        H_up // 16
                    )  # We have hardcoded this to the patch size
                    if T == 1:
                        T_up_patches = 1  # In this case, it is a LVD image
                    else:
                        T_up_patches = int(
                            T // 2
                        )  # We have hardcoded this to the tubelet size
                    out_3d = out.view(
                        B, T_up_patches, H_up_patches, W_up_patches, D
                    )  # (bs, T, H, W, D)
                    out_3d = out_3d.permute(0, 4, 1, 2, 3)  # (bs, D, T, H, W)

                    # Downscale to original 2D size
                    out_2d = out_3d.permute(0, 2, 1, 3, 4).reshape(
                        B * T_up_patches, D, H_up_patches, W_up_patches
                    )  # (B*T, C, H_up, W_up)
                    out = F.interpolate(
                        out_2d,
                        size=(int(H_up_patches // 2), int(W_up_patches // 2)),
                        mode="bicubic",
                        align_corners=False,
                    )
                    out = out.view(
                        B,
                        T_up_patches,
                        D,
                        int(H_up_patches // 2),
                        int(W_up_patches // 2),
                    ).permute(
                        0, 2, 1, 3, 4
                    )  # (B,C,T,H,W)
                    out = out.permute(0, 2, 3, 4, 1).reshape(
                        B,
                        T_up_patches * int(H_up_patches // 2) * int(W_up_patches // 2),
                        D,
                    )  # (B,C,T,H,W) -> (B, T, H, W, C) -> (B, T * H * W, C)
                    outputs.append(out)
                else:
                    outputs.append(self.backbone(x_fpc, training=training_mode))
            return outputs

        outs = [[] for _ in x]
        for i, (x_fpc, m_fpc) in enumerate(zip(x, masks)):
            for m in m_fpc:
                outs[i] += [self.backbone(x_fpc, masks=m, training=training_mode)]
        return outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, mod="video"):
        """
        :param x: [list] List of encoder outputs for different seq lengths
        :param masks_x: [list] List of encoder masks
        :param masks_y: [list] List of predictor masks
        """
        n = 0
        outs_pred = [[] for _ in x]
        outs_context = [[] for _ in x]
        for i, (x_fpc, mx_fpc, my_fpc) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mx, my in zip(x_fpc, mx_fpc, my_fpc):
                x_pred, x_context = self.backbone(xij, mx, my, mask_index=i, mod=mod)
                outs_pred[i] += [x_pred]
                outs_context[i] += [x_context]
                n += 1
        return outs_pred, outs_context
