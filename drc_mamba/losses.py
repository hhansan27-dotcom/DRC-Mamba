"""Loss functions for DRC-Mamba.

The primary segmentation objective is Soft-IoU + binary focal loss. Auxiliary
response, boundary, and candidate-retention heads are supervised only during training.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_list(value):
    if value is None:
        return None
    return list(value) if isinstance(value, (list, tuple)) else [value]


class DRCMambaLoss(nn.Module):
    """
    Training objective used by DRC-Mamba.

    Main:   SoftIoU(logits) + Focal(logits)
    Aux 1:  Edge loss (optional): BCEWithLogits or BCE (auto-detect)
    Aux 2:  Boundary-band loss (optional): same as edge, usually uses ring-band GT

    Notes:
      - preds: segmentation logits (B,1,H,W)
      - targets: mask in {0,1} float (B,1,H,W)
      - edge_preds/bnd_preds may be logits OR probabilities in [0,1]
        If *_from_logits is None -> auto-detect by range.
      - You may warm-up by updating self.edge_weight / self.bnd_weight each epoch.
    """

    def __init__(self,
                 iou_weight=1.0,
                 focal_weight=1.0,
                 edge_weight=1.0,
                 bnd_weight=0.1,
                 bnd_from_logits=True,
                 bnd_bg_weight=0.0,
                 bnd_bg_dilate_ks=9,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 outer_weight=0.0,
                 inner_weight=0.0,
                 tail_weight=0.0,
                 outer_r=1,
                 inner_r=1,
                 far_r=7,
                 tail_k=64,
                 outer_margin: float = 0.0,
                 inner_margin: float = 0.0,
                 halo_weight: float = 0.0,
                 support_weight: float = 0.05,
                 center_weight: float = 0.03,
                 interference_weight: float = 0.03,
                 boundary_rejection_weight: float = 0.04,
                 confidence_weight: float = 0.02,
                 rejection_weight: float = 0.0,
                 risk_weight: float = 0.0,
                 radial_weight: float = 0.0
                 ):
        super().__init__()
        self.iou_weight = float(iou_weight)
        self.focal_weight = float(focal_weight)
        self.edge_weight = float(edge_weight)
        self.bnd_weight = float(bnd_weight)

        self.bnd_from_logits = bool(bnd_from_logits)
        self.bnd_bg_weight = float(bnd_bg_weight)
        self.bnd_bg_dilate_ks = int(bnd_bg_dilate_ks)
        if self.bnd_bg_dilate_ks < 1:
            self.bnd_bg_dilate_ks = 1
        self.bnd_bg_pad = self.bnd_bg_dilate_ks // 2

        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

        self.outer_weight = float(outer_weight)
        self.inner_weight = float(inner_weight)
        self.tail_weight = float(tail_weight)

        self.outer_r = int(outer_r)
        self.inner_r = int(inner_r)
        self.far_r = int(far_r)
        self.tail_k = int(tail_k)

        self.outer_margin = float(outer_margin)
        self.inner_margin = float(inner_margin)
        self.halo_weight = float(halo_weight)

        self.support_weight = float(support_weight)
        self.center_weight = float(center_weight)
        self.interference_weight = float(interference_weight)
        self.boundary_rejection_weight = float(boundary_rejection_weight)
        self.confidence_weight = float(confidence_weight)

        # Additional response-head weights used by the deferred-decision training objective.
        self.rejection_weight = float(rejection_weight)
        self.risk_weight = float(risk_weight)
        self.radial_weight = float(radial_weight)

        self.outer_r = max(0, self.outer_r)
        self.inner_r = max(0, self.inner_r)
        self.far_r = max(0, self.far_r)
        self.tail_k = max(1, self.tail_k)
    # -------------------------
    # Core losses
    # -------------------------
    def _soft_iou_loss(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target - pred * target).sum(dim=(2, 3))
        iou = (intersection + smooth) / (union + smooth)
        return 1.0 - iou.mean()

    def _focal_loss(self, pred_logits, target):
        # Binary focal loss on logits
        # FL = alpha * (1-pt)^gamma * BCE
        # where pt = sigmoid(logits) if y=1 else 1-sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        p = torch.sigmoid(pred_logits)
        pt = p * target + (1.0 - p) * (1.0 - target)
        alpha = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        modulation = (1.0 - pt).clamp_min(1e-6) ** self.focal_gamma
        w = alpha * modulation
        return (w * bce).mean()

    # -------------------------
    # Helpers for aux losses
    # -------------------------
    @staticmethod
    def _auto_is_prob(x: torch.Tensor) -> bool:
        # Heuristic: probabilities should lie in [0,1] with small tolerance
        if x.numel() == 0:
            return False
        x_min = float(x.min().detach().cpu())
        x_max = float(x.max().detach().cpu())
        return (x_min >= -1e-4) and (x_max <= 1.0 + 1e-4)

    @staticmethod
    def _dynamic_pos_weight(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # target: (B,1,H,W) with {0,1}
        with torch.no_grad():
            pos = target.sum()
            neg = target.numel() - pos
            w = (neg + eps) / (pos + eps)
            # clamp to avoid extreme instabilities
            w = torch.clamp(w, 1.0, 50.0)
        return w

    def _dynamic_bce(self, pred, target, from_logits: bool) -> torch.Tensor:
        pos_w = self._dynamic_pos_weight(target).to(pred.device)
        if from_logits:
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_w)
        else:
            pred = pred.clamp(1e-6, 1.0 - 1e-6)
            # Note: BCE doesn't support pos_weight; emulate with per-element weights
            # weight = pos_w for positive, 1 for negative
            w = torch.ones_like(target)
            w = w + (pos_w - 1.0) * target
            return F.binary_cross_entropy(pred, target, weight=w)

    @staticmethod
    def _dilate(mask01: torch.Tensor, r: int) -> torch.Tensor:
        """mask01: (B,1,H,W) float {0,1} -> dilated float {0,1}"""
        if r <= 0:
            return (mask01 > 0.5).float()
        ks = 2 * int(r) + 1
        pad = ks // 2
        return (F.max_pool2d(mask01, kernel_size=ks, stride=1, padding=pad) > 0.5).float()

    @staticmethod
    def _erode(mask01: torch.Tensor, r: int) -> torch.Tensor:
        """mask01: (B,1,H,W) float {0,1} -> eroded float {0,1}"""
        if r <= 0:
            return (mask01 > 0.5).float()
        ks = 2 * int(r) + 1
        pad = ks // 2
        return (-F.max_pool2d(-mask01, kernel_size=ks, stride=1, padding=pad) > 0.5).float()

    @staticmethod
    def _masked_bce_with_logits(pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute BCEWithLogits only on mask==1 region, normalized by mask area."""
        loss_map = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        denom = mask.sum().clamp_min(1.0)
        return (loss_map * mask).sum() / denom

    def _make_outer_inner_far_masks(self, targets01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        targets01: (B,1,H,W) float {0,1}
        outer = dilate(GT,outer_r) - GT
        inner = GT - erode(GT,inner_r)
        far   = 1 - dilate(GT,far_r)
        """
        gt = (targets01 > 0.5).float()

        dil_out = self._dilate(gt, self.outer_r)
        outer = dil_out * (1.0 - gt)

        ero_in = self._erode(gt, self.inner_r)
        inner = gt * (1.0 - ero_in)

        dil_far = self._dilate(gt, self.far_r)
        far = (1.0 - dil_far).clamp(0.0, 1.0)

        return outer, inner, far

    def _topk_tail_penalty(self, pred_logits: torch.Tensor, far_mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize top-k probabilities on far background.
        pred_logits: (B,1,H,W)
        far_mask:    (B,1,H,W) float {0,1}
        """
        p = torch.sigmoid(pred_logits)
        B = p.shape[0]
        losses = []
        for b in range(B):
            m = far_mask[b, 0] > 0.5
            if m.sum().item() <= 0:
                continue
            vals = p[b, 0][m]  # (N,)
            k = min(self.tail_k, vals.numel())
            topk = torch.topk(vals, k=k, largest=True).values
            losses.append(topk.mean())
        if len(losses) == 0:
            return pred_logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    # -------------------------
    # Forward
    # -------------------------
    def forward(self,
                preds, targets,
                edge_preds=None, edge_targets=None, edge_from_logits=None,
                bnd_preds=None, bnd_targets=None, bnd_from_logits=None,
                support_preds=None, support_targets=None,
                center_preds=None, center_targets=None,
                interference_preds=None, interference_targets=None,
                boundary_rejection_preds=None, boundary_rejection_targets=None,
                rejection_preds=None, rejection_targets=None,
                risk_preds=None, risk_targets=None,
                conf_preds=None, conf_targets=None,
                radial_preds=None, radial_targets=None,
                ring_tail: bool = True):
        """
        preds: logits (B,1,H,W)
        targets: mask in {0,1} float (B,1,H,W)

        edge_preds/bnd_preds: logits OR probabilities
        edge_targets/bnd_targets: {0,1} float
        """
        loss = 0.0

        # 1) main loss: IoU + Focal
        loss_iou = self._soft_iou_loss(preds, targets)
        loss_focal = self._focal_loss(preds, targets)
        loss = loss + self.iou_weight * loss_iou + self.focal_weight * loss_focal

        # 2) edge loss (optional)
        if edge_preds is not None and edge_targets is not None and self.edge_weight > 0:
            if edge_preds.shape[2:] != edge_targets.shape[2:]:
                edge_preds = F.interpolate(
                    edge_preds,
                    size=edge_targets.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if edge_from_logits is None:
                edge_from_logits = not self._auto_is_prob(edge_preds)
            loss_edge = self._dynamic_bce(edge_preds, edge_targets, bool(edge_from_logits))
            loss = loss + self.edge_weight * loss_edge

        # --- bnd (ring) supervision + background sparsity regularization ---
        bnd_preds_list = _as_list(bnd_preds)
        bnd_tgts_list = _as_list(bnd_targets)

        if (bnd_preds_list is not None) and (bnd_tgts_list is not None) and (self.bnd_weight > 0):
            # allow bnd_targets broadcast (single gt for multi-scale preds)
            if len(bnd_tgts_list) == 1 and len(bnd_preds_list) > 1:
                bnd_tgts_list = bnd_tgts_list * len(bnd_preds_list)

            bnd_losses = []
            bnd_bg_pens = []

            for bp, bt in zip(bnd_preds_list, bnd_tgts_list):
                # resize pred to target
                if bp.shape[2:] != bt.shape[2:]:
                    bp = F.interpolate(bp, size=bt.shape[2:], mode='bilinear', align_corners=False)

                # logits/prob handling (default: logits)
                use_logits = self.bnd_from_logits if (bnd_from_logits is None) else bool(bnd_from_logits)
                if (bnd_from_logits is None) and self._auto_is_prob(bp):
                    use_logits = False

                bp_logits = bp if use_logits else torch.logit(bp.clamp(1e-4, 1 - 1e-4))

                pos_w = self._dynamic_pos_weight(bt)
                bnd_losses.append(F.binary_cross_entropy_with_logits(bp_logits, bt, pos_weight=pos_w))

                # bnd background sparsity: penalize bnd activation outside dilated GT mask
                if self.bnd_bg_weight > 0:
                    # downsample main mask target to this scale
                    gt_ds = targets
                    if gt_ds.shape[2:] != bt.shape[2:]:
                        gt_ds = F.interpolate(gt_ds, size=bt.shape[2:], mode='nearest')

                    # dilate GT mask -> "allowed region" (avoid suppressing near target)
                    dil = F.max_pool2d(gt_ds, kernel_size=self.bnd_bg_dilate_ks, stride=1,
                                       padding=self.bnd_bg_dilate_ks // 2)
                    bg_mask = (1.0 - dil).clamp(0.0, 1.0)

                    bp_prob = torch.sigmoid(bp_logits)
                    bnd_bg_pens.append((bp_prob * bg_mask).mean())

            loss = loss + self.bnd_weight * (sum(bnd_losses) / (len(bnd_losses) + 1e-6))

            if (self.bnd_bg_weight > 0) and (len(bnd_bg_pens) > 0):
                loss = loss + self.bnd_bg_weight * (sum(bnd_bg_pens) / (len(bnd_bg_pens) + 1e-6))

        gt01 = (targets > 0.5).float()

        use_ring_tail = ring_tail and (
                (self.outer_weight > 0) or
                (self.inner_weight > 0) or
                (self.tail_weight > 0)
        )

        if use_ring_tail:
            outer, inner, far = self._make_outer_inner_far_masks(gt01)

            # (1) Outer ring: force negatives
            if self.outer_weight > 0 and outer.sum().item() > 0:
                if self.outer_margin > 0:
                    pen = F.softplus(preds + self.outer_margin)
                    loss_out = (pen * outer).sum() / outer.sum().clamp_min(1.0)
                else:
                    loss_out = self._masked_bce_with_logits(preds, torch.zeros_like(preds), outer)
                loss = loss + self.outer_weight * loss_out

            # (2) Inner ring: optional only
            if self.inner_weight > 0 and inner.sum().item() > 0:
                if self.inner_margin > 0:
                    pen = F.softplus(-preds + self.inner_margin)
                    loss_in = (pen * inner).sum() / inner.sum().clamp_min(1.0)
                else:
                    loss_in = self._masked_bce_with_logits(preds, torch.ones_like(preds), inner)
                loss = loss + self.inner_weight * loss_in

            # (3) Far tail
            if self.tail_weight > 0:
                loss_tail = self._topk_tail_penalty(preds, far)
                loss = loss + self.tail_weight * loss_tail

        # (4) Halo penalty follows ring_tail switch:
        # training: ring_tail=True  -> enable halo suppression
        # testing : ring_tail=False -> pure base segmentation loss only
        if ring_tail and self.halo_weight > 0:
            outer_halo = self._dilate(gt01, self.outer_r) * (1.0 - gt01)
            if outer_halo.sum().item() > 0:
                halo_pen = (torch.sigmoid(preds) * outer_halo).sum() / outer_halo.sum().clamp_min(1.0)
                loss = loss + self.halo_weight * halo_pen

        # 4) optional head supervision (response heads)
        if support_preds is not None and support_targets is not None and self.support_weight > 0:
            if support_preds.shape[2:] != support_targets.shape[2:]:
                support_preds = F.interpolate(
                    support_preds, size=support_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.support_weight * F.binary_cross_entropy(
                support_preds.clamp(1e-4, 1.0 - 1e-4), support_targets
            )

        if center_preds is not None and center_targets is not None and self.center_weight > 0:
            if center_preds.shape[2:] != center_targets.shape[2:]:
                center_preds = F.interpolate(
                    center_preds, size=center_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.center_weight * F.binary_cross_entropy(
                center_preds.clamp(1e-4, 1.0 - 1e-4), center_targets
            )

        if interference_preds is not None and interference_targets is not None and self.interference_weight > 0:
            if interference_preds.shape[2:] != interference_targets.shape[2:]:
                interference_preds = F.interpolate(
                    interference_preds, size=interference_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.interference_weight * F.binary_cross_entropy(
                interference_preds.clamp(1e-4, 1.0 - 1e-4), interference_targets
            )

        if boundary_rejection_preds is not None and boundary_rejection_targets is not None and self.boundary_rejection_weight > 0:
            if boundary_rejection_preds.shape[2:] != boundary_rejection_targets.shape[2:]:
                boundary_rejection_preds = F.interpolate(
                    boundary_rejection_preds, size=boundary_rejection_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.boundary_rejection_weight * F.binary_cross_entropy(
                boundary_rejection_preds.clamp(1e-4, 1.0 - 1e-4), boundary_rejection_targets
            )

        if conf_preds is not None and conf_targets is not None and self.confidence_weight > 0:
            if conf_preds.shape[2:] != conf_targets.shape[2:]:
                conf_preds = F.interpolate(
                    conf_preds, size=conf_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.confidence_weight * F.binary_cross_entropy(
                conf_preds.clamp(1e-4, 1.0 - 1e-4), conf_targets
            )

        # Optional response-head supervision.
        if rejection_preds is not None and rejection_targets is not None and self.rejection_weight > 0:
            if rejection_preds.shape[2:] != rejection_targets.shape[2:]:
                rejection_preds = F.interpolate(
                    rejection_preds, size=rejection_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.rejection_weight * F.binary_cross_entropy(
                rejection_preds.clamp(1e-4, 1.0 - 1e-4), rejection_targets
            )

        if risk_preds is not None and risk_targets is not None and self.risk_weight > 0:
            if risk_preds.shape[2:] != risk_targets.shape[2:]:
                risk_preds = F.interpolate(
                    risk_preds, size=risk_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.risk_weight * F.binary_cross_entropy(
                risk_preds.clamp(1e-4, 1.0 - 1e-4), risk_targets
            )

        if radial_preds is not None and radial_targets is not None and self.radial_weight > 0:
            if radial_preds.shape[2:] != radial_targets.shape[2:]:
                radial_preds = F.interpolate(
                    radial_preds, size=radial_targets.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + self.radial_weight * F.binary_cross_entropy(
                radial_preds.clamp(1e-4, 1.0 - 1e-4), radial_targets
            )

        return loss
