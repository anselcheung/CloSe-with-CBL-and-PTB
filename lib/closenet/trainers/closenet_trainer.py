from collections import defaultdict

import torch
import torch.nn.functional as F

from lib.utils.metrics import (
    boundary_inner_mIoU,
    boundary_iou,
    contrastive_boundary_loss,
    cross_entropy,
    labels_from_logits,
    select_cbl_features,
)
from lib.utils.types import EasierDict
from lib.closenet.postprocess import segfix_relabel
from lib.closenet.trainers.base_trainer import BaseTrainer
from typing import Any


class CloSeNetTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        cfg: EasierDict,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            cfg,
            cfg.training.loss_weights,
            **kwargs,
        )

        self.training_cfg = cfg.training
        self.cbl_cfg = cfg.training.get('cbl', EasierDict(enabled=False))

        def _optim_params(x) -> dict[str, Any]:
            return {
                'lr': x.get('lr', cfg.optim.lr),
                'weight_decay': x.get('weight_decay', cfg.optim.weight_decay),
            }

        param_groups = [
            {
                'name': 'pc_params',
                'params': model.pc_enc.parameters() if model.pc_enc is not None else [],
                **_optim_params(cfg.model_arch['pc_enc']),
            },
            {
                'name': 'garm_params',
                'params': model.garm_enc.parameters() if model.garm_enc is not None else [],
                **_optim_params(cfg.model_arch['garm_enc']),
            },
            {
                'name': 'part_params',
                'params': model.part_enc.parameters() if model.part_enc is not None else [],
                **_optim_params(cfg.model_arch['part_enc']),
            },
            {
                'name': 'dec_params',
                'params': model.segm_dec.parameters() if model.segm_dec is not None else [],
                **_optim_params(cfg.model_arch['segm_dec']),
            },
        ]
        # * PTB auxiliary heads (Phase 1) share the decoder's optimiser settings.
        if getattr(model, 'use_aux_heads', False):
            param_groups.append(
                {
                    'name': 'aux_head_params',
                    'params': list(model.boundary_head.parameters())
                    + list(model.direction_head.parameters()),
                    **_optim_params(cfg.model_arch['segm_dec']),
                }
            )

        self.optimizer = torch.optim.Adam(param_groups)

        # * Phase 1 / Phase 3 knobs (all optional; defaults keep baseline behaviour).
        self.boundary_beta = float(self.training_cfg.get('boundary_beta', 0.6))
        self.dir_mask_dist = self.training_cfg.get('dir_mask_dist', None)
        self.pp_cfg = cfg.get('post_processing', None)

    def _pack_data(
        self,
        batch: EasierDict,
        outp: dict,
    ) -> dict:
        pred_labels = labels_from_logits(outp['logits'])

        # * PTB auxiliary outputs (Phase 1): boundary probability + interior directions.
        has_aux = outp.get('boundary_logits', None) is not None
        points_xyz = batch.points[..., :3] if hasattr(batch, 'points') else None
        if has_aux:
            boundary_prob = F.softmax(outp['boundary_logits'], dim=1)[:, 1, :]  # (B, N)
            pred_dirs = outp['pred_dirs'].permute(0, 2, 1).contiguous()  # (B, N, 3)

            # * Phase 3: SegFix relabeling at eval time only.
            pp = self.pp_cfg
            if pp is not None and pp.get('enabled', False) and not self.model.training:
                pred_labels = segfix_relabel(
                    pred_labels,
                    points_xyz,
                    boundary_prob,
                    pred_dirs,
                    threshold=float(pp.get('threshold', 0.7)),
                    step=pp.get('step', None),
                    n_iters=int(pp.get('n_iters', 2)),
                )
        elif self.pp_cfg is not None and self.pp_cfg.get('enabled', False):
            raise ValueError(
                'post_processing.enabled=True but the model has no boundary/direction '
                'heads. Post-processing requires a PTB-trained checkpoint '
                '(model_arch.aux_heads.enabled=True).'
            )

        ret_dict = {
            'points': batch.points.cpu() if hasattr(batch, 'points') else batch.x.cpu(),
            'pred_labels': pred_labels.cpu(),
            'tgts': batch.y.cpu(),
            'pred_logits': outp['logits'].cpu(),
        }
        if outp.get('attn_weights', None) is not None:
            ret_dict['attn_weights'] = outp['attn_weights'].cpu()
        if has_aux:
            ret_dict['boundary_prob'] = boundary_prob.cpu()
            ret_dict['pred_dirs'] = pred_dirs.cpu()
        if hasattr(batch, 'boundary_dist') and batch.boundary_dist is not None:
            ret_dict['boundary_dist'] = batch.boundary_dist.cpu()
        if hasattr(batch, 'idx'):
            ret_dict['idx'] = batch.idx.cpu()
        if hasattr(batch, 'scan_id'):
            ret_dict['scan_id'] = batch.scan_id
        return ret_dict

    def _boundary_loss(self, boundary_logits: torch.Tensor, boundary_gt: torch.Tensor):
        # * Weighted 2-class cross-entropy (weighted-BCE equivalent). The boundary
        # * class is up-weighted by beta to counter its scarcity.
        weight = torch.tensor(
            [1.0 - self.boundary_beta, self.boundary_beta], device=boundary_logits.device
        )
        return F.cross_entropy(boundary_logits, boundary_gt.long(), weight=weight)

    def _direction_loss(self, pred_dirs: torch.Tensor, batch: EasierDict):
        # * MSE against unit GT directions. GT is zero at boundary/degenerate points
        # * (nearest boundary is the point itself); those are masked out since a
        # * unit-normalised prediction can never match a zero target.
        dir_gt = batch.direction.float()  # (B, N, 3)
        pred = pred_dirs.permute(0, 2, 1).contiguous()  # (B, N, 3)
        valid = dir_gt.norm(dim=-1) > 0.5  # (B, N)
        if self.dir_mask_dist is not None and hasattr(batch, 'boundary_dist'):
            valid = valid & (batch.boundary_dist >= float(self.dir_mask_dist))
        if valid.any():
            return F.mse_loss(pred[valid], dir_gt[valid])
        return pred.new_zeros(())

    def step(self, batch: EasierDict, skip_loss: bool = False, **kwargs: dict) -> dict:
        batch = batch.to(self.device)

        outp_dict = self.model(batch)

        loss_dict = defaultdict()
        if not skip_loss:
            loss_dict['segm_loss'] = cross_entropy(outp_dict['logits'], batch.y, smoothing=False)
            if self.cbl_cfg.get('enabled', False):
                # select_cbl_features returns 1 tensor (decoder / encoder_stage3) or 3
                # (encoder_all_stages); summing the per-tensor losses matches the
                # paper's multi-scale form (Eq. 7): L = L_segm + weight * sum_n(L_CBL^n).
                loss_dict['cbl_loss'] = sum(
                    contrastive_boundary_loss(
                        feat,
                        batch.points,
                        batch.y,
                        k=self.cbl_cfg.get('k', 40),
                        radius=self.cbl_cfg.get('radius', 0.1),
                        temperature=self.cbl_cfg.get('temperature', 1.0),
                    )
                    for feat in select_cbl_features(
                        self.cbl_cfg.get('feature_source', 'decoder'),
                        outp_dict,
                        emb_dim=self.model.pc_enc.emb_dim,
                    )
                )

            # * PTB auxiliary losses (Phase 1), only when the heads exist and GT is present.
            if outp_dict.get('boundary_logits', None) is not None:
                if hasattr(batch, 'boundary') and batch.boundary is not None:
                    loss_dict['boundary_loss'] = self._boundary_loss(
                        outp_dict['boundary_logits'], batch.boundary
                    )
                if hasattr(batch, 'direction') and batch.direction is not None:
                    loss_dict['direction_loss'] = self._direction_loss(
                        outp_dict['pred_dirs'], batch
                    )

            # * Sum only the weighted losses that were actually computed this step.
            loss_dict['total_loss'] = sum(
                [
                    self.loss_weights[k] * loss_dict[k]
                    for k in self.loss_weights.keys()
                    if k in loss_dict
                ]
            )

            # boundary_iou / mIoU_boundary / mIoU_inner are eval-only metrics - only
            # computed at validation/test time (self.model.training is False there).
            # They only need points + predicted labels + GT labels, so they apply to
            # every ablation (baseline, PTB, CBL, PTB+CBL), not just CBL-enabled runs.
            if not self.model.training:
                eval_preds = labels_from_logits(outp_dict['logits'])
                loss_dict['boundary_iou'] = torch.tensor(
                    boundary_iou(
                        batch.points,
                        eval_preds,
                        batch.y,
                        k=self.cbl_cfg.get('k', 40),
                        radius=self.cbl_cfg.get('radius', 0.1),
                    ),
                    device=self.device,
                )
                m_boundary, m_inner = boundary_inner_mIoU(
                    batch.points,
                    eval_preds,
                    batch.y,
                    num_classes=self.cfg.data.n_classes,
                    k=self.cbl_cfg.get('k', 40),
                    radius=self.cbl_cfg.get('radius', 0.1),
                )
                loss_dict['mIoU_boundary'] = torch.tensor(m_boundary, device=self.device)
                loss_dict['mIoU_inner'] = torch.tensor(m_inner, device=self.device)

        return loss_dict, self._pack_data(batch, outp_dict)
