from collections import defaultdict

import torch

from lib.utils.metrics import (
    boundary_iou,
    contrastive_boundary_loss,
    cross_entropy,
    labels_from_logits,
    select_cbl_features,
)
from lib.utils.types import EasierDict
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

        self.optimizer = torch.optim.Adam(
            [
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
        )

    def _pack_data(
        self,
        batch: EasierDict,
        outp: dict,
    ) -> dict:
        ret_dict = {
            'points': batch.points.cpu() if hasattr(batch, 'points') else batch.x.cpu(),
            'pred_labels': labels_from_logits(outp['logits']).cpu(),
            'tgts': batch.y.cpu(),
            'pred_logits': outp['logits'].cpu(),
        }
        if outp.get('attn_weights', None) is not None:
            ret_dict['attn_weights'] = outp['attn_weights'].cpu()
        if hasattr(batch, 'idx'):
            ret_dict['idx'] = batch.idx.cpu()
        if hasattr(batch, 'scan_id'):
            ret_dict['scan_id'] = batch.scan_id
        return ret_dict

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

            loss_dict['total_loss'] = sum(
                [
                    self.loss_weights[k] * loss_dict[k]
                    for k in self.loss_weights.keys()
                    if k in loss_dict
                ]
            )

            # boundary_iou is a metric, not a training loss - only computed at
            # validation/test time (self.model.training is False there).
            if self.cbl_cfg.get('enabled', False) and not self.model.training:
                loss_dict['boundary_iou'] = torch.tensor(
                    boundary_iou(
                        batch.points,
                        labels_from_logits(outp_dict['logits']),
                        batch.y,
                        k=self.cbl_cfg.get('k', 40),
                        radius=self.cbl_cfg.get('radius', 0.1),
                    ),
                    device=self.device,
                )

        return loss_dict, self._pack_data(batch, outp_dict)
