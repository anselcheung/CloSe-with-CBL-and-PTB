from typing import Optional

import torch
from torch.nn import functional as F

from ..utils.types import EasierDict
from .nets import (
    AttentionGarmentEncoder,
    BoundaryRefinementModule,
    CanonEncoder,
    DGCNNBase,
    MLPDecoder,
)


class CloSeNet(torch.nn.Module):
    def __init__(self, cfg: EasierDict) -> None:
        super(CloSeNet, self).__init__()
        # * Shortened
        model_cfg = cfg.model_arch
        pc_enc_cfg = model_cfg.pc_enc
        garm_enc_cfg = model_cfg.garm_enc
        segm_dec_cfg = model_cfg.segm_dec
        n_classes = 18

        # * Setup encoders
        self.pc_enc = DGCNNBase(
            inp_dim=pc_enc_cfg.inp_dim,
            emb_dim=pc_enc_cfg.emb_dim,
            k=pc_enc_cfg.k,
            use_tnet=pc_enc_cfg.use_tnet,
        )
        self.part_enc = CanonEncoder()
        self.garm_enc = AttentionGarmentEncoder(garm_enc_cfg)

        # * Setup MLP decoder
        segm_input = (
            self.pc_enc.emb_dim
            + self.pc_enc.channels_sum
            + self.part_enc.emb_dim
            + self.garm_enc.emb_dim
        )
        self.segm_dec = MLPDecoder(
            channels=[segm_input] + segm_dec_cfg.channels + [n_classes],
            dropout=segm_dec_cfg.dropout,
            slope=segm_dec_cfg.slope,
        )

        # * Optional PTB auxiliary heads (Phase 1). Gated by config so the baseline
        # * architecture and pretrained checkpoints are unaffected when disabled.
        aux_cfg = model_cfg.get('aux_heads', None)
        self.use_aux_heads = bool(aux_cfg is not None and aux_cfg.get('enabled', False))
        if self.use_aux_heads:
            boundary_channels = aux_cfg.get('boundary_channels', [256, 128])
            direction_channels = aux_cfg.get('direction_channels', [256, 128])
            self.boundary_head = MLPDecoder(
                channels=[segm_input] + list(boundary_channels) + [2],
                dropout=segm_dec_cfg.dropout,
                slope=segm_dec_cfg.slope,
            )
            self.direction_head = MLPDecoder(
                channels=[segm_input] + list(direction_channels) + [3],
                dropout=segm_dec_cfg.dropout,
                slope=segm_dec_cfg.slope,
            )
        else:
            self.boundary_head = None
            self.direction_head = None

        # * Optional Phase-2 BRM (adapted guided feature propagation). Requires the aux
        # * heads, since it consumes predicted boundary probability / direction.
        brm_cfg = model_cfg.get('brm', None)
        self.use_brm = bool(brm_cfg is not None and brm_cfg.get('enabled', False))
        if self.use_brm and not self.use_aux_heads:
            raise ValueError(
                'model_arch.brm.enabled=True requires model_arch.aux_heads.enabled=True '
                '(BRM consumes the predicted boundary probability and direction).'
            )
        if self.use_brm:
            self.brm = BoundaryRefinementModule(
                channels=segm_dec_cfg.channels[-1],
                k=brm_cfg.get('k', 6),
                alpha=brm_cfg.get('alpha', 1.0),
                r_init=brm_cfg.get('r_init', 0.05),
            )
        else:
            self.brm = None

    def _encode(self, data: EasierDict, **kwargs) -> EasierDict:
        x_max, conv_out, data = self.pc_enc(data, **kwargs)
        # * Get per-point features
        pc_features = (
            torch.cat([x_max] + conv_out, dim=1).permute(0, 2, 1).contiguous()
        )  # B, n_samples, K

        # * Get part features
        part_features = self.part_enc(data, **kwargs)

        # * Get garment features
        garm_features, attn_weights = self.garm_enc(data, [x_max] + conv_out, **kwargs)

        return EasierDict(
            pc_features=pc_features,
            part_features=part_features,
            garm_features=garm_features,
            attn_weights=attn_weights,
        )

    def _decode(
        self, data: EasierDict, points_xyz: Optional[torch.Tensor] = None, **kwargs
    ) -> EasierDict:
        feat = (
            torch.cat([data.pc_features, data.part_features, data.garm_features], dim=-1)
            .permute(0, 2, 1)
            .contiguous()
        )
        logits, decoder_features = self.segm_dec(feat, return_features=True, **kwargs)
        out = EasierDict(logits=logits, decoder_features=decoder_features)
        if self.use_aux_heads:
            out.boundary_logits = self.boundary_head(feat, **kwargs)  # (B, 2, N)
            # * Unit-normalise the regressed directions along the channel axis.
            out.pred_dirs = F.normalize(
                self.direction_head(feat, **kwargs), dim=1, eps=1e-8
            )  # (B, 3, N)
            if self.use_brm:
                # * Phase 2: refine the pre-logit features with the BRM, then re-run them
                # * through the *same* trained classifier layer (mirrors PTB, where GFP
                # * replaces what feeds the existing classifier rather than adding a new one).
                pb = F.softmax(out.boundary_logits, dim=1)[:, 1, :]  # (B, N)
                refined = self.brm(decoder_features, points_xyz, pb, out.pred_dirs)
                out.logits = self.segm_dec.layers[-1](refined)
        return out

    def forward(self, data: EasierDict, **kwargs) -> EasierDict:
        encodings = self._encode(data, **kwargs)
        points_xyz = data.points[..., :3] if hasattr(data, 'points') else None
        decoded = self._decode(encodings, points_xyz=points_xyz, **kwargs)
        logits = decoded.logits
        out = EasierDict(
            **data,
            logits=logits,
            labels=F.softmax(logits, dim=1).argmax(dim=1),
            encodings=encodings,
            decoder_features=decoded.decoder_features,
        )
        if self.use_aux_heads:
            out.boundary_logits = decoded.boundary_logits
            out.pred_dirs = decoded.pred_dirs
        return out
