import torch
from torch.nn import functional as F

from ..utils.types import EasierDict
from .nets import AttentionGarmentEncoder, CanonEncoder, DGCNNBase, MLPDecoder


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

    def _decode(self, data: EasierDict, **kwargs) -> EasierDict:
        encodings = (
            torch.cat([data.pc_features, data.part_features, data.garm_features], dim=-1)
            .permute(0, 2, 1)
            .contiguous()
        )
        logits, decoder_features = self.segm_dec(encodings, return_features=True, **kwargs)
        return EasierDict(logits=logits, decoder_features=decoder_features)

    def forward(self, data: EasierDict, **kwargs) -> EasierDict:
        encodings = self._encode(data, **kwargs)
        decoded = self._decode(encodings, **kwargs)
        logits = decoded.logits
        return EasierDict(
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
