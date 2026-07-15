import torch
from torch import nn


class BoundaryRefinementModule(nn.Module):
    """Boundary-aware Refinement Module (BRM) -- CloSeNet's stand-in for PTB's guided
    feature propagation (GFP).

    PTB's GFP reweights the *upsampling* interpolation of an encoder-decoder FCN using
    predicted boundary probability / direction (`PointNetFeaturePropagation.forward`'s
    guided branch in the reference `pointnet2_utils_boundary.py`). CloSeNet has no such
    upsampling stage (its DGCNN encoder runs at full resolution and the segmentation
    decoder is a pointwise MLP), so this module re-homes the same weighting formula onto
    a spatial-kNN aggregation of the pre-logit decoder features instead:

        w = clamp(exp(-dist / r) + alpha * exp(P_b - 1) * cos(neighbor_dir, pred_dir), min=0)

    normalized to sum to 1 over the k neighbours, matching PTB's Eq. 2 modulo the
    upsampling-vs-fixed-neighborhood difference (see `ptb_phase2_postproc_fix_brief.md`,
    Part A).
    """

    def __init__(self, channels: int, k: int = 6, alpha: float = 1.0, r_init: float = 0.05):
        super().__init__()
        self.k = k
        self.alpha = alpha
        # r is scale-sensitive (PTB tuned it for room-scale S3DIS in meters; CloSe scans
        # are unit-bbox normalised), so keep it learnable rather than copying PTB's r=0.125.
        self.log_r = nn.Parameter(torch.log(torch.tensor(float(r_init))))
        self.merge = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        feat: torch.Tensor,
        xyz: torch.Tensor,
        boundary_prob: torch.Tensor,
        pred_dirs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat:          (B, C, N) pre-logit decoder features.
            xyz:           (B, N, 3) point positions.
            boundary_prob: (B, N)    predicted P(boundary).
            pred_dirs:     (B, 3, N) predicted unit interior directions.

        Returns:
            (B, C, N) refined features (residual-merged with the input).
        """
        B, C, N = feat.shape
        k = min(self.k, N)
        r = torch.exp(self.log_r).clamp(min=1e-4)

        dist = torch.cdist(xyz, xyz)  # (B, N, N)
        nn_dist, nn_idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k)

        nbr_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )  # (B, N, k, 3)
        rel = nbr_xyz - xyz.unsqueeze(2)  # (B, N, k, 3)
        dirs = pred_dirs.permute(0, 2, 1)  # (B, N, 3)
        cos = (rel * dirs.unsqueeze(2)).sum(-1)  # (B, N, k)
        cos = cos / (nn_dist + 1e-8)  # matches PTB's dir_cos / dists

        w_spatial = torch.exp(-nn_dist / r)
        w_guide = torch.exp(boundary_prob - 1.0).unsqueeze(-1) * cos
        w = (w_spatial + self.alpha * w_guide).clamp(min=0.0)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)  # (B, N, k)

        feat_t = feat.permute(0, 2, 1)  # (B, N, C)
        nbr_feat = torch.gather(
            feat_t.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )  # (B, N, k, C)
        agg = (w.unsqueeze(-1) * nbr_feat).sum(2).permute(0, 2, 1)  # (B, C, N)

        return self.merge(agg) + feat
