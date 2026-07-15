"""SegFix-style test-time relabeling (brief Phase 3).

Descends from PTB's direction scheme / SegFix: for every point predicted to lie on
a semantic boundary (P_b > threshold), we step a short distance along the regressed
interior direction and copy the predicted label of the nearest point at that offset
location. This costs one kNN query at inference and requires no retraining beyond the
Phase-1 boundary/direction heads. It is a lower bound on what the boundary/direction
information is worth and a sanity check that the heads learned something meaningful.
"""

from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree


def _relabel_one(labels, xyz, boundary_prob, dirs, threshold, step, n_iters):
    """Relabel a single point cloud in place-safe fashion; returns new labels (N,)."""
    n = labels.shape[0]
    if n == 0:
        return labels

    tree = cKDTree(xyz)

    # Local point spacing -> default step when none is configured.
    if step is None:
        nn_dist, _ = tree.query(xyz, k=min(2, n))
        if nn_dist.ndim == 2 and nn_dist.shape[1] > 1:
            median_spacing = float(np.median(nn_dist[:, 1]))
        else:
            median_spacing = 0.0
        step = 2.0 * median_spacing if median_spacing > 0 else 0.0

    mask = boundary_prob > threshold
    if not mask.any() or step == 0.0:
        return labels

    move_idx = np.nonzero(mask)[0]
    # Directions are unit vectors; `pos` is a virtual position that walks further along
    # them each iteration (total distance after i iterations is i * step), re-querying the
    # tree each time. This lets n_iters actually control how far the label search travels,
    # and lets chains of adjacent boundary points hand labels down the line via `out`.
    pos = xyz[mask].copy()
    out = labels.copy()
    for _ in range(max(1, n_iters)):
        pos = pos + step * dirs[mask]
        _, nn_idx = tree.query(pos, k=1)  # index (into all points) of the interior neighbour
        out[move_idx] = out[nn_idx]
    return out


def segfix_relabel(
    pred_labels: torch.Tensor,
    points_xyz: torch.Tensor,
    boundary_prob: torch.Tensor,
    pred_dirs: torch.Tensor,
    threshold: float = 0.7,
    step: Optional[float] = None,
    n_iters: int = 2,
) -> torch.Tensor:
    """Apply SegFix relabeling to a (batched) prediction.

    Args:
        pred_labels:   (..., N) integer predicted labels.
        points_xyz:    (..., N, 3) posed xyz coordinates.
        boundary_prob: (..., N) predicted boundary probability P_b.
        pred_dirs:     (..., N, 3) predicted unit interior directions.
        threshold:     P_b cut-off above which a point is relabeled.
        step:          step length along the direction; ``None`` => 2x median local
                       nearest-neighbour spacing, computed per cloud.
        n_iters:       number of propagation iterations.

    Returns:
        Tensor of the same shape/dtype/device as ``pred_labels``.
    """
    device = pred_labels.device
    dtype = pred_labels.dtype

    labels_np = pred_labels.detach().cpu().numpy()
    xyz_np = points_xyz.detach().cpu().numpy()
    prob_np = boundary_prob.detach().cpu().numpy()
    dirs_np = pred_dirs.detach().cpu().numpy()

    flat_labels = labels_np.reshape(-1, labels_np.shape[-1])
    flat_xyz = xyz_np.reshape(-1, xyz_np.shape[-2], xyz_np.shape[-1])
    flat_prob = prob_np.reshape(-1, prob_np.shape[-1])
    flat_dirs = dirs_np.reshape(-1, dirs_np.shape[-2], dirs_np.shape[-1])

    out = np.empty_like(flat_labels)
    for i in range(flat_labels.shape[0]):
        out[i] = _relabel_one(
            flat_labels[i], flat_xyz[i], flat_prob[i], flat_dirs[i], threshold, step, n_iters
        )

    out = out.reshape(labels_np.shape)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


if __name__ == '__main__':
    # Regression + bug-catching check for _relabel_one's iteration semantics: a chain of
    # 9 points along a line (x=0..8) with a 3-point-wide boundary band (idx 3,4,5) between
    # two differently-labeled interior regions, direction pointing toward the higher-x side,
    # step=1 (one grid spacing).
    xyz = np.stack([np.arange(9, dtype=np.float64), np.zeros(9), np.zeros(9)], axis=1)
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    boundary_prob = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0])
    dirs = np.zeros((9, 3))
    dirs[3] = dirs[4] = dirs[5] = [1.0, 0.0, 0.0]

    out1 = _relabel_one(labels, xyz, boundary_prob, dirs, threshold=0.5, step=1.0, n_iters=1)
    # n_iters=1 hops every boundary point exactly one grid step forward (identical to the
    # pre-fix code, which always got the first iteration right): 3->4 (still 0), 4->5
    # (still 0), 5->6 (already 1).
    assert out1.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1], out1

    out2 = _relabel_one(labels, xyz, boundary_prob, dirs, threshold=0.5, step=1.0, n_iters=2)
    # n_iters=2 should walk boundary points a cumulative 2 steps and re-query at each hop,
    # so point 3 (the innermost boundary point, 2 hops from the label-1 region) picks up the
    # propagated label. The pre-fix code queried a *fixed* single-step target [4,5,6] on every
    # iteration instead of advancing further, so point 3 (whose fixed target is point 4) only
    # ever saw point 4's label, which reaches 1 only via a same-iteration chain coincidence
    # that does NOT occur here -- pre-fix this assertion fails with out2[3] == 0.
    assert out2.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1], (
        f'n_iters=2 should propagate the interior label a second hop in, got {out2.tolist()}'
    )

    print('postprocess._relabel_one iteration checks passed:', out1.tolist(), out2.tolist())
