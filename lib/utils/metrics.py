import torch
from torchmetrics.functional import jaccard_index
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def IoU(
    preds: torch.Tensor, tgts: torch.Tensor, num_classes: int, **kwargs
) -> tuple[list[float], float]:
    n_shapes = preds.size(0)
    assert n_shapes == tgts.size(0)
    ious = []
    for i in range(n_shapes):
        ious.append(calc_IoU(preds[i], tgts[i], average='none', num_classes=num_classes, **kwargs))
    t_ious = torch.stack(ious).mean(0)

    return t_ious.tolist(), t_ious.mean(-1).item()


def calc_IoU(preds: torch.Tensor, tgts: torch.Tensor, num_classes: int, **kwargs) -> torch.Tensor:
    return jaccard_index(preds, tgts, num_classes, absent_score=1.0, **kwargs)


def frequency_weighted_IoU(pred_labels, gt_labels, num_classes):
    ious = torch.tensor(IoU(pred_labels, gt_labels, num_classes)[0])
    frequencies = torch.bincount(gt_labels.flatten(), minlength=num_classes)

    fw_miou = torch.sum(ious * frequencies) / torch.sum(frequencies)
    return fw_miou.item()


def cross_entropy(preds, tgts, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    if smoothing:
        eps = 0.2
        n_class = preds.size(1)

        one_hot = torch.zeros_like(preds).scatter(1, tgts.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(preds, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(preds, tgts, reduction='mean')
    return loss


def labels_from_logits(scores: Tensor) -> Tensor:
    return F.softmax(scores, dim=1).argmax(dim=1)


def _local_neighborhood(
    points: Tensor,
    labels: Tensor,
    k: int,
    radius: Optional[float],
) -> tuple:
    xyz = points[..., :3].float()
    batch_size, num_points, _ = xyz.shape
    k = min(k + 1, num_points)

    dists = torch.cdist(xyz, xyz)
    knn_dists, knn_idx = dists.topk(k=k, dim=-1, largest=False)

    # Remove the anchor point itself. In degenerate duplicate-point cases this still
    # leaves enough neighbors for a useful loss while avoiding trivial positives.
    knn_dists = knn_dists[..., 1:]
    knn_idx = knn_idx[..., 1:]

    batch_idx = torch.arange(batch_size, device=points.device).view(batch_size, 1, 1)
    neighbor_labels = labels[batch_idx, knn_idx]
    valid = torch.ones_like(knn_dists, dtype=torch.bool)
    if radius is not None and radius > 0:
        valid = knn_dists <= radius

    return neighbor_labels, valid


def boundary_mask(
    points: Tensor,
    labels: Tensor,
    k: int = 40,
    radius: Optional[float] = 0.1,
) -> Tensor:
    """Ground-truth or prediction boundary mask from local label disagreement."""
    neighbor_labels, valid = _local_neighborhood(points, labels, k, radius)
    center_labels = labels.unsqueeze(-1)
    return ((neighbor_labels != center_labels) & valid).any(dim=-1)


def contrastive_boundary_loss(
    features: Tensor,
    points: Tensor,
    labels: Tensor,
    k: int = 40,
    radius: Optional[float] = 0.1,
    temperature: float = 1.0,
) -> Tensor:
    """CBL loss from arXiv:2203.05272 for a single point-cloud scale.

    The paper defines supervised InfoNCE on ground-truth boundary points. This
    implementation uses the k nearest points within the optional radius as the
    local neighborhood and cosine distance for feature discrimination.
    """
    if features.dim() != 3 or points.dim() != 3 or labels.dim() != 2:
        raise ValueError('Expected features [B, C, N], points [B, N, D], labels [B, N].')

    features = F.normalize(features.transpose(1, 2).contiguous(), dim=-1)
    batch_size, num_points, _ = features.shape
    k = min(k + 1, num_points)

    xyz = points[..., :3].float()
    dists = torch.cdist(xyz, xyz)
    knn_dists, knn_idx = dists.topk(k=k, dim=-1, largest=False)
    knn_dists = knn_dists[..., 1:]
    knn_idx = knn_idx[..., 1:]

    batch_idx = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1)
    neighbor_features = features[batch_idx, knn_idx]
    neighbor_labels = labels[batch_idx, knn_idx]

    valid = torch.ones_like(knn_dists, dtype=torch.bool)
    if radius is not None and radius > 0:
        valid = knn_dists <= radius

    center_labels = labels.unsqueeze(-1)
    same_label = (neighbor_labels == center_labels) & valid
    diff_label = (neighbor_labels != center_labels) & valid
    anchors = same_label.any(dim=-1) & diff_label.any(dim=-1)

    if not anchors.any():
        return features.sum() * 0.0

    center_features = features.unsqueeze(2)
    cosine_distance = 1.0 - (center_features * neighbor_features).sum(dim=-1)
    logits = -cosine_distance / temperature
    exp_logits = torch.exp(logits) * valid.float()

    numerator = (exp_logits * same_label.float()).sum(dim=-1)
    denominator = exp_logits.sum(dim=-1)
    loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))

    return loss[anchors].mean()


def boundary_iou(
    points: Tensor,
    pred_labels: Tensor,
    gt_labels: Tensor,
    k: int = 40,
    radius: Optional[float] = 0.1,
) -> float:
    gt_boundary = boundary_mask(points, gt_labels, k=k, radius=radius)
    pred_boundary = boundary_mask(points, pred_labels, k=k, radius=radius)
    intersection = (gt_boundary & pred_boundary).sum().float()
    union = (gt_boundary | pred_boundary).sum().float()
    if union.item() == 0:
        return 1.0
    return (intersection / union).item()
