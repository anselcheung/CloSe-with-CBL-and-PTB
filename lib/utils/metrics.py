import torch
from torchmetrics.functional import jaccard_index
import torch.nn.functional as F
from torch import Tensor


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


def boundary_region_mIoU(preds, tgts, boundary_dist, radius, num_classes):
    """mIoU restricted to points within ``radius`` of a GT boundary (brief section 5).

    Global mIoU barely moves under boundary-aware training; this measures the effect
    where it happens. ``boundary_dist`` is the per-point distance to the nearest GT
    boundary (precomputed by prep_boundaries.py). Shapes with no points inside the
    radius are skipped. Returns NaN if no shape contributes.
    """
    n_shapes = preds.size(0)
    ious = []
    for i in range(n_shapes):
        mask = boundary_dist[i] < radius
        if mask.sum() == 0:
            continue
        ious.append(
            calc_IoU(preds[i][mask], tgts[i][mask], average='none', num_classes=num_classes)
        )
    if len(ious) == 0:
        return float('nan')
    return torch.stack(ious).mean(0).mean(-1).item()


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
