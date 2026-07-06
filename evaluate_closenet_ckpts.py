#!/usr/bin/env python3
"""Evaluate a base (no-CBL) and a CBL CloSeNet checkpoint on the CloSe-Di test split.

Each model is loaded with its own training config (architecture is identical between
the two; only the CBL loss/branch differs), run once over the test split, and scored
with mIoU, per-class IoU, frequency-weighted IoU and boundary IoU. Results are written
to a single JSON file for later comparison.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from lib import factory
from lib.utils.metrics import (
    IoU,
    boundary_iou,
    contrastive_boundary_loss,
    cross_entropy,
    frequency_weighted_IoU,
    labels_from_logits,
    select_cbl_features,
)
from lib.utils.types import EasierDict

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_path(path: str, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_config(config_path: Path) -> EasierDict:
    cfg = EasierDict(yaml.load(config_path.read_text(encoding='utf-8'), Loader=yaml.FullLoader))
    cfg.data.split_file = str(_resolve_path(cfg.data.split_file))
    cfg.data.data_path = str(_resolve_path(cfg.data.data_path))
    cfg.exp_logs_path = str(_resolve_path(cfg.exp_logs_path))
    return cfg


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Pick the checkpoint with the highest mIoU encoded in its filename.

    Training saves one `valmin_*_mIoU=<value>_freq_IoU=<value>.pt` file every time
    validation mIoU or freq_IoU improves, plus a final `model_<epoch>_<it>.pt` with no
    metric in its name. We only rank the metric-tagged files.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_dir}')

    scored = []
    for ckpt in checkpoint_dir.glob('valmin_*.pt'):
        match = re.search(r'mIoU=([0-9.]+)', ckpt.name)
        if match:
            scored.append((float(match.group(1)), ckpt))

    if not scored:
        raise FileNotFoundError(f'No scored (valmin_*mIoU=*.pt) checkpoints found in {checkpoint_dir}')

    return max(scored, key=lambda x: x[0])[1]


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    # torch<2.0 (this cluster runs 1.11) doesn't have the weights_only kwarg at all.
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    # CheckpointIO.save() nests the model weights under "<model_name>_model".
    state_dict = checkpoint.get('closenet_model', checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'Failed to load checkpoint {ckpt_path}: '
            f'missing_keys={incompatible.missing_keys[:10]}, '
            f'unexpected_keys={incompatible.unexpected_keys[:10]}'
        )


@torch.no_grad()
def evaluate_checkpoint(cfg: EasierDict, model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    load_checkpoint_weights(model, ckpt_path, str(device))
    model.to(device)
    model.eval()

    cbl_cfg = cfg.training.get('cbl', EasierDict(enabled=False))
    cbl_enabled = bool(cbl_cfg.get('enabled', False))

    test_dataset = factory.get_dataset(cfg, 'test')
    test_loader = test_dataset.get_loader(shuffle=False)

    segm_losses: List[float] = []
    cbl_losses: List[float] = []
    boundary_ious: List[float] = []
    all_preds, all_tgts = [], []

    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        logits = outputs['logits']
        targets = batch.y
        preds = labels_from_logits(logits)

        segm_losses.append(float(cross_entropy(logits, targets, smoothing=False)))

        if cbl_enabled:
            cbl_loss = sum(
                contrastive_boundary_loss(
                    feat,
                    batch.points,
                    targets,
                    k=cbl_cfg.get('k', 40),
                    radius=cbl_cfg.get('radius', 0.1),
                    temperature=cbl_cfg.get('temperature', 1.0),
                )
                for feat in select_cbl_features(
                    cbl_cfg.get('feature_source', 'decoder'),
                    outputs,
                    emb_dim=model.pc_enc.emb_dim,
                )
            )
            cbl_losses.append(float(cbl_loss))

        boundary_ious.append(float(boundary_iou(
            batch.points, preds, targets,
            k=cbl_cfg.get('k', 40), radius=cbl_cfg.get('radius', 0.1),
        )))

        all_preds.append(preds.cpu())
        all_tgts.append(targets.cpu())

    preds_all = torch.cat(all_preds, dim=0)
    tgts_all = torch.cat(all_tgts, dim=0)
    per_class_iou, mean_iou = IoU(preds_all, tgts_all, num_classes=cfg.data.n_classes)

    metrics: Dict[str, Any] = {
        'checkpoint': str(ckpt_path),
        'num_test_samples': len(test_dataset),
        'num_evaluated_shapes': int(preds_all.size(0)),
        'device': str(device),
        'cbl_enabled': cbl_enabled,
        'segm_loss_mean': float(torch.tensor(segm_losses).mean()),
        'mIoU': float(mean_iou),
        'IoU_per_class': [float(x) for x in per_class_iou],
        'freq_IoU': float(frequency_weighted_IoU(preds_all, tgts_all, num_classes=cfg.data.n_classes)),
        'boundary_iou_mean': float(torch.tensor(boundary_ious).mean()),
    }
    if cbl_enabled:
        metrics['cbl_loss_mean'] = float(torch.tensor(cbl_losses).mean()) if cbl_losses else float('nan')

    return metrics


def build_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['segm_loss_mean', 'mIoU', 'freq_IoU', 'boundary_iou_mean', 'cbl_loss_mean']
    return {k: metrics[k] for k in keys if k in metrics}


def evaluate_variant(config_path: Path, ckpt_path: Optional[Path], device: torch.device) -> Dict[str, Any]:
    cfg = load_config(config_path)
    ckpt_path = ckpt_path or find_best_checkpoint(Path(cfg.exp_logs_path) / 'checkpoints')
    model = factory.get_model(cfg)
    return evaluate_checkpoint(cfg, model, ckpt_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate base vs. CBL CloSeNet checkpoints on the test split.')
    parser.add_argument('--base-config', default='cfg/closenet.yaml', help='Config used to train the no-CBL model')
    parser.add_argument('--cbl-config', default='cfg/closenet_cbl.yaml', help='Config used to train the CBL model')
    parser.add_argument('--base-ckpt', default=None, help='Base checkpoint path (default: best mIoU checkpoint in its exp dir)')
    parser.add_argument('--cbl-ckpt', default=None, help='CBL checkpoint path (default: best mIoU checkpoint in its exp dir)')
    parser.add_argument('--output-json', default='results/closenet_di_test_metrics.json', help='Where to save results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    base_metrics = evaluate_variant(
        _resolve_path(args.base_config), args.base_ckpt and _resolve_path(args.base_ckpt), device,
    )
    cbl_metrics = evaluate_variant(
        _resolve_path(args.cbl_config), args.cbl_ckpt and _resolve_path(args.cbl_ckpt), device,
    )

    results = {
        'baseline': base_metrics,
        'cbl': cbl_metrics,
        'summary': {
            'baseline': build_metric_summary(base_metrics),
            'cbl': build_metric_summary(cbl_metrics),
        },
    }

    output_path = _resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    print(json.dumps(results['summary'], indent=2))
    print(f'\nSaved full metrics to {output_path}')


if __name__ == '__main__':
    main()
