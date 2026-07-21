#!/usr/bin/env python3
"""Evaluate CloSeNet checkpoints (baseline / PTB / CBL / PTB+CBL / PTB+postproc) on the
CloSe-Di test split.

Two modes, both routed through lib.factory + BaseTrainer.evaluate_model() so every
ablation shares the exact same metric computation (mIoU, per-class IoU,
frequency-weighted IoU, boundary_iou, mIoU@boundary, mIoU@inner, per-class
boundary/inner IoU, and boundary_mIoU@rho whenever the scans carry boundary_dist,
i.e. prep_boundaries.py has been run):

  Single-checkpoint mode (--config/--ckpt) - evaluate one checkpoint against one config,
  e.g. a PTB checkpoint with SegFix post-processing enabled via a dedicated test config:
      python evaluate_closenet_ckpts.py --config cfg/closenet_test_ptb.yaml \\
          --ckpt closenet_ptb_train/checkpoints/<BEST>.pt

  Paired mode (--base-config/--cbl-config, the default) - compare a no-CBL and a CBL
  checkpoint side by side, auto-picking the best-mIoU checkpoint from each config's
  exp_logs_path unless --base-ckpt/--cbl-ckpt are given:
      python evaluate_closenet_ckpts.py --base-config cfg/closenet.yaml \\
          --cbl-config cfg/closenet_cbl.yaml

Results are written to a single JSON file for later comparison.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from lib import factory
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

    cbl_cfg = cfg.training.get('cbl', EasierDict(enabled=False))
    cbl_enabled = bool(cbl_cfg.get('enabled', False))

    test_dataset = factory.get_dataset(cfg, 'test')
    trainer = factory.get_trainer(model, None, None, test_dataset, cfg)

    # BaseTrainer.__init__ -> CheckpointIO._check_ckpts() auto-scans
    # cfg.exp_logs_path/checkpoints and may silently overwrite our weights with the
    # "latest by iteration" (not "best by mIoU") checkpoint it finds there. Re-load
    # explicitly, after trainer construction, so the checkpoint we picked always wins,
    # regardless of whether cfg.exp_logs_path is a real training dir (paired mode) or a
    # fresh scratch dir (single-checkpoint mode).
    load_checkpoint_weights(trainer.model, ckpt_path, str(device))

    val_dict, outp_dict = trainer.evaluate_model(dataset='test', seed=42)

    metrics: Dict[str, Any] = {
        'checkpoint': str(ckpt_path),
        'num_test_samples': len(test_dataset),
        'num_evaluated_shapes': int(outp_dict['pred_labels'].shape[0]),
        'device': str(device),
        'cbl_enabled': cbl_enabled,
        'segm_loss_mean': float(val_dict['segm_loss']),
        'mIoU': float(val_dict['mIoU']),
        'IoU_per_class': [float(x) for x in val_dict['IoU']],
        'freq_IoU': float(val_dict['freq_IoU']),
        'boundary_iou_mean': float(val_dict['boundary_iou']),
        'mIoU_boundary_mean': float(val_dict['mIoU_boundary']),
        'mIoU_inner_mean': float(val_dict['mIoU_inner']),
        'boundary_IoU_per_class': [float(x) for x in val_dict['IoU_boundary_per_class']],
        'inner_IoU_per_class': [float(x) for x in val_dict['IoU_inner_per_class']],
    }
    if cbl_enabled:
        metrics['cbl_loss_mean'] = float(val_dict.get('cbl_loss', float('nan')))
    for key, value in val_dict.items():
        if key.startswith('boundary_mIoU@'):
            metrics[key] = float(value)

    return metrics


def build_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        'segm_loss_mean', 'mIoU', 'freq_IoU', 'boundary_iou_mean',
        'mIoU_boundary_mean', 'mIoU_inner_mean', 'cbl_loss_mean',
    ]
    summary = {k: metrics[k] for k in keys if k in metrics}
    summary.update({k: v for k, v in metrics.items() if k.startswith('boundary_mIoU@')})
    return summary


def evaluate_variant(config_path: Path, ckpt_path: Optional[Path], device: torch.device) -> Dict[str, Any]:
    cfg = load_config(config_path)
    ckpt_path = ckpt_path or find_best_checkpoint(Path(cfg.exp_logs_path) / 'checkpoints')
    model = factory.get_model(cfg)
    return evaluate_checkpoint(cfg, model, ckpt_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate CloSeNet checkpoints (baseline / PTB / CBL / PTB+CBL / PTB+postproc) on the test split.'
    )
    parser.add_argument('--config', default=None, help='Single-checkpoint mode: eval config (baseline/PTB/PTB+postproc/...)')
    parser.add_argument('--ckpt', default=None, help='Single-checkpoint mode: checkpoint path (required with --config)')
    parser.add_argument('--base-config', default='cfg/closenet.yaml', help='Paired mode: config used to train the no-CBL model')
    parser.add_argument('--cbl-config', default='cfg/closenet_cbl.yaml', help='Paired mode: config used to train the CBL model')
    parser.add_argument('--base-ckpt', default=None, help='Paired mode: base checkpoint path (default: best mIoU checkpoint in its exp dir)')
    parser.add_argument('--cbl-ckpt', default=None, help='Paired mode: CBL checkpoint path (default: best mIoU checkpoint in its exp dir)')
    parser.add_argument('--output-json', default='results/closenet_di_test_metrics.json', help='Where to save results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.config:
        if not args.ckpt:
            parser.error('--config requires --ckpt')
        metrics = evaluate_variant(_resolve_path(args.config), _resolve_path(args.ckpt), device)
        results = {'metrics': metrics, 'summary': build_metric_summary(metrics)}
    else:
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
