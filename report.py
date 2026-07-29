#!/usr/bin/env python3
"""Turn the per-run result JSONs from evaluate_run.sbatch into report tables.

Reads every results/<ablation>_<timestamp>_<eval-config-stem>.json (30 files: 10
ablations x 3 seeds) plus its results_clip/ counterpart (10 ablations x 3 seeds,
CLIP-initialized codebook), and writes:

  table_headline.tex   20-row headline table (10 non-CLIP + 10 CLIP configs), same
                        6 metrics, plus a paired Delta B-IoU column vs. each row's
                        same-architecture non-CLIP counterpart.

This script only computes tables/figures -- no prose. reports/report.tex is a
separate, hand-authored document that \\input{}s whatever this script writes.

Usage:
    python report.py [--results-dir results] [--results-clip-dir results_clip]
                      [--runs-dir runs] [--output-dir reports]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Longest-prefix-first so "ptb_brm_cbl" isn't mis-parsed as "ptb_brm" + leftover.
ABLATION_PREFIXES = ['ptb_brm_cbl', 'ptb_brm', 'ptb_cbl', 'ptb', 'cbl', 'baseline']

METRIC_KEYS = ['mIoU', 'freq_IoU', 'boundary_iou_mean', 'mIoU_boundary_mean', 'mIoU_inner_mean', 'segm_loss_mean']
METRIC_LABELS = {
    'mIoU': 'mIoU',
    'freq_IoU': 'freq\\_IoU',
    'boundary_iou_mean': 'B-IoU',
    'mIoU_boundary_mean': 'mIoU@boundary',
    'mIoU_inner_mean': 'mIoU@inner',
    'segm_loss_mean': 'segm\\_loss',
}
# Columns reported as percentages (mean/std x100); segm_loss stays a raw decimal.
PCT_METRICS = {'mIoU', 'freq_IoU', 'boundary_iou_mean', 'mIoU_boundary_mean', 'mIoU_inner_mean'}
LOWER_IS_BETTER = {'segm_loss_mean'}

HEADLINE_ROW_ORDER = [
    'baseline', 'cbl', 'ptb', 'ptb_postproc', 'ptb_cbl', 'ptb_cbl_postproc',
    'ptb_brm', 'ptb_brm_postproc', 'ptb_brm_cbl', 'ptb_brm_cbl_postproc',
]
HEADLINE_ROW_LABELS = {
    'baseline': 'CloSeNet',
    'cbl': '+ CBL',
    'ptb': '+ PTB',
    'ptb_postproc': '+ PTB + SegFix',
    'ptb_cbl': '+ PTB + CBL',
    'ptb_cbl_postproc': '+ PTB + CBL + SegFix',
    'ptb_brm': '+ PTB + BRM',
    'ptb_brm_postproc': '+ PTB + BRM + SegFix',
    'ptb_brm_cbl': '+ PTB + BRM + CBL',
    'ptb_brm_cbl_postproc': '+ PTB + BRM + CBL + SegFix',
}

# Class index -> name, in label order. Source of truth: prep_scan.py's `label2class`
# array (not imported directly -- it pulls in pytorch3d/smplx, unwanted here).
CLASS_NAMES = [
    'Hat', 'Body', 'Shirt', 'TShirt', 'Vest', 'Coat', 'Dress', 'Skirt', 'Pants',
    'ShortPants', 'Shoes', 'Hoodies', 'Hair', 'Swimwear', 'Underwear', 'Scarf',
    'Jumpsuits', 'Jacket',
]

# Four-config ladder for the per-class analysis: increasing boundary supervision
# alongside CBL, ending at the paper's proposed method.
PER_CLASS_CONFIGS = ['baseline', 'cbl', 'ptb_cbl', 'ptb_brm_cbl']
TAIL_CLASSES = ['Dress', 'Jumpsuits', 'Scarf']
BEST_BOUNDARY_CONFIG = 'ptb_brm_cbl'

# The four PTB-family configs SegFix-style post-processing is evaluated against
# (each has a matching '<config>_postproc' eval, same checkpoint, paired by timestamp).
PTB_FAMILY_CONFIGS = ['ptb', 'ptb_cbl', 'ptb_brm', 'ptb_brm_cbl']

# Non-CLIP training-dynamics configs, same ladder as PER_CLASS_CONFIGS for visual
# consistency across the report.
TRAINING_DYNAMICS_CONFIGS = ['baseline', 'cbl', 'ptb_cbl', 'ptb_brm_cbl']
MAX_EPOCHS = 50  # training.max_epochs, shared by every training config in cfg/

TIMESTAMP_RE = re.compile(r'_(\d{8}_\d{6})_')
CHECKPOINT_EPOCH_RE = re.compile(r'valmin_(\d+)_')


def parse_filename(path: Path):
    name = path.name
    match = TIMESTAMP_RE.search(name)
    if not match:
        raise ValueError(f'Could not find a <YYYYMMDD_HHMMSS> timestamp in {name}')
    timestamp = match.group(1)

    prefix = next((p for p in ABLATION_PREFIXES if name.startswith(p + '_')), None)
    if prefix is None:
        raise ValueError(f'Could not match a known ablation prefix in {name}')

    is_postproc = 'postproc' in name
    ablation_id = prefix + ('_postproc' if is_postproc else '')
    return ablation_id, timestamp


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    groups = {}
    for path in sorted(results_dir.glob('*.json')):
        ablation_id, timestamp = parse_filename(path)
        groups.setdefault(ablation_id, []).append((timestamp, path))

    expected = set(HEADLINE_ROW_ORDER)
    missing = expected - groups.keys()
    unexpected = groups.keys() - expected
    if missing:
        raise ValueError(f'Missing result groups: {sorted(missing)}')
    if unexpected:
        raise ValueError(f'Unrecognized result groups: {sorted(unexpected)}')

    for ablation_id, entries in groups.items():
        if len(entries) != 3:
            raise ValueError(
                f'Expected exactly 3 seed runs for "{ablation_id}", found {len(entries)}: '
                f'{[p.name for _, p in entries]}'
            )
        # Sort by timestamp ascending: position 0/1/2 = seed 1/2/3, matched across
        # ablations because all 10 were submitted together per seed batch.
        entries.sort(key=lambda e: e[0])
        for seed_index, (timestamp, path) in enumerate(entries):
            data = json.loads(path.read_text())
            summary = data['summary']
            metrics = data['metrics']
            row = {'ablation_id': ablation_id, 'seed_index': seed_index, 'timestamp': timestamp}
            row.update({key: summary[key] for key in METRIC_KEYS})
            row['IoU_per_class'] = metrics['IoU_per_class']
            row['boundary_IoU_per_class'] = metrics['boundary_IoU_per_class']

            epoch_match = CHECKPOINT_EPOCH_RE.search(metrics['checkpoint'])
            if not epoch_match:
                raise ValueError(f'Could not parse checkpoint epoch from {metrics["checkpoint"]!r}')
            row['checkpoint_epoch'] = int(epoch_match.group(1))

            rows.append(row)

    return pd.DataFrame(rows)


def _fmt_cell(mean: float, std: float, metric: str, pm: str) -> str:
    if metric in PCT_METRICS:
        return f'{mean * 100:.2f} {pm} {std * 100:.2f}'
    return f'{mean:.4f} {pm} {std:.4f}'


def compute_headline_stats(df: pd.DataFrame):
    grouped = df.groupby('ablation_id')[METRIC_KEYS].agg(['mean', 'std'])

    best_ablation = {}
    for metric in METRIC_KEYS:
        means = grouped[(metric, 'mean')]
        best_ablation[metric] = means.idxmin() if metric in LOWER_IS_BETTER else means.idxmax()

    baseline_biou_mean = grouped.loc['baseline', ('boundary_iou_mean', 'mean')]
    return grouped, best_ablation, baseline_biou_mean


def paired_delta(df: pd.DataFrame, metric: str, group_a: str, group_b: str, pair_on: str = 'seed_index'):
    """mean/std of (group_a - group_b) on `metric`, paired row-by-row on `pair_on`.

    `pair_on='seed_index'` matches ablations by their seed batch (all 10 ablations
    were submitted together per seed, seed_index 0/1/2 = seed 1/2/3). `pair_on=
    'timestamp'` matches a config's postproc/non-postproc pair by their shared run
    timestamp (same trained checkpoint, evaluated twice) -- an exact, tighter pairing.
    """
    pivot = df[df['ablation_id'].isin([group_a, group_b])].pivot(
        index=pair_on, columns='ablation_id', values=metric
    )
    diff = pivot[group_a] - pivot[group_b]
    return float(diff.mean()), float(diff.std(ddof=1))


def _assert_seed_alignment(df: pd.DataFrame, df_clip: pd.DataFrame, runs_dir: Path) -> None:
    """Guards every cross-universe paired comparison in the headline table: that
    seed_index N means the same literal training seed in df and df_clip. Verifies
    this directly from each run's saved config.yaml `seed:` field rather than
    trusting timestamp order alone."""
    def _seed_of(ablation_id: str, timestamp: str, clip: bool) -> int:
        training_id = ablation_id[:-len('_postproc')] if ablation_id.endswith('_postproc') else ablation_id
        suffix = '_clip' if clip else ''
        config_path = runs_dir / f'{training_id}{suffix}' / timestamp / 'config.yaml'
        cfg = yaml.safe_load(config_path.read_text())
        return int(cfg['seed'])

    for ablation_id in HEADLINE_ROW_ORDER:
        base_rows = df[df['ablation_id'] == ablation_id].sort_values('seed_index')
        clip_rows = df_clip[df_clip['ablation_id'] == ablation_id].sort_values('seed_index')
        for (_, base_row), (_, clip_row) in zip(base_rows.iterrows(), clip_rows.iterrows()):
            base_seed = _seed_of(ablation_id, base_row['timestamp'], clip=False)
            clip_seed = _seed_of(ablation_id, clip_row['timestamp'], clip=True)
            if base_seed != clip_seed:
                raise ValueError(
                    f'Seed alignment broken for "{ablation_id}" seed_index='
                    f'{base_row["seed_index"]}: non-CLIP run used seed {base_seed}, '
                    f'CLIP run used seed {clip_seed}. Every cross-universe comparison '
                    f'assumes matching seed_index -> same literal seed.'
                )


def _clip_vs_noclip_biou_deltas(df: pd.DataFrame, df_clip: pd.DataFrame) -> dict:
    """Paired-seed (CLIP - non-CLIP) delta in boundary_iou_mean for each of the 10
    HEADLINE_ROW_ORDER ids. Builds a small, local combined frame (df_clip's
    ablation_id suffixed '_clip', concatenated with df) purely so paired_delta's pivot
    never sees a duplicate 'baseline'/'ptb'/... value; this frame is not returned or
    reused anywhere else."""
    df_clip_tagged = df_clip.copy()
    df_clip_tagged['ablation_id'] = df_clip_tagged['ablation_id'] + '_clip'
    combined = pd.concat([df, df_clip_tagged], ignore_index=True)

    return {
        ablation_id: paired_delta(combined, 'boundary_iou_mean', f'{ablation_id}_clip', ablation_id, pair_on='seed_index')
        for ablation_id in HEADLINE_ROW_ORDER
    }


def _clip_vs_noclip_all_deltas(df: pd.DataFrame, df_clip: pd.DataFrame) -> dict:
    """Paired-seed (CLIP - non-CLIP) delta for every METRIC_KEYS entry (not just
    boundary_iou_mean), for each of the 10 HEADLINE_ROW_ORDER ids. Generalizes
    _clip_vs_noclip_biou_deltas so claims like "CLIP wins on metric X for config Y"
    can be checked against the same paired 2-sigma convention as the B-IoU column,
    rather than just comparing raw best-per-column cells (which may be different,
    unpaired configs across the two campaign blocks)."""
    df_clip_tagged = df_clip.copy()
    df_clip_tagged['ablation_id'] = df_clip_tagged['ablation_id'] + '_clip'
    combined = pd.concat([df, df_clip_tagged], ignore_index=True)

    return {
        ablation_id: {
            metric: paired_delta(combined, metric, f'{ablation_id}_clip', ablation_id, pair_on='seed_index')
            for metric in METRIC_KEYS
        }
        for ablation_id in HEADLINE_ROW_ORDER
    }


def _fmt_effect(mean: float, std: float, pm: str, no_effect_phrase: str) -> str:
    if abs(mean) < 2 * std:
        return no_effect_phrase
    return f'{mean * 100:+.2f} {pm} {std * 100:.2f}'


def _fmt_paired_cell(mean: float, std: float, metric: str, pm: str, bold) -> str:
    """Like _fmt_cell, but for a signed paired delta: correct decimal scale per
    metric (segm_loss_mean is a raw decimal, not a percentage), and bolds the cell
    when the delta clears the report's 2-sigma detectability threshold, rather than
    replacing non-detectable cells with a phrase (too wide for a dense 6-metric
    table; the single-column Delta B-IoU cell in table_headline.tex can afford that,
    this table can't)."""
    if metric in PCT_METRICS:
        text = f'{mean * 100:+.2f} {pm} {std * 100:.2f}'
    else:
        text = f'{mean:+.4f} {pm} {std:.4f}'
    return bold(text) if abs(mean) >= 2 * std else text


def _combined_headline_row_cells(ablation_id: str, grouped, best_ablation: dict, delta, pm: str, bold) -> list:
    """delta is None for non-CLIP rows (no non-CLIP counterpart to pair against);
    otherwise a (mean, std) paired B-IoU delta vs. the same architecture's non-CLIP
    run, same seed."""
    cells = [HEADLINE_ROW_LABELS[ablation_id]]
    for metric in METRIC_KEYS:
        mean = grouped.loc[ablation_id, (metric, 'mean')]
        std = grouped.loc[ablation_id, (metric, 'std')]
        cell = _fmt_cell(mean, std, metric, pm)
        if best_ablation[metric] == ablation_id:
            cell = bold(cell)
        cells.append(cell)

    if delta is None:
        cells.append('--')
    else:
        delta_mean, delta_std = delta
        cells.append(_fmt_effect(delta_mean, delta_std, pm, 'no detectable effect at $n{=}3$ seeds'))
    return cells


def render_combined_headline_latex(grouped, best_ablation: dict, grouped_clip, best_ablation_clip: dict, clip_vs_noclip_deltas: dict) -> str:
    ncols = len(METRIC_KEYS) + 2
    lines = ['\\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + 'c}', '\\toprule']
    header = ['Model'] + [METRIC_LABELS[m] for m in METRIC_KEYS] + ['$\\Delta$ B-IoU vs.\\ non-CLIP']
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\midrule')

    lines.append(f'\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Non-CLIP campaign (random codebook init)}}}} \\\\')
    lines.append('\\midrule')
    for ablation_id in HEADLINE_ROW_ORDER:
        cells = _combined_headline_row_cells(
            ablation_id, grouped, best_ablation, None,
            pm='$\\pm$', bold=lambda c: f'\\textbf{{{c}}}',
        )
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{ncols}}}{{l}}{{\\textit{{CLIP campaign (CLIP-initialized codebook)}}}} \\\\')
    lines.append('\\midrule')
    for ablation_id in HEADLINE_ROW_ORDER:
        cells = _combined_headline_row_cells(
            ablation_id, grouped_clip, best_ablation_clip, clip_vs_noclip_deltas[ablation_id],
            pm='$\\pm$', bold=lambda c: f'\\textbf{{{c}}}',
        )
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def build_headline_table(df: pd.DataFrame, df_clip: pd.DataFrame, output_dir: Path, runs_dir: Path):
    """20-row headline table (10 non-CLIP + 10 CLIP configs), same 6 metrics, plus a
    paired Delta B-IoU column against each row's same-architecture non-CLIP counterpart
    (blank for the non-CLIP rows themselves, which have no such counterpart)."""
    _assert_seed_alignment(df, df_clip, runs_dir)

    grouped, best_ablation, _ = compute_headline_stats(df)
    grouped_clip, best_ablation_clip, _ = compute_headline_stats(df_clip)
    clip_vs_noclip_deltas = _clip_vs_noclip_biou_deltas(df, df_clip)

    output_path = output_dir / 'table_headline.tex'
    output_path.write_text(
        render_combined_headline_latex(grouped, best_ablation, grouped_clip, best_ablation_clip, clip_vs_noclip_deltas)
    )
    print(f'Wrote {output_path}')

    return best_ablation


def render_clip_paired_latex(deltas: dict) -> str:
    """10 configs x 6 metrics, each cell the paired (CLIP - non-CLIP) delta for that
    metric on that config, bolded where it clears the report's 2-sigma detectability
    threshold. Companion to table_headline.tex's single Delta B-IoU column -- this
    is what lets a claim like "CLIP wins on metric X" be checked against the same
    paired convention as B-IoU, instead of just comparing raw best-per-column cells
    across the two campaign blocks (which need not be the same config)."""
    lines = ['\\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + '}', '\\toprule']
    header = ['Model'] + [METRIC_LABELS[m] for m in METRIC_KEYS]
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\midrule')
    for ablation_id in HEADLINE_ROW_ORDER:
        cells = [HEADLINE_ROW_LABELS[ablation_id]]
        for metric in METRIC_KEYS:
            mean, std = deltas[ablation_id][metric]
            cells.append(_fmt_paired_cell(mean, std, metric, '$\\pm$', bold=lambda c: f'\\textbf{{{c}}}'))
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def build_clip_paired_table(df: pd.DataFrame, df_clip: pd.DataFrame, output_dir: Path) -> dict:
    """Paired (CLIP - non-CLIP) delta for all 6 metrics, all 10 configs. Seed
    alignment is already asserted by build_headline_table on this same df/df_clip
    pair, so it is not re-asserted here."""
    deltas = _clip_vs_noclip_all_deltas(df, df_clip)

    output_path = output_dir / 'table_clip_paired.tex'
    output_path.write_text(render_clip_paired_latex(deltas))
    print(f'Wrote {output_path}')

    return deltas


def _class_order_by_baseline_iou(df: pd.DataFrame) -> list:
    baseline = np.stack(df[df['ablation_id'] == 'baseline']['IoU_per_class'].to_numpy())
    order = np.argsort(-baseline.mean(axis=0))
    return [CLASS_NAMES[i] for i in order]


def build_per_class_heatmap(df: pd.DataFrame, output_dir: Path, suffix: str = '') -> list:
    """Per-class IoU heatmap across the 4-config ladder (baseline -> +CBL ->
    +PTB+CBL -> +PTB+BRM+CBL), rows sorted by baseline IoU descending."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_order = _class_order_by_baseline_iou(df)
    class_idx = [CLASS_NAMES.index(c) for c in class_order]

    heat = np.zeros((len(class_order), len(PER_CLASS_CONFIGS)))
    for j, ablation_id in enumerate(PER_CLASS_CONFIGS):
        per_class = np.stack(df[df['ablation_id'] == ablation_id]['IoU_per_class'].to_numpy())
        heat[:, j] = per_class.mean(axis=0)[class_idx]

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(heat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(PER_CLASS_CONFIGS)))
    ax.set_xticklabels([HEADLINE_ROW_LABELS[c] for c in PER_CLASS_CONFIGS], rotation=30, ha='right')
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(class_order)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f'{heat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=7)
    fig.colorbar(im, ax=ax, label='mean IoU')
    ax.set_title('Per-class IoU')
    fig.tight_layout()
    fig_path = figures_dir / f'per_class_heatmap{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    return class_order


def build_per_class_boundary_delta(df: pd.DataFrame, output_dir: Path, class_order: list, suffix: str = '') -> np.ndarray:
    """Per-class delta in boundary-region IoU (mIoU@boundary's per-class components),
    BEST_BOUNDARY_CONFIG minus baseline, ordered to match build_per_class_heatmap."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_idx = [CLASS_NAMES.index(c) for c in class_order]
    baseline = np.stack(df[df['ablation_id'] == 'baseline']['boundary_IoU_per_class'].to_numpy())
    best = np.stack(df[df['ablation_id'] == BEST_BOUNDARY_CONFIG]['boundary_IoU_per_class'].to_numpy())
    delta = (best.mean(axis=0) - baseline.mean(axis=0))[class_idx]

    colors = ['tab:green' if d >= 0 else 'tab:red' for d in delta]
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(range(len(class_order)), delta, color=colors)
    ax.set_yticks(range(len(class_order)))
    ax.set_yticklabels(class_order)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(f'{HEADLINE_ROW_LABELS[BEST_BOUNDARY_CONFIG]} $-$ baseline (boundary-region IoU)')
    ax.set_title('Per-class boundary-region IoU delta')
    fig.tight_layout()
    fig_path = figures_dir / f'per_class_boundary_delta{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    return delta


def build_tail_class_closeup(df: pd.DataFrame, output_dir: Path, suffix: str = '') -> None:
    """Per-seed boundary-region IoU for the 3 tail classes, baseline vs. best boundary
    config, paired by seed_index so same-seed movement is visible rather than hidden
    behind a mean-only bar."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(TAIL_CLASSES), figsize=(3 * len(TAIL_CLASSES), 4), sharey=True)
    for ax, class_name in zip(axes, TAIL_CLASSES):
        idx = CLASS_NAMES.index(class_name)
        pivot = df[df['ablation_id'].isin(['baseline', BEST_BOUNDARY_CONFIG])].copy()
        pivot['value'] = pivot['boundary_IoU_per_class'].apply(lambda a: a[idx])
        pivot = pivot.pivot(index='seed_index', columns='ablation_id', values='value')

        for seed_index, row in pivot.iterrows():
            ax.plot([0, 1], [row['baseline'], row[BEST_BOUNDARY_CONFIG]], color='gray', linewidth=0.8, zorder=1)
        ax.scatter([0] * len(pivot), pivot['baseline'], color='tab:blue', zorder=2, label='baseline')
        ax.scatter([1] * len(pivot), pivot[BEST_BOUNDARY_CONFIG], color='tab:orange', zorder=2, label=HEADLINE_ROW_LABELS[BEST_BOUNDARY_CONFIG])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['baseline', HEADLINE_ROW_LABELS[BEST_BOUNDARY_CONFIG]], rotation=20, ha='right')
        ax.set_title(class_name)
        ax.set_xlim(-0.3, 1.3)

    axes[0].set_ylabel('boundary-region IoU')
    axes[0].legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig_path = figures_dir / f'tail_class_closeup{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')


def build_variance_scatter(df: pd.DataFrame, output_dir: Path, suffix: str = '') -> float:
    """Baseline per-class mean IoU vs. per-class seed-to-seed std: the 'ceiling'
    story -- classes already close to 1.0 have little room left to vary."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline = np.stack(df[df['ablation_id'] == 'baseline']['IoU_per_class'].to_numpy())
    means = baseline.mean(axis=0)
    stds = baseline.std(axis=0, ddof=1)
    corr = float(np.corrcoef(means, stds)[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(means, stds, color='tab:blue')
    ax.set_xlabel('baseline mean IoU (per class)')
    ax.set_ylabel('baseline seed std (per class)')
    ax.set_title(f'Mean vs. seed variance across classes ($r$={corr:.2f})')
    fig.tight_layout()
    fig_path = figures_dir / f'variance_scatter{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    return corr


def build_boundary_inner_tradeoff_scatter(df: pd.DataFrame, df_clip: pd.DataFrame, output_dir: Path) -> None:
    """Mean Delta-mIoU@inner vs. mean Delta-mIoU@boundary, each non-baseline config
    vs. its own campaign's baseline (raw mean differences -- descriptive, not a
    paired significance test, same scope as the per-class heatmap/variance scatter).
    Makes Section 5's "gain inner, hold boundary" claim visible at a glance."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    markers = {'non-CLIP': 'o', 'CLIP': '^'}
    colors = {'non-CLIP': 'tab:blue', 'CLIP': 'tab:orange'}
    for label, frame in [('non-CLIP', df), ('CLIP', df_clip)]:
        grouped = frame.groupby('ablation_id')[['mIoU_inner_mean', 'mIoU_boundary_mean']].mean()
        baseline_inner = grouped.loc['baseline', 'mIoU_inner_mean']
        baseline_boundary = grouped.loc['baseline', 'mIoU_boundary_mean']
        configs = [c for c in HEADLINE_ROW_ORDER if c != 'baseline']
        delta_inner = grouped.loc[configs, 'mIoU_inner_mean'] - baseline_inner
        delta_boundary = grouped.loc[configs, 'mIoU_boundary_mean'] - baseline_boundary
        ax.scatter(delta_boundary * 100, delta_inner * 100, marker=markers[label], color=colors[label], label=label, s=50)

    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(r'$\Delta$ mIoU@boundary vs. own baseline (pp)')
    ax.set_ylabel(r'$\Delta$ mIoU@inner vs. own baseline (pp)')
    ax.set_title('Boundary vs. interior movement, per config')
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / 'boundary_inner_tradeoff.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')


def render_segfix_latex(deltas: dict) -> str:
    """Shows the raw paired delta always (bolded only if it clears 2-sigma), rather
    than replacing non-detectable cells with a text phrase -- unlike table_headline's
    single Delta B-IoU column, this table's whole point is to let the reader see the
    (non-detectable) magnitude of the SegFix trend, not just whether it's detectable."""
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Config & Non-CLIP $\\Delta$ B-IoU & CLIP $\\Delta$ B-IoU \\\\', '\\midrule']
    for config in PTB_FAMILY_CONFIGS:
        noclip_mean, noclip_std = deltas['non-CLIP'][config]
        clip_mean, clip_std = deltas['CLIP'][config]
        noclip_cell = _fmt_paired_cell(noclip_mean, noclip_std, 'boundary_iou_mean', '$\\pm$', bold=lambda c: f'\\textbf{{{c}}}')
        clip_cell = _fmt_paired_cell(clip_mean, clip_std, 'boundary_iou_mean', '$\\pm$', bold=lambda c: f'\\textbf{{{c}}}')
        lines.append(f'{HEADLINE_ROW_LABELS[config]} & {noclip_cell} & {clip_cell} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def build_segfix_table(df: pd.DataFrame, df_clip: pd.DataFrame, output_dir: Path) -> dict:
    """Paired-by-timestamp (same checkpoint, with vs. without SegFix post-processing)
    B-IoU delta for the 4 PTB-family configs, both campaigns. pair_on='timestamp' is
    the tighter pairing paired_delta already supports (exact same trained checkpoint,
    evaluated twice), not seed_index."""
    deltas = {
        'non-CLIP': {
            config: paired_delta(df, 'boundary_iou_mean', f'{config}_postproc', config, pair_on='timestamp')
            for config in PTB_FAMILY_CONFIGS
        },
        'CLIP': {
            config: paired_delta(df_clip, 'boundary_iou_mean', f'{config}_postproc', config, pair_on='timestamp')
            for config in PTB_FAMILY_CONFIGS
        },
    }

    output_path = output_dir / 'table_segfix.tex'
    output_path.write_text(render_segfix_latex(deltas))
    print(f'Wrote {output_path}')

    return deltas


def _average_curve(runs: list):
    """runs: list of (steps, values) from seed-matched runs sharing the same step axis."""
    if not runs:
        return None
    steps = runs[0][0]
    values = np.stack([v for _, v in runs])  # (n_seeds, n_steps)
    return steps, values.mean(axis=0), values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(values[0])


def build_training_dynamics(runs_dir: Path, output_dir: Path) -> None:
    """Val loss and val B-IoU over training, mean +/- seed-std ribbon, for the
    non-CLIP TRAINING_DYNAMICS_CONFIGS ladder. Reads TensorBoard summaries directly
    (no checkpoint files needed)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    tags = ['val/total_loss', 'val/boundary_iou']
    titles = ['Val loss', 'Val B-IoU']

    curves = {}
    for ablation_id in TRAINING_DYNAMICS_CONFIGS:
        ablation_dir = runs_dir / ablation_id
        run_dirs = sorted(p for p in ablation_dir.iterdir() if p.is_dir())
        curves[ablation_id] = {tag: [] for tag in tags}
        for run_dir in run_dirs:
            ea = EventAccumulator(str(run_dir / 'summary'), size_guidance={'scalars': 0})
            ea.Reload()
            available = set(ea.Tags().get('scalars', []))
            for tag in tags:
                if tag not in available:
                    continue
                events = ea.Scalars(tag)
                curves[ablation_id][tag].append((
                    np.array([e.step for e in events]),
                    np.array([e.value for e in events]),
                ))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, tag, title in zip(axes, tags, titles):
        for ablation_id in TRAINING_DYNAMICS_CONFIGS:
            averaged = _average_curve(curves[ablation_id][tag])
            if averaged is None:
                continue
            steps, mean, std = averaged
            line, = ax.plot(steps, mean, label=HEADLINE_ROW_LABELS[ablation_id])
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel('iteration')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(TRAINING_DYNAMICS_CONFIGS), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig_path = figures_dir / 'training_dynamics.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')


def render_best_epoch_latex(epoch_stats) -> str:
    lines = ['\\begin{tabular}{lc}', '\\toprule', 'Model & Best-checkpoint epoch \\\\', '\\midrule']
    any_flagged = False
    for ablation_id in TRAINING_DYNAMICS_CONFIGS:
        mean = epoch_stats.loc[ablation_id, 'mean']
        std = epoch_stats.loc[ablation_id, 'std']
        is_flagged = mean >= 0.9 * MAX_EPOCHS
        any_flagged = any_flagged or is_flagged
        flag = '$^\\dagger$' if is_flagged else ''
        lines.append(f'{HEADLINE_ROW_LABELS[ablation_id]} & {mean:.1f} $\\pm$ {std:.1f}{flag} \\\\')
    lines.append('\\bottomrule')
    if any_flagged:
        lines.append(
            f'\\multicolumn{{2}}{{l}}{{\\footnotesize $^\\dagger$mean epoch $\\geq$ 90\\% of '
            f'max\\_epochs ({MAX_EPOCHS}); may not have fully converged.}} \\\\'
        )
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def build_best_epoch_table(df: pd.DataFrame, output_dir: Path) -> dict:
    """Mean +/- std best-checkpoint epoch (non-CLIP), for the same 4-config ladder as
    the training-dynamics figure. Flags configs whose mean checkpoint epoch is late in
    the fixed MAX_EPOCHS budget -- grounds the under-training caveat named in
    Discussion, rather than asserting convergence without evidence."""
    epoch_stats = df.groupby('ablation_id')['checkpoint_epoch'].agg(['mean', 'std'])

    output_path = output_dir / 'table_best_epoch.tex'
    output_path.write_text(render_best_epoch_latex(epoch_stats))
    print(f'Wrote {output_path}')

    flagged = [
        ablation_id for ablation_id in TRAINING_DYNAMICS_CONFIGS
        if epoch_stats.loc[ablation_id, 'mean'] >= 0.9 * MAX_EPOCHS
    ]
    return {'epoch_stats': epoch_stats, 'flagged': flagged}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--results-clip-dir', default='results_clip')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_clip_dir = Path(args.results_clip_dir)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    df_clip = load_results(results_clip_dir)
    build_headline_table(df, df_clip, output_dir, runs_dir)
    build_clip_paired_table(df, df_clip, output_dir)

    for frame, suffix in [(df, ''), (df_clip, '_clip')]:
        class_order = build_per_class_heatmap(frame, output_dir, suffix)
        build_per_class_boundary_delta(frame, output_dir, class_order, suffix)
        build_tail_class_closeup(frame, output_dir, suffix)
        build_variance_scatter(frame, output_dir, suffix)

    build_boundary_inner_tradeoff_scatter(df, df_clip, output_dir)
    build_segfix_table(df, df_clip, output_dir)
    build_training_dynamics(runs_dir, output_dir)
    build_best_epoch_table(df, output_dir)


if __name__ == '__main__':
    main()
