#!/usr/bin/env python3
"""Turn the per-run result JSONs from evaluate_run.sbatch into a paper-ready report.

Reads every results/<ablation>_<timestamp>_<eval-config-stem>.json (30 files: 10
ablations x 3 seeds) plus the TensorBoard event files under
runs/<ablation>/<timestamp>/summary/, and writes one table/figure pair per section:

  Section 1 (headline table):         table_headline.tex
  Section 2 (CBL x PTB factorial):    table_factorial_effects.tex, figures/interaction_cbl_ptb.png
  Section 3 (BRM results):            table_brm_pairwise.tex, figures/brm_grouped_bars.png
  Section 4 (SegFix effect):          table_segfix_effect.tex, figures/segfix_delta.png
  Section 5 (per-class breakdown):    figures/per_class_heatmap.png, figures/per_class_delta.png
  Section 6 (training dynamics):      table_best_epoch.tex, figures/training_dynamics.png
  Section 7 (boundary/inner tradeoff): figures/boundary_inner_scatter.png

  reports/report.tex   Standalone document \\input-ing/\\includegraphics-ing all of the
                        above (plus data-driven prose per section), for a quick
                        `pdflatex report.tex` view.

Usage:
    python report.py [--results-dir results] [--runs-dir runs] [--output-dir reports]
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

FACTORIAL_CELLS = ['baseline', 'cbl', 'ptb', 'ptb_cbl']
PTB_FAMILY_CELLS = ['ptb', 'ptb_cbl', 'ptb_brm', 'ptb_brm_cbl']

# Class index -> name, in label order. Source of truth: prep_scan.py's `label2class`
# array (not imported directly -- it pulls in pytorch3d/smplx, unwanted here).
CLASS_NAMES = [
    'Hat', 'Body', 'Shirt', 'TShirt', 'Vest', 'Coat', 'Dress', 'Skirt', 'Pants',
    'ShortPants', 'Shoes', 'Hoodies', 'Hair', 'Swimwear', 'Underwear', 'Scarf',
    'Jumpsuits', 'Jacket',
]

# Section 8: the "tail classes" the CLIP-init narrative (semantically related garments
# starting closer together in codebook space) is expected to help most.
TAIL_CLASSES = ['Dress', 'Jumpsuits', 'Scarf']

# Placeholder for a per-class boundary-region metric that evaluate_closenet_ckpts.py
# doesn't save yet (a follow-up task will add it, likely by exposing the per-class
# breakdown boundary_inner_mIoU() already computes internally -- see
# lib/utils/metrics.py:246 -- before it's averaged away via .mean(-1)). Reading this
# key is optional: it's None for every row until that follow-up lands, and
# section5_per_class() skips the B-IoU delta chart cleanly when it's unavailable.
PER_CLASS_BOUNDARY_KEY = 'boundary_IoU_per_class'

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
            row['boundary_IoU_per_class'] = metrics.get(PER_CLASS_BOUNDARY_KEY)  # None until available

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


def _headline_row_cells(ablation_id: str, grouped, best_ablation: dict, baseline_biou_mean: float, pm: str, bold):
    cells = [HEADLINE_ROW_LABELS[ablation_id]]
    for metric in METRIC_KEYS:
        mean = grouped.loc[ablation_id, (metric, 'mean')]
        std = grouped.loc[ablation_id, (metric, 'std')]
        cell = _fmt_cell(mean, std, metric, pm)
        if best_ablation[metric] == ablation_id:
            cell = bold(cell)
        cells.append(cell)

    if ablation_id == 'baseline':
        cells.append('--')
    else:
        delta = (grouped.loc[ablation_id, ('boundary_iou_mean', 'mean')] - baseline_biou_mean) * 100
        cells.append(f'{delta:+.2f}')
    return cells


def render_headline_latex(grouped, best_ablation: dict, baseline_biou_mean: float) -> str:
    lines = ['\\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + 'c}', '\\toprule']
    header = ['Model'] + [METRIC_LABELS[m] for m in METRIC_KEYS] + ['$\\Delta$ B-IoU vs.\\ baseline']
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\midrule')
    for ablation_id in HEADLINE_ROW_ORDER:
        cells = _headline_row_cells(
            ablation_id, grouped, best_ablation, baseline_biou_mean,
            pm='$\\pm$', bold=lambda c: f'\\textbf{{{c}}}',
        )
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def section1_headline_table(df: pd.DataFrame, output_dir: Path):
    grouped, best_ablation, baseline_biou_mean = compute_headline_stats(df)

    output_path = output_dir / 'table_headline.tex'
    output_path.write_text(render_headline_latex(grouped, best_ablation, baseline_biou_mean))
    print(f'Wrote {output_path}')

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


def _paired_effects(df: pd.DataFrame, metric: str) -> dict:
    pivot = df[df['ablation_id'].isin(FACTORIAL_CELLS)].pivot(
        index='seed_index', columns='ablation_id', values=metric
    )
    baseline, cbl, ptb, ptb_cbl = pivot['baseline'], pivot['cbl'], pivot['ptb'], pivot['ptb_cbl']

    cbl_effect = 0.5 * (cbl + ptb_cbl) - 0.5 * (baseline + ptb)
    ptb_effect = 0.5 * (ptb + ptb_cbl) - 0.5 * (baseline + cbl)
    interaction = (ptb_cbl - ptb) - (cbl - baseline)

    def summarize(effect_per_seed):
        mean = float(effect_per_seed.mean())
        std = float(effect_per_seed.std(ddof=1))
        return mean, std

    return {
        'CBL main effect': summarize(cbl_effect),
        'PTB main effect': summarize(ptb_effect),
        'Interaction': summarize(interaction),
    }


def _fmt_effect(mean: float, std: float, pm: str, no_effect_phrase: str) -> str:
    if abs(mean) < 2 * std:
        return no_effect_phrase
    return f'{mean * 100:+.2f} {pm} {std * 100:.2f}'


def render_factorial_latex(biou_effects: dict, miou_effects: dict) -> str:
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Effect & B-IoU & mIoU \\\\', '\\midrule']
    for effect_name in ['CBL main effect', 'PTB main effect', 'Interaction']:
        biou_mean, biou_std = biou_effects[effect_name]
        miou_mean, miou_std = miou_effects[effect_name]
        biou_cell = _fmt_effect(biou_mean, biou_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        miou_cell = _fmt_effect(miou_mean, miou_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        lines.append(f'{effect_name} & {biou_cell} & {miou_cell} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def section2_factorial(df: pd.DataFrame, output_dir: Path):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    cell_stats = (
        df[df['ablation_id'].isin(FACTORIAL_CELLS)]
        .groupby('ablation_id')[['boundary_iou_mean', 'mIoU']]
        .agg(['mean', 'std'])
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = [0, 1]  # no PTB, +PTB
    x_labels = ['no PTB', '+PTB']

    for ax, metric, title in zip(axes, ['boundary_iou_mean', 'mIoU'], ['B-IoU', 'mIoU']):
        no_cbl = [cell_stats.loc['baseline', (metric, 'mean')], cell_stats.loc['ptb', (metric, 'mean')]]
        no_cbl_err = [cell_stats.loc['baseline', (metric, 'std')], cell_stats.loc['ptb', (metric, 'std')]]
        with_cbl = [cell_stats.loc['cbl', (metric, 'mean')], cell_stats.loc['ptb_cbl', (metric, 'mean')]]
        with_cbl_err = [cell_stats.loc['cbl', (metric, 'std')], cell_stats.loc['ptb_cbl', (metric, 'std')]]

        ax.errorbar(x, no_cbl, yerr=no_cbl_err, marker='o', capsize=4, label='no CBL')
        ax.errorbar(x, with_cbl, yerr=with_cbl_err, marker='s', capsize=4, label='+CBL')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(title)
        ax.set_title(f'CBL x PTB interaction ({title})')
        ax.legend()

    fig.tight_layout()
    fig_path = figures_dir / 'interaction_cbl_ptb.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    biou_effects = _paired_effects(df, 'boundary_iou_mean')
    miou_effects = _paired_effects(df, 'mIoU')

    output_path = output_dir / 'table_factorial_effects.tex'
    output_path.write_text(render_factorial_latex(biou_effects, miou_effects))
    print(f'Wrote {output_path}')

    return biou_effects, miou_effects


BRM_LABELS = {'ptb': 'PTB', 'ptb_cbl': 'PTB+CBL', 'ptb_brm': 'PTB+BRM', 'ptb_brm_cbl': 'PTB+BRM+CBL'}


def render_pairwise_latex(rows: list) -> str:
    """rows: list of (label, biou_mean, biou_std, miou_mean, miou_std)."""
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Comparison & B-IoU & mIoU \\\\', '\\midrule']
    for label, biou_mean, biou_std, miou_mean, miou_std in rows:
        biou_cell = _fmt_effect(biou_mean, biou_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        miou_cell = _fmt_effect(miou_mean, miou_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        lines.append(f'{label} & {biou_cell} & {miou_cell} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def section3_brm(df: pd.DataFrame, output_dir: Path):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    cell_stats = (
        df[df['ablation_id'].isin(PTB_FAMILY_CELLS)]
        .groupby('ablation_id')[['mIoU', 'boundary_iou_mean']]
        .agg(['mean', 'std'])
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(PTB_FAMILY_CELLS))
    width = 0.35

    miou_means = [cell_stats.loc[c, ('mIoU', 'mean')] for c in PTB_FAMILY_CELLS]
    miou_stds = [cell_stats.loc[c, ('mIoU', 'std')] for c in PTB_FAMILY_CELLS]
    biou_means = [cell_stats.loc[c, ('boundary_iou_mean', 'mean')] for c in PTB_FAMILY_CELLS]
    biou_stds = [cell_stats.loc[c, ('boundary_iou_mean', 'std')] for c in PTB_FAMILY_CELLS]

    ax.bar(x - width / 2, miou_means, width, yerr=miou_stds, capsize=4, label='mIoU')
    ax.bar(x + width / 2, biou_means, width, yerr=biou_stds, capsize=4, label='B-IoU')
    ax.set_xticks(x)
    ax.set_xticklabels([BRM_LABELS[c] for c in PTB_FAMILY_CELLS])
    ax.set_ylabel('Score')
    ax.set_title('BRM ablation: mIoU and B-IoU per config')
    ax.legend()

    fig.tight_layout()
    fig_path = figures_dir / 'brm_grouped_bars.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    rows = []
    for other in ['ptb', 'ptb_cbl', 'ptb_brm']:
        biou_mean, biou_std = paired_delta(df, 'boundary_iou_mean', 'ptb_brm_cbl', other)
        miou_mean, miou_std = paired_delta(df, 'mIoU', 'ptb_brm_cbl', other)
        rows.append((f'PTB+BRM+CBL vs.\\ {BRM_LABELS[other]}', biou_mean, biou_std, miou_mean, miou_std))

    output_path = output_dir / 'table_brm_pairwise.tex'
    output_path.write_text(render_pairwise_latex(rows))
    print(f'Wrote {output_path}')

    return rows


def section4_segfix(df: pd.DataFrame, output_dir: Path):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    deltas = {}
    for config in PTB_FAMILY_CELLS:
        biou = paired_delta(df, 'boundary_iou_mean', f'{config}_postproc', config, pair_on='timestamp')
        miou = paired_delta(df, 'mIoU', f'{config}_postproc', config, pair_on='timestamp')
        deltas[config] = {'boundary_iou_mean': biou, 'mIoU': miou}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(PTB_FAMILY_CELLS))
    means = [deltas[c]['boundary_iou_mean'][0] * 100 for c in PTB_FAMILY_CELLS]
    stds = [deltas[c]['boundary_iou_mean'][1] * 100 for c in PTB_FAMILY_CELLS]

    ax.errorbar(x, means, yerr=stds, marker='o', capsize=4, linestyle='-')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BRM_LABELS[c] for c in PTB_FAMILY_CELLS])
    ax.set_ylabel('B-IoU improvement from SegFix (pp)')
    ax.set_title('SegFix post-processing effect (paired by run)')

    fig.tight_layout()
    fig_path = figures_dir / 'segfix_delta.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    rows = []
    for config in PTB_FAMILY_CELLS:
        biou_mean, biou_std = deltas[config]['boundary_iou_mean']
        miou_mean, miou_std = deltas[config]['mIoU']
        rows.append((f'{BRM_LABELS[config]} + SegFix vs.\\ {BRM_LABELS[config]}', biou_mean, biou_std, miou_mean, miou_std))

    output_path = output_dir / 'table_segfix_effect.tex'
    output_path.write_text(render_pairwise_latex(rows))
    print(f'Wrote {output_path}')

    return deltas


PER_CLASS_HEATMAP_CONFIGS = ['baseline', 'ptb', 'ptb_cbl', 'ptb_brm', 'ptb_brm_cbl']


def _plot_per_class_delta(delta: np.ndarray, class_order_names: list, fig_path: Path, xlabel: str, title: str) -> None:
    colors = ['tab:green' if d >= 0 else 'tab:red' for d in delta]
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(range(len(class_order_names)), delta, color=colors)
    ax.set_yticks(range(len(class_order_names)))
    ax.set_yticklabels(class_order_names)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def section5_per_class(df: pd.DataFrame, output_dir: Path, best_ablation: dict):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_class = np.stack(df['IoU_per_class'].to_numpy())  # (n_rows, 18)
    for i, class_name in enumerate(CLASS_NAMES):
        df[f'class_{class_name}'] = per_class[:, i]

    class_cols = [f'class_{c}' for c in CLASS_NAMES]
    grouped = df.groupby('ablation_id')[class_cols].agg(['mean', 'std'])

    baseline_means = grouped.loc['baseline'].xs('mean', level=1)
    class_order = baseline_means.sort_values(ascending=False).index.tolist()
    class_order_names = [c.replace('class_', '') for c in class_order]

    # Plot D: heatmap, rows sorted by baseline IoU descending, curated config columns.
    heat = np.array([
        [grouped.loc[config, (col, 'mean')] for config in PER_CLASS_HEATMAP_CONFIGS]
        for col in class_order
    ])

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(heat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(PER_CLASS_HEATMAP_CONFIGS)))
    ax.set_xticklabels([BRM_LABELS.get(c, HEADLINE_ROW_LABELS[c]) for c in PER_CLASS_HEATMAP_CONFIGS], rotation=30, ha='right')
    ax.set_yticks(range(len(class_order_names)))
    ax.set_yticklabels(class_order_names)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f'{heat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=7)
    fig.colorbar(im, ax=ax, label='mean IoU')
    ax.set_title('Per-class IoU (rows sorted by baseline IoU, descending)')
    fig.tight_layout()
    fig_path = figures_dir / 'per_class_heatmap.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {fig_path}')

    # Plot E: delta bar chart, best-mIoU config (from Section 1) minus baseline.
    best_config = best_ablation['mIoU']
    delta = np.array([
        grouped.loc[best_config, (col, 'mean')] - grouped.loc['baseline', (col, 'mean')]
        for col in class_order
    ])
    fig_path = figures_dir / 'per_class_delta.png'
    _plot_per_class_delta(
        delta, class_order_names, fig_path,
        xlabel=f'{HEADLINE_ROW_LABELS[best_config]} $-$ baseline (IoU)',
        title='Per-class IoU delta: best config vs. baseline',
    )
    print(f'Wrote {fig_path}')

    worse_classes = [name for name, d in zip(class_order_names, delta) if d < 0]

    baseline_std = grouped.loc['baseline'].xs('std', level=1)
    variance_corr = float(np.corrcoef(baseline_means.to_numpy(), baseline_std.to_numpy())[0, 1])

    # Plot E2: same delta chart for per-class B-IoU, once that data exists in the
    # result JSONs (see PER_CLASS_BOUNDARY_KEY) -- skipped cleanly until then.
    biou_available = df['boundary_IoU_per_class'].notna().all()
    biou_worse_classes = []
    if biou_available:
        biou_per_class = np.stack(df['boundary_IoU_per_class'].to_numpy())
        for i, class_name in enumerate(CLASS_NAMES):
            df[f'biou_class_{class_name}'] = biou_per_class[:, i]
        biou_class_cols = [f'biou_class_{c}' for c in CLASS_NAMES]
        biou_grouped = df.groupby('ablation_id')[biou_class_cols].agg('mean')

        biou_delta = np.array([
            biou_grouped.loc[best_config, f'biou_class_{name}'] - biou_grouped.loc['baseline', f'biou_class_{name}']
            for name in class_order_names
        ])
        fig_path = figures_dir / 'per_class_delta_biou.png'
        _plot_per_class_delta(
            biou_delta, class_order_names, fig_path,
            xlabel=f'{HEADLINE_ROW_LABELS[best_config]} $-$ baseline (B-IoU)',
            title='Per-class B-IoU delta: best config vs. baseline',
        )
        print(f'Wrote {fig_path}')
        biou_worse_classes = [name for name, d in zip(class_order_names, biou_delta) if d < 0]
    else:
        print(
            f'Skipping per-class B-IoU delta chart: "{PER_CLASS_BOUNDARY_KEY}" not present '
            f'in the result JSONs yet (planned follow-up to evaluate_closenet_ckpts.py).'
        )

    return {
        'best_config': best_config,
        'worse_classes': worse_classes,
        'variance_corr': variance_corr,
        'biou_available': biou_available,
        'biou_worse_classes': biou_worse_classes,
    }


TB_SCALAR_TAGS = ['val/total_loss', 'val/mIoU', 'val/boundary_iou', 'train/total_loss', 'val/cbl_loss']
TRAINING_DYNAMICS_CONFIGS = ['baseline', 'cbl', 'ptb_cbl', 'ptb_brm_cbl']
MAX_EPOCHS = 50  # training.max_epochs, shared by every training config in cfg/


def load_tb_curves(runs_dir: Path, ablation_ids: list) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    curves = {}  # {ablation_id: {tag: [(steps, values) per seed run]}}
    for ablation_id in ablation_ids:
        ablation_dir = runs_dir / ablation_id
        run_dirs = sorted(p for p in ablation_dir.iterdir() if p.is_dir())
        curves[ablation_id] = {tag: [] for tag in TB_SCALAR_TAGS}
        for run_dir in run_dirs:
            summary_dir = run_dir / 'summary'
            ea = EventAccumulator(str(summary_dir), size_guidance={'scalars': 0})
            ea.Reload()
            available_tags = set(ea.Tags().get('scalars', []))
            for tag in TB_SCALAR_TAGS:
                if tag not in available_tags:
                    continue
                events = ea.Scalars(tag)
                steps = np.array([e.step for e in events])
                values = np.array([e.value for e in events])
                curves[ablation_id][tag].append((steps, values))
    return curves


def _average_curve(runs: list):
    """runs: list of (steps, values) from seed-matched runs sharing the same step axis."""
    if not runs:
        return None
    steps = runs[0][0]
    values = np.stack([v for _, v in runs])  # (n_seeds, n_steps)
    return steps, values.mean(axis=0), values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(values[0])


def section6_training_dynamics(df: pd.DataFrame, runs_dir: Path, output_dir: Path):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    curves = load_tb_curves(runs_dir, TRAINING_DYNAMICS_CONFIGS)

    plot_tags = ['val/total_loss', 'val/mIoU', 'val/boundary_iou', 'train/total_loss']
    plot_titles = ['Val loss', 'Val mIoU', 'Val B-IoU', 'Train loss']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, tag, title in zip(axes.flat, plot_tags, plot_titles):
        for ablation_id in TRAINING_DYNAMICS_CONFIGS:
            averaged = _average_curve(curves[ablation_id][tag])
            if averaged is None:
                continue
            steps, mean, std = averaged
            label = HEADLINE_ROW_LABELS[ablation_id]
            line, = ax.plot(steps, mean, label=label)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel('iteration')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(TRAINING_DYNAMICS_CONFIGS), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig_path = figures_dir / 'training_dynamics.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')

    cbl_trend = None
    cbl_runs = curves.get('cbl', {}).get('val/cbl_loss', [])
    if cbl_runs:
        averaged = _average_curve(cbl_runs)
        if averaged is not None:
            _, mean, _ = averaged
            cbl_trend = float(mean[0] - mean[-1])  # positive => loss decreased

    # Best-epoch table: all 10 ablations, from checkpoint filenames already loaded
    # (no TensorBoard access needed for this part).
    epoch_stats = df.groupby('ablation_id')['checkpoint_epoch'].agg(['mean', 'std'])
    rows = []
    for ablation_id in HEADLINE_ROW_ORDER:
        mean = epoch_stats.loc[ablation_id, 'mean']
        std = epoch_stats.loc[ablation_id, 'std']
        flag = '$^\\dagger$' if mean >= 0.9 * MAX_EPOCHS else ''
        rows.append((HEADLINE_ROW_LABELS[ablation_id], mean, std, flag))

    lines = ['\\begin{tabular}{lc}', '\\toprule', 'Model & Best-checkpoint epoch \\\\', '\\midrule']
    for label, mean, std, flag in rows:
        lines.append(f'{label} & {mean:.1f} $\\pm$ {std:.1f}{flag} \\\\')
    lines.append('\\bottomrule')
    lines.append(
        f'\\multicolumn{{2}}{{l}}{{\\footnotesize $^\\dagger$mean epoch $\\geq$ 90\\% of '
        f'max\\_epochs ({MAX_EPOCHS}); may benefit from longer training.}} \\\\'
    )
    lines.append('\\end{tabular}')

    output_path = output_dir / 'table_best_epoch.tex'
    output_path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {output_path}')

    flagged = [label for label, mean, _, flag in rows if flag]
    return {'cbl_trend': cbl_trend, 'flagged_configs': flagged}


def section7_boundary_inner_scatter(df: pd.DataFrame, output_dir: Path):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap('tab20')
    for i, ablation_id in enumerate(HEADLINE_ROW_ORDER):
        subset = df[df['ablation_id'] == ablation_id]
        ax.scatter(
            subset['mIoU_inner_mean'], subset['mIoU_boundary_mean'],
            color=cmap(i), label=HEADLINE_ROW_LABELS[ablation_id], s=40,
        )

    ax.set_xlabel('mIoU@inner')
    ax.set_ylabel('mIoU@boundary')
    ax.set_title('Boundary vs. inner accuracy tradeoff (one point per config/seed)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig_path = figures_dir / 'boundary_inner_scatter.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_path}')


# ---------------------------------------------------------------------------
# Section 8: CLIP-initialized codebook. A self-contained extension that reuses the
# generic helpers above (paired_delta, _paired_effects, _plot_per_class_delta,
# compute_headline_stats, _fmt_cell, _fmt_effect, _headline_row_cells) on a second,
# independently-loaded DataFrame (results_clip/). Sections 1-7's own output files are
# never re-written by anything below -- section3_brm/section4_segfix are deliberately
# never called a second time; their underlying comparisons are recomputed standalone.
# ---------------------------------------------------------------------------

THREE_WAY_LABELS = {
    'A': 'CloSeNet + CLIP',
    'B': 'CloSeNet + PTB + BRM + CBL',
    'C': 'CloSeNet + PTB + BRM + CBL + CLIP',
}


def _assert_seed_alignment(df: pd.DataFrame, df_clip: pd.DataFrame, runs_dir: Path) -> None:
    """Guards every cross-universe paired comparison in Section 8: that seed_index N
    means the same literal training seed in df and df_clip. Verifies this directly
    from each run's saved config.yaml `seed:` field rather than trusting timestamp
    order alone."""
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
                    f'CLIP run used seed {clip_seed}. Every Section 8 cross-universe '
                    f'comparison assumes matching seed_index -> same literal seed.'
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


def _clip_headline_row_cells(ablation_id, grouped_clip, best_ablation_clip, baseline_biou_mean_clip, clip_vs_noclip_deltas, pm, bold):
    cells = _headline_row_cells(ablation_id, grouped_clip, best_ablation_clip, baseline_biou_mean_clip, pm, bold)
    delta_mean, delta_std = clip_vs_noclip_deltas[ablation_id]
    cells.append(_fmt_effect(delta_mean, delta_std, pm, 'no detectable effect at $n{=}3$ seeds'))
    return cells


def render_clip_headline_latex(grouped_clip, best_ablation_clip, baseline_biou_mean_clip, clip_vs_noclip_deltas) -> str:
    lines = ['\\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + 'cc}', '\\toprule']
    header = (
        ['Model'] + [METRIC_LABELS[m] for m in METRIC_KEYS]
        + ['$\\Delta$ B-IoU vs.\\ CLIP baseline', '$\\Delta$ B-IoU vs.\\ non-CLIP']
    )
    lines.append(' & '.join(header) + ' \\\\')
    lines.append('\\midrule')
    for ablation_id in HEADLINE_ROW_ORDER:
        cells = _clip_headline_row_cells(
            ablation_id, grouped_clip, best_ablation_clip, baseline_biou_mean_clip,
            clip_vs_noclip_deltas, pm='$\\pm$', bold=lambda c: f'\\textbf{{{c}}}',
        )
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def _three_way_frame(df: pd.DataFrame, df_clip: pd.DataFrame) -> pd.DataFrame:
    """9-row frame (3 groups x 3 seeds) with generic A/B/C labels so paired_delta can
    run directly: A = CloSeNet+CLIP, B = our best method without CLIP
    (PTB+BRM+CBL), C = our best method with CLIP."""
    a = df_clip[df_clip['ablation_id'] == 'baseline'].assign(ablation_id='A')
    b = df[df['ablation_id'] == 'ptb_brm_cbl'].assign(ablation_id='B')
    c = df_clip[df_clip['ablation_id'] == 'ptb_brm_cbl'].assign(ablation_id='C')
    return pd.concat([a, b, c], ignore_index=True)


def section8_3_three_way(df: pd.DataFrame, df_clip: pd.DataFrame) -> dict:
    frame = _three_way_frame(df, df_clip)

    biou_stats = frame.groupby('ablation_id')['boundary_iou_mean'].agg(['mean', 'std'])

    pairwise = {
        (a, b): paired_delta(frame, 'boundary_iou_mean', a, b, pair_on='seed_index')
        for a, b in [('C', 'A'), ('C', 'B'), ('A', 'B')]
    }

    # Per-class *regular* IoU (not B-IoU) for the tail classes, computed fresh from
    # each group's raw IoU_per_class -- not df's section5-mutated class_<name>
    # columns, so this has no ordering dependency on section5_per_class having run.
    tail_indices = [CLASS_NAMES.index(name) for name in TAIL_CLASSES]
    tail_stats = {}
    for group in ['A', 'B', 'C']:
        per_class = np.stack(frame[frame['ablation_id'] == group]['IoU_per_class'].to_numpy())
        tail_stats[group] = {
            name: (float(per_class[:, idx].mean()), float(per_class[:, idx].std(ddof=1)))
            for name, idx in zip(TAIL_CLASSES, tail_indices)
        }

    return {'biou_stats': biou_stats, 'pairwise': pairwise, 'tail_stats': tail_stats}


def render_clip_three_way_latex(three_way_info: dict) -> str:
    biou_stats = three_way_info['biou_stats']
    pairwise = three_way_info['pairwise']
    tail_stats = three_way_info['tail_stats']

    lines = ['\\begin{tabular}{lc}', '\\toprule', 'Config & B-IoU \\\\', '\\midrule']
    for group in ['A', 'B', 'C']:
        mean, std = biou_stats.loc[group, 'mean'], biou_stats.loc[group, 'std']
        cell = _fmt_cell(mean, std, 'boundary_iou_mean', '$\\pm$')
        lines.append(f'{THREE_WAY_LABELS[group]} & {cell} \\\\')
    lines.append('\\midrule')
    for (a, b), (mean, std) in pairwise.items():
        label = f'{THREE_WAY_LABELS[a]} vs.\\ {THREE_WAY_LABELS[b]}'
        cell = _fmt_effect(mean, std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        lines.append(f'{label} & {cell} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    tail_lines = ['\\begin{tabular}{l' + 'c' * len(TAIL_CLASSES) + '}', '\\toprule']
    tail_lines.append('Config & ' + ' & '.join(TAIL_CLASSES) + ' \\\\')
    tail_lines.append('\\midrule')
    for group in ['A', 'B', 'C']:
        cells = [THREE_WAY_LABELS[group]]
        for name in TAIL_CLASSES:
            mean, std = tail_stats[group][name]
            cells.append(f'{mean * 100:.2f} $\\pm$ {std * 100:.2f}')
        tail_lines.append(' & '.join(cells) + ' \\\\')
    tail_lines.append('\\bottomrule')
    tail_lines.append('\\end{tabular}')

    return '\n'.join(lines) + '\n\n' + '\n'.join(tail_lines) + '\n'


def _class_order_from_baseline(df: pd.DataFrame) -> list:
    """Same descending-baseline-mean-IoU class order Section 5 uses, computed fresh
    from df's raw IoU_per_class (not df's section5-mutated class_<name> columns) so
    this has no hidden run-order dependency on section5_per_class."""
    baseline_rows = df[df['ablation_id'] == 'baseline']
    per_class = np.stack(baseline_rows['IoU_per_class'].to_numpy())
    order = np.argsort(-per_class.mean(axis=0))
    return [CLASS_NAMES[i] for i in order]


def _clip_baseline_biou_per_class_delta(df: pd.DataFrame, df_clip: pd.DataFrame, class_order_names: list):
    """mean(df_clip baseline biou per class) - mean(df baseline biou per class),
    reordered into class_order_names. Returns None if either side's
    boundary_IoU_per_class isn't fully populated (mirrors section5_per_class's
    biou_available gate)."""
    if not df['boundary_IoU_per_class'].notna().all() or not df_clip['boundary_IoU_per_class'].notna().all():
        return None

    def _mean_per_class(frame: pd.DataFrame) -> np.ndarray:
        rows = frame[frame['ablation_id'] == 'baseline']
        return np.stack(rows['boundary_IoU_per_class'].to_numpy()).mean(axis=0)

    base_means = _mean_per_class(df)
    clip_means = _mean_per_class(df_clip)
    name_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    return np.array([clip_means[name_to_index[n]] - base_means[name_to_index[n]] for n in class_order_names])


def _brm_main_effect(df: pd.DataFrame, metric: str = 'boundary_iou_mean'):
    """0.5*((ptb_brm - ptb) + (ptb_brm_cbl - ptb_cbl)) per seed -- BRM's main effect,
    mirroring _paired_effects's averaging pattern. Does not call section3_brm (which
    would overwrite table_brm_pairwise.tex)."""
    pivot = df[df['ablation_id'].isin(PTB_FAMILY_CELLS)].pivot(index='seed_index', columns='ablation_id', values=metric)
    effect_per_seed = 0.5 * ((pivot['ptb_brm'] - pivot['ptb']) + (pivot['ptb_brm_cbl'] - pivot['ptb_cbl']))
    return float(effect_per_seed.mean()), float(effect_per_seed.std(ddof=1))


def _segfix_overall_effect(df: pd.DataFrame, metric: str = 'boundary_iou_mean'):
    """Pools the postproc-vs-non-postproc diff (paired by timestamp) across all 4
    PTB_FAMILY_CELLS into one mean/std, mirroring section4_segfix's per-config deltas.
    Does not call section4_segfix (which would overwrite table_segfix_effect.tex)."""
    diffs = []
    for config in PTB_FAMILY_CELLS:
        pivot = df[df['ablation_id'].isin([f'{config}_postproc', config])].pivot(index='timestamp', columns='ablation_id', values=metric)
        diffs.append(pivot[f'{config}_postproc'] - pivot[config])
    pooled = pd.concat(diffs, ignore_index=True)
    return float(pooled.mean()), float(pooled.std(ddof=1))


def _interaction_summary_rows(df: pd.DataFrame, df_clip: pd.DataFrame) -> list:
    noclip_cbl = _paired_effects(df, 'boundary_iou_mean')['CBL main effect']
    clip_cbl = _paired_effects(df_clip, 'boundary_iou_mean')['CBL main effect']
    noclip_ptb = _paired_effects(df, 'boundary_iou_mean')['PTB main effect']
    clip_ptb = _paired_effects(df_clip, 'boundary_iou_mean')['PTB main effect']
    noclip_brm = _brm_main_effect(df)
    clip_brm = _brm_main_effect(df_clip)
    noclip_segfix = _segfix_overall_effect(df)
    clip_segfix = _segfix_overall_effect(df_clip)

    return [
        ('CBL', *noclip_cbl, *clip_cbl),
        ('PTB', *noclip_ptb, *clip_ptb),
        ('BRM', *noclip_brm, *clip_brm),
        ('SegFix', *noclip_segfix, *clip_segfix),
    ]


def render_clip_interaction_latex(rows: list) -> str:
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'Contribution & Without CLIP & With CLIP \\\\', '\\midrule']
    for label, noclip_mean, noclip_std, clip_mean, clip_std in rows:
        noclip_cell = _fmt_effect(noclip_mean, noclip_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        clip_cell = _fmt_effect(clip_mean, clip_std, '$\\pm$', 'no detectable effect at $n{=}3$ seeds')
        lines.append(f'{label} & {noclip_cell} & {clip_cell} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def _clip_discussion_paragraph(clip_vs_noclip_deltas, three_way_info, per_class_biou_delta, class_order_names, interaction_rows) -> str:
    baseline_mean, baseline_std = clip_vs_noclip_deltas['baseline']
    baseline_sentence = (
        f'CLIP init alone shifts baseline B-IoU by {baseline_mean * 100:+.2f} $\\pm$ '
        f'{baseline_std * 100:.2f} pp relative to the random-init baseline.'
    )

    a_vs_b_mean, a_vs_b_std = three_way_info['pairwise'][('A', 'B')]
    c_vs_b_mean, c_vs_b_std = three_way_info['pairwise'][('C', 'B')]
    c_vs_a_mean, c_vs_a_std = three_way_info['pairwise'][('C', 'A')]
    money_sentence = (
        f'CLIP alone vs.\\ our best non-CLIP method (PTB+BRM+CBL): {a_vs_b_mean * 100:+.2f} '
        f'$\\pm$ {a_vs_b_std * 100:.2f} pp B-IoU; combining both vs.\\ PTB+BRM+CBL alone: '
        f'{c_vs_b_mean * 100:+.2f} $\\pm$ {c_vs_b_std * 100:.2f} pp; combining both vs.\\ '
        f'CLIP alone: {c_vs_a_mean * 100:+.2f} $\\pm$ {c_vs_a_std * 100:.2f} pp.'
    )

    if per_class_biou_delta is None:
        tail_sentence = 'Per-class B-IoU deltas were unavailable for this comparison.'
    else:
        tail_str = ', '.join(
            f'{name}: {per_class_biou_delta[class_order_names.index(name)] * 100:+.2f} pp'
            for name in TAIL_CLASSES if name in class_order_names
        )
        tail_sentence = (
            f'On the tail classes, CLIP-alone shifts baseline B-IoU by {tail_str} '
            f'(compare against the best-config-vs-baseline deltas for the same classes in Section 5).'
        )

    composing = [label for label, _, _, clip_mean, clip_std in interaction_rows if abs(clip_mean) >= 2 * clip_std]
    if len(composing) > 2:
        composing_str = ', '.join(composing[:-1]) + f', and {composing[-1]}'
    elif len(composing) == 2:
        composing_str = f'{composing[0]} and {composing[1]}'
    elif composing:
        composing_str = composing[0]
    if composing:
        compose_sentence = f'Under CLIP init, {composing_str} retain a detectable B-IoU effect (i.e.\\ compose with CLIP rather than being made redundant by it).'
    else:
        compose_sentence = 'None of CBL/PTB/BRM/SegFix retain a detectable B-IoU effect once CLIP init is applied, at $n{=}3$ seeds.'

    return ' '.join([baseline_sentence, money_sentence, tail_sentence, compose_sentence])


def section8_clip_comparison(df: pd.DataFrame, df_clip: pd.DataFrame, output_dir: Path, runs_dir: Path) -> dict:
    _assert_seed_alignment(df, df_clip, runs_dir)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 8.2
    grouped_clip, best_ablation_clip, baseline_biou_mean_clip = compute_headline_stats(df_clip)
    clip_vs_noclip_deltas = _clip_vs_noclip_biou_deltas(df, df_clip)
    output_path = output_dir / 'table_clip_headline.tex'
    output_path.write_text(render_clip_headline_latex(grouped_clip, best_ablation_clip, baseline_biou_mean_clip, clip_vs_noclip_deltas))
    print(f'Wrote {output_path}')

    # 8.3
    three_way_info = section8_3_three_way(df, df_clip)
    output_path = output_dir / 'table_clip_three_way.tex'
    output_path.write_text(render_clip_three_way_latex(three_way_info))
    print(f'Wrote {output_path}')

    # 8.4
    class_order_names = _class_order_from_baseline(df)
    per_class_biou_delta = _clip_baseline_biou_per_class_delta(df, df_clip, class_order_names)
    if per_class_biou_delta is not None:
        fig_path = figures_dir / 'clip_per_class_delta_biou.png'
        _plot_per_class_delta(
            per_class_biou_delta, class_order_names, fig_path,
            xlabel='CloSeNet+CLIP $-$ CloSeNet (B-IoU)',
            title='Per-class B-IoU delta: CLIP-init vs.\\ random-init codebook',
        )
        print(f'Wrote {fig_path}')
    else:
        print('Skipping Section 8 per-class B-IoU delta chart: boundary_IoU_per_class not fully populated.')

    # 8.5
    interaction_rows = _interaction_summary_rows(df, df_clip)
    output_path = output_dir / 'table_clip_interaction.tex'
    output_path.write_text(render_clip_interaction_latex(interaction_rows))
    print(f'Wrote {output_path}')

    # 8.6
    discussion = _clip_discussion_paragraph(clip_vs_noclip_deltas, three_way_info, per_class_biou_delta, class_order_names, interaction_rows)

    return {
        'clip_vs_noclip_deltas': clip_vs_noclip_deltas,
        'three_way_info': three_way_info,
        'per_class_biou_delta_available': per_class_biou_delta is not None,
        'interaction_rows': interaction_rows,
        'discussion': discussion,
    }


def render_section8(clip_info: dict) -> str:
    biou_figure_block = (
        """\\begin{figure}[h]
\\centering
\\includegraphics[width=0.65\\linewidth]{figures/clip_per_class_delta_biou.png}
\\caption{Per-class B-IoU delta: CLIP-init vs.\\ random-init codebook (both plain baselines, no PTB/BRM/CBL).}
\\end{figure}"""
        if clip_info['per_class_biou_delta_available'] else ''
    )

    return f"""
\\section{{CLIP-Initialized Codebook}}

\\subsection{{Setup}}

Instead of a random initialization, the garment codebook's 18 rows are seeded from the
frozen OpenAI CLIP text encoder's embeddings of each class name (prompt: ``a photo of''
plus the lowercased class name), projected down to the codebook dimension via a fixed
random projection, and left trainable during training. This tests whether a semantic
prior over garment classes is a cheaper or complementary route to the same
boundary-quality gains as CBL/PTB/BRM/SegFix.

\\subsection{{Headline Results with Paired Deltas}}

Same 10-config table as Section 1, computed on the CLIP-initialized campaign, with an
extra paired-seed column: the B-IoU delta of each CLIP-initialized config vs.\\ its
non-CLIP counterpart (same architecture, same seed, random codebook init).

\\begin{{table}}[h]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{\\input{{table_clip_headline.tex}}}}
\\end{{table}}

\\subsection{{Does CLIP Alone Match Our Best Method?}}

The critical head-to-head: CloSeNet+CLIP vs.\\ our best non-CLIP method
(PTB+BRM+CBL) vs.\\ combining both, paired by seed, plus per-class IoU on the tail
classes (Dress, Jumpsuits, Scarf).

\\begin{{table}}[h]
\\centering
\\input{{table_clip_three_way.tex}}
\\end{{table}}

\\subsection{{Per-Class Delta}}

Reproducing Section 5's per-class B-IoU delta chart, but for CLIP-init vs.\\
random-init on the plain baseline (no PTB/BRM/CBL), using the same class ordering
(sorted by the non-CLIP baseline's mean IoU, descending) for visual comparability with
Section 5's own chart.

{biou_figure_block}

\\subsection{{Interaction Summary}}

For each of the four boundary contributions, the mean B-IoU effect with and without
CLIP init, using the same effect definitions as Sections 2-4 (main effects /
postproc-vs-non-postproc deltas), so a reader can see at a glance which contributions
compose with CLIP and which become redundant to it.

\\begin{{table}}[h]
\\centering
\\input{{table_clip_interaction.tex}}
\\end{{table}}

\\subsection{{Discussion}}

{clip_info['discussion']}
"""


REPORT_TEX_TEMPLATE = """\\documentclass[11pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{amsmath}

\\title{CloSeNet Ablation Report}
\\date{}

\\begin{document}
\\maketitle
\\sloppy

\\section{Headline Results}

Mean $\\pm$ std across 3 seeds for all 10 ablations, on mIoU, freq\\_IoU, B-IoU,
mIoU@boundary, mIoU@inner, and segm\\_loss. The best value per column is bolded. This
table is the anchor of the paper -- everything else supports it. Since B-IoU is the
headline metric, a $\\Delta$ B-IoU vs.\\ baseline column is also reported.

\\begin{table}[h]
\\centering
\\resizebox{\\textwidth}{!}{\\input{table_headline.tex}}
\\end{table}

Section~8 extends this analysis to a CLIP-initialized codebook baseline.

\\section{CBL $\\times$ PTB Factorial}

This is the central scientific claim: does combining CBL and PTB help more, less, or
the same as their individual effects would predict? The interaction plot below shows
the four cells (baseline, +CBL, +PTB, +CBL+PTB) as two lines -- \\{no CBL, +CBL\\} --
over \\{no PTB, +PTB\\}, for both B-IoU and mIoU, with error bars from seed std.
Parallel lines indicate additive (independent) effects; crossing or diverging lines
indicate an interaction.

\\begin{figure}[h]
\\centering
\\includegraphics[width=\\linewidth]{figures/interaction_cbl_ptb.png}
\\caption{CBL $\\times$ PTB interaction plot (error bars: seed std).}
\\end{figure}

The table below is the quantitative version of the plot: the main effect of CBL
(average of CBL rows minus non-CBL rows), the main effect of PTB (average of PTB
rows minus non-PTB rows), and the interaction term
$(\\text{PTB+CBL} - \\text{PTB}) - (\\text{CBL} - \\text{baseline})$, each reported as
mean $\\pm$ std across the 3 matched seeds. Where an effect is smaller than 2$\\times$
its std, the cell states ``no detectable effect at $n{=}3$ seeds'' rather than
reporting a possibly spurious sign.

\\begin{table}[h]
\\centering
\\input{table_factorial_effects.tex}
\\end{table}
"""

REPORT_TEX_FOOTER = "\n\\end{document}\n"


def _brm_beats_all_sentence(brm_rows: list) -> str:
    # brm_rows: (label, biou_mean, biou_std, miou_mean, miou_std), biou/miou are
    # PTB+BRM+CBL minus the comparison config -- positive means PTB+BRM+CBL wins.
    biou_losses = [label for label, biou_mean, biou_std, _, _ in brm_rows if biou_mean <= 0]
    if not biou_losses:
        return 'PTB+BRM+CBL beats all three other PTB-family configs on B-IoU.'
    return (
        'PTB+BRM+CBL does not clearly beat every comparison on B-IoU (see: '
        + ', '.join(biou_losses) + ').'
    )


def _segfix_trend_sentence(segfix_deltas: dict) -> str:
    biou_means = [segfix_deltas[c]['boundary_iou_mean'][0] for c in PTB_FAMILY_CELLS]
    if all(x >= y for x, y in zip(biou_means, biou_means[1:])):
        trend = 'shrinks monotonically'
    elif all(x <= y for x, y in zip(biou_means, biou_means[1:])):
        trend = 'grows monotonically'
    else:
        trend = 'is not monotonic'
    values = ', '.join(f'{BRM_LABELS[c]}: {segfix_deltas[c]["boundary_iou_mean"][0] * 100:+.2f}' for c in PTB_FAMILY_CELLS)
    return f'Going from PTB to PTB+BRM+CBL, the SegFix B-IoU delta {trend} ({values} pp).'


def _per_class_sentence(per_class_info: dict) -> str:
    worse = per_class_info['worse_classes']
    if not worse:
        worse_sentence = f'No class regresses under {HEADLINE_ROW_LABELS[per_class_info["best_config"]]} relative to baseline.'
    else:
        worse_sentence = f'Classes that regress under {HEADLINE_ROW_LABELS[per_class_info["best_config"]]}: ' + ', '.join(worse) + '.'
    corr = per_class_info['variance_corr']
    corr_sentence = (
        f'Correlation between a class\'s baseline mean IoU and its baseline seed-to-seed std is '
        f'{corr:.2f} ({"lower-IoU classes do show higher seed variance" if corr < -0.2 else "no strong relationship is evident at n=18 classes"}).'
    )
    if per_class_info['biou_available']:
        biou_worse = per_class_info['biou_worse_classes']
        biou_sentence = (
            'No class regresses on B-IoU.' if not biou_worse
            else 'Classes that regress on B-IoU: ' + ', '.join(biou_worse) + '.'
        )
    else:
        key_escaped = PER_CLASS_BOUNDARY_KEY.replace('_', '\\_')
        biou_sentence = (
            f'Per-class B-IoU is not yet available ("{key_escaped}" is not saved '
            f'by evaluate\\_closenet\\_ckpts.py yet) -- planned as a follow-up.'
        )
    return worse_sentence + ' ' + corr_sentence + ' ' + biou_sentence


def _per_class_biou_figure_block(per_class_info: dict) -> str:
    if not per_class_info['biou_available']:
        return ''
    return """\\begin{figure}[h]
\\centering
\\includegraphics[width=0.65\\linewidth]{figures/per_class_delta_biou.png}
\\caption{Per-class B-IoU delta: best config vs.\\ baseline.}
\\end{figure}"""


def _training_dynamics_sentence(training_info: dict) -> str:
    trend = training_info['cbl_trend']
    if trend is None:
        cbl_sentence = 'val/cbl\\_loss was not available to check the CBL loss trend.'
    elif trend > 0:
        cbl_sentence = f'val/cbl\\_loss decreases by {trend:.4f} from start to end of training, evidence CBL is doing something.'
    else:
        cbl_sentence = f'val/cbl\\_loss does not decrease (net change {trend:.4f}) -- worth checking whether CBL is contributing as intended.'
    flagged = training_info['flagged_configs']
    flag_sentence = (
        'No config\'s best checkpoint was selected from a late epoch.' if not flagged
        else 'Configs whose best checkpoint came from a late epoch ($\\geq$90\\% of max\\_epochs): ' + ', '.join(flagged) + '.'
    )
    return cbl_sentence + ' ' + flag_sentence


def render_dynamic_sections(brm_rows, segfix_deltas, per_class_info, training_info) -> str:
    return f"""
\\section{{BRM Results}}

This is the likely headline finding: does the Boundary-aware Refinement Module,
combined with CBL, beat PTB alone, PTB+CBL, and PTB+BRM individually? {_brm_beats_all_sentence(brm_rows)}
The synergy claim being tested is whether adding CBL on top of BRM specifically
helps more than adding CBL elsewhere (compare the CBL main effect from Section 2 to
the PTB+BRM+CBL vs.\\ PTB+BRM row below).

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\linewidth]{{figures/brm_grouped_bars.png}}
\\caption{{mIoU and B-IoU per PTB-family config, error bars: seed std.}}
\\end{{figure}}

\\begin{{table}}[h]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{\\input{{table_brm_pairwise.tex}}}}
\\end{{table}}

\\section{{SegFix Post-Processing Effect}}

Does post-processing help uniformly, or does its benefit shrink once the model
itself is already boundary-aware? {_segfix_trend_sentence(segfix_deltas)}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.7\\linewidth]{{figures/segfix_delta.png}}
\\caption{{B-IoU improvement from SegFix, paired by run (same checkpoint, with vs.\\ without post-processing).}}
\\end{{figure}}

\\begin{{table}}[h]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{\\input{{table_segfix_effect.tex}}}}
\\end{{table}}

\\section{{Per-Class Breakdown}}

{_per_class_sentence(per_class_info)}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.65\\linewidth]{{figures/per_class_heatmap.png}}
\\caption{{Per-class IoU, rows sorted by baseline IoU descending.}}
\\end{{figure}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.65\\linewidth]{{figures/per_class_delta.png}}
\\caption{{Per-class IoU delta: best config vs.\\ baseline.}}
\\end{{figure}}

{_per_class_biou_figure_block(per_class_info)}

\\section{{Training Dynamics}}

Training curves (mean $\\pm$ seed std ribbon) for baseline, +CBL, +PTB+CBL, and
+PTB+BRM+CBL. {_training_dynamics_sentence(training_info)}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=\\linewidth]{{figures/training_dynamics.png}}
\\caption{{Val loss, val mIoU, val B-IoU, and train loss over training.}}
\\end{{figure}}

\\begin{{table}}[h]
\\centering
\\input{{table_best_epoch.tex}}
\\end{{table}}

\\section{{Boundary vs.\\ Inner Tradeoff}}

Do boundary-region improvements come at the cost of inner-region accuracy? Each
point is one (config, seed) run; an ideal method moves points up and to the right
(better at both), while a method trading inner accuracy for boundary accuracy would
move points up and to the left.

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.75\\linewidth]{{figures/boundary_inner_scatter.png}}
\\caption{{mIoU@inner vs.\\ mIoU@boundary, one point per (config, seed).}}
\\end{{figure}}
"""


def write_latex_report(output_dir: Path, brm_rows, segfix_deltas, per_class_info, training_info, clip_info) -> None:
    # table_*.tex / figures/*.png are \\input / \\includegraphics'd rather than
    # re-rendered here, so this file always matches what each sectionN_xxx wrote.
    # Only the prose (worse classes, trend directions, flagged configs) is computed
    # here from each section's return value, so it always reflects the actual numbers.
    content = (
        REPORT_TEX_TEMPLATE
        + render_dynamic_sections(brm_rows, segfix_deltas, per_class_info, training_info)
        + render_section8(clip_info)
        + REPORT_TEX_FOOTER
    )
    output_path = output_dir / 'report.tex'
    output_path.write_text(content)
    print(f'Wrote {output_path}')


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
    grouped, best_ablation, baseline_biou_mean = section1_headline_table(df, output_dir)
    section2_factorial(df, output_dir)
    brm_rows = section3_brm(df, output_dir)
    segfix_deltas = section4_segfix(df, output_dir)
    per_class_info = section5_per_class(df, output_dir, best_ablation)
    training_info = section6_training_dynamics(df, runs_dir, output_dir)
    section7_boundary_inner_scatter(df, output_dir)
    df_clip = load_results(results_clip_dir)
    clip_info = section8_clip_comparison(df, df_clip, output_dir, runs_dir)
    write_latex_report(output_dir, brm_rows, segfix_deltas, per_class_info, training_info, clip_info)


if __name__ == '__main__':
    main()
