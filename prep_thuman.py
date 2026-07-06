"""
Convert raw THuman2.0 scans + CloSe-D++_updated labels into CloSe-Di-style
per-scan .npz files, and build a train/val/test split for them.

Produced fields match the CloSe-Di schema (docs/dataset.md) *except* for the
SMPL-derived fields (pose, betas, trans, canon_pose): THuman2.0_Release_copy.zip
ships scans only, with no SMPL registrations, so those fields are intentionally
omitted rather than faked. See data/CloSe-Di-THuman/README.md.

Example:
    python prep_thuman.py
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh

N_CLASSES = 18


def load_scan_mesh(zf: zipfile.ZipFile, scan_id: str, tmp_dir: Path) -> trimesh.Trimesh:
    scan_dir = tmp_dir / scan_id
    for name in zf.namelist():
        if name.startswith(f'{scan_id}/') and not name.endswith('/'):
            zf.extract(name, tmp_dir)
    mesh = trimesh.load(scan_dir / f'{scan_id}.obj', process=False)
    shutil.rmtree(scan_dir)
    return mesh


def prep_scan(mesh: trimesh.Trimesh, labels: np.ndarray) -> dict:
    colors = mesh.visual.to_color().vertex_colors[:, :3].astype(np.uint8)
    normals = mesh.vertex_normals.astype(np.float32)

    # Center on the bounding-box midpoint and scale by its largest extent, matching
    # the CloSe-Di normalization convention so points end up in the same coordinate
    # frame CloSeNet was trained on (roughly unit scale, origin-centered).
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    total_size = (bounds[1] - bounds[0]).max()
    points = ((mesh.vertices - center) / total_size).astype(np.float32)

    # Multi-hot vector of which garment classes appear anywhere on this scan
    # (CloSe-Di schema field, consumed by the garment encoder).
    garments = np.zeros(N_CLASSES, dtype=np.int32)
    garments[np.unique(labels)] = 1

    return dict(
        points=points,
        normals=normals,
        colors=colors,
        faces=mesh.faces.astype(np.int32),
        labels=labels.astype(np.int64),
        garments=garments,
        scale=np.array(1.0 / total_size, dtype=np.float32),
        centers=center.astype(np.float32),
    )


def make_split(scan_ids: list, train_ratio: float, val_ratio: float, seed: int) -> dict:
    """Shuffle (fixed seed, for a reproducible split) then cut into train/val/test."""
    ids = list(scan_ids)
    np.random.default_rng(seed).shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val = int(len(ids) * val_ratio)
    return {
        'train': ids[:n_train],
        'val': ids[n_train:n_train + n_val],
        'test': ids[n_train + n_val:],
    }


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix='thuman_prep_'))

    labels_npz = np.load(args.labels_path, allow_pickle=True)
    zf = zipfile.ZipFile(args.zip_path)
    zip_scan_ids = {n.split('/')[0] for n in zf.namelist() if n.endswith('.obj')}
    scan_ids = sorted(set(labels_npz.keys()) & zip_scan_ids)
    if args.limit:
        scan_ids = scan_ids[:args.limit]
    print(f'{len(scan_ids)} scans with both mesh and labels')

    processed = []
    for i, scan_id in enumerate(scan_ids):
        out_path = out_dir / f'{scan_id}.npz'
        if out_path.exists() and not args.overwrite:
            processed.append(scan_id)
            continue

        mesh = load_scan_mesh(zf, scan_id, tmp_dir)
        labels = labels_npz[scan_id]
        if labels.shape[0] != mesh.vertices.shape[0]:
            print(f'[skip] {scan_id}: label/vertex count mismatch '
                  f'({labels.shape[0]} vs {mesh.vertices.shape[0]})')
            continue

        np.savez(out_path, **prep_scan(mesh, labels))
        processed.append(scan_id)
        if (i + 1) % 25 == 0:
            print(f'[{i + 1}/{len(scan_ids)}] processed {scan_id}')

    shutil.rmtree(tmp_dir, ignore_errors=True)

    split = make_split(processed, args.train_ratio, args.val_ratio, args.seed)
    split_paths = {
        k: [str(out_dir / f'{sid}.npz') for sid in v] for k, v in split.items()
    }
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.split_out, 'w') as f:
        json.dump(split_paths, f, indent=2)

    print(f'wrote {len(processed)} scans to {out_dir}')
    print(f'split -> train={len(split["train"])} val={len(split["val"])} test={len(split["test"])}')
    print(f'split file: {args.split_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', default='data/THuman2.0_Release_copy.zip')
    parser.add_argument('--labels_path', default='data/CloSe-D++_updated/THuman2.0_labels.npz')
    parser.add_argument('--out_dir', default='data/CloSe-Di-THuman')
    parser.add_argument('--split_out', default='cfg/data_split_thuman.json')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None, help='process only the first N scans (debugging)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args)
