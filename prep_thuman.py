"""Wrapper around prep_scan.py's preprocessing pipeline for the THuman2.0 dataset.

Produces per-scan .npz files matching the CloSe-Di schema, then builds a combined
train/val/test split file that additively folds THuman2.0 scans into the existing
CloSe-Di split (cfg/data_split.json is never modified; a new split file is written).

Does not modify prep_scan.py, lib/closed/dataset.py, or any existing config/data file.

Example:
  python prep_thuman.py --bm_dir_path $SMPL_PATH/models --stage all

Must be run in an environment with CUDA + pytorch3d + smplx installed (same
requirement as prep_scan.py itself); see prep_thuman.sbatch for the cluster job wrapper.
"""

import argparse
import datetime
import json
import os
import time
from collections import Counter

import numpy as np
import trimesh

from prep_scan import create_smpl, humanbody_data, load_mesh


def make_logger(log_file):
    def log(msg):
        line = f'[{datetime.datetime.now().isoformat(timespec="seconds")}] {msg}'
        print(line)
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(line + '\n')

    return log


def process_scan(scan_id, thuman_root, smpl_dir, labels_npz, bm_dir_path, output_dir,
                  device, overwrite, log):
    """Preprocesses a single THuman2.0 scan into the CloSe-Di npz schema.

    Returns one of: 'ok', 'skip_exists', 'skip_no_labels', 'skip_no_smpl',
    'skip_mismatch', 'skip_error'.
    """
    out_path = os.path.join(output_dir, f'{scan_id}.npz')
    if not overwrite and os.path.exists(out_path):
        return 'skip_exists'

    if scan_id not in labels_npz.files:
        log(f'{scan_id}: no label entry, skipping')
        return 'skip_no_labels'

    smpl_path = os.path.join(smpl_dir, f'{scan_id}_smpl.pkl')
    if not os.path.exists(smpl_path):
        log(f'{scan_id}: no SMPL fit at {smpl_path}, skipping')
        return 'skip_no_smpl'

    obj_path = os.path.join(thuman_root, scan_id, f'{scan_id}.obj')
    tex_path = os.path.join(thuman_root, scan_id, 'material0.jpeg')

    try:
        labels = labels_npz[scan_id].astype(np.int64)

        _, mesh_verts, mesh_faces, col_val, norms = load_mesh(obj_path, tex_path, device=device)

        if labels.shape[0] != mesh_verts.shape[0]:
            log(
                f'{scan_id}: vertex/label count mismatch '
                f'({mesh_verts.shape[0]} verts vs {labels.shape[0]} labels), skipping'
            )
            return 'skip_mismatch'

        smpl_verts, _smpl_scale, smpl_trans, full_pose, betas = create_smpl(
            smpl_path, bm_dir_path, device
        )
        canon_pose = humanbody_data(smpl_verts, mesh_verts)

        # Bounding-box normalization, identical math to prep_scan.py's main().
        scan_mesh = trimesh.Trimesh(
            vertices=mesh_verts.detach().cpu().numpy(),
            faces=mesh_faces.detach().cpu().numpy(),
            maintain_order=True,
            process=False,
        )
        total_size = (scan_mesh.bounds[1] - scan_mesh.bounds[0]).max()
        centers = (scan_mesh.bounds[1] + scan_mesh.bounds[0]) / 2
        scan_mesh.apply_translation(-centers)
        scan_mesh.apply_scale(1 / total_size)

        garments = np.zeros(18, dtype=np.int64)
        garments[np.unique(labels)] = 1

        np.savez(
            out_path,
            points=scan_mesh.vertices,
            colors=col_val.detach().cpu().numpy()[0].astype(np.float32),
            normals=norms.detach().cpu().numpy()[0].astype(np.float32),
            labels=labels,
            canon_pose=canon_pose.astype(np.float64),
            garments=garments.astype(np.int32),
            faces=scan_mesh.faces.astype(np.int64),
            betas=betas.detach().cpu().numpy()[0].astype(np.float32),
            pose=full_pose.astype(np.float32),
            trans=np.asarray(smpl_trans, dtype=np.float32).reshape(3),
            scale=np.float64(1.0 / total_size),
        )
        return 'ok'
    except Exception as e:  # noqa: BLE001 - one bad scan must not abort the batch
        log(f'{scan_id}: unexpected error: {e!r}')
        return 'skip_error'


def run_prep(args, log):
    labels_npz = np.load(args.labels_file, allow_pickle=True)
    os.makedirs(args.output_dir, exist_ok=True)

    scan_ids = args.scan_ids or sorted(
        d for d in os.listdir(args.thuman_root)
        if os.path.isdir(os.path.join(args.thuman_root, d))
    )
    if args.limit is not None:
        scan_ids = scan_ids[: args.limit]

    counts = Counter()
    start = time.time()
    for scan_id in scan_ids:
        status = process_scan(
            scan_id,
            args.thuman_root,
            args.smpl_dir,
            labels_npz,
            args.bm_dir_path,
            args.output_dir,
            args.device,
            args.overwrite,
            log,
        )
        counts[status] += 1

    elapsed = time.time() - start
    log(f'SUMMARY: {dict(counts)} over {len(scan_ids)} scans in {elapsed:.1f}s')


def run_split(args, log):
    with open(args.closedi_split) as f:
        closedi = json.load(f)

    npz_files = sorted(f for f in os.listdir(args.output_dir) if f.endswith('.npz'))
    scan_ids = [os.path.splitext(f)[0] for f in npz_files]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(scan_ids)
    n = len(perm)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val :]

    def rel_path(scan_id):
        return os.path.join(args.output_dir, f'{scan_id}.npz')

    merged = {
        'train': list(closedi['train']) + [rel_path(i) for i in train_ids],
        'val': list(closedi['val']) + [rel_path(i) for i in val_ids],
        'test': list(closedi['test']) + [rel_path(i) for i in test_ids],
    }

    with open(args.out_split, 'w') as f:
        json.dump(merged, f, indent=2)

    log(
        f'Wrote {args.out_split}: train={len(merged["train"])} '
        f'val={len(merged["val"])} test={len(merged["test"])} '
        f'(added {len(train_ids)}/{len(val_ids)}/{len(test_ids)} THuman scans)'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thuman_root', default='data/THuman2.0_Release_copy')
    parser.add_argument('--smpl_dir', default='data/THuman2.0_smpl')
    parser.add_argument('--labels_file', default='data/CloSe-D++_updated/THuman2.0_labels.npz')
    parser.add_argument('--bm_dir_path', required=True, help='Path to SMPL body model directory')
    parser.add_argument('--output_dir', default='data/THuman2.0_preprocessed')
    parser.add_argument('--closedi_split', default='cfg/data_split.json')
    parser.add_argument('--out_split', default='cfg/data_split_thuman.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', choices=['prep', 'split', 'all'], default='all')
    parser.add_argument('--scan_ids', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    if args.log_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.log_file = os.path.join(args.output_dir, 'prep_thuman_log.txt')

    log = make_logger(args.log_file)

    if args.stage in ('prep', 'all'):
        run_prep(args, log)
    if args.stage in ('split', 'all'):
        run_split(args, log)


if __name__ == '__main__':
    main()
