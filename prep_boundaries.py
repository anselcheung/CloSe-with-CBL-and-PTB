"""Ground-truth boundary/direction precomputation for boundary-aware CloSeNet.

Port of Push-the-Boundary's ``preprocessing_boundaries.py`` (Du et al., WACV 2023)
adapted to the CloSe-Di npz schema. For every scan it augments the npz *in place*
with three per-point arrays, computed at full scan resolution (before the 2048-point
training subsample), in the posed world-xyz frame:

  boundary      (N,) uint8    1 if any of the point's k nearest neighbours carries a
                              different (valid) label, else 0. Background points
                              (label == -1) are treated as "ignore": they are never
                              boundary, and disagreements involving -1 do not count,
                              so the outer scan silhouette is not marked a boundary.
  direction     (N, 3) f32    unit vector from the nearest boundary point to the
                              point (i.e. pointing into the object interior). Zero
                              for boundary points, background points, and scans with
                              no boundary.
  boundary_dist (N,) f32      Euclidean distance to the nearest boundary point.
                              +inf for background points / scans without boundaries;
                              consumed by the boundary-region mIoU metric at eval.

Scans are unit-bbox normalised (max extent == 1), so directions/distances are in
normalised units.

Does not modify any existing config, dataset, or model file.

Examples:
  # Augment every scan referenced by a split file (train+val+test):
  python prep_boundaries.py --split cfg/data_split.json

  # Augment every .npz under a directory:
  python prep_boundaries.py --data_dir data/THuman2.0_preprocessed

  # Augment specific scans, overwriting existing boundary fields:
  python prep_boundaries.py --scan_ids data/CloSe-Di/10009_2182.npz --overwrite
"""

import argparse
import datetime
import json
import os
import time
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree


def make_logger(log_file):
    def log(msg):
        line = f'[{datetime.datetime.now().isoformat(timespec="seconds")}] {msg}'
        print(line)
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(line + '\n')

    return log


def compute_boundary_fields(points, labels, k=4):
    """Compute (boundary, direction, boundary_dist) for one scan at full resolution.

    Args:
        points: (N, 3) posed xyz coordinates.
        labels: (N,) int labels; -1 denotes background/ignore.
        k:      number of nearest neighbours (excluding self) used to detect a
                label disagreement.

    Returns:
        boundary:      (N,) uint8
        direction:     (N, 3) float32
        boundary_dist: (N,) float32
    """
    n = points.shape[0]
    labels = labels.astype(np.int64)

    tree = cKDTree(points)
    # k + 1 so that the first (self) neighbour can be dropped.
    _, idx = tree.query(points, k=min(k + 1, n))
    if idx.ndim == 1:  # degenerate: n == 1
        idx = idx[:, None]
    neigh_idx = idx[:, 1:]  # drop self, (N, k)

    neigh_labels = labels[neigh_idx]  # (N, k)
    self_valid = labels != -1  # (N,)
    neigh_valid = neigh_labels != -1  # (N, k)
    disagree = (neigh_labels != labels[:, None]) & neigh_valid  # (N, k)
    boundary = (self_valid & disagree.any(axis=1)).astype(np.uint8)

    direction = np.zeros((n, 3), dtype=np.float32)
    boundary_dist = np.full(n, np.inf, dtype=np.float32)

    b_idx = np.nonzero(boundary)[0]
    if b_idx.size > 0:
        b_tree = cKDTree(points[b_idx])
        dist, nn = b_tree.query(points, k=1)  # (N,), (N,)
        nearest = points[b_idx[nn]]  # (N, 3)
        vec = points - nearest
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        nonzero = norm[:, 0] > 1e-8
        direction[nonzero] = (vec[nonzero] / norm[nonzero]).astype(np.float32)
        boundary_dist = dist.astype(np.float32)

    # Background points are ignored everywhere downstream.
    bg = ~self_valid
    direction[bg] = 0.0
    boundary_dist[bg] = np.inf

    return boundary, direction, boundary_dist


def process_scan(scan_path, k, overwrite, log):
    """Augment a single npz with boundary fields.

    Returns one of: 'ok', 'skip_exists', 'skip_no_labels', 'skip_error'.
    """
    try:
        with np.load(scan_path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}
    except Exception as e:  # noqa: BLE001
        log(f'{scan_path}: failed to load: {e!r}')
        return 'skip_error'

    if not overwrite and 'boundary' in data and 'direction' in data and 'boundary_dist' in data:
        return 'skip_exists'

    if 'labels' not in data:
        log(f'{scan_path}: no labels, skipping (cannot define boundaries)')
        return 'skip_no_labels'

    try:
        points = np.asarray(data['points']).squeeze().astype(np.float64)[:, :3]
        labels = np.asarray(data['labels']).squeeze()
        boundary, direction, boundary_dist = compute_boundary_fields(points, labels, k=k)

        data['boundary'] = boundary
        data['direction'] = direction
        data['boundary_dist'] = boundary_dist
        np.savez(scan_path, **data)

        frac = float(boundary.mean())
        log(f'{os.path.basename(scan_path)}: N={len(labels)} boundary_frac={frac:.3f}')
        return 'ok'
    except Exception as e:  # noqa: BLE001 - one bad scan must not abort the batch
        log(f'{scan_path}: unexpected error: {e!r}')
        return 'skip_error'


def collect_scan_paths(args):
    if args.scan_ids:
        return list(args.scan_ids)
    if args.split:
        loader = (
            (lambda p: dict(np.load(p, allow_pickle=True)))
            if args.split.endswith('.npz')
            else (lambda p: json.load(open(p)))
        )
        split = loader(args.split)
        paths = []
        for mode in ('train', 'val', 'test'):
            paths.extend(list(split.get(mode, [])))
        # De-duplicate while preserving order.
        return list(dict.fromkeys(paths))
    if args.data_dir:
        return sorted(
            os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npz')
        )
    raise SystemExit('Provide one of --split, --data_dir, or --scan_ids')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group('scan source (choose one)')
    src.add_argument('--split', default=None, help='split json/npz; augments train+val+test')
    src.add_argument('--data_dir', default=None, help='directory of .npz scans')
    src.add_argument('--scan_ids', nargs='*', default=None, help='explicit .npz paths')
    parser.add_argument('--k', type=int, default=4, help='kNN neighbours for boundary detection')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true',
                        help='recompute even if boundary fields already exist')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    log = make_logger(args.log_file)

    scan_paths = collect_scan_paths(args)
    if args.limit is not None:
        scan_paths = scan_paths[: args.limit]

    counts = Counter()
    start = time.time()
    for scan_path in scan_paths:
        counts[process_scan(scan_path, args.k, args.overwrite, log)] += 1

    elapsed = time.time() - start
    log(f'SUMMARY: {dict(counts)} over {len(scan_paths)} scans in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
