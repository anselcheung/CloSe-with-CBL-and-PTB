"""Counts per-class segmentation label occurrences in a data split file.

For each split in the given split JSON (e.g. cfg/data_split_thuman.json), loads every
scan's `labels` array and reports, per class: how many scans contain it and how many
points/vertices belong to it. Scans referenced in the split but missing on disk (e.g.
THuman2.0 scans not yet preprocessed by prep_thuman.py) are skipped and counted.

Example:
  python count_class_labels.py --split_file cfg/data_split_thuman.json
"""

import argparse
import json
import os

import numpy as np

CLASS_NAMES = [
    'Hat',
    'Body',
    'Shirt',
    'TShirt',
    'Vest',
    'Coat',
    'Dress',
    'Skirt',
    'Pants',
    'ShortPants',
    'Shoes',
    'Hoodies',
    'Hair',
    'Swimwear',
    'Underwear',
    'Scarf',
    'Jumpsuits',
    'Jacket',
]


def count_split(paths, class_names):
    n_classes = len(class_names)
    scan_counts = np.zeros(n_classes, dtype=np.int64)
    point_counts = np.zeros(n_classes, dtype=np.int64)
    n_missing = 0
    n_loaded = 0

    for path in paths:
        if not os.path.exists(path):
            n_missing += 1
            continue

        with np.load(path, allow_pickle=True) as data:
            labels = data['labels']

        present = np.unique(labels)
        present = present[(present >= 0) & (present < n_classes)]
        scan_counts[present] += 1
        point_counts += np.bincount(labels[labels >= 0], minlength=n_classes)[:n_classes]
        n_loaded += 1

    return scan_counts, point_counts, n_loaded, n_missing


def print_split_report(split_name, paths, class_names):
    scan_counts, point_counts, n_loaded, n_missing = count_split(paths, class_names)

    print(f'\n=== {split_name} ===')
    print(f'{len(paths)} scans listed, {n_loaded} loaded, {n_missing} missing on disk')

    n_classes_present = int((scan_counts > 0).sum())
    print(f'Classes present: {n_classes_present} / {len(class_names)}')

    header = f'{"class":<12}{"scans":>10}{"points":>14}'
    print(header)
    print('-' * len(header))
    for name, n_scans, n_points in zip(class_names, scan_counts, point_counts):
        print(f'{name:<12}{n_scans:>10}{n_points:>14}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split_file', default='cfg/data_split_thuman.json')
    parser.add_argument('--splits', nargs='*', default=['train', 'test'])
    args = parser.parse_args()

    with open(args.split_file) as f:
        split_data = json.load(f)

    for split_name in args.splits:
        print_split_report(split_name, split_data[split_name], CLASS_NAMES)


if __name__ == '__main__':
    main()
