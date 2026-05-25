from collections import defaultdict

import numpy as np


def sample_balanced_shot_indices(labels, groups, shots, seed):
    shots = int(shots)
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length.")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    by_label_group = defaultdict(lambda: defaultdict(list))
    for idx, (label, group) in enumerate(zip(labels.tolist(), groups)):
        by_label_group[int(label)][group].append(idx)

    sampled = []
    for label in sorted(by_label_group):
        group_indices = by_label_group[label]
        capacities = {
            group: len(indices)
            for group, indices in group_indices.items()
        }
        target = min(shots, sum(capacities.values()))
        target_by_group = balanced_counts(capacities, target)

        for group in sorted(target_by_group, key=repr):
            sample_count = target_by_group[group]
            if sample_count == 0:
                continue
            indices = np.asarray(group_indices[group], dtype=np.int64)
            chosen = rng.choice(indices, size=sample_count, replace=False)
            sampled.extend(chosen.tolist())

    sampled = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(sampled)
    return sampled


def sample_balanced_fraction_indices(indices, groups, fraction, seed):
    fraction = float(fraction)
    if fraction <= 0:
        raise ValueError("fraction must be positive.")
    if fraction > 1:
        raise ValueError("fraction must not exceed 1.")

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return np.asarray([], dtype=np.int64)

    target = max(1, int(np.round(len(indices) * fraction)))
    target = min(target, len(indices))

    rng = np.random.default_rng(seed)
    by_label_group = defaultdict(lambda: defaultdict(list))
    for idx in indices.tolist():
        group = groups[int(idx)]
        by_label_group[group_label_key(group)][group].append(int(idx))

    label_capacities = {
        label: sum(len(group_indices) for group_indices in group_map.values())
        for label, group_map in by_label_group.items()
    }
    target_by_label = balanced_counts(label_capacities, target)

    sampled = []
    for label in sorted(target_by_label, key=repr):
        label_target = target_by_label[label]
        if label_target == 0:
            continue
        group_map = by_label_group[label]
        group_capacities = {
            group: len(group_indices)
            for group, group_indices in group_map.items()
        }
        target_by_group = balanced_counts(group_capacities, label_target)
        for group in sorted(target_by_group, key=repr):
            sample_count = target_by_group[group]
            if sample_count == 0:
                continue
            group_indices = np.asarray(group_map[group], dtype=np.int64)
            chosen = rng.choice(group_indices, size=sample_count, replace=False)
            sampled.extend(chosen.tolist())

    sampled = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(sampled)
    return sampled


def group_label_key(group):
    if isinstance(group, tuple):
        if len(group) == 1:
            return group[0]
        return group[:-1]
    return group


def balanced_counts(capacities, target):
    counts = {group: 0 for group in capacities}
    remaining = int(target)
    groups = sorted(capacities, key=repr)

    while remaining > 0:
        progressed = False
        for group in groups:
            if remaining == 0:
                break
            if counts[group] >= capacities[group]:
                continue
            counts[group] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return counts
