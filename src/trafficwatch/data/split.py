"""Sequence-level splits and the leakage check (the project rules ML rules 1 and 2)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SequenceInfo:
    seq_id: str
    official_split: str
    weather: str


def choose_val_sequences(
    sequences: Sequence[SequenceInfo], n_val: int, seed: int
) -> tuple[str, ...]:
    """Pick whole official-train sequences for validation, stratified by weather, seeded."""
    strata: dict[str, list[str]] = defaultdict(list)
    for seq in sequences:
        if seq.official_split == "train":
            strata[seq.weather].append(seq.seq_id)
    total = sum(len(ids) for ids in strata.values())
    if not 1 <= n_val <= total:
        raise ValueError(f"n_val must be between 1 and {total}, got {n_val}")

    names = sorted(strata)
    quota = {name: 1 if n_val >= len(names) else 0 for name in names}
    remaining = n_val - sum(quota.values())
    share = {name: remaining * len(strata[name]) / total for name in names}
    for name in names:
        quota[name] += min(int(share[name]), len(strata[name]) - quota[name])
    leftover = n_val - sum(quota.values())
    by_fraction = sorted(names, key=lambda n: (-(share[n] - int(share[n])), n))
    for name in by_fraction:
        if leftover == 0:
            break
        if quota[name] < len(strata[name]):
            quota[name] += 1
            leftover -= 1

    chosen: list[str] = []
    for name in names:
        ids = sorted(strata[name])
        random.Random(f"{seed}:{name}").shuffle(ids)
        chosen.extend(ids[: quota[name]])
    return tuple(sorted(chosen))


def assign_splits(sequences: Sequence[SequenceInfo], val_ids: Collection[str]) -> dict[str, str]:
    val = set(val_ids)
    return {
        seq.seq_id: "test"
        if seq.official_split == "test"
        else "val"
        if seq.seq_id in val
        else "train"
        for seq in sequences
    }


def check_leakage(
    assignment: Mapping[str, str],
    sequences: Sequence[SequenceInfo],
    official_test_ids: Collection[str],
) -> list[str]:
    """Return every violation; the build must stop if the list is not empty."""
    violations: list[str] = []
    known = {seq.seq_id: seq for seq in sequences}
    official_test = set(official_test_ids)

    for seq_id in sorted(set(known) - set(assignment)):
        violations.append(f"sequence {seq_id} is in no split")
    for seq_id, split in sorted(assignment.items()):
        if split not in SPLITS:
            violations.append(f"sequence {seq_id} has unknown split {split!r}")
        if seq_id not in known:
            violations.append(f"sequence {seq_id} is not in the dataset listing")
    in_test = {s for s, split in assignment.items() if split == "test"}
    for seq_id in sorted(in_test - official_test):
        violations.append(f"sequence {seq_id} is in test but not in the official test list")
    for seq_id in sorted(official_test - in_test):
        violations.append(f"official test sequence {seq_id} is not in the test split")
    for seq_id in sorted(official_test & {s for s, sp in assignment.items() if sp != "test"}):
        violations.append(f"official test sequence {seq_id} leaks into training or validation")
    for seq_id, split in sorted(assignment.items()):
        seq = known.get(seq_id)
        if split == "val" and seq is not None and seq.official_split != "train":
            violations.append(f"validation sequence {seq_id} is not from the official train set")
    return violations


def split_manifest(
    assignment: Mapping[str, str], *, seed: int, stride: int, git_sha: str | None
) -> dict[str, object]:
    return {
        "seed": seed,
        "stride": stride,
        "git_sha": git_sha,
        "sequences": {
            split: sorted(s for s, sp in assignment.items() if sp == split) for split in SPLITS
        },
    }


def cluster_by_hash(hashes: Mapping[str, int], max_distance: int) -> dict[str, int]:
    """Group images whose 64-bit perceptual hashes are within ``max_distance`` bits (union-find)."""
    ids = sorted(hashes)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1 :]:
            if (hashes[a] ^ hashes[b]).bit_count() <= max_distance:
                parent[find(b)] = find(a)
    roots = sorted({find(i) for i in ids})
    number = {root: n for n, root in enumerate(roots)}
    return {i: number[find(i)] for i in ids}


def assign_clusters(
    cluster_of: Mapping[str, int], fractions: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Assign whole clusters to splits so near-duplicates never straddle a boundary."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions.values())}")
    members: dict[int, list[str]] = defaultdict(list)
    for image, cluster in cluster_of.items():
        members[cluster].append(image)
    order = sorted(members)
    random.Random(seed).shuffle(order)

    total = len(cluster_of)
    filled = dict.fromkeys(fractions, 0)
    result: dict[str, str] = {}
    for cluster in order:
        target = max(fractions, key=lambda s: (fractions[s] * total - filled[s], s))
        for image in members[cluster]:
            result[image] = target
        filled[target] += len(members[cluster])
    return result


def check_hash_leakage(
    hashes: Mapping[str, int], assignment: Mapping[str, str], max_distance: int
) -> list[str]:
    ids = sorted(hashes)
    violations = []
    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1 :]:
            if (
                assignment[a] != assignment[b]
                and (hashes[a] ^ hashes[b]).bit_count() <= max_distance
            ):
                violations.append(
                    f"{a} ({assignment[a]}) and {b} ({assignment[b]}) are near-duplicates"
                )
    return violations
