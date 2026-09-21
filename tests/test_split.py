"""Sequence-level splitting and the leakage check (rules 1 and 2, risk R-10)."""

from __future__ import annotations

import random

import pytest

from trafficwatch.data import split as sp
from trafficwatch.data.split import SequenceInfo


def make_sequences() -> list[SequenceInfo]:
    train = (
        [SequenceInfo(f"S{i:02d}", "train", "sunny") for i in range(6)]
        + [SequenceInfo(f"C{i:02d}", "train", "cloudy") for i in range(3)]
        + [SequenceInfo(f"R{i:02d}", "train", "rainy") for i in range(2)]
        + [SequenceInfo("N00", "train", "night")]
    )
    test = [SequenceInfo(f"T{i:02d}", "test", "sunny") for i in range(4)]
    return train + test


def test_val_is_stratified_seeded_and_only_from_official_train():
    seqs = make_sequences()
    val = sp.choose_val_sequences(seqs, n_val=6, seed=0)
    assert len(val) == 6
    assert all(v[0] in "SCRN" for v in val)
    weather = {s.seq_id: s.weather for s in seqs}
    assert {weather[v] for v in val} == {"sunny", "cloudy", "rainy", "night"}
    assert val == sp.choose_val_sequences(seqs, n_val=6, seed=0)
    assert val != sp.choose_val_sequences(seqs, n_val=6, seed=1)


def test_val_size_is_exact_when_quotas_need_rounding():
    for n_val in (4, 5, 7, 12):
        assert len(sp.choose_val_sequences(make_sequences(), n_val=n_val, seed=3)) == n_val


@pytest.mark.parametrize("n_val", [0, 13])
def test_impossible_val_size_is_rejected(n_val):
    with pytest.raises(ValueError, match="n_val"):
        sp.choose_val_sequences(make_sequences(), n_val=n_val, seed=0)


def test_clean_assignment_has_no_violations():
    seqs = make_sequences()
    assignment = sp.assign_splits(seqs, sp.choose_val_sequences(seqs, 6, 0))
    test_ids = [s.seq_id for s in seqs if s.official_split == "test"]
    assert sp.check_leakage(assignment, seqs, test_ids) == []
    assert sorted(set(assignment.values())) == ["test", "train", "val"]


def leaky(mutate):
    seqs = make_sequences()
    assignment = sp.assign_splits(seqs, sp.choose_val_sequences(seqs, 6, 0))
    test_ids = [s.seq_id for s in seqs if s.official_split == "test"]
    mutate(assignment)
    return sp.check_leakage(assignment, seqs, test_ids)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda a: a.__setitem__("T00", "train"), "leaks into training or validation"),
        (lambda a: a.__setitem__("T01", "val"), "leaks into training or validation"),
        (lambda a: a.__setitem__("S00", "test"), "not in the official test list"),
        (lambda a: a.pop("S01"), "is in no split"),
        (lambda a: a.__setitem__("T02", "holdout"), "unknown split"),
        (lambda a: a.__setitem__("X99", "train"), "not in the dataset listing"),
    ],
    ids=["test_in_train", "test_in_val", "train_in_test", "unassigned", "bad_split", "unknown_id"],
)
def test_leakage_is_detected(mutate, fragment):
    assert any(fragment in v for v in leaky(mutate))


def test_val_sequence_from_official_test_is_flagged():
    seqs = make_sequences()
    assignment = sp.assign_splits(seqs, ["T00"])
    assert assignment["T00"] == "test"
    assignment["T00"] = "val"
    violations = sp.check_leakage(assignment, seqs, ["T00", "T01", "T02", "T03"])
    assert any("not from the official train set" in v for v in violations)


def test_split_manifest_lists_sorted_ids_per_split():
    manifest = sp.split_manifest(
        {"b": "val", "a": "train", "c": "train"}, seed=0, stride=5, git_sha=None
    )
    assert manifest["sequences"] == {"train": ["a", "c"], "val": ["b"], "test": []}
    assert manifest["git_sha"] is None


HASHES = {
    "a1": 0b0000_0000,
    "a2": 0b0000_0011,  # 2 bits from a1: same cluster at distance 4
    "a3": 0b0000_1111,  # 2 bits from a2, 4 from a1: chained into the same cluster
    "b1": (1 << 60) - 1,  # far from everything in the a cluster
}


def test_clusters_join_by_distance_and_chain():
    clusters = sp.cluster_by_hash(HASHES, max_distance=4)
    assert clusters["a1"] == clusters["a2"] == clusters["a3"]
    assert clusters["b1"] != clusters["a1"]
    tight = sp.cluster_by_hash(HASHES, max_distance=1)
    assert len({tight["a1"], tight["a2"], tight["a3"]}) == 3


def test_whole_clusters_are_assigned_and_no_near_duplicate_straddles():
    rng = random.Random(0)
    bases = [rng.getrandbits(64) for _ in range(10)]
    hashes = {f"img{i}": bases[i // 4] ^ (i % 4) for i in range(40)}
    clusters = sp.cluster_by_hash(hashes, max_distance=4)
    assignment = sp.assign_clusters(
        clusters, {"train": 0.8, "dhaka_val": 0.1, "dhaka_test": 0.1}, seed=0
    )
    assert sp.check_hash_leakage(hashes, assignment, max_distance=4) == []
    counts = {s: list(assignment.values()).count(s) for s in set(assignment.values())}
    assert counts["train"] >= 28


def test_hash_leakage_is_detected():
    assignment = {"a1": "train", "a2": "dhaka_val", "a3": "train", "b1": "train"}
    violations = sp.check_hash_leakage(HASHES, assignment, max_distance=4)
    assert any("a1" in v and "a2" in v for v in violations)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        sp.assign_clusters({"a": 0}, {"train": 0.9, "dhaka_val": 0.2}, seed=0)
