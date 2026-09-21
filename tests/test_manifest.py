"""Dataset manifest: hashing, immutability of recorded archives, completeness check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trafficwatch.data import manifest as mf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_datasets.py"
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "sample.zip"
    path.write_bytes(b"abc")
    return path


def test_sha256_matches_known_vector(archive):
    assert mf.sha256_file(archive) == ABC_SHA256


def test_sha256_streams_across_chunk_boundaries(archive):
    assert mf.sha256_file(archive, chunk_bytes=2) == ABC_SHA256


def test_build_entry_records_hash_size_and_defaults_to_verify(archive):
    entry = mf.build_entry(archive, dataset="ua-detrac", retrieval="manual")
    assert entry["sha256"] == ABC_SHA256
    assert entry["size_bytes"] == 3
    assert entry["licence"] == mf.VERIFY


@pytest.mark.parametrize("retrieved", ["21-09-2026", "yesterday"])
def test_retrieved_must_be_an_iso_date(archive, retrieved):
    with pytest.raises(mf.ManifestError, match="YYYY-MM-DD"):
        mf.build_entry(archive, dataset="d", retrieval="manual", retrieved=retrieved)


def test_unknown_retrieval_kind_and_missing_archive_are_rejected(archive, tmp_path):
    with pytest.raises(mf.ManifestError, match="retrieval"):
        mf.build_entry(archive, dataset="d", retrieval="download")
    with pytest.raises(mf.ManifestError, match="not found"):
        mf.build_entry(tmp_path / "nope.zip", dataset="d", retrieval="manual")


def test_changed_hash_for_a_recorded_archive_is_refused(archive):
    manifest = mf.empty_manifest()
    mf.add_entry(manifest, mf.build_entry(archive, dataset="d", retrieval="manual"))
    archive.write_bytes(b"abd")
    with pytest.raises(mf.ManifestError, match="immutable"):
        mf.add_entry(manifest, mf.build_entry(archive, dataset="d", retrieval="manual"))


def test_re_adding_fills_verify_fields_but_keeps_recorded_values(archive):
    manifest = mf.empty_manifest()
    mf.add_entry(manifest, mf.build_entry(archive, dataset="d", retrieval="manual", doi="10.1/x"))
    mf.add_entry(
        manifest,
        mf.build_entry(archive, dataset="d", retrieval="manual", doi="10.9/y", licence="L"),
    )
    (entry,) = manifest["sources"]
    assert entry["doi"] == "10.1/x"
    assert entry["licence"] == "L"


def test_incomplete_fields_lists_every_verify(archive):
    manifest = mf.empty_manifest()
    mf.add_entry(manifest, mf.build_entry(archive, dataset="d", retrieval="manual"))
    assert len(mf.incomplete_fields(manifest)) == 4


def test_save_and_load_round_trip_and_schema_check(tmp_path, archive):
    path = tmp_path / "MANIFEST.json"
    manifest = mf.empty_manifest()
    mf.add_entry(manifest, mf.build_entry(archive, dataset="d", retrieval="mirror"))
    mf.save_manifest(path, manifest)
    assert mf.load_manifest(path) == manifest
    path.write_text('{"schema_version": 99, "sources": []}', encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="schema_version"):
        mf.load_manifest(path)


def cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_cli_verify_then_strict_check_fails_until_metadata_is_complete(tmp_path, archive):
    path = tmp_path / "MANIFEST.json"
    mf.save_manifest(path, mf.empty_manifest())
    base = ["--manifest", str(path)]
    assert cli(*base, "check", "--strict").returncode == 0
    verify = cli(*base, "verify", str(archive), "--dataset", "d", "--retrieval", "manual")
    assert verify.returncode == 0
    assert ABC_SHA256 in verify.stdout
    strict = cli(*base, "check", "--strict")
    assert strict.returncode == 1
    assert "VERIFY: d/sample.zip: licence" in strict.stdout
    assert cli(*base, "check").returncode == 0
    full = ["--source-url", "u", "--doi", "10.1/x", "--licence", "L", "--retrieved", "2026-09-21"]
    assert (
        cli(
            *base, "verify", str(archive), "--dataset", "d", "--retrieval", "manual", *full
        ).returncode
        == 0
    )
    assert cli(*base, "check", "--strict").returncode == 0
