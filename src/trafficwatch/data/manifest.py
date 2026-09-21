"""data/MANIFEST.json: provenance for every raw and derived dataset archive."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VERIFY = "VERIFY"
RETRIEVAL_KINDS = ("manual", "mirror", "derived")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_METADATA_FIELDS = ("source_url", "doi", "licence", "retrieved")
_ENTRY_FIELDS = ("dataset", "artifact", "sha256", "size_bytes", "retrieval", *_METADATA_FIELDS)


class ManifestError(ValueError):
    pass


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def empty_manifest() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "sources": []}


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(
        manifest.get("sources"), list
    ):
        raise ManifestError(f"manifest {path} does not match schema_version {SCHEMA_VERSION}")
    for entry in manifest["sources"]:
        missing = [f for f in _ENTRY_FIELDS if f not in entry]
        if missing:
            raise ManifestError(f"manifest entry {entry.get('artifact')!r} lacks {missing}")
    return manifest


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def manifest_hash(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_entry(
    archive: str | Path,
    *,
    dataset: str,
    retrieval: str,
    source_url: str = VERIFY,
    doi: str = VERIFY,
    licence: str = VERIFY,
    retrieved: str = VERIFY,
) -> dict[str, Any]:
    archive = Path(archive)
    if retrieval not in RETRIEVAL_KINDS:
        raise ManifestError(f"retrieval must be one of {list(RETRIEVAL_KINDS)}, got {retrieval!r}")
    if retrieved != VERIFY and not _DATE.match(retrieved):
        raise ManifestError(f"retrieved must be YYYY-MM-DD, got {retrieved!r}")
    if not archive.is_file():
        raise ManifestError(f"archive not found: {archive}")
    return {
        "dataset": dataset,
        "artifact": archive.name,
        "sha256": sha256_file(archive),
        "size_bytes": archive.stat().st_size,
        "retrieval": retrieval,
        "source_url": source_url,
        "doi": doi,
        "licence": licence,
        "retrieved": retrieved,
    }


def add_entry(manifest: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Add an entry, or fill in VERIFY metadata of an existing one. A changed hash is refused."""
    for existing in manifest["sources"]:
        if (existing["dataset"], existing["artifact"]) != (entry["dataset"], entry["artifact"]):
            continue
        if existing["sha256"] != entry["sha256"]:
            raise ManifestError(
                f"{entry['dataset']}/{entry['artifact']}: hash differs from the recorded one; "
                "raw archives are immutable"
            )
        for field in _METADATA_FIELDS:
            if existing[field] == VERIFY:
                existing[field] = entry[field]
        return manifest
    manifest["sources"].append(entry)
    return manifest


def incomplete_fields(manifest: dict[str, Any]) -> list[str]:
    return [
        f"{entry['dataset']}/{entry['artifact']}: {field}"
        for entry in manifest["sources"]
        for field in _METADATA_FIELDS
        if entry[field] == VERIFY
    ]
