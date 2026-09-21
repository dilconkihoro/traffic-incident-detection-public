"""Deterministic dataset archive for Colab: same tree in, same bytes and SHA-256 out."""

from __future__ import annotations

import tarfile
from pathlib import Path

from trafficwatch.data.manifest import sha256_file


def deterministic_tar(src_dir: str | Path, out_tar: str | Path) -> str:
    """Write ``src_dir`` as ``<name>/...`` with sorted entries and zeroed mtime and owner."""
    src_dir, out_tar = Path(src_dir), Path(out_tar)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {src_dir}")
    with tarfile.open(out_tar, "w", format=tarfile.GNU_FORMAT) as archive:
        for path in [src_dir, *sorted(src_dir.rglob("*"))]:
            arcname = (Path(src_dir.name) / path.relative_to(src_dir)).as_posix()
            info = tarfile.TarInfo(arcname)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            else:
                info.size = path.stat().st_size
                info.mode = 0o644
                with open(path, "rb") as handle:
                    archive.addfile(info, handle)
    return sha256_file(out_tar)


def write_sha256_file(tar_path: str | Path) -> Path:
    """Write ``<hash>  <name>`` beside the tar so ``sha256sum -c`` works from its directory."""
    tar_path = Path(tar_path)
    target = tar_path.with_suffix(".sha256")
    target.write_text(f"{sha256_file(tar_path)}  {tar_path.name}\n", encoding="utf-8")
    return target
