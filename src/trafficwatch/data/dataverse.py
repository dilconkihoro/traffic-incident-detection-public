"""Harvard Dataverse's documented public API (verified 2026-09-22 against DOI 10.7910/DVN/POREXF,
the DhakaAI citation in docs/03_DATA_PIPELINE.md): a dataset's metadata, including its licence
and file listing, is a public unauthenticated GET; a file download is a public unauthenticated
redirect to a time-limited object-store URL. No account, login, or request step is involved for
a dataset whose licence metadata reports fileAccessRequest: false, as DhakaAI's does.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

API_ROOT = "https://dataverse.harvard.edu"
# Dataverse's edge blocks the default urllib User-Agent (verified 2026-09-22); an identifying
# one is enough, no need to imitate a browser.
USER_AGENT = (
    "trafficwatch-acquire-script/1.0 (+https://github.com/dilconkihoro/traffic-incident-detection)"
)
Opener = Callable[[str], IO[bytes]]


class DataverseError(ValueError):
    pass


@dataclass(frozen=True)
class DataverseFile:
    file_id: int
    filename: str
    size_bytes: int
    restricted: bool


def metadata_url(doi: str) -> str:
    return f"{API_ROOT}/api/datasets/:persistentId/?persistentId=doi:{doi}"


def access_url(file_id: int) -> str:
    return f"{API_ROOT}/api/access/datafile/{file_id}"


def parse_licence(payload: dict[str, Any]) -> str:
    """The published licence name and URI, as the dataset's own metadata states it."""
    try:
        licence = payload["data"]["latestVersion"]["license"]
        return f"{licence['name']} ({licence['uri']})"
    except (KeyError, TypeError) as exc:
        raise DataverseError("dataset metadata has no license block") from exc


def parse_title(payload: dict[str, Any]) -> str:
    try:
        version = payload["data"]["latestVersion"]
        return version["metadataBlocks"]["citation"]["fields"][0]["value"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DataverseError("dataset metadata has no citation title") from exc


def requires_access_request(payload: dict[str, Any]) -> bool:
    try:
        return bool(payload["data"]["latestVersion"]["fileAccessRequest"])
    except (KeyError, TypeError) as exc:
        raise DataverseError("dataset metadata has no fileAccessRequest flag") from exc


def parse_files(payload: dict[str, Any]) -> tuple[DataverseFile, ...]:
    try:
        entries = payload["data"]["latestVersion"]["files"]
    except (KeyError, TypeError) as exc:
        raise DataverseError("dataset metadata has no file listing") from exc
    files = []
    for entry in entries:
        data_file = entry["dataFile"]
        files.append(
            DataverseFile(
                file_id=data_file["id"],
                filename=data_file["filename"],
                size_bytes=data_file.get("filesize", 0),
                restricted=bool(entry.get("restricted", False)),
            )
        )
    return tuple(files)


def fetch_metadata(
    doi: str, *, opener: Opener | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    """The dataset's own metadata document; the network boundary, thin by design."""
    url = metadata_url(doi)
    if opener is not None:
        with opener(url) as handle:
            data = handle.read()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as handle:  # noqa: S310 - fixed https host
            data = handle.read()
    payload = json.loads(data)
    if payload.get("status") != "OK":
        raise DataverseError(f"unexpected response from {url}: {payload}")
    return payload


def download_file(
    file_id: int,
    dest: Path,
    *,
    opener: Opener | None = None,
    chunk_bytes: int = 1 << 20,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    """Stream one file to ``dest``; refuses to overwrite an archive already on disk."""
    if dest.exists():
        raise DataverseError(f"refusing to overwrite an existing file: {dest}")
    url = access_url(file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if opener is not None:
        opened = opener(url)
    else:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        opened = urllib.request.urlopen(request, timeout=60)  # noqa: S310 - fixed https host
    with opened as source, open(dest, "wb") as out:
        while chunk := source.read(chunk_bytes):
            out.write(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))


def refuse_if_gated(payload: dict[str, Any], files: Iterable[DataverseFile]) -> None:
    """Fail closed: only proceed on a dataset the API itself reports needs no access request."""
    if requires_access_request(payload):
        raise DataverseError(
            "this dataset's own metadata reports fileAccessRequest: true; "
            "it is not a documented no-account source, download it manually"
        )
    if any(f.restricted for f in files):
        raise DataverseError("a file in this dataset is marked restricted; download it manually")
