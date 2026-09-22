"""Verify raw dataset archives and record them in data/MANIFEST.json.

Downloads only from a documented, stable, unauthenticated source (verified 2026-09-22): DhakaAI
on Harvard Dataverse (fetch-dhakaai). UA-DETRAC has no such source right now (its original host
is gone; the live replacement is Google Drive, which is not reliably scriptable, and a Kaggle
mirror needs the user's own account) and is never downloaded by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficwatch.data import dataverse as dv
from trafficwatch.data import manifest as mf

DEFAULT_MANIFEST = Path("data/MANIFEST.json")
DHAKAAI_DOI = "10.7910/DVN/POREXF"  # docs/03 section 2, Shihavuddin and Rashid (2020)


def _verify(args: argparse.Namespace) -> int:
    manifest = mf.load_manifest(args.manifest)
    entry = mf.build_entry(
        args.archive,
        dataset=args.dataset,
        retrieval=args.retrieval,
        source_url=args.source_url,
        doi=args.doi,
        licence=args.licence,
        retrieved=args.retrieved,
    )
    mf.add_entry(manifest, entry)
    mf.save_manifest(args.manifest, manifest)
    print(f"recorded {entry['dataset']}/{entry['artifact']} sha256={entry['sha256']}")
    return 0


def _check(args: argparse.Namespace) -> int:
    manifest = mf.load_manifest(args.manifest)
    gaps = mf.incomplete_fields(manifest)
    count, digest = len(manifest["sources"]), mf.manifest_hash(manifest)
    print(f"{count} archive(s) recorded, manifest hash {digest}")
    for gap in gaps:
        print(f"VERIFY: {gap}")
    return 1 if gaps and args.strict else 0


def _fetch_dhakaai(args: argparse.Namespace) -> int:
    payload = dv.fetch_metadata(args.doi)
    files = dv.parse_files(payload)
    dv.refuse_if_gated(payload, files)
    title, licence = dv.parse_title(payload), dv.parse_licence(payload)
    print(f"{title}: {len(files)} file(s), licence {licence}")
    for f in files:
        print(f"  {f.filename}  {f.size_bytes / 1e6:.1f} MB  id={f.file_id}")
    if args.dry_run:
        print("dry run: nothing downloaded")
        return 0

    manifest = mf.load_manifest(args.manifest)
    for f in files:
        dest = args.out / f.filename
        if dest.is_file():
            print(f"already on disk, skipping download: {dest}")
        else:
            print(f"downloading {f.filename} ({f.size_bytes / 1e6:.1f} MB) ...")
            dv.download_file(f.file_id, dest)
        entry = mf.build_entry(
            dest,
            dataset="dhakaai",
            retrieval="manual",
            source_url=f"https://doi.org/{args.doi}",
            doi=args.doi,
            licence=licence,
            retrieved=args.retrieved,
        )
        mf.add_entry(manifest, entry)
        print(f"recorded dhakaai/{entry['artifact']} sha256={entry['sha256']}")
    mf.save_manifest(args.manifest, manifest)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="hash one already-downloaded archive and record it")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--retrieval", required=True, choices=mf.RETRIEVAL_KINDS)
    verify.add_argument("--source-url", default=mf.VERIFY)
    verify.add_argument("--doi", default=mf.VERIFY)
    verify.add_argument("--licence", default=mf.VERIFY)
    verify.add_argument("--retrieved", default=mf.VERIFY, help="YYYY-MM-DD")
    verify.set_defaults(func=_verify)

    check = sub.add_parser("check", help="report entries with unresolved VERIFY fields")
    check.add_argument("--strict", action="store_true", help="exit 1 if any field is VERIFY")
    check.set_defaults(func=_check)

    fetch = sub.add_parser(
        "fetch-dhakaai", help="download DhakaAI from Harvard Dataverse and record it"
    )
    fetch.add_argument("--doi", default=DHAKAAI_DOI)
    fetch.add_argument("--out", type=Path, default=Path("data/raw/dhakaai"))
    fetch.add_argument("--retrieved", default=mf.VERIFY, help="YYYY-MM-DD")
    fetch.add_argument(
        "--dry-run", action="store_true", help="list files and licence, download nothing"
    )
    fetch.set_defaults(func=_fetch_dhakaai)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (mf.ManifestError, dv.DataverseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
