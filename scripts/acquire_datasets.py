"""Verify raw dataset archives and record them in data/MANIFEST.json. Downloads nothing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trafficwatch.data import manifest as mf

DEFAULT_MANIFEST = Path("data/MANIFEST.json")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="hash one archive and record it")
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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except mf.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
