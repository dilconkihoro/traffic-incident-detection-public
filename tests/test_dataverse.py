"""Harvard Dataverse API parsing and the fail-closed gate, offline against a real-shaped fixture.

The fixture mirrors the actual response from the DhakaAI dataset's metadata endpoint
(DOI 10.7910/DVN/POREXF, verified 2026-09-22): CC0 1.0, fileAccessRequest false, two files.
Network calls are never made in these tests; fetch_metadata and download_file take an injectable
opener.
"""

from __future__ import annotations

import io
import json

import pytest

from trafficwatch.data import dataverse as dv


def real_shaped_payload(*, file_access_request=False, restricted=(False, False)):
    return {
        "status": "OK",
        "data": {
            "latestVersion": {
                "fileAccessRequest": file_access_request,
                "license": {
                    "name": "CC0 1.0",
                    "uri": "http://creativecommons.org/publicdomain/zero/1.0",
                },
                "metadataBlocks": {
                    "citation": {"fields": [{"typeName": "title", "value": "DhakaAI"}]}
                },
                "files": [
                    {
                        "dataFile": {"id": 4080443, "filename": "test1.rar", "filesize": 51771445},
                        "restricted": restricted[0],
                    },
                    {
                        "dataFile": {
                            "id": 4080444,
                            "filename": "train.rar",
                            "filesize": 1443360509,
                        },
                        "restricted": restricted[1],
                    },
                ],
            }
        },
    }


def test_urls_are_built_from_the_doi_and_file_id():
    assert dv.metadata_url("10.7910/DVN/POREXF") == (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi:10.7910/DVN/POREXF"
    )
    assert dv.access_url(4080443) == "https://dataverse.harvard.edu/api/access/datafile/4080443"


def test_parses_licence_title_and_files_from_the_real_shaped_payload():
    payload = real_shaped_payload()
    assert dv.parse_licence(payload) == "CC0 1.0 (http://creativecommons.org/publicdomain/zero/1.0)"
    assert dv.parse_title(payload) == "DhakaAI"
    assert dv.requires_access_request(payload) is False
    files = dv.parse_files(payload)
    assert files == (
        dv.DataverseFile(4080443, "test1.rar", 51771445, False),
        dv.DataverseFile(4080444, "train.rar", 1443360509, False),
    )


@pytest.mark.parametrize(
    ("field", "fragment"),
    [("license", "license"), ("metadataBlocks", "citation title"), ("files", "file listing")],
)
def test_missing_field_is_reported_by_name(field, fragment):
    payload = real_shaped_payload()
    del payload["data"]["latestVersion"][field]
    parser = {
        "license": dv.parse_licence,
        "metadataBlocks": dv.parse_title,
        "files": dv.parse_files,
    }[field]
    with pytest.raises(dv.DataverseError, match=fragment):
        parser(payload)


def test_gated_dataset_is_refused():
    payload = real_shaped_payload(file_access_request=True)
    with pytest.raises(dv.DataverseError, match="fileAccessRequest"):
        dv.refuse_if_gated(payload, dv.parse_files(payload))


def test_restricted_file_is_refused():
    payload = real_shaped_payload(restricted=(True, False))
    with pytest.raises(dv.DataverseError, match="restricted"):
        dv.refuse_if_gated(payload, dv.parse_files(payload))


def test_open_dataset_passes_the_gate():
    payload = real_shaped_payload()
    dv.refuse_if_gated(payload, dv.parse_files(payload))  # does not raise


def test_non_ok_status_is_rejected():
    def opener(url):
        return io.BytesIO(json.dumps({"status": "ERROR", "message": "no such dataset"}).encode())

    with pytest.raises(dv.DataverseError, match="unexpected response"):
        dv.fetch_metadata("0000/DOES-NOT-EXIST", opener=opener)


def test_fetch_metadata_uses_the_injected_opener_and_parses_json():
    seen = {}

    def opener(url):
        seen["url"] = url
        return io.BytesIO(json.dumps(real_shaped_payload()).encode())

    payload = dv.fetch_metadata("10.7910/DVN/POREXF", opener=opener)
    assert seen["url"] == dv.metadata_url("10.7910/DVN/POREXF")
    assert dv.parse_title(payload) == "DhakaAI"


def test_download_streams_all_chunks_to_disk(tmp_path):
    content = b"a" * 2500

    def opener(url):
        return io.BytesIO(content)

    dest = tmp_path / "file.rar"
    seen_chunks = []
    dv.download_file(4080443, dest, opener=opener, chunk_bytes=1000, on_chunk=seen_chunks.append)
    assert dest.read_bytes() == content
    assert seen_chunks == [1000, 1000, 500]


def test_download_refuses_to_overwrite_an_existing_file(tmp_path):
    dest = tmp_path / "file.rar"
    dest.write_bytes(b"already here")

    def opener(url):
        raise AssertionError("should not be called")

    with pytest.raises(dv.DataverseError, match="overwrite"):
        dv.download_file(4080443, dest, opener=opener)
    assert dest.read_bytes() == b"already here"
