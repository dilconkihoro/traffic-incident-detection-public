"""Pascal VOC parsing against the real schema confirmed in train.rar (docs/03 section 2)."""

from __future__ import annotations

import pytest

from trafficwatch.data import voc

VALID = """<annotation>
\t<folder>Images</folder>
\t<filename>traffic-jam-feni-wb.jpg</filename>
\t<path>E:\\Datasets\\Dataset\\Images\\traffic-jam-feni-wb.jpg</path>
\t<source><database>Unknown</database></source>
\t<size><width>1300</width><height>728</height><depth>3</depth></size>
\t<segmented>0</segmented>
\t<object>
\t\t<name>truck</name>
\t\t<pose>Unspecified</pose>
\t\t<truncated>1</truncated>
\t\t<difficult>0</difficult>
\t\t<bndbox><xmin>814</xmin><ymin>60</ymin><xmax>1300</xmax><ymax>514</ymax></bndbox>
\t</object>
\t<object>
\t\t<name>three wheelers (CNG)</name>
\t\t<pose>Unspecified</pose>
\t\t<truncated>0</truncated>
\t\t<difficult>0</difficult>
\t\t<bndbox><xmin>333</xmin><ymin>247</ymin><xmax>539</xmax><ymax>432</ymax></bndbox>
\t</object>
</annotation>
"""

# The real anomaly found in train.rar: 231.xml holds YOLO-format label lines under a .xml name.
YOLO_TEXT_MISLABELLED_XML = """4 0.183988 0.853687 0.115473 0.147465
4 0.267129 0.947005 0.157044 0.105991
4 0.392225 0.518433 0.057737 0.080645
"""


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_the_real_confirmed_schema(tmp_path):
    result = voc.parse_voc_annotation(write(tmp_path, "100.xml", VALID))
    assert result.filename == "traffic-jam-feni-wb.jpg"
    assert (result.width, result.height) == (1300, 728)
    assert len(result.objects) == 2
    truck, cng = result.objects
    assert truck.name == "truck"
    assert truck.bndbox == (814.0, 60.0, 1300.0, 514.0)
    assert truck.truncated is True and truck.difficult is False
    assert cng.name == "three wheelers (CNG)"  # exact string: spaces and parentheses preserved


def test_truncated_and_difficult_default_to_false_when_absent(tmp_path):
    minimal = VALID.replace("<truncated>1</truncated>\n\t\t<difficult>0</difficult>\n\t\t", "")
    result = voc.parse_voc_annotation(write(tmp_path, "x.xml", minimal))
    assert result.objects[0].truncated is False
    assert result.objects[0].difficult is False


def test_the_real_mislabelled_yolo_text_file_is_reported_not_crashed_on(tmp_path):
    path = write(tmp_path, "231.xml", YOLO_TEXT_MISLABELLED_XML)
    with pytest.raises(voc.VocParseError, match="not well-formed XML"):
        voc.parse_voc_annotation(path)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda t: t.replace("<size>", "<sizeX>").replace("</size>", "</sizeX>"), "missing <size>"),
        (lambda t: t.replace("<width>1300</width>", ""), "missing <width>"),
        (lambda t: t.replace("<width>1300</width>", "<width>wide</width>"), "not an integer"),
        (
            lambda t: t.replace(
                "<filename>traffic-jam-feni-wb.jpg</filename>", "<filename></filename>"
            ),
            "missing or empty <filename>",
        ),
        (lambda t: t.replace("<name>truck</name>", "<name></name>"), "no <name>"),
        (
            lambda t: t.replace(
                "<bndbox><xmin>814</xmin><ymin>60</ymin><xmax>1300</xmax><ymax>514</ymax></bndbox>",
                "",
            ),
            "no <bndbox>",
        ),
        (lambda t: t.replace("<xmax>1300</xmax>", "<xmax>800</xmax>"), "degenerate box"),
        (lambda t: t.replace("<xmin>814</xmin>", "<xmin>wide</xmin>"), "not a number"),
    ],
    ids=[
        "missing_size",
        "missing_width",
        "non_integer_width",
        "empty_filename",
        "empty_object_name",
        "missing_bndbox",
        "degenerate_box",
        "non_numeric_coordinate",
    ],
)
def test_each_malformed_field_is_reported_by_name(tmp_path, mutate, fragment):
    with pytest.raises(voc.VocParseError, match=fragment):
        voc.parse_voc_annotation(write(tmp_path, "bad.xml", mutate(VALID)))


def test_parse_directory_separates_successes_from_failures_and_keeps_going(tmp_path):
    write(tmp_path, "100.xml", VALID)
    write(tmp_path, "101.xml", VALID)
    write(tmp_path, "231.xml", YOLO_TEXT_MISLABELLED_XML)

    ok, failed = voc.parse_directory(tmp_path)
    assert len(ok) == 2
    assert [f.name for f, _ in failed] == ["231.xml"]
    assert "not well-formed XML" in failed[0][1]


def test_parse_directory_on_an_empty_directory(tmp_path):
    assert voc.parse_directory(tmp_path) == ([], [])
