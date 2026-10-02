"""UA-DETRAC parsing against the real schema confirmed in the train and test annotation
archives (docs/03 section 2). The fixture below is a trimmed excerpt of the real
MVI_20011.xml (two frames, two targets), not invented."""

from __future__ import annotations

import pytest

from trafficwatch.data import ua_detrac

VALID = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_20011">
   <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
   <ignored_region>
      <box left="778.75" top="24.75" width="181.75" height="63.5"/>
      <box left="930.75" top="94.75" width="29.75" height="33.5"/>
   </ignored_region>
   <frame density="7" num="1">
      <target_list>
         <target id="1">
            <box left="592.75" top="378.8" width="160.05" height="162.2"/>
            <attribute vehicle_type="car" truncation_ratio="0.1"/>
         </target>
         <target id="3">
            <box left="545.2" top="88.27" width="35.25" height="30.08"/>
            <attribute vehicle_type="car" truncation_ratio="0"/>
         </target>
      </target_list>
   </frame>
   <frame density="7" num="2">
      <target_list>
         <target id="1">
            <box left="594.2" top="391.81" width="161.5" height="149.19"/>
            <attribute vehicle_type="car" truncation_ratio="0.2"/>
         </target>
      </target_list>
   </frame>
</sequence>
"""


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_parses_the_real_confirmed_schema(tmp_path):
    result = ua_detrac.parse_sequence(write(tmp_path, "MVI_20011.xml", VALID))
    assert result.name == "MVI_20011"
    assert result.camera_state == "unstable"
    assert result.weather == "sunny"  # the source attribute is sence_weather, a misspelling
    assert len(result.ignored_regions) == 2
    assert result.ignored_regions[0] == (778.75, 24.75, 778.75 + 181.75, 24.75 + 63.5)
    assert len(result.frames) == 2

    frame1 = result.frames[0]
    assert frame1.num == 1
    assert len(frame1.targets) == 2
    t1, t3 = frame1.targets
    assert t1.track_id == 1
    assert t1.xyxy == (592.75, 378.8, 592.75 + 160.05, 378.8 + 162.2)  # left/top/w/h -> xyxy
    assert t1.vehicle_type == "car"
    assert t1.truncation_ratio == 0.1
    assert t3.track_id == 3
    assert t3.truncation_ratio == 0.0

    # track id 1 persists across frames, matching the project's own "track IDs" claim
    assert result.frames[1].targets[0].track_id == 1


def test_truncation_ratio_defaults_to_zero_when_absent(tmp_path):
    minimal = VALID.replace(' truncation_ratio="0.1"', "")
    result = ua_detrac.parse_sequence(write(tmp_path, "x.xml", minimal))
    assert result.frames[0].targets[0].truncation_ratio == 0.0


def test_sequence_with_no_ignored_region_is_valid(tmp_path):
    no_ignored = VALID.replace(
        "<ignored_region>\n"
        '      <box left="778.75" top="24.75" width="181.75" height="63.5"/>\n'
        '      <box left="930.75" top="94.75" width="29.75" height="33.5"/>\n'
        "   </ignored_region>\n",
        "",
    )
    result = ua_detrac.parse_sequence(write(tmp_path, "x.xml", no_ignored))
    assert result.ignored_regions == ()


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda t: t.replace("<sequence ", "<sequenceX ").replace("</sequence>", "</sequenceX>"),
            "expected 'sequence'",
        ),
        (lambda t: t.replace('name="MVI_20011"', 'name=""'), "no name attribute"),
        (
            lambda t: t.replace(
                '<sequence_attribute camera_state="unstable" sence_weather="sunny"/>', ""
            ),
            "missing <sequence_attribute>",
        ),
        (
            lambda t: t.replace(' sence_weather="sunny"', ""),
            "missing camera_state or sence_weather",
        ),
        (lambda t: t.replace('num="1"', 'num="one"'), "not an integer"),
        (lambda t: t.replace('id="1"', 'id="one"', 1), "not an integer"),
        (
            lambda t: t.replace(
                '<box left="592.75" top="378.8" width="160.05" height="162.2"/>', "", 1
            ),
            "has no <box>",
        ),
        (lambda t: t.replace('width="160.05"', 'width="0"'), "non-positive width or height"),
        (
            lambda t: t.replace('<attribute vehicle_type="car" truncation_ratio="0.1"/>', "", 1),
            "has no <attribute>",
        ),
        (lambda t: t.replace('vehicle_type="car"', 'vehicle_type=""', 1), "has no vehicle_type"),
    ],
    ids=[
        "wrong_root_tag",
        "empty_sequence_name",
        "missing_sequence_attribute",
        "missing_weather",
        "non_integer_frame_num",
        "non_integer_target_id",
        "missing_box",
        "non_positive_box_size",
        "missing_attribute",
        "empty_vehicle_type",
    ],
)
def test_each_malformed_field_is_reported_by_name(tmp_path, mutate, fragment):
    with pytest.raises(ua_detrac.UaDetracParseError, match=fragment):
        ua_detrac.parse_sequence(write(tmp_path, "bad.xml", mutate(VALID)))


def test_a_sequence_with_no_frames_is_rejected(tmp_path):
    no_frames = VALID[: VALID.index("<frame")] + "</sequence>\n"
    with pytest.raises(ua_detrac.UaDetracParseError, match="no <frame>"):
        ua_detrac.parse_sequence(write(tmp_path, "empty.xml", no_frames))


def test_not_well_formed_xml_is_reported_not_crashed_on(tmp_path):
    with pytest.raises(ua_detrac.UaDetracParseError, match="not well-formed XML"):
        ua_detrac.parse_sequence(write(tmp_path, "broken.xml", "<sequence name='x'>"))


def test_parse_directory_separates_successes_from_failures_and_keeps_going(tmp_path):
    write(tmp_path, "MVI_20011.xml", VALID)
    write(tmp_path, "MVI_20012.xml", VALID)
    write(tmp_path, "broken.xml", "not xml at all")

    ok, failed = ua_detrac.parse_directory(tmp_path)
    assert len(ok) == 2
    assert [f.name for f, _ in failed] == ["broken.xml"]


def test_parse_directory_on_an_empty_directory(tmp_path):
    assert ua_detrac.parse_directory(tmp_path) == ([], [])
