"""Deterministic ingestion of the CHLA cover-test video dataset.

Every physical video is one independent subject.  Patient identifiers remain
available solely for source auditing; ``video_uid`` and ``subject_uid`` are the
only modeling identifiers and are identical by design.

The implementation uses only the Python standard library, including a small
streaming OOXML reader, so the large source workbooks are never rewritten and
do not require Excel, pandas, or openpyxl at runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import csv
from hashlib import sha256
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .labels import (
    ClinicalManifestClass,
    HorizontalDirection,
    Job1ManifestLabel,
    MixedFlickPolicy,
    PrismDiopterInterval,
    horizontal_direction,
    WELL_CONTROLLED_MAXIMUM,
    job1_manifest_label,
    job3_clinical_class,
    normalize_horizontal_type,
    observable_in_clip,
    parse_horizontal_pd,
)


EXPECTED_VIDEO_COUNT = 846
# The v2 release (September 2026) removed 164 clips and added 96 edited ones.
EXPECTED_VIDEO_COUNT_V2 = 778
_AUTO_VIDEO_COUNT = -1


class DataValidationError(ValueError):
    """Raised when source files fail a deterministic join or schema check."""


@dataclass(frozen=True, slots=True)
class VideoSummary:
    n_frames: int
    fps: float
    duration_s: float
    faces_in_scene: int
    pct_frames_with_face: float
    pct_frames_flow_held: float
    mesh_stability: float
    occluder_type: str
    right_covered_frames: int
    left_covered_frames: int
    right_cover_episodes: int
    left_cover_episodes: int
    n_blinks: int
    total_events: int
    quality_flag: str


@dataclass(frozen=True, slots=True)
class EventAnnotation:
    video_uid: str
    sequence: int
    frame: int
    time_s: float
    event: str
    eye: str
    episode_frames: int
    episode_seconds: float


@dataclass(frozen=True, slots=True)
class CoverEpisode:
    """A paired cover/uncover episode.

    ``cover_frame=None`` denotes a clip that was already covered before its
    first annotated transition.  ``uncover_frame=None`` denotes a
    right-censored cover that remains open at the end of the clip.
    """

    video_uid: str
    eye: str
    cover_frame: int | None
    uncover_frame: int | None
    begins_covered: bool
    right_censored: bool


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    video_uid: str
    frame: int
    time_s: float
    face_detected: bool
    right_eye_state: str
    left_eye_state: str
    landmark_source: str
    right_gets_covered: bool
    right_gets_uncovered: bool
    left_gets_covered: bool
    left_gets_uncovered: bool
    blink: bool
    n_events: int


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_uid: str
    subject_uid: str
    cohort: str
    relative_path: str
    video_path: Path
    source_video_key: str
    filename: str
    patient_id: str
    exam_date: date
    condition: str
    distance: str
    #: "filename" when the condition code decides, "clinician_override" when a
    #: confirmed acquisition correction does (configs/distance_overrides_v2.csv).
    distance_source: str
    correction: str
    test_type: str
    repeat: int
    raw_horizontal_type: str
    raw_horizontal_pd: str | None
    #: Chart control score.  Defines the Job 1 target for intermittent tropias
    #: (labels.job1_manifest_label); barred from every feature view.
    control: float | None
    horizontal_direction: HorizontalDirection
    horizontal_pd: PrismDiopterInterval | None
    job1_manifest_label: Job1ManifestLabel | None
    observable_in_clip: bool | None
    job3_clinical_class: ClinicalManifestClass | None
    summary: VideoSummary
    first_right_eye_state: str
    first_left_eye_state: str
    last_right_eye_state: str
    last_left_eye_state: str
    begins_right_covered: bool | None
    begins_left_covered: bool | None
    ends_right_covered: bool | None
    ends_left_covered: bool | None
    right_censored_right_cover_events: int
    right_censored_left_cover_events: int

    @property
    def horizontal_pd_lower(self) -> float | None:
        return None if self.horizontal_pd is None else self.horizontal_pd.lower

    @property
    def horizontal_pd_upper(self) -> float | None:
        return None if self.horizontal_pd is None else self.horizontal_pd.upper


@dataclass(frozen=True, slots=True)
class ValidationReport:
    video_count: int
    event_count: int
    frame_count: int
    cover_episode_count: int
    clips_beginning_covered: int
    right_censored_cover_events: int


@dataclass(frozen=True, slots=True)
class IngestedDataset:
    root: Path
    manifest: tuple[VideoRecord, ...]
    events: tuple[EventAnnotation, ...]
    cover_episodes: tuple[CoverEpisode, ...]
    validation: ValidationReport
    records_by_uid: Mapping[str, VideoRecord]
    events_by_uid: Mapping[str, tuple[EventAnnotation, ...]]
    cover_episodes_by_uid: Mapping[str, tuple[CoverEpisode, ...]]
    _source_key_to_uid: Mapping[tuple[str, str], str]

    def iter_frames(self) -> Iterator[FrameAnnotation]:
        """Stream the joined frame annotations without retaining them in RAM."""

        return iter_frame_annotations(self)


@dataclass(frozen=True, slots=True)
class _CohortSpec:
    name: str
    # Candidate source tables in priority order; the first existing file wins.
    # The v2 (September 2026) release renamed the older-cohort tables.
    clinical_csv_candidates: tuple[str, ...]
    annotations_xlsx_candidates: tuple[str, ...]
    video_root: str

    def _resolve(self, root: Path, candidates: tuple[str, ...]) -> str:
        for candidate in candidates:
            if (root / candidate).is_file():
                return candidate
        return candidates[0]

    def clinical_csv(self, root: Path) -> str:
        return self._resolve(root, self.clinical_csv_candidates)

    def annotations_xlsx(self, root: Path) -> str:
        return self._resolve(root, self.annotations_xlsx_candidates)


_COHORTS = (
    _CohortSpec(
        "older",
        ("Older data/CHLA v2.csv", "Older data/CHLA.csv"),
        (
            "Older data/CHLA frame annotations v2.xlsx",
            "Older data/CHLA frame annotations.xlsx",
        ),
        "Older data/data",
    ),
    _CohortSpec(
        "new",
        ("New data/Additional CHLA dataset.csv",),
        ("New data/Additional frame annotations.xlsx",),
        "New data",
    ),
)

_VIDEO_SUFFIXES = (".mov", ".mp4")

# Clinician-confirmed near/distance corrections: clips whose filename says near but whose
# camera geometry shows a distance recording.  Only the near/distance covariate changes;
# no clinical label does.
_DISTANCE_OVERRIDE_CSV = Path(__file__).resolve().parents[2] / "configs" / "distance_overrides_v2.csv"


def _well_controlled_maximum() -> float:
    """Control score at or below which an intermittent tropia counts as orthotropic for Job 1.

    Defaults to 2.0 (clinician's rule).  The ``STRA_WELL_CONTROLLED_MAX`` environment variable
    overrides it for sensitivity analyses.
    """

    import os

    raw = os.environ.get("STRA_WELL_CONTROLLED_MAX")
    if raw is None:
        return WELL_CONTROLLED_MAXIMUM
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataValidationError(f"invalid STRA_WELL_CONTROLLED_MAX: {raw!r}") from exc
    if not 0.0 <= value <= 5.0:
        raise DataValidationError(f"STRA_WELL_CONTROLLED_MAX out of range: {value}")
    return value


def _distance_overrides() -> dict[str, str]:
    if not _DISTANCE_OVERRIDE_CSV.is_file():
        return {}
    overrides: dict[str, str] = {}
    with _DISTANCE_OVERRIDE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row.get("source_video_key") or "").strip()
            value = (row.get("corrected_distance") or "").strip()
            if key and value:
                if value not in {"near", "distance"}:
                    raise DataValidationError(f"invalid corrected_distance for {key!r}: {value!r}")
                overrides[key] = value
    return overrides

# Edited v2 clips append a free-text suffix after the optional repeat number,
# e.g. ``_SN``, ``_SN-1``, ``_split1_SN-1``, ``_mislabeled Nsc_SN`` or
# ``_replaceSN``.  The leading condition code stays authoritative.
_FILENAME_PATTERN = re.compile(
    r"^(?P<patient>\d+)_"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{2})_"
    r"(?P<condition>[DN](?:cc|sc)) "
    r"(?P<test>Cover-Uncover|Alternate-Cover)"
    r"(?: (?P<repeat>\d+))?"
    r"(?:_(?P<suffix>[^./]+))?"
    r"\.(?:mov|mp4)$"
)

_EVENTS = {
    "left_gets_covered",
    "left_gets_uncovered",
    "right_gets_covered",
    "right_gets_uncovered",
    "blink",
}
_EYE_STATES = {"covered", "uncovered", "no_face"}
_LANDMARK_SOURCES = {"detected", "flow_held", "none"}


@dataclass(frozen=True, slots=True)
class _ClinicalRow:
    video_uid: str
    cohort: str
    relative_path: str
    video_path: Path
    source_video_key: str
    filename: str
    patient_id: str
    exam_date: date
    condition: str
    distance: str
    correction: str
    test_type: str
    repeat: int
    raw_horizontal_type: str
    raw_horizontal_pd: str | None
    control: float | None
    distance_source: str
    horizontal_direction: HorizontalDirection
    horizontal_pd: PrismDiopterInterval | None
    job1_label: Job1ManifestLabel | None
    observable: bool | None
    job3_class: ClinicalManifestClass | None


@dataclass(slots=True)
class _FrameBoundary:
    count: int = 0
    first_right: str | None = None
    first_left: str | None = None
    last_right: str | None = None
    last_left: str | None = None


_XML_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XML_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XML_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class _XlsxReader:
    """Minimal streaming reader for the static OOXML tables in this dataset."""

    def __init__(self, path: Path):
        self.path = path
        self._archive = ZipFile(path)
        self._shared_strings = self._read_shared_strings()
        self._sheet_paths = self._read_sheet_paths()

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> _XlsxReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self._archive.namelist():
            return []
        root = ET.parse(self._archive.open("xl/sharedStrings.xml")).getroot()
        text_tag = f"{{{_XML_MAIN}}}t"
        return [
            "".join(node.text or "" for node in item.iter(text_tag)) for item in root
        ]

    def _read_sheet_paths(self) -> dict[str, str]:
        workbook = ET.parse(self._archive.open("xl/workbook.xml")).getroot()
        relationships = ET.parse(
            self._archive.open("xl/_rels/workbook.xml.rels")
        ).getroot()
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall(f"{{{_XML_PKG_REL}}}Relationship")
        }
        result: dict[str, str] = {}
        for sheet in workbook.findall(f".//{{{_XML_MAIN}}}sheet"):
            relation_id = sheet.attrib[f"{{{_XML_DOC_REL}}}id"]
            target = rel_targets[relation_id].lstrip("/")
            if not target.startswith("xl/"):
                target = posixpath.normpath(posixpath.join("xl", target))
            result[sheet.attrib["name"]] = target
        return result

    @staticmethod
    def _column_index(cell_reference: str) -> int:
        letters = "".join(
            character for character in cell_reference if character.isalpha()
        )
        index = 0
        for character in letters.upper():
            index = index * 26 + ord(character) - ord("A") + 1
        return index - 1

    def _cell_value(self, cell: ET.Element) -> Any:
        cell_type = cell.attrib.get("t", "n")
        if cell_type == "inlineStr":
            return "".join(node.text or "" for node in cell.iter(f"{{{_XML_MAIN}}}t"))
        value_node = cell.find(f"{{{_XML_MAIN}}}v")
        if value_node is None or value_node.text is None:
            return None
        text = value_node.text
        if cell_type == "s":
            return self._shared_strings[int(text)]
        if cell_type in {"str", "e"}:
            return text
        if cell_type == "b":
            return text == "1"
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number

    def iter_rows(self, sheet_name: str) -> Iterator[list[Any]]:
        try:
            sheet_path = self._sheet_paths[sheet_name]
        except KeyError as exc:
            raise DataValidationError(
                f"{self.path}: missing sheet {sheet_name!r}"
            ) from exc
        row_tag = f"{{{_XML_MAIN}}}row"
        cell_tag = f"{{{_XML_MAIN}}}c"
        with self._archive.open(sheet_path) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != row_tag:
                    continue
                values: list[Any] = []
                for cell in element.findall(cell_tag):
                    index = self._column_index(cell.attrib.get("r", "A1"))
                    if index >= len(values):
                        values.extend([None] * (index + 1 - len(values)))
                    values[index] = self._cell_value(cell)
                element.clear()
                if any(value is not None for value in values):
                    yield values

    def iter_dicts(self, sheet_name: str) -> Iterator[dict[str, Any]]:
        rows = self.iter_rows(sheet_name)
        try:
            header_values = next(rows)
        except StopIteration as exc:
            raise DataValidationError(
                f"{self.path}: sheet {sheet_name!r} is empty"
            ) from exc
        headers = [str(value) if value is not None else "" for value in header_values]
        if (
            not headers
            or any(not header for header in headers)
            or len(set(headers)) != len(headers)
        ):
            raise DataValidationError(
                f"{self.path}: sheet {sheet_name!r} has an invalid header"
            )
        for values in rows:
            yield {
                header: values[index] if index < len(values) else None
                for index, header in enumerate(headers)
            }


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field} must be an integer, got {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise DataValidationError(f"{field} must be an integer, got {value!r}")
    return int(number)


def _as_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise DataValidationError(f"{field} must be finite, got {value!r}")
    return number


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise DataValidationError(f"missing required field {field!r}")
    return str(value).strip()


def parse_condition(condition: str) -> tuple[str, str]:
    """Parse D/N and cc/sc into explicit distance and correction fields."""

    match = re.fullmatch(r"(?P<distance>[DN])(?P<correction>cc|sc)", condition)
    if match is None:
        raise DataValidationError(f"invalid examination condition: {condition!r}")
    distance = "distance" if match.group("distance") == "D" else "near"
    correction = (
        "with_correction" if match.group("correction") == "cc" else "without_correction"
    )
    return distance, correction


def make_video_uid(relative_path: str | PurePosixPath) -> str:
    """Create the stable opaque identifier used for both video and subject."""

    canonical = PurePosixPath(relative_path).as_posix()
    if (
        canonical.startswith("/")
        or canonical in {"", "."}
        or ".." in PurePosixPath(canonical).parts
    ):
        raise DataValidationError(
            f"video path must be a safe relative path: {relative_path!r}"
        )
    digest = sha256(
        f"strabismus-screen/video/v1\0{canonical}".encode("utf-8")
    ).hexdigest()
    return f"vid_{digest}"


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataValidationError(f"invalid ISO clinical date: {value!r}") from exc


def _parse_annotation_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%m.%d.%y").date()
    except ValueError as exc:
        raise DataValidationError(f"invalid annotation date: {value!r}") from exc


def _load_clinical_rows(
    root: Path,
    mixed_flick_policy: MixedFlickPolicy | str,
    apply_distance_overrides: bool = True,
) -> tuple[dict[tuple[str, str], _ClinicalRow], set[str]]:
    rows_by_source_key: dict[tuple[str, str], _ClinicalRow] = {}
    expected_relative_paths: set[str] = set()
    uids: set[str] = set()
    overrides = _distance_overrides() if apply_distance_overrides else {}
    applied_overrides: set[str] = set()

    for spec in _COHORTS:
        clinical_path = root / spec.clinical_csv(root)
        if not clinical_path.is_file():
            raise DataValidationError(f"missing clinical CSV: {clinical_path}")
        with clinical_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "Filename",
                "Patient ID",
                "Date",
                "Visual acuity",
                "Test type",
                "Horizontal type",
                "Horizontal PD",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise DataValidationError(
                    f"{clinical_path}: expected clinical columns {sorted(required)}"
                )
            for source_row in reader:
                filename = _required_text(source_row, "Filename")
                patient_id = _required_text(source_row, "Patient ID")
                match = _FILENAME_PATTERN.fullmatch(filename)
                if match is None:
                    raise DataValidationError(f"invalid video filename: {filename!r}")
                if match.group("patient") != patient_id:
                    raise DataValidationError(
                        f"filename/patient mismatch: {filename!r} vs {patient_id!r}"
                    )
                condition = _required_text(source_row, "Visual acuity")
                test_type = _required_text(source_row, "Test type")
                if condition != match.group("condition") or test_type != match.group(
                    "test"
                ):
                    raise DataValidationError(
                        f"filename protocol metadata disagree for {filename!r}"
                    )
                exam_date = _parse_iso_date(_required_text(source_row, "Date"))
                if exam_date != _parse_annotation_date(match.group("date")):
                    raise DataValidationError(
                        f"filename/date mismatch for {filename!r}"
                    )
                repeat = int(match.group("repeat") or "1")
                distance, correction = parse_condition(condition)
                source_video_key = f"{patient_id}/{filename}"
                distance_source = "filename"
                if source_video_key in overrides and overrides[source_video_key] != distance:
                    distance = overrides[source_video_key]
                    distance_source = "clinician_override"
                    applied_overrides.add(source_video_key)
                control_text = (source_row.get("Control") or "").strip()
                control = float(control_text) if control_text else None
                raw_horizontal_type = normalize_horizontal_type(
                    _required_text(source_row, "Horizontal type")
                )
                raw_pd_text = (source_row.get("Horizontal PD") or "").strip() or None
                pd_interval = parse_horizontal_pd(raw_pd_text, raw_horizontal_type)
                relative_path = PurePosixPath(
                    spec.video_root, patient_id, filename
                ).as_posix()
                video_path = root / Path(relative_path)
                if not video_path.is_file():
                    raise DataValidationError(f"missing video file: {video_path}")
                uid = make_video_uid(relative_path)
                if uid in uids:
                    raise DataValidationError(
                        f"video_uid collision for {relative_path}"
                    )
                uids.add(uid)
                source_key = (spec.name, source_video_key)
                if source_key in rows_by_source_key:
                    raise DataValidationError(
                        f"duplicate clinical video key: {source_key}"
                    )
                expected_relative_paths.add(relative_path)
                rows_by_source_key[source_key] = _ClinicalRow(
                    video_uid=uid,
                    cohort=spec.name,
                    relative_path=relative_path,
                    video_path=video_path,
                    source_video_key=source_video_key,
                    filename=filename,
                    patient_id=patient_id,
                    exam_date=exam_date,
                    condition=condition,
                    distance=distance,
                    correction=correction,
                    test_type=test_type,
                    repeat=repeat,
                    raw_horizontal_type=raw_horizontal_type,
                    raw_horizontal_pd=raw_pd_text,
                    control=control,
                    distance_source=distance_source,
                    horizontal_direction=horizontal_direction(raw_horizontal_type),
                    horizontal_pd=pd_interval,
                    job1_label=job1_manifest_label(
                        raw_horizontal_type,
                        test_type,
                        control,
                        well_controlled_maximum=_well_controlled_maximum(),
                    ),
                    observable=observable_in_clip(raw_horizontal_type, test_type),
                    job3_class=job3_clinical_class(
                        raw_horizontal_type, mixed_flick_policy=mixed_flick_policy
                    ),
                )

    actual_relative_paths = {
        path.relative_to(root).as_posix()
        for spec in _COHORTS
        for path in (root / spec.video_root).rglob("*")
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
    }
    if actual_relative_paths != expected_relative_paths:
        missing = sorted(expected_relative_paths - actual_relative_paths)
        orphaned = sorted(actual_relative_paths - expected_relative_paths)
        raise DataValidationError(
            f"clinical/video mismatch; missing={missing[:5]}, orphaned={orphaned[:5]}"
        )
    unknown = sorted(set(overrides) - applied_overrides - {r.source_video_key for r in rows_by_source_key.values()})
    if unknown:
        raise DataValidationError(f"distance overrides name videos absent from this dataset: {unknown[:5]}")
    return rows_by_source_key, expected_relative_paths


def _validate_annotation_metadata(
    source: _ClinicalRow,
    row: Mapping[str, Any],
    *,
    include_repeat: bool,
) -> None:
    checks = {
        "patient_id": source.patient_id,
        "test_type": source.test_type,
    }
    if "condition" in row:
        checks["condition"] = source.condition
    for field, expected in checks.items():
        if _required_text(row, field) != expected:
            raise DataValidationError(
                f"{source.source_video_key}: annotation {field} "
                "disagrees with clinical CSV"
            )
    if (
        "exam_date" in row
        and _parse_annotation_date(_required_text(row, "exam_date")) != source.exam_date
    ):
        raise DataValidationError(
            f"{source.source_video_key}: annotation date disagrees with clinical CSV"
        )
    if include_repeat and _as_int(row.get("repeat"), field="repeat") != source.repeat:
        raise DataValidationError(
            f"{source.source_video_key}: annotation repeat disagrees with filename"
        )


def _summary_from_row(row: Mapping[str, Any]) -> VideoSummary:
    return VideoSummary(
        n_frames=_as_int(row.get("n_frames"), field="n_frames"),
        fps=_as_float(row.get("fps"), field="fps"),
        duration_s=_as_float(row.get("duration_s"), field="duration_s"),
        faces_in_scene=_as_int(row.get("faces_in_scene"), field="faces_in_scene"),
        pct_frames_with_face=_as_float(
            row.get("pct_frames_with_face"), field="pct_frames_with_face"
        ),
        pct_frames_flow_held=_as_float(
            row.get("pct_frames_flow_held"), field="pct_frames_flow_held"
        ),
        mesh_stability=_as_float(row.get("mesh_stability"), field="mesh_stability"),
        occluder_type=_required_text(row, "occluder_type"),
        right_covered_frames=_as_int(
            row.get("right_covered_frames"), field="right_covered_frames"
        ),
        left_covered_frames=_as_int(
            row.get("left_covered_frames"), field="left_covered_frames"
        ),
        right_cover_episodes=_as_int(
            row.get("right_cover_episodes"), field="right_cover_episodes"
        ),
        left_cover_episodes=_as_int(
            row.get("left_cover_episodes"), field="left_cover_episodes"
        ),
        n_blinks=_as_int(row.get("n_blinks"), field="n_blinks"),
        total_events=_as_int(row.get("total_events"), field="total_events"),
        quality_flag=_required_text(row, "quality_flag"),
    )


def _canonical_eye(value: str, event: str) -> str:
    mapping = {"Right (OD)": "right", "Left (OS)": "left", "Both": "both"}
    try:
        eye = mapping[value]
    except KeyError as exc:
        raise DataValidationError(f"unsupported event eye: {value!r}") from exc
    if event == "blink" and eye != "both":
        raise DataValidationError("blink event must apply to both eyes")
    if event != "blink" and eye == "both":
        raise DataValidationError("cover transition cannot apply to both eyes")
    if event.startswith("right_") and eye != "right":
        raise DataValidationError("right-eye transition has a non-right eye label")
    if event.startswith("left_") and eye != "left":
        raise DataValidationError("left-eye transition has a non-left eye label")
    return eye


def _frame_values(
    source: _ClinicalRow,
    row: Mapping[str, Any],
) -> FrameAnnotation:
    right_state = _required_text(row, "right_eye_state")
    left_state = _required_text(row, "left_eye_state")
    landmark_source = _required_text(row, "landmark_source")
    if right_state not in _EYE_STATES or left_state not in _EYE_STATES:
        raise DataValidationError(
            f"{source.source_video_key}: unsupported eye state "
            f"{right_state!r}/{left_state!r}"
        )
    if landmark_source not in _LANDMARK_SOURCES:
        raise DataValidationError(
            f"{source.source_video_key}: unsupported landmark source "
            f"{landmark_source!r}"
        )
    flags = {
        field: bool(_as_int(row.get(field), field=field))
        for field in (
            "right_gets_covered",
            "right_gets_uncovered",
            "left_gets_covered",
            "left_gets_uncovered",
            "blink",
        )
    }
    n_events = _as_int(row.get("n_events"), field="n_events")
    if n_events != sum(flags.values()):
        raise DataValidationError(
            f"{source.source_video_key}: n_events does not equal event flags"
        )
    face_detected = bool(_as_int(row.get("face_detected"), field="face_detected"))
    if face_detected != (landmark_source != "none"):
        raise DataValidationError(
            f"{source.source_video_key}: face_detected/landmark_source disagree"
        )
    return FrameAnnotation(
        video_uid=source.video_uid,
        frame=_as_int(row.get("frame"), field="frame"),
        time_s=_as_float(row.get("time_s"), field="time_s"),
        face_detected=face_detected,
        right_eye_state=right_state,
        left_eye_state=left_state,
        landmark_source=landmark_source,
        right_gets_covered=flags["right_gets_covered"],
        right_gets_uncovered=flags["right_gets_uncovered"],
        left_gets_covered=flags["left_gets_covered"],
        left_gets_uncovered=flags["left_gets_uncovered"],
        blink=flags["blink"],
        n_events=n_events,
    )


def _pair_cover_episodes(
    video_uid: str,
    events: Iterable[EventAnnotation],
    *,
    begins_covered: Mapping[str, bool | None],
) -> tuple[CoverEpisode, ...]:
    episodes: list[CoverEpisode] = []
    for eye in ("right", "left"):
        open_frame: int | None = None
        open_begins_covered = False
        for event in events:
            if event.eye != eye:
                continue
            if event.event == f"{eye}_gets_covered":
                if open_frame is not None:
                    raise DataValidationError(
                        f"{video_uid}: repeated {eye} cover without an uncover"
                    )
                open_frame = event.frame
                open_begins_covered = (
                    event.frame == 0 and begins_covered.get(eye) is True
                )
            elif event.event == f"{eye}_gets_uncovered":
                if open_frame is None:
                    episodes.append(
                        CoverEpisode(
                            video_uid=video_uid,
                            eye=eye,
                            cover_frame=None,
                            uncover_frame=event.frame,
                            begins_covered=True,
                            right_censored=False,
                        )
                    )
                else:
                    episodes.append(
                        CoverEpisode(
                            video_uid=video_uid,
                            eye=eye,
                            cover_frame=open_frame,
                            uncover_frame=event.frame,
                            begins_covered=open_begins_covered,
                            right_censored=False,
                        )
                    )
                    open_frame = None
                    open_begins_covered = False
        if open_frame is not None:
            episodes.append(
                CoverEpisode(
                    video_uid=video_uid,
                    eye=eye,
                    cover_frame=open_frame,
                    uncover_frame=None,
                    begins_covered=open_begins_covered,
                    right_censored=True,
                )
            )
    return tuple(sorted(episodes, key=lambda item: (item.cover_frame or -1, item.eye)))


def _covered_flag(state: str) -> bool | None:
    if state == "no_face":
        return None
    return state == "covered"


def load_dataset(
    root: str | Path,
    *,
    mixed_flick_policy: MixedFlickPolicy | str = MixedFlickPolicy.PHORIA,
    expected_video_count: int | None = _AUTO_VIDEO_COUNT,
    apply_distance_overrides: bool = True,
) -> IngestedDataset:
    """Load, join, and validate the complete clip-level dataset.

    ``apply_distance_overrides`` applies the confirmed acquisition corrections in
    configs/distance_overrides_v2.csv to the near/distance covariate.  Pass False
    to reproduce results published before 2026-09-19.
    """

    dataset_root = Path(root).expanduser().resolve()
    clinical, physical_paths = _load_clinical_rows(dataset_root, mixed_flick_policy, apply_distance_overrides)
    if expected_video_count == _AUTO_VIDEO_COUNT:
        older = _COHORTS[0]
        expected_video_count = (
            EXPECTED_VIDEO_COUNT_V2
            if older.clinical_csv(dataset_root).endswith("v2.csv")
            else EXPECTED_VIDEO_COUNT
        )
    if expected_video_count is not None and len(clinical) != expected_video_count:
        raise DataValidationError(
            f"expected {expected_video_count} videos, found {len(clinical)}"
        )
    if len(physical_paths) != len(clinical):
        raise DataValidationError("one-video/one-subject invariant failed")

    summaries: dict[tuple[str, str], VideoSummary] = {}
    events_by_source: dict[tuple[str, str], list[EventAnnotation]] = defaultdict(list)
    boundaries: dict[tuple[str, str], _FrameBoundary] = defaultdict(_FrameBoundary)

    for spec in _COHORTS:
        annotation_path = dataset_root / spec.annotations_xlsx(dataset_root)
        if not annotation_path.is_file():
            raise DataValidationError(f"missing annotation workbook: {annotation_path}")
        with _XlsxReader(annotation_path) as workbook:
            for row in workbook.iter_dicts("video_summary"):
                source_video_key = _required_text(row, "video")
                source_key = (spec.name, source_video_key)
                try:
                    source = clinical[source_key]
                except KeyError as exc:
                    raise DataValidationError(
                        f"orphaned video_summary row: {source_key}"
                    ) from exc
                if source_key in summaries:
                    raise DataValidationError(
                        f"duplicate video_summary row: {source_key}"
                    )
                _validate_annotation_metadata(source, row, include_repeat=True)
                annotated_distance = _required_text(row, "distance").casefold()
                annotated_correction = (
                    _required_text(row, "correction").casefold().replace(" ", "_")
                )
                # The workbook derives distance from the same condition code, so this
                # cross-check must use the filename-derived value; a confirmed
                # acquisition override is expected to differ from both.
                filename_distance = parse_condition(source.condition)[0]
                if annotated_distance != filename_distance:
                    raise DataValidationError(
                        f"{source_video_key}: decoded distance disagrees with condition"
                    )
                if annotated_correction != source.correction:
                    raise DataValidationError(
                        f"{source_video_key}: decoded correction "
                        "disagrees with condition"
                    )
                summaries[source_key] = _summary_from_row(row)

            for row in workbook.iter_dicts("events"):
                source_video_key = _required_text(row, "video")
                source_key = (spec.name, source_video_key)
                try:
                    source = clinical[source_key]
                    summary = summaries[source_key]
                except KeyError as exc:
                    raise DataValidationError(
                        f"orphaned event row: {source_key}"
                    ) from exc
                _validate_annotation_metadata(source, row, include_repeat=False)
                event_name = _required_text(row, "event")
                if event_name not in _EVENTS:
                    raise DataValidationError(f"unsupported event: {event_name!r}")
                frame = _as_int(row.get("frame"), field="frame")
                if frame < 0 or frame >= summary.n_frames:
                    raise DataValidationError(
                        f"{source_video_key}: event frame {frame} is out of bounds"
                    )
                event = EventAnnotation(
                    video_uid=source.video_uid,
                    sequence=len(events_by_source[source_key]),
                    frame=frame,
                    time_s=_as_float(row.get("time_s"), field="time_s"),
                    event=event_name,
                    eye=_canonical_eye(_required_text(row, "eye"), event_name),
                    episode_frames=_as_int(
                        row.get("episode_frames"), field="episode_frames"
                    ),
                    episode_seconds=_as_float(
                        row.get("episode_seconds"), field="episode_seconds"
                    ),
                )
                events_by_source[source_key].append(event)

            for row in workbook.iter_dicts("frame_annotations"):
                source_video_key = _required_text(row, "video")
                source_key = (spec.name, source_video_key)
                try:
                    source = clinical[source_key]
                    summary = summaries[source_key]
                except KeyError as exc:
                    raise DataValidationError(
                        f"orphaned frame row: {source_key}"
                    ) from exc
                boundary = boundaries[source_key]
                frame = _frame_values(source, row)
                if frame.frame != boundary.count:
                    raise DataValidationError(
                        f"{source_video_key}: expected frame {boundary.count}, "
                        f"got {frame.frame}"
                    )
                if abs(frame.time_s - frame.frame / summary.fps) > 0.002:
                    raise DataValidationError(
                        f"{source_video_key}: frame/time_s disagree at "
                        f"frame {frame.frame}"
                    )
                if boundary.count == 0:
                    _validate_annotation_metadata(source, row, include_repeat=True)
                    boundary.first_right = frame.right_eye_state
                    boundary.first_left = frame.left_eye_state
                boundary.last_right = frame.right_eye_state
                boundary.last_left = frame.left_eye_state
                boundary.count += 1

    clinical_keys = set(clinical)
    if set(summaries) != clinical_keys:
        raise DataValidationError(
            f"video_summary join mismatch: {len(summaries)} summaries for "
            f"{len(clinical)} videos"
        )
    if set(boundaries) != clinical_keys:
        raise DataValidationError(
            f"frame join mismatch: {len(boundaries)} frame groups for "
            f"{len(clinical)} videos"
        )

    manifest: list[VideoRecord] = []
    all_events: list[EventAnnotation] = []
    all_episodes: list[CoverEpisode] = []
    events_by_uid: dict[str, tuple[EventAnnotation, ...]] = {}
    episodes_by_uid: dict[str, tuple[CoverEpisode, ...]] = {}

    for source in sorted(clinical.values(), key=lambda item: item.relative_path):
        source_key = (source.cohort, source.source_video_key)
        summary = summaries[source_key]
        boundary = boundaries[source_key]
        if boundary.count != summary.n_frames:
            raise DataValidationError(
                f"{source.source_video_key}: frame count {boundary.count} "
                f"!= summary {summary.n_frames}"
            )
        source_events = tuple(events_by_source[source_key])
        if len(source_events) != summary.total_events:
            raise DataValidationError(
                f"{source.source_video_key}: event count {len(source_events)} "
                f"!= summary {summary.total_events}"
            )
        if any(
            earlier.frame > later.frame
            for earlier, later in zip(source_events, source_events[1:])
        ):
            raise DataValidationError(
                f"{source.source_video_key}: events are not frame ordered"
            )
        first_right = boundary.first_right or "no_face"
        first_left = boundary.first_left or "no_face"
        last_right = boundary.last_right or "no_face"
        last_left = boundary.last_left or "no_face"
        begins_covered = {
            "right": _covered_flag(first_right),
            "left": _covered_flag(first_left),
        }
        episodes = _pair_cover_episodes(
            source.video_uid,
            source_events,
            begins_covered=begins_covered,
        )
        right_censored = Counter(
            episode.eye for episode in episodes if episode.right_censored
        )
        record = VideoRecord(
            video_uid=source.video_uid,
            subject_uid=source.video_uid,
            cohort=source.cohort,
            relative_path=source.relative_path,
            video_path=source.video_path,
            source_video_key=source.source_video_key,
            filename=source.filename,
            patient_id=source.patient_id,
            exam_date=source.exam_date,
            condition=source.condition,
            distance=source.distance,
            distance_source=source.distance_source,
            correction=source.correction,
            test_type=source.test_type,
            repeat=source.repeat,
            raw_horizontal_type=source.raw_horizontal_type,
            raw_horizontal_pd=source.raw_horizontal_pd,
            control=source.control,
            horizontal_direction=source.horizontal_direction,
            horizontal_pd=source.horizontal_pd,
            job1_manifest_label=source.job1_label,
            observable_in_clip=source.observable,
            job3_clinical_class=source.job3_class,
            summary=summary,
            first_right_eye_state=first_right,
            first_left_eye_state=first_left,
            last_right_eye_state=last_right,
            last_left_eye_state=last_left,
            begins_right_covered=begins_covered["right"],
            begins_left_covered=begins_covered["left"],
            ends_right_covered=_covered_flag(last_right),
            ends_left_covered=_covered_flag(last_left),
            right_censored_right_cover_events=right_censored["right"],
            right_censored_left_cover_events=right_censored["left"],
        )
        manifest.append(record)
        all_events.extend(source_events)
        all_episodes.extend(episodes)
        events_by_uid[source.video_uid] = source_events
        episodes_by_uid[source.video_uid] = episodes

    records_by_uid = {record.video_uid: record for record in manifest}
    if len(records_by_uid) != len(manifest) or any(
        record.subject_uid != record.video_uid for record in manifest
    ):
        raise DataValidationError("one-video/one-subject invariant failed")
    report = ValidationReport(
        video_count=len(manifest),
        event_count=len(all_events),
        frame_count=sum(boundary.count for boundary in boundaries.values()),
        cover_episode_count=len(all_episodes),
        clips_beginning_covered=sum(
            record.begins_right_covered is True or record.begins_left_covered is True
            for record in manifest
        ),
        right_censored_cover_events=sum(
            episode.right_censored for episode in all_episodes
        ),
    )
    source_uid = {key: source.video_uid for key, source in clinical.items()}
    return IngestedDataset(
        root=dataset_root,
        manifest=tuple(manifest),
        events=tuple(all_events),
        cover_episodes=tuple(all_episodes),
        validation=report,
        records_by_uid=MappingProxyType(records_by_uid),
        events_by_uid=MappingProxyType(events_by_uid),
        cover_episodes_by_uid=MappingProxyType(episodes_by_uid),
        _source_key_to_uid=MappingProxyType(source_uid),
    )


def build_manifest(
    root: str | Path,
    *,
    mixed_flick_policy: MixedFlickPolicy | str = MixedFlickPolicy.PHORIA,
    expected_video_count: int | None = _AUTO_VIDEO_COUNT,
) -> tuple[VideoRecord, ...]:
    """Build the validated deterministic manifest."""

    return load_dataset(
        root,
        mixed_flick_policy=mixed_flick_policy,
        expected_video_count=expected_video_count,
    ).manifest


def iter_frame_annotations(dataset: IngestedDataset) -> Iterator[FrameAnnotation]:
    """Stream every frame row joined to its opaque ``video_uid``."""

    for spec in _COHORTS:
        annotation_path = dataset.root / spec.annotations_xlsx(dataset.root)
        with _XlsxReader(annotation_path) as workbook:
            for row in workbook.iter_dicts("frame_annotations"):
                source_key = (spec.name, _required_text(row, "video"))
                try:
                    uid = dataset._source_key_to_uid[source_key]
                    record = dataset.records_by_uid[uid]
                except KeyError as exc:
                    raise DataValidationError(
                        f"orphaned streamed frame: {source_key}"
                    ) from exc
                source = _ClinicalRow(
                    video_uid=record.video_uid,
                    cohort=record.cohort,
                    relative_path=record.relative_path,
                    video_path=record.video_path,
                    source_video_key=record.source_video_key,
                    filename=record.filename,
                    patient_id=record.patient_id,
                    exam_date=record.exam_date,
                    condition=record.condition,
                    distance=record.distance,
                    correction=record.correction,
                    test_type=record.test_type,
                    repeat=record.repeat,
                    raw_horizontal_type=record.raw_horizontal_type,
                    raw_horizontal_pd=record.raw_horizontal_pd,
                    control=record.control,
                    distance_source=record.distance_source,
                    horizontal_direction=record.horizontal_direction,
                    horizontal_pd=record.horizontal_pd,
                    job1_label=record.job1_manifest_label,
                    observable=record.observable_in_clip,
                    job3_class=record.job3_clinical_class,
                )
                yield _frame_values(source, row)


__all__ = [
    "CoverEpisode",
    "DataValidationError",
    "EventAnnotation",
    "EXPECTED_VIDEO_COUNT",
    "FrameAnnotation",
    "IngestedDataset",
    "ValidationReport",
    "VideoRecord",
    "VideoSummary",
    "build_manifest",
    "iter_frame_annotations",
    "load_dataset",
    "make_video_uid",
    "parse_condition",
]
