"""Controlled vocabularies and validation for crowd-survey labels."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

VERDICTS = ("confirmed_drone", "false_positive", "unsure")
FP_CLASSES = (
    "wifi_ap",
    "wifi_client",
    "bluetooth",
    "ble",
    "zigbee",
    "video_sender_5g8",
    "cordless_phone",
    "microwave_oven",
    "rc_toy",
    "other",
)
DRONE_MODEL_SUGGESTIONS = (
    "DJI Mavic 3",
    "DJI Mini",
    "DJI Air",
    "Autel EVO",
    "analog FPV 5.8 GHz",
    "generic 2.4 GHz control link",
    "other/unknown",
)
SIGNATURE_SCHEMA_VERSION = "1.0"

LEGACY_TO_VERDICT = {
    "drone": "confirmed_drone",
    "true_positive": "confirmed_drone",
    "false_positive": "false_positive",
    "not_drone": "false_positive",
}
VERDICT_TO_LEGACY = {
    "confirmed_drone": "drone",
    "false_positive": "false_positive",
    "unsure": "",
}

_MAX_MODEL = 200
_MAX_FP_DETAIL = 200
_MAX_NOTES = 2000
_MAX_CONTRIBUTOR = 100


class LabelValidationError(ValueError):
    """A structured survey label failed validation."""


def vocabulary_payload() -> dict[str, list[str]]:
    """JSON-ready vocabulary for the dependency-free dashboard."""
    return {
        "verdicts": list(VERDICTS),
        "fp_classes": list(FP_CLASSES),
        "drone_models": list(DRONE_MODEL_SUGGESTIONS),
    }


def _text(value: Any, field: str, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length:
        raise LabelValidationError(
            f"{field} must be at most {max_length} characters"
        )
    return text


def validate_structured_label(
    *,
    verdict: Any,
    drone_model: Any = None,
    fp_class: Any = None,
    fp_class_detail: Any = None,
    label_notes: Any = None,
    labeled_by: Any = None,
) -> dict[str, str | None]:
    """Normalize a structured label and enforce conditional fields."""
    normalized_verdict = str(verdict or "").strip()
    if normalized_verdict not in VERDICTS:
        raise LabelValidationError(f"unknown verdict {normalized_verdict!r}")

    model = _text(drone_model, "drone_model", _MAX_MODEL)
    fp = _text(fp_class, "fp_class", _MAX_MODEL)
    fp_detail = _text(fp_class_detail, "fp_class_detail", _MAX_FP_DETAIL)
    notes = _text(label_notes, "label_notes", _MAX_NOTES)
    contributor = _text(labeled_by, "labeled_by", _MAX_CONTRIBUTOR)

    if normalized_verdict == "confirmed_drone":
        if model is None:
            raise LabelValidationError(
                "drone_model is required for confirmed_drone"
            )
        fp = None
        fp_detail = None
    elif normalized_verdict == "false_positive":
        if fp is None:
            raise LabelValidationError(
                "fp_class is required for false_positive"
            )
        if fp not in FP_CLASSES:
            raise LabelValidationError(f"unknown fp_class {fp!r}")
        if fp != "other":
            fp_detail = None
        model = None
    else:
        model = None
        fp = None
        fp_detail = None

    return {
        "verdict": normalized_verdict,
        "drone_model": model,
        "fp_class": fp,
        "fp_class_detail": fp_detail,
        "label_notes": notes,
        "labeled_by": contributor,
    }


def legacy_label_payload(feedback: str) -> dict[str, str | None] | None:
    """Convert the old flat labels without breaking existing callers."""
    label = (feedback or "").strip().lower()
    if not label:
        return None
    verdict = LEGACY_TO_VERDICT.get(label)
    if verdict is None:
        raise LabelValidationError(f"unknown label {feedback!r}")
    return validate_structured_label(
        verdict=verdict,
        drone_model="other/unknown" if verdict == "confirmed_drone" else None,
        fp_class="other" if verdict == "false_positive" else None,
    )


def export_json(records: Iterable[dict[str, Any]]) -> str:
    return json.dumps(list(records), indent=2, ensure_ascii=False) + "\n"


def export_csv(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    if not rows:
        return ""
    output = io.StringIO(newline="")
    fieldnames = list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        flat = {
            key: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        writer.writerow(flat)
    return output.getvalue()
