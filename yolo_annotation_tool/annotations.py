"""In-memory annotations and YOLO detection serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class Detection:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Polygon:
    class_id: int
    points: tuple[tuple[float, float], ...]


Annotation: TypeAlias = Detection | Polygon


def load_detections(path: str | Path) -> list[Detection]:
    path = Path(path)
    if not path.exists():
        return []
    annotations: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            annotations.append(Detection(int(fields[0]), *(float(value) for value in fields[1:])))
        except ValueError:
            continue
    return annotations


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load both YOLO boxes and polygon rows from one label file."""
    path = Path(path)
    if not path.exists():
        return []
    annotations: list[Annotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            try:
                annotations.append(Detection(int(fields[0]), *(float(value) for value in fields[1:])))
            except ValueError:
                pass
        elif len(fields) >= 7 and len(fields[1:]) % 2 == 0:
            try:
                values = [float(value) for value in fields[1:]]
                if all(0.0 <= value <= 1.0 for value in values):
                    annotations.append(Polygon(int(fields[0]), tuple(zip(values[::2], values[1::2]))))
            except ValueError:
                pass
    return annotations


def save_detections(path: str | Path, annotations: list[Detection]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        f"{item.class_id} {item.x_center:.6f} {item.y_center:.6f} {item.width:.6f} {item.height:.6f}"
        for item in annotations
    )
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def save_annotations(path: str | Path, annotations: list[Annotation]) -> None:
    """Persist mixed detection and segmentation annotations as YOLO rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for item in annotations:
        if isinstance(item, Detection):
            rows.append(f"{item.class_id} {item.x_center:.6f} {item.y_center:.6f} {item.width:.6f} {item.height:.6f}")
        else:
            values = " ".join(f"{value:.6f}" for point in item.points for value in point)
            rows.append(f"{item.class_id} {values}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
