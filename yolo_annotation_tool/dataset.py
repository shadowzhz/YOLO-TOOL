"""YOLO dataset validation and label writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Iterable, Sequence


class DatasetError(ValueError):
    """Raised when a dataset does not satisfy the YOLO contract."""


@dataclass(frozen=True)
class DatasetReport:
    images: int
    labels: int
    missing_labels: tuple[str, ...]
    malformed_labels: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_labels and not self.malformed_labels and self.images > 0


@dataclass(frozen=True)
class DatasetSplitReport:
    source_images: int
    train_images: int
    val_images: int
    skipped_unlabeled: tuple[str, ...]
    yaml_path: Path


def prepare_project_structure(root: str | Path) -> Path:
    """Create the standard project folders without touching existing files."""
    root = Path(root).expanduser()
    if root.exists() and not root.is_dir():
        raise DatasetError(f"项目路径不是文件夹：{root}")
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("images", "labels", "train/images", "train/labels", "val/images", "val/labels"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _image_files(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _valid_label_line(line: str, class_count: int | None) -> bool:
    fields = line.split()
    is_detection = len(fields) == 5
    is_polygon = len(fields) >= 7 and len(fields[1:]) % 2 == 0
    if not (is_detection or is_polygon):
        return False
    try:
        cls = int(fields[0])
        values = [float(v) for v in fields[1:]]
    except ValueError:
        return False
    if class_count is not None and not 0 <= cls < class_count:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def validate_dataset(root: str | Path, class_count: int | None = None) -> DatasetReport:
    root = Path(root)
    images = _image_files(root / "images") if (root / "images").exists() else _image_files(root)
    labels_root = root / "labels"
    missing: list[str] = []
    malformed: list[str] = []
    labels = 0
    for image in images:
        label = labels_root / image.relative_to(root / "images").with_suffix(".txt") if (root / "images").exists() else image.with_suffix(".txt")
        if not label.exists():
            missing.append(str(image))
            continue
        labels += 1
        if any(not _valid_label_line(line, class_count) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()):
            malformed.append(str(label))
    return DatasetReport(len(images), labels, tuple(missing), tuple(malformed))


def write_detection_labels(path: str | Path, boxes: Iterable[Sequence[float]], classes: Iterable[int]) -> None:
    """Write normalized ``class x_center y_center width height`` rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, box in zip(classes, boxes):
        if len(box) != 4:
            raise DatasetError("A detection box must contain four normalized values")
        values = [float(v) for v in box]
        if not all(0 <= value <= 1 for value in values):
            raise DatasetError("YOLO box values must be normalized to [0, 1]")
        rows.append(f"{int(cls)} " + " ".join(f"{value:.6f}" for value in values))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def make_dataset_yaml(root: str | Path, names: Sequence[str], output: str | Path | None = None) -> Path:
    """Create a minimal Ultralytics dataset YAML without requiring PyYAML."""
    root = prepare_project_structure(root).resolve()
    target = Path(output) if output else root / "data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {root.as_posix()}", "train: train/images", "val: val/images", f"nc: {len(names)}", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def split_annotated_dataset(
    root: str | Path,
    names: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplitReport:
    """Copy labeled images and labels into the standard train/val folders.

    Images without a label file are deliberately excluded so unfinished images
    are not silently treated as negative samples. Empty label files are kept as
    valid negative samples.
    """
    if not 0 <= val_fraction < 1:
        raise DatasetError("验证集比例必须大于等于 0 且小于 1")
    root = prepare_project_structure(root)
    image_root = root / "images"
    label_root = root / "labels"
    labeled: list[tuple[Path, Path, Path]] = []
    skipped: list[str] = []
    for image in _image_files(image_root):
        relative = image.relative_to(image_root)
        label = label_root / relative.with_suffix(".txt")
        if not label.exists():
            skipped.append(relative.as_posix())
            continue
        labeled.append((image, label, relative))
    if not labeled:
        raise DatasetError("没有找到已标注图片。请先在 images 中完成至少一张图片的标注。")

    random.Random(seed).shuffle(labeled)
    if len(labeled) < 2 or val_fraction == 0:
        val_count = 0
    else:
        val_count = min(len(labeled) - 1, max(1, round(len(labeled) * val_fraction)))
    val_items = labeled[:val_count]
    train_items = labeled[val_count:]

    for split_name, items in (("train", train_items), ("val", val_items)):
        split_images = root / split_name / "images"
        split_labels = root / split_name / "labels"
        shutil.rmtree(split_images, ignore_errors=True)
        shutil.rmtree(split_labels, ignore_errors=True)
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)
        for image, label, relative in items:
            destination_image = split_images / relative
            destination_label = split_labels / relative.with_suffix(".txt")
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination_image)
            shutil.copy2(label, destination_label)

    yaml_path = make_dataset_yaml(root, names)
    return DatasetSplitReport(len(labeled), len(train_items), len(val_items), tuple(skipped), yaml_path)
