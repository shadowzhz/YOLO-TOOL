"""SAM-assisted pre-annotation using the Ultralytics model API."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from threading import Event
from typing import Callable, Sequence

from .annotations import Annotation, Detection, Polygon, load_annotations, save_annotations
from .dataset import write_detection_labels


_APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
os.environ.setdefault("YOLO_CONFIG_DIR", str(_APP_ROOT / "config"))


class SamAnnotator:
    def __init__(self, model: str = "sam2_b.pt") -> None:
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
        self.model = SAM(model)

    def segment(
        self,
        image: str | Path,
        boxes: Sequence[Sequence[float]] | None = None,
        points: Sequence[Sequence[float]] | None = None,
        labels: Sequence[int] | None = None,
    ):
        kwargs = {}
        if boxes:
            kwargs["bboxes"] = boxes
        if points:
            kwargs["points"] = points
        if labels:
            kwargs["labels"] = labels
        return self.model(str(image), **kwargs)


class AutoAnnotator:
    """Detect objects with YOLO and use SAM to refine their masks."""

    def __init__(self, detector: str = "yolo11n.pt", segmenter: str = "sam2_b.pt", confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
        self.detector = YOLO(detector)
        self.segmenter: SamAnnotator | None = None
        self.segmenter_path = segmenter
        self.confidence = confidence
        self.last_batch_stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0, "labels": 0}

    def annotate_detection(self, image: str | Path, label_path: str | Path, class_ids: Sequence[int] | None = None) -> int:
        results = self.detector.predict(str(image), conf=self.confidence, verbose=False)
        if not results:
            write_detection_labels(label_path, [], [])
            return 0
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            write_detection_labels(label_path, [], [])
            return 0
        xywhn = boxes.xywhn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        if class_ids is not None:
            allowed = set(class_ids)
            pairs = [(box, int(cls)) for box, cls in zip(xywhn, classes) if int(cls) in allowed]
            xywhn = [box for box, _ in pairs]
            classes = [cls for _, cls in pairs]
        write_detection_labels(label_path, xywhn, [int(value) for value in classes])
        return len(xywhn)

    def annotate_segmentation(self, image: str | Path, label_path: str | Path, class_ids: Sequence[int] | None = None) -> int:
        """Write YOLO polygon labels from detector boxes refined by SAM."""
        from PIL import Image

        detection_results = self.detector.predict(str(image), conf=self.confidence, verbose=False)
        if not detection_results:
            write_detection_labels(label_path, [], [])
            return 0
        detected = detection_results[0].boxes
        if detected is None or len(detected) == 0:
            write_detection_labels(label_path, [], [])
            return 0
        xyxy = detected.xyxy.cpu().tolist()
        classes = [int(value) for value in detected.cls.cpu().tolist()]
        if class_ids is not None:
            allowed = set(class_ids)
            pairs = [(box, cls) for box, cls in zip(xyxy, classes) if cls in allowed]
            xyxy = [box for box, _ in pairs]
            classes = [cls for _, cls in pairs]
            if not xyxy:
                write_detection_labels(label_path, [], [])
                return 0
        if self.segmenter is None:
            self.segmenter = SamAnnotator(self.segmenter_path)
        segmented = self.segmenter.segment(image, boxes=xyxy)
        if not segmented or segmented[0].masks is None:
            write_detection_labels(label_path, [], [])
            return 0
        width, height = Image.open(image).size
        rows: list[str] = []
        polygons = segmented[0].masks.xy
        for cls, polygon in zip(classes, polygons):
            values = []
            for x, y in polygon:
                values.extend((float(x) / width, float(y) / height))
            if len(values) >= 6:
                rows.append(f"{cls} " + " ".join(f"{value:.6f}" for value in values))
        path = Path(label_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        return len(rows)

    def annotate_folder(
        self,
        images: str | Path,
        labels: str | Path,
        task: str = "detect",
        merge: str = "replace",
        progress_callback: Callable[[int, int, str, str], None] | None = None,
        stop_event: Event | None = None,
        pause_event: Event | None = None,
        class_ids: Sequence[int] | None = None,
        merge_iou: float = 0.45,
    ) -> int:
        """Annotate a folder with optional progress and cooperative controls.

        ``merge`` is one of ``replace``, ``skip`` or ``append``.  The return
        value remains the generated-label count for compatibility with the
        command-line and Tk frontends; detailed counters are available in
        ``last_batch_stats``.
        """
        images = Path(images)
        labels = Path(labels)
        count = 0
        image_files = sorted(p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        total = len(image_files)
        stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0, "labels": 0}
        self.last_batch_stats = stats
        for position, image in enumerate(image_files, 1):
            if stop_event and stop_event.is_set():
                break
            if pause_event:
                pause_event.wait()
                if stop_event and stop_event.is_set():
                    break
            relative = image.relative_to(images).with_suffix(".txt")
            target = labels / relative
            if merge == "skip" and target.exists() and target.read_text(encoding="utf-8").strip():
                stats["skipped"] += 1
                if progress_callback:
                    progress_callback(position, total, image.name, "跳过：已有标注")
                continue
            previous = target.read_text(encoding="utf-8").strip() if merge in {"append", "smart_dedup", "append_new"} and target.exists() else ""
            try:
                writer = self.annotate_segmentation if task == "segment" else self.annotate_detection
                generated = writer(image, target, class_ids)
                if merge in {"smart_dedup", "append_new"}:
                    manual = load_annotations_from_text(previous)
                    predicted = load_annotations(target)
                    combined = self._merge_annotations(manual, predicted, merge, merge_iou)
                    added = max(0, len(combined) - len(manual))
                    save_annotations(target, combined)
                    generated = added
                elif merge == "append" and previous:
                    generated_rows = target.read_text(encoding="utf-8").strip()
                    combined = "\n".join(row for row in (previous, generated_rows) if row)
                    target.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
                count += generated
                stats["labels"] += generated
                stats["success"] += 1
                state = f"完成：{generated} 条"
            except Exception as exc:
                stats["failed"] += 1
                state = f"失败：{exc}"
            stats["processed"] += 1
            if progress_callback:
                progress_callback(position, total, image.name, state)
        self.last_batch_stats = stats
        return count

    @staticmethod
    def _merge_annotations(manual: list[Annotation], predicted: list[Annotation], mode: str, threshold: float) -> list[Annotation]:
        result = list(manual)
        for candidate in predicted:
            if any(_bbox_iou(candidate, existing) >= threshold for existing in manual):
                continue
            if mode == "smart_dedup":
                replaced = False
                for index, existing in enumerate(result[len(manual):], len(manual)):
                    if _bbox_iou(candidate, existing) < threshold:
                        continue
                    if _annotation_area(candidate) > _annotation_area(existing):
                        result[index] = candidate
                    replaced = True
                    break
                if replaced:
                    continue
            result.append(candidate)
        return result


def load_annotations_from_text(text: str) -> list[Annotation]:
    """Parse a temporary label buffer using the same loader as disk labels."""
    if not text:
        return []
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return load_annotations(path)
    finally:
        path.unlink(missing_ok=True)


def _annotation_bbox(item: Annotation) -> tuple[float, float, float, float]:
    if isinstance(item, Detection):
        return (item.x_center - item.width / 2, item.y_center - item.height / 2, item.x_center + item.width / 2, item.y_center + item.height / 2)
    xs = [point[0] for point in item.points]
    ys = [point[1] for point in item.points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(first: Annotation, second: Annotation) -> float:
    ax1, ay1, ax2, ay2 = _annotation_bbox(first)
    bx1, by1, bx2, by2 = _annotation_bbox(second)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _annotation_area(item: Annotation) -> float:
    x1, y1, x2, y2 = _annotation_bbox(item)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
