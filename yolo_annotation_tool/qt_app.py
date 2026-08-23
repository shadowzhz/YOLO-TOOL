"""Qt desktop reconstruction of the annotation, training, and validation workflow."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import re
import sys
import threading
from threading import Event

from .annotations import Annotation, Detection, Polygon, load_annotations, save_annotations
from .dataset import DatasetError, make_dataset_yaml, prepare_project_structure, split_annotated_dataset
from .export import ExportConfig, export_model
from .sam import AutoAnnotator, SamAnnotator
from .settings import AppSettings, SettingsStore
from .training import TrainingConfig, TrainingStopped, detect_training_device, train


try:
    from PySide6.QtCore import QPointF, QUrl, Qt, Signal
    from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QKeySequenceEdit,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QListWidgetItem,
        QScrollArea,
        QSplitter,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QStyle,
    )
except ImportError as exc:  # pragma: no cover - depends on local desktop runtime
    raise RuntimeError("Install the desktop extra: python -m pip install PySide6") from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class _TrainingLogStream:
    """Forward console output from Ultralytics to the training log widget."""

    _ansi_pattern = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self, callback) -> None:
        self._callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += self._ansi_pattern.sub("", text).replace("\r", "\n")
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            line = line.strip()
            if line:
                self._callback(line)
        return len(text)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            self._callback(line)
        self._buffer = ""


class ImageCanvas(QLabel):
    """A compact rectangle annotation canvas that reads and writes YOLO labels."""

    changed = Signal()
    sam_point_added = Signal(float, float, int)
    view_changed = Signal(float)
    selection_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__("请先打开图片目录")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background: #161b22; color: #c9d1d9;")
        self._image = QPixmap()
        self._annotations: list[Annotation] = []
        self._class_id = 0
        self._mode = "detect"
        self._start = None
        self._draft = None
        self._polygon_draft: list[tuple[float, float]] = []
        self._sam_points: list[tuple[float, float]] = []
        self._sam_labels: list[int] = []
        self._zoom = 1.0
        self._grid_enabled = False
        self._show_guide_lines = True
        self._guide_position: tuple[float, float] | None = None
        self._zoom_step_percent = 25
        self._keep_zoom_on_image_switch = False
        self._annotation_line_width = 2
        self._annotation_fill_enabled = False
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_start = None
        self._pan_origin = (0.0, 0.0)
        self._selected_index: int | None = None
        self._drag_vertex: tuple[int, int] | None = None
        self._undo_stack: list[tuple[list[Annotation], list[tuple[float, float]], int | None]] = []
        self._redo_stack: list[tuple[list[Annotation], list[tuple[float, float]], int | None]] = []

    def set_class_id(self, class_id: int) -> None:
        self._class_id = class_id

    def set_display_settings(
        self,
        *,
        show_guide_lines: bool,
        zoom_step_percent: int,
        keep_zoom_on_image_switch: bool,
        annotation_line_width: int,
        annotation_fill_enabled: bool,
    ) -> None:
        self._show_guide_lines = show_guide_lines
        self._zoom_step_percent = max(5, min(50, zoom_step_percent))
        self._keep_zoom_on_image_switch = keep_zoom_on_image_switch
        self._annotation_line_width = max(1, min(6, annotation_line_width))
        self._annotation_fill_enabled = annotation_fill_enabled
        self.update()

    @staticmethod
    def _class_color(class_id: int) -> str:
        palette = ("#52c41a", "#409eff", "#faad14", "#9254de", "#13c2c2", "#eb2f96", "#fa541c", "#2f54eb")
        return palette[class_id % len(palette)]

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._start = None
        self._draft = None
        self.select_annotation(-1)
        self._polygon_draft.clear()
        self._sam_points.clear()
        self._sam_labels.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.update()

    def set_image(self, image_path: Path, annotations: list[Annotation]) -> None:
        self._image = QPixmap(str(image_path))
        self._annotations = annotations
        self._start = None
        self._draft = None
        self.select_annotation(-1)
        self._sam_points.clear()
        self._sam_labels.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        if not self._keep_zoom_on_image_switch:
            self.reset_view()
        self.update()

    @property
    def annotations(self) -> list[Annotation]:
        return list(self._annotations)

    @property
    def selected_annotation(self) -> Annotation | None:
        if self._selected_index is None or not 0 <= self._selected_index < len(self._annotations):
            return None
        return self._annotations[self._selected_index]

    @property
    def selected_index(self) -> int | None:
        return self._selected_index

    def select_annotation(self, index: int) -> None:
        self._selected_index = index if 0 <= index < len(self._annotations) else None
        self.selection_changed.emit(self._selected_index if self._selected_index is not None else -1)
        self.update()

    def append_annotations(self, annotations: list[Annotation]) -> None:
        if not annotations:
            return
        self._record_change()
        self._annotations.extend(annotations)
        self.select_annotation(len(self._annotations) - 1)
        self.changed.emit()
        self.update()

    def nudge_selected(self, dx: float, dy: float) -> bool:
        item = self.selected_annotation
        if item is None:
            return False
        self._record_change()
        if isinstance(item, Detection):
            self._annotations[self._selected_index] = replace(item, x_center=max(0.0, min(1.0, item.x_center + dx)), y_center=max(0.0, min(1.0, item.y_center + dy)))
        else:
            self._annotations[self._selected_index] = Polygon(item.class_id, tuple((max(0.0, min(1.0, x + dx)), max(0.0, min(1.0, y + dy))) for x, y in item.points))
        self.changed.emit()
        self.update()
        return True

    @property
    def sam_points(self) -> list[tuple[float, float]]:
        return list(self._sam_points)

    @property
    def sam_labels(self) -> list[int]:
        return list(self._sam_labels)

    def clear_sam_points(self) -> None:
        self._sam_points.clear()
        self._sam_labels.clear()
        self.update()

    def undo_sam_point(self) -> bool:
        if not self._sam_points:
            return False
        self._sam_points.pop()
        self._sam_labels.pop()
        self.update()
        return True

    def apply_sam_result(self, payload: object) -> None:
        """Add a normalized SAM mask as either a box or a polygon."""
        points, output = payload  # type: ignore[misc]
        normalized = tuple((float(x), float(y)) for x, y in points)  # type: ignore[misc]
        if len(normalized) < 3:
            return
        self._record_change()
        if output == "detect":
            min_x = min(x for x, _ in normalized)
            max_x = max(x for x, _ in normalized)
            min_y = min(y for _, y in normalized)
            max_y = max(y for _, y in normalized)
            self._annotations.append(Detection(self._class_id, (min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y))
        else:
            self._annotations.append(Polygon(self._class_id, normalized))
        self.select_annotation(len(self._annotations) - 1)
        self._sam_points.clear()
        self._sam_labels.clear()
        self.changed.emit()
        self.update()

    def apply_sam_polygon(self, points: object) -> None:
        """Compatibility wrapper for callers that only return a polygon."""
        self.apply_sam_result((points, "segment"))

    def clear_annotations(self) -> None:
        if not self._annotations and not self._polygon_draft:
            return
        self._record_change()
        self._annotations.clear()
        self._polygon_draft.clear()
        self.select_annotation(-1)
        self.changed.emit()
        self.update()

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom * (1 + self._zoom_step_percent / 100))

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom / (1 + self._zoom_step_percent / 100))

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.view_changed.emit(self._zoom)
        self.update()

    def toggle_grid(self) -> None:
        self._grid_enabled = not self._grid_enabled
        self.update()

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.2, min(zoom, 8.0))
        self.view_changed.emit(self._zoom)
        self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self.changed.emit()
        self.update()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self.changed.emit()
        self.update()

    def _snapshot(self) -> tuple[list[Annotation], list[tuple[float, float]], int | None]:
        return list(self._annotations), list(self._polygon_draft), self._selected_index

    def _restore(self, state: tuple[list[Annotation], list[tuple[float, float]], int | None]) -> None:
        annotations, draft, selected = state
        self._annotations = list(annotations)
        self._polygon_draft = list(draft)
        self.select_annotation(selected if selected is not None else -1)

    def _record_change(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def delete_selected(self) -> None:
        if self._selected_index is None:
            return
        self._record_change()
        del self._annotations[self._selected_index]
        self.select_annotation(-1)
        self.changed.emit()
        self.update()

    def set_selected_class(self, class_id: int) -> bool:
        if self._selected_index is None:
            return False
        item = self._annotations[self._selected_index]
        if item.class_id == class_id:
            return True
        self._record_change()
        self._annotations[self._selected_index] = replace(item, class_id=class_id)
        self.changed.emit()
        self.update()
        return True

    def _draw_rect(self):
        if self._image.isNull():
            return None
        scale = min(self.width() / self._image.width(), self.height() / self._image.height()) * self._zoom
        width = self._image.width() * scale
        height = self._image.height() * scale
        left = (self.width() - width) / 2 + self._pan_x
        top = (self.height() - height) / 2 + self._pan_y
        return left, top, width, height, scale

    def _image_pos(self, point):
        area = self._draw_rect()
        if not area:
            return None
        left, top, width, height, _ = area
        x = (point.x() - left) / width
        y = (point.y() - top) / height
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x, y
        return None

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161b22"))
        area = self._draw_rect()
        if not area:
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return
        left, top, width, height, _ = area
        painter.drawPixmap(int(left), int(top), int(width), int(height), self._image)
        if self._grid_enabled:
            painter.setPen(QPen(QColor(255, 255, 255, 55), 1, Qt.PenStyle.DotLine))
            for step in range(1, 10):
                x = int(left + width * step / 10)
                y = int(top + height * step / 10)
                painter.drawLine(x, int(top), x, int(top + height))
                painter.drawLine(int(left), y, int(left + width), y)
        for index, item in enumerate(self._annotations):
            color = "#f0883e" if index == self._selected_index else self._class_color(item.class_id)
            painter.setPen(QPen(QColor(color), self._annotation_line_width + (1 if index == self._selected_index else 0)))
            fill = QColor(color)
            fill.setAlpha(40)
            painter.setBrush(fill if self._annotation_fill_enabled else Qt.BrushStyle.NoBrush)
            if isinstance(item, Detection):
                x = left + (item.x_center - item.width / 2) * width
                y = top + (item.y_center - item.height / 2) * height
                painter.drawRect(int(x), int(y), int(item.width * width), int(item.height * height))
                painter.drawText(int(x) + 3, int(y) + 16, str(item.class_id))
            else:
                points = [(left + x * width, top + y * height) for x, y in item.points]
                painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
                if points:
                    painter.drawText(int(points[0][0]) + 3, int(points[0][1]) + 16, str(item.class_id))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._show_guide_lines and self._guide_position is not None:
            guide_x, guide_y = self._guide_position
            if left <= guide_x <= left + width and top <= guide_y <= top + height:
                painter.setPen(QPen(QColor(64, 158, 255, 170), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(left), int(guide_y), int(left + width), int(guide_y))
                painter.drawLine(int(guide_x), int(top), int(guide_x), int(top + height))
        if self._polygon_draft:
            painter.setPen(QPen(QColor("#58a6ff"), 2, Qt.PenStyle.DashLine))
            points = [(left + x * width, top + y * height) for x, y in self._polygon_draft]
            for first, second in zip(points, points[1:]):
                painter.drawLine(int(first[0]), int(first[1]), int(second[0]), int(second[1]))
        if self._sam_points:
            for index, (x, y) in enumerate(self._sam_points):
                color = "#22c55e" if self._sam_labels[index] == 1 else "#ef4444"
                painter.setPen(QPen(QColor(color), 3))
                painter.drawEllipse(int(left + x * width - 5), int(top + y * height - 5), 10, 10)
        if self._selected_index is not None and 0 <= self._selected_index < len(self._annotations):
            selected = self._annotations[self._selected_index]
            if isinstance(selected, Polygon):
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#58a6ff"), 2))
                for x, y in selected.points:
                    painter.drawEllipse(int(left + x * width - 4), int(top + y * height - 4), 8, 8)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._draft:
            x1, y1, x2, y2 = self._draft
            painter.setPen(QPen(QColor("#58a6ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1)))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        point = self._image_pos(event.position())
        if self._mode == "sam" and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and point:
            self._sam_points.append(point)
            label = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self._sam_labels.append(label)
            self.sam_point_added.emit(point[0], point[1], label)
            self.update()
        elif self._mode == "segment" and event.button() == Qt.MouseButton.LeftButton and point:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                hit = self._hit_test(point)
                if hit is not None:
                    self.select_annotation(hit)
                return
            vertex = self._hit_vertex(point)
            if vertex is not None:
                self._record_change()
                shape_index, vertex_index = vertex
                self.select_annotation(shape_index)
                self._drag_vertex = (shape_index, vertex_index)
                return
            self._record_change()
            self._polygon_draft.append(point)
            self.update()
        elif self._mode == "segment" and event.button() == Qt.MouseButton.RightButton:
            self._finish_polygon()
        elif self._mode == "detect" and event.button() == Qt.MouseButton.RightButton:
            self._pan_start = event.position()
            self._pan_origin = (self._pan_x, self._pan_y)
        elif self._mode == "detect" and event.button() == Qt.MouseButton.LeftButton and point:
            hit = self._hit_test(point)
            if hit is not None:
                self.select_annotation(hit)
                return
            self.select_annotation(-1)
            self._start = event.position()
            self._draft = (self._start.x(), self._start.y(), self._start.x(), self._start.y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._show_guide_lines:
            self._guide_position = (event.position().x(), event.position().y())
        if self._drag_vertex:
            point = self._image_pos(event.position())
            if point:
                shape_index, vertex_index = self._drag_vertex
                item = self._annotations[shape_index]
                if isinstance(item, Polygon):
                    points = list(item.points)
                    points[vertex_index] = point
                    self._annotations[shape_index] = Polygon(item.class_id, tuple(points))
                    self.update()
        elif self._pan_start:
            self._pan_x = self._pan_origin[0] + event.position().x() - self._pan_start.x()
            self._pan_y = self._pan_origin[1] + event.position().y() - self._pan_start.y()
            self.update()
        elif self._mode == "detect" and self._start:
            self._draft = (self._start.x(), self._start.y(), event.position().x(), event.position().y())
            self.update()
        elif self._show_guide_lines:
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._guide_position = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_vertex:
            self._drag_vertex = None
            self.changed.emit()
            self.update()
            return
        if event.button() == Qt.MouseButton.RightButton and self._pan_start:
            self._pan_start = None
            return
        if self._mode == "segment":
            return
        if not self._start or event.button() != Qt.MouseButton.LeftButton:
            return
        start = self._image_pos(self._start)
        end = self._image_pos(event.position())
        self._start = None
        self._draft = None
        if start and end:
            x1, y1 = start
            x2, y2 = end
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if width > 0.003 and height > 0.003:
                self._record_change()
                self._annotations.append(Detection(self._class_id, (x1 + x2) / 2, (y1 + y2) / 2, width, height))
                self.select_annotation(len(self._annotations) - 1)
                self.changed.emit()
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1 + self._zoom_step_percent / 100
        if event.angleDelta().y() > 0:
            self._set_zoom(self._zoom * factor)
        elif event.angleDelta().y() < 0:
            self._set_zoom(self._zoom / factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._mode == "segment" and event.button() == Qt.MouseButton.LeftButton:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        if len(self._polygon_draft) < 3:
            self._polygon_draft.clear()
            self.update()
            return
        self._record_change()
        self._annotations.append(Polygon(self._class_id, tuple(self._polygon_draft)))
        self.select_annotation(len(self._annotations) - 1)
        self._polygon_draft.clear()
        self.changed.emit()
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._mode == "sam" and event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.undo_sam_point():
                event.accept()
                return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if self._mode == "sam" and self.undo_sam_point():
                event.accept()
                return
            self.undo()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Y:
            self.redo()
            event.accept()
            return
        nudge = {Qt.Key.Key_Left: (-0.005, 0.0), Qt.Key.Key_Right: (0.005, 0.0), Qt.Key.Key_Up: (0.0, -0.005), Qt.Key.Key_Down: (0.0, 0.005)}
        if event.key() in nudge:
            if self.nudge_selected(*nudge[event.key()]):
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def _hit_test(self, point: tuple[float, float]) -> int | None:
        x, y = point
        for index in range(len(self._annotations) - 1, -1, -1):
            item = self._annotations[index]
            if isinstance(item, Detection):
                if item.x_center - item.width / 2 <= x <= item.x_center + item.width / 2 and item.y_center - item.height / 2 <= y <= item.y_center + item.height / 2:
                    return index
            elif self._point_in_polygon(x, y, item.points):
                return index
        return None

    def _hit_vertex(self, point: tuple[float, float]) -> tuple[int, int] | None:
        area = self._draw_rect()
        if not area:
            return None
        _, _, width, height, _ = area
        tolerance = max(10 / width, 10 / height)
        x, y = point
        for shape_index in range(len(self._annotations) - 1, -1, -1):
            item = self._annotations[shape_index]
            if not isinstance(item, Polygon):
                continue
            for vertex_index, (vertex_x, vertex_y) in enumerate(item.points):
                if ((x - vertex_x) / tolerance) ** 2 + ((y - vertex_y) / tolerance) ** 2 <= 1:
                    return shape_index, vertex_index
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, points: tuple[tuple[float, float], ...]) -> bool:
        inside = False
        previous_x, previous_y = points[-1]
        for current_x, current_y in points:
            crosses = (current_y > y) != (previous_y > y)
            if crosses and x < (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x:
                inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside


class AnnotationPage(QWidget):
    status_changed = Signal(str)
    classes_changed = Signal(object)
    dataset_ready = Signal(str)
    stats_changed = Signal(int, int, int, int)
    sam_result_ready = Signal(object)
    sam_failed = Signal(str)
    sam_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.images: list[Path] = []
        self.visible_indices: list[int] = []
        self.index = -1
        self.project_root = QLineEdit()
        self.class_names = QLineEdit("目标")
        self.class_combo = QComboBox()
        self.class_list = QListWidget()
        self.class_list.setObjectName("classList")
        self.class_list.setMinimumHeight(110)
        self.annotation_list = QListWidget()
        self.annotation_list.setObjectName("currentAnnotationList")
        self.annotation_list.setMinimumHeight(110)
        self.class_color_swatch = QFrame()
        self.class_color_swatch.setObjectName("classColorSwatch")
        self.class_color_swatch.setFixedSize(22, 22)
        self.class_color_swatch.setToolTip("当前类别颜色")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("矩形框", "detect")
        self.mode_combo.addItem("分割多边形", "segment")
        self.mode_combo.addItem("SAM 点选", "sam")
        self.sam_model = QLineEdit("sam2_b.pt")
        self.sam_model.setPlaceholderText("SAM 模型文件或模型名")
        self.sam_output = QComboBox()
        self.sam_output.addItem("SAM 输出矩形框", "detect")
        self.sam_output.addItem("SAM 输出多边形", "segment")
        self.image_list = QListWidget()
        self.image_list.setObjectName("imageList")
        self.image_list.setMinimumHeight(180)
        self.image_status_filter = QComboBox()
        self.image_status_filter.addItem("全部状态", "all")
        self.image_status_filter.addItem("未标注", "unlabeled")
        self.image_status_filter.addItem("已标注", "labeled")
        self.image_status_filter.addItem("负样本", "negative")
        self.image_search = QLineEdit()
        self.image_search.setPlaceholderText("按文件名筛选图片")
        self.split_ratio = QComboBox()
        self.split_ratio.addItem("验证 20%", 0.2)
        self.split_ratio.addItem("验证 10%", 0.1)
        self.split_ratio.addItem("验证 30%", 0.3)
        self.status_stats_label = QLabel("图片总数: 0  已标注: 0  未标注: 0  负样本: 0")
        self.status_stats_label.setObjectName("statusStats")
        self.canvas = ImageCanvas()
        self.canvas.changed.connect(lambda: self.status_changed.emit("有未保存的标注"))
        self.canvas.sam_point_added.connect(self._sam_point_clicked)
        self.sam_result_ready.connect(self.canvas.apply_sam_result)
        self.sam_failed.connect(lambda message: self.status_changed.emit(f"SAM 失败：{message}"))
        self._sam_annotator: SamAnnotator | None = None
        self._sam_model_path = ""
        self._build()
        self.canvas.changed.connect(self._update_canvas_status)
        self.canvas.changed.connect(self.refresh_annotation_list)
        self.refresh_classes()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        header = QFrame()
        header.setObjectName("annotationHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 9, 12, 10)
        header_layout.setSpacing(7)
        title = QLabel("数据集标注")
        title.setObjectName("annotationTitle")
        header_layout.addWidget(title)
        project = QHBoxLayout()
        project.addWidget(QLabel("项目目录"))
        project.addWidget(self.project_root, 1)
        choose = QPushButton("打开目录")
        choose.clicked.connect(self.open_folder)
        project.addWidget(choose)
        generate_yaml = QPushButton("生成 data.yaml")
        generate_yaml.clicked.connect(self.generate_yaml)
        project.addWidget(generate_yaml)
        project.addWidget(self.split_ratio)
        split_dataset = QPushButton("划分训练/验证")
        split_dataset.setObjectName("primaryButton")
        split_dataset.clicked.connect(self.split_dataset)
        project.addWidget(split_dataset)
        project.addWidget(self.status_stats_label)
        header_layout.addLayout(project)
        classes = QHBoxLayout()
        classes.addWidget(QLabel("类别列表"))
        classes.addWidget(self.class_names, 1)
        refresh = QPushButton("应用类别")
        refresh.clicked.connect(self.refresh_classes)
        classes.addWidget(refresh)
        add_class = QPushButton("添加类别")
        add_class.clicked.connect(self.add_class)
        classes.addWidget(add_class)
        remove_class = QPushButton("删除当前类别")
        remove_class.clicked.connect(self.remove_current_class)
        classes.addWidget(remove_class)
        save_classes = QPushButton("保存类别")
        save_classes.clicked.connect(self.save_classes)
        classes.addWidget(save_classes)
        classes.addWidget(self.class_combo)
        classes.addWidget(self.class_color_swatch)
        classes.addWidget(QLabel("标注模式"))
        classes.addWidget(self.mode_combo)
        header_layout.addLayout(classes)
        layout.addWidget(header)
        splitter = QSplitter()
        side = QFrame()
        side.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(6)

        def section(title: str) -> QLabel:
            label = QLabel(title)
            label.setObjectName("sideSection")
            return label

        def list_header(*labels: str) -> QFrame:
            header = QFrame()
            header.setObjectName("listHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(6, 3, 6, 3)
            header_layout.setSpacing(6)
            for position, text in enumerate(labels):
                label = QLabel(text)
                label.setObjectName("listHeaderLabel")
                header_layout.addWidget(label, 1 if position == len(labels) - 1 else 0)
            return header

        side_layout.addWidget(section("图片导航"))
        side_layout.addWidget(self.image_search)
        side_layout.addWidget(self.image_status_filter)
        side_layout.addWidget(list_header("状态", "图片名称"))
        side_layout.addWidget(self.image_list)
        side_layout.addWidget(section("类别"))
        side_layout.addWidget(list_header("ID", "分类名"))
        side_layout.addWidget(self.class_list)
        side_layout.addWidget(section("当前图片标注"))
        side_layout.addWidget(list_header("序号", "类别 / 形状"))
        side_layout.addWidget(self.annotation_list)
        previous = QPushButton("上一张")
        previous.clicked.connect(self.previous_image)
        next_button = QPushButton("下一张")
        next_button.clicked.connect(self.next_image)
        clear = QPushButton("清空标注")
        clear.clicked.connect(self.canvas.clear_annotations)
        set_negative = QPushButton("设置负样本")
        set_negative.clicked.connect(self.set_negative)
        unset_negative = QPushButton("取消负样本")
        unset_negative.clicked.connect(self.unset_negative)
        delete_selected = QPushButton("删除选中")
        delete_selected.clicked.connect(self.canvas.delete_selected)
        change_selected_class = QPushButton("选中标注改为当前类别")
        change_selected_class.clicked.connect(self.change_selected_class)
        sam_run = QPushButton("运行 SAM")
        sam_run.setObjectName("primaryButton")
        sam_run.clicked.connect(self.run_sam)
        sam_clear = QPushButton("清除 SAM 点")
        sam_clear.clicked.connect(self.canvas.clear_sam_points)
        reset_view = QPushButton("适应窗口")
        reset_view.clicked.connect(self.canvas.reset_view)
        grid = QPushButton("显示/隐藏网格")
        grid.clicked.connect(self.canvas.toggle_grid)
        jump_unlabeled = QPushButton("下一张未标注")
        jump_unlabeled.clicked.connect(self.jump_unlabeled)
        settings = QFrame()
        settings.setObjectName("settingsPanel")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(6)
        settings_layout.addWidget(section("SAM 交互"))
        settings_layout.addWidget(QLabel("SAM 模型"))
        settings_layout.addWidget(self.sam_model)
        settings_layout.addWidget(QLabel("SAM 输出类型"))
        settings_layout.addWidget(self.sam_output)
        settings_layout.addWidget(sam_run)
        settings_layout.addWidget(sam_clear)
        settings_layout.addWidget(section("标注编辑"))
        for button in (previous, next_button, jump_unlabeled, delete_selected, change_selected_class, clear, set_negative, unset_negative, reset_view, grid):
            settings_layout.addWidget(button)
        settings_layout.addStretch(1)
        side_layout.addStretch(1)
        side_scroll = QScrollArea()
        side_scroll.setObjectName("sideScroll")
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setWidget(side)
        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setWidget(settings)
        splitter.addWidget(side_scroll)
        self.canvas.setObjectName("canvasPanel")
        canvas_workspace = QWidget()
        canvas_workspace.setObjectName("canvasWorkspace")
        canvas_layout = QVBoxLayout(canvas_workspace)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(5)
        canvas_context = QFrame()
        canvas_context.setObjectName("classifyBar")
        context_layout = QHBoxLayout(canvas_context)
        context_layout.setContentsMargins(8, 5, 8, 5)
        context_layout.setSpacing(6)
        self.canvas_class_chip = QFrame()
        self.canvas_class_chip.setObjectName("classifyColorChip")
        self.canvas_class_chip.setFixedSize(13, 13)
        self.canvas_class_label = QLabel("当前类别：—")
        self.canvas_class_label.setObjectName("classifyBarLabel")
        self.canvas_mode_label = QLabel()
        self.canvas_mode_label.setObjectName("canvasModeLabel")
        context_layout.addWidget(self.canvas_class_chip)
        context_layout.addWidget(self.canvas_class_label)
        context_layout.addStretch(1)
        context_layout.addWidget(self.canvas_mode_label)
        canvas_layout.addWidget(canvas_context)
        canvas_layout.addWidget(self.canvas, 1)
        self.canvas_status = QLabel()
        self.canvas_status.setObjectName("canvasStatus")
        canvas_layout.addWidget(self.canvas_status)
        splitter.addWidget(canvas_workspace)
        splitter.addWidget(settings_scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 700, 230])
        layout.addWidget(splitter, 1)
        self.image_list.currentRowChanged.connect(self._select_visible_image)
        self.image_search.textChanged.connect(self.apply_image_filter)
        self.image_status_filter.currentIndexChanged.connect(lambda _index: self.apply_image_filter())
        self.class_combo.currentIndexChanged.connect(self._set_current_class)
        self.class_list.currentRowChanged.connect(self._select_class_from_list)
        self.annotation_list.currentRowChanged.connect(self.canvas.select_annotation)
        self.canvas.selection_changed.connect(self._sync_annotation_selection)
        self.mode_combo.currentIndexChanged.connect(lambda index: self._set_mode(self.mode_combo.itemData(index)))
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self.save_current)
        self._previous_shortcut = QShortcut(QKeySequence("A"), self.canvas)
        self._previous_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._previous_shortcut.activated.connect(self.previous_image)
        self._next_shortcut = QShortcut(QKeySequence("D"), self.canvas)
        self._next_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._next_shortcut.activated.connect(self.next_image)
        self._sam_run_button = sam_run
        self._sam_busy = False
        self.sam_finished.connect(lambda: self._sam_run_button.setEnabled(True))
        self._update_canvas_status()
        self.refresh_annotation_list()

    def open_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择项目或图片目录", self.project_root.text())
        if not selected:
            return
        if self.index >= 0:
            self.save_current()
        root = Path(selected)
        direct_images = any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in root.iterdir())
        if not direct_images:
            try:
                prepare_project_structure(root)
            except DatasetError as exc:
                QMessageBox.warning(self, "打开项目", str(exc))
                return
            image_root = root / "images"
        else:
            image_root = root
        self.project_root.setText(str(root))
        self.index = -1
        self.images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        self.visible_indices = list(range(len(self.images)))
        self.image_search.clear()
        self.apply_image_filter()
        self.load_classes()
        self.status_changed.emit(f"已加载 {len(self.images)} 张图片")
        self.update_stats()

    def apply_image_filter(self) -> None:
        query = self.image_search.text().strip().lower()
        status_filter = self.image_status_filter.currentData()
        self.visible_indices = [
            index
            for index, path in enumerate(self.images)
            if (not query or query in path.name.lower()) and (status_filter == "all" or self._image_state(path) == status_filter)
        ]
        current = self.index
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self.image_list.addItems([self._image_list_text(self.images[index]) for index in self.visible_indices])
        for row, image_index in enumerate(self.visible_indices):
            self._style_image_item(self.image_list.item(row), self.images[image_index])
        if current in self.visible_indices:
            self.image_list.setCurrentRow(self.visible_indices.index(current))
        self.image_list.blockSignals(False)
        if current not in self.visible_indices and self.visible_indices:
            self.select_image(self.visible_indices[0])

    def _image_state(self, image: Path) -> str:
        label = self._label_path(image) if self.project_root.text().strip() else None
        if label is None or not label.exists():
            return "unlabeled"
        return "labeled" if label.read_text(encoding="utf-8").strip() else "negative"

    def _image_list_text(self, image: Path) -> str:
        labels = {"unlabeled": "未标注", "labeled": "已标注", "negative": "负样本"}
        return f"[{labels[self._image_state(image)]}] {image.name}"

    def _style_image_item(self, item, image: Path) -> None:
        if item is None:
            return
        state = self._image_state(image)
        colors = {"unlabeled": "#e6a23c", "labeled": "#67c23a", "negative": "#909399"}
        descriptions = {"unlabeled": "尚未创建标签文件", "labeled": "已有标注", "negative": "已标记为负样本"}
        item.setForeground(QBrush(QColor(colors[state])))
        item.setToolTip(descriptions[state])

    def _select_visible_image(self, visible_index: int) -> None:
        if 0 <= visible_index < len(self.visible_indices):
            self.select_image(self.visible_indices[visible_index])

    def previous_image(self) -> None:
        self._step_image(-1)

    def next_image(self) -> None:
        self._step_image(1)

    def jump_unlabeled(self) -> None:
        if not self.visible_indices:
            self.status_changed.emit("当前没有图片")
            return
        try:
            position = self.visible_indices.index(self.index)
        except ValueError:
            position = -1
        for offset in range(1, len(self.visible_indices) + 1):
            candidate_position = (position + offset) % len(self.visible_indices)
            image = self.images[self.visible_indices[candidate_position]]
            if not self._label_path(image).exists():
                self.select_image(self.visible_indices[candidate_position])
                return
        self.status_changed.emit("没有找到未标注图片")

    def _step_image(self, delta: int) -> None:
        if not self.visible_indices:
            self.status_changed.emit("当前没有可翻页的图片")
            return
        try:
            position = self.visible_indices.index(self.index)
        except ValueError:
            position = 0 if delta > 0 else len(self.visible_indices) - 1
        target = max(0, min(len(self.visible_indices) - 1, position + delta))
        if target == position and self.index == self.visible_indices[position]:
            self.status_changed.emit("已经是第一张" if delta < 0 else "已经是最后一张")
            return
        self.select_image(self.visible_indices[target])

    def refresh_classes(self) -> None:
        names = self.current_class_names()
        if not names:
            names = ["目标"]
            self.class_names.setText("目标")
        self.class_combo.clear()
        self.class_combo.addItems([f"{index}: {name}" for index, name in enumerate(names)])
        self.class_list.blockSignals(True)
        self.class_list.clear()
        self.class_list.addItems([f"{index}: {name}" for index, name in enumerate(names)])
        for class_id in range(self.class_list.count()):
            self.class_list.item(class_id).setForeground(QBrush(QColor(self.canvas._class_color(class_id))))
        self.class_list.setCurrentRow(0)
        self.class_list.blockSignals(False)
        self._set_current_class(0)
        self.refresh_annotation_list()
        self.classes_changed.emit(names)

    def _set_current_class(self, class_id: int) -> None:
        if class_id < 0:
            return
        self.canvas.set_class_id(class_id)
        color = self.canvas._class_color(class_id)
        self.class_color_swatch.setStyleSheet(f"background-color: {color}; border: 1px solid rgba(0, 0, 0, 0.22); border-radius: 4px;")
        item = self.class_list.item(class_id)
        if item is not None:
            item.setForeground(QBrush(QColor(color)))
        self.class_list.blockSignals(True)
        self.class_list.setCurrentRow(class_id)
        self.class_list.blockSignals(False)
        self._update_canvas_status()

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self._update_canvas_status()

    def _update_canvas_status(self) -> None:
        names = {"detect": "矩形框", "segment": "分割多边形", "sam": "SAM 点选"}
        mode = names.get(self.mode_combo.currentData(), "标注")
        current = self.class_combo.currentText() or "—"
        self.canvas_class_label.setText(f"当前类别：{current}")
        color = self.canvas._class_color(self.class_combo.currentIndex()) if self.class_combo.currentIndex() >= 0 else "#909399"
        self.canvas_class_chip.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
        self.canvas_mode_label.setText(mode)
        if 0 <= self.index < len(self.images):
            image = self.images[self.index]
            context = f"图片：{self.index + 1}/{len(self.images)} · {image.name} · 标注：{len(self.canvas.annotations)}"
        else:
            context = "尚未打开图片"
        self.canvas_status.setText(f"{mode}  |  当前类别：{current}  |  {context}")

    def _select_class_from_list(self, class_id: int) -> None:
        if 0 <= class_id < self.class_combo.count():
            self.class_combo.setCurrentIndex(class_id)

    def refresh_annotation_list(self) -> None:
        names = self.current_class_names()
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for index, annotation in enumerate(self.canvas.annotations):
            class_id = annotation.class_id
            class_name = names[class_id] if 0 <= class_id < len(names) else f"类别 {class_id}"
            shape_name = "矩形框" if isinstance(annotation, Detection) else "分割多边形"
            self.annotation_list.addItem(f"{index + 1}. {class_name} · {shape_name}")
            item = self.annotation_list.item(index)
            item.setForeground(QBrush(QColor(self.canvas._class_color(class_id))))
        selected = self.canvas.selected_index
        if selected is not None and selected < self.annotation_list.count():
            self.annotation_list.setCurrentRow(selected)
        self.annotation_list.blockSignals(False)

    def _sync_annotation_selection(self, index: int) -> None:
        self.annotation_list.blockSignals(True)
        self.annotation_list.setCurrentRow(index if 0 <= index < self.annotation_list.count() else -1)
        self.annotation_list.blockSignals(False)

    def current_class_names(self) -> list[str]:
        return [name.strip() for name in self.class_names.text().split(",") if name.strip()]

    def class_file_path(self) -> Path | None:
        root_text = self.project_root.text().strip()
        return Path(root_text) / "classes.txt" if root_text else None

    def load_classes(self) -> None:
        path = self.class_file_path()
        if path and path.exists():
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if names:
                self.class_names.setText(",".join(names))
        self.refresh_classes()

    def save_classes(self) -> None:
        path = self.class_file_path()
        names = self.current_class_names()
        if path is None:
            QMessageBox.warning(self, "类别管理", "请先打开项目目录。")
            return
        if not names:
            QMessageBox.warning(self, "类别管理", "请至少保留一个类别。")
            return
        path.write_text("\n".join(names) + "\n", encoding="utf-8")
        self.status_changed.emit(f"类别已保存：{path.name}")

    def add_class(self) -> None:
        name, accepted = QInputDialog.getText(self, "添加类别", "类别名称：")
        name = name.strip()
        if not accepted or not name:
            return
        names = self.current_class_names()
        if name in names:
            QMessageBox.information(self, "类别管理", "该类别已存在。")
            return
        names.append(name)
        self.class_names.setText(",".join(names))
        self.refresh_classes()
        self.class_combo.setCurrentIndex(len(names) - 1)

    def remove_current_class(self) -> None:
        names = self.current_class_names()
        index = self.class_combo.currentIndex()
        if len(names) <= 1:
            QMessageBox.warning(self, "类别管理", "至少需要保留一个类别。")
            return
        if not 0 <= index < len(names):
            return
        del names[index]
        self.class_names.setText(",".join(names))
        self.refresh_classes()

    def generate_yaml(self) -> None:
        root_text = self.project_root.text().strip()
        if not root_text:
            QMessageBox.warning(self, "数据集配置", "请先打开项目目录。")
            return
        names = [name.strip() for name in self.class_names.text().split(",") if name.strip()]
        if not names:
            QMessageBox.warning(self, "数据集配置", "请至少填写一个类别名称。")
            return
        class_path = self.class_file_path()
        if class_path:
            class_path.write_text("\n".join(names) + "\n", encoding="utf-8")
        target = make_dataset_yaml(root_text, names)
        self.dataset_ready.emit(str(target))
        self.status_changed.emit(f"已保存类别并生成数据集配置：{target}")

    def split_dataset(self) -> None:
        root_text = self.project_root.text().strip()
        if not root_text:
            QMessageBox.warning(self, "划分数据集", "请先打开项目目录。")
            return
        if self.index >= 0:
            self.save_current()
        root = Path(root_text)
        outputs = (root / "train" / "images", root / "train" / "labels", root / "val" / "images", root / "val" / "labels")
        has_existing_output = any(directory.exists() and any(directory.rglob("*")) for directory in outputs)
        if has_existing_output:
            answer = QMessageBox.question(
                self,
                "重新划分数据集",
                "现有 train 和 val 内容将被重新生成，原始 images 和 labels 不会改变。是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        names = self.current_class_names()
        if not names:
            QMessageBox.warning(self, "划分数据集", "请至少保留一个类别。")
            return
        try:
            report = split_annotated_dataset(root, names, val_fraction=float(self.split_ratio.currentData()))
        except DatasetError as exc:
            QMessageBox.warning(self, "划分数据集", str(exc))
            return
        self.dataset_ready.emit(str(report.yaml_path))
        skipped = f"，跳过未标注 {len(report.skipped_unlabeled)} 张" if report.skipped_unlabeled else ""
        self.status_changed.emit(f"数据集已划分：训练 {report.train_images} 张，验证 {report.val_images} 张{skipped}")

    def _label_path(self, image: Path) -> Path:
        root = Path(self.project_root.text())
        image_root = root / "images" if (root / "images").exists() else root
        relative = image.relative_to(image_root).with_suffix(".txt")
        return root / "labels" / relative

    def select_image(self, index: int) -> None:
        if not 0 <= index < len(self.images):
            return
        if self.index >= 0:
            self.save_current()
        self.index = index
        self.image_list.blockSignals(True)
        if index in self.visible_indices:
            self.image_list.setCurrentRow(self.visible_indices.index(index))
        self.image_list.blockSignals(False)
        image = self.images[index]
        self.canvas.set_image(image, load_annotations(self._label_path(image)))
        self.refresh_annotation_list()
        self._update_canvas_status()
        self.status_changed.emit(f"第 {index + 1}/{len(self.images)} 张：{image.name}")

    def save_current(self) -> None:
        if self.index < 0:
            return
        save_annotations(self._label_path(self.images[self.index]), self.canvas.annotations)
        self.update_stats()
        self.status_changed.emit("标注已保存")

    def update_stats(self) -> None:
        total = len(self.images)
        labeled = 0
        negative = 0
        for image in self.images:
            label = self._label_path(image)
            if not label.exists():
                continue
            if label.read_text(encoding="utf-8").strip():
                labeled += 1
            else:
                negative += 1
        unlabeled = max(0, total - labeled - negative)
        self.status_stats_label.setText(f"图片总数: {total}  已标注: {labeled}  未标注: {unlabeled}  负样本: {negative}")
        self.stats_changed.emit(total, labeled, unlabeled, negative)
        self.image_list.blockSignals(True)
        current = self.image_list.currentRow()
        for row, image_index in enumerate(self.visible_indices):
            item = self.image_list.item(row)
            item.setText(self._image_list_text(self.images[image_index]))
            self._style_image_item(item, self.images[image_index])
        if 0 <= current < self.image_list.count():
            self.image_list.setCurrentRow(current)
        self.image_list.blockSignals(False)

    def set_negative(self) -> None:
        if self.index < 0:
            QMessageBox.warning(self, "负样本", "请先打开一张图片。")
            return
        answer = QMessageBox.question(self, "设置负样本", "将清空当前图片的全部标注并设置为负样本，是否继续？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.canvas.clear_annotations()
        path = self._label_path(self.images[self.index])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self.update_stats()
        self.status_changed.emit("当前图片已设置为负样本")

    def unset_negative(self) -> None:
        if self.index < 0:
            return
        path = self._label_path(self.images[self.index])
        if not path.exists():
            self.status_changed.emit("当前图片不是负样本")
            return
        if path.read_text(encoding="utf-8").strip():
            QMessageBox.information(self, "负样本", "当前图片有实际标注，不能按负样本取消。")
            return
        path.unlink()
        self.update_stats()
        self.status_changed.emit("已取消当前图片的负样本标记")

    def run_auto_current(self) -> None:
        if self.index < 0:
            self.status_changed.emit("请先打开一张图片")
            return
        if self._auto_busy:
            self.status_changed.emit("当前图片自动标注仍在进行中")
            return
        detector = self.auto_model.text().strip()
        if not detector:
            QMessageBox.warning(self, "当前图自动标注", "请填写 YOLO 检测模型。")
            return
        image = self.images[self.index]
        mode = self.mode_combo.currentData()
        output = self.sam_output.currentData() if mode == "sam" else mode
        task = "segment" if output == "segment" else "detect"
        self._auto_busy = True
        self._auto_run_button.setEnabled(False)
        self.status_changed.emit("正在自动标注当前图片")
        threading.Thread(
            target=self._auto_work,
            args=(image, self._label_path(image), detector, self.sam_model.text().strip(), task),
            daemon=True,
        ).start()

    def _auto_work(self, image: Path, label_path: Path, detector: str, segmenter: str, task: str) -> None:
        try:
            annotator = AutoAnnotator(detector, segmenter)
            writer = annotator.annotate_segmentation if task == "segment" else annotator.annotate_detection
            generated = writer(image, label_path)
            self.auto_result_ready.emit((image, load_annotations(label_path), generated))
        except Exception as exc:
            self.auto_failed.emit(str(exc))
        finally:
            self.auto_finished.emit()

    def _apply_auto_result(self, payload: object) -> None:
        image, annotations, generated = payload  # type: ignore[misc]
        if self.index >= 0 and self.images[self.index] == image:
            self.canvas.set_image(image, annotations)
            self.update_stats()
        self.status_changed.emit(f"当前图自动标注完成，共生成 {generated} 条标注")

    def _auto_done(self) -> None:
        self._auto_busy = False
        self._auto_run_button.setEnabled(True)

    @staticmethod
    def _shape_bounds(item: Annotation) -> tuple[float, float, float, float]:
        if isinstance(item, Detection):
            return (item.x_center - item.width / 2, item.y_center - item.height / 2, item.x_center + item.width / 2, item.y_center + item.height / 2)
        xs = [point[0] for point in item.points]
        ys = [point[1] for point in item.points]
        return min(xs), min(ys), max(xs), max(ys)

    def lock_match_template(self) -> None:
        if self.index < 0:
            QMessageBox.warning(self, "标注匹配", "请先打开图片。")
            return
        item = self.canvas.selected_annotation
        if item is None:
            QMessageBox.warning(self, "标注匹配", "请先在矩形模式下点击一个已有标注。")
            return
        self._match_template = (self.images[self.index], self._shape_bounds(item), item.class_id)
        self.status_changed.emit(f"已锁定匹配模板：{self.images[self.index].name}")

    def run_match_current(self) -> None:
        if self._match_template is None:
            QMessageBox.warning(self, "标注匹配", "请先锁定一个已有标注作为模板。")
            return
        if self.index < 0:
            return
        if self._match_busy:
            self.status_changed.emit("标注匹配仍在进行中")
            return
        try:
            confidence = float(self.match_conf.text())
            if not 0 <= confidence <= 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "标注匹配", "匹配阈值必须是 0 到 1 之间的数字。")
            return
        template_image, bounds, class_id = self._match_template
        self._match_busy = True
        self._match_run_button.setEnabled(False)
        self.status_changed.emit("正在匹配当前图片")
        threading.Thread(
            target=self._match_work,
            args=(template_image, self.images[self.index], bounds, class_id, confidence),
            daemon=True,
        ).start()

    def _match_work(self, template_image: Path, target_image: Path, bounds: tuple[float, float, float, float], class_id: int, confidence: float) -> None:
        try:
            import cv2
            import numpy as np

            source = cv2.imread(str(template_image), cv2.IMREAD_COLOR)
            target = cv2.imread(str(target_image), cv2.IMREAD_COLOR)
            if source is None or target is None:
                raise RuntimeError("无法读取模板或目标图片")
            source_height, source_width = source.shape[:2]
            target_height, target_width = target.shape[:2]
            x1 = max(0, min(source_width - 1, int(bounds[0] * source_width)))
            y1 = max(0, min(source_height - 1, int(bounds[1] * source_height)))
            x2 = max(x1 + 1, min(source_width, int(bounds[2] * source_width)))
            y2 = max(y1 + 1, min(source_height, int(bounds[3] * source_height)))
            template = source[y1:y2, x1:x2]
            template_height, template_width = template.shape[:2]
            if template_width >= target_width or template_height >= target_height:
                raise RuntimeError("模板尺寸大于当前图片")
            scores = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(np.isfinite(scores) & (scores >= confidence))
            candidates = sorted(((float(scores[y, x]), int(x), int(y)) for y, x in zip(ys, xs)), reverse=True)
            accepted: list[tuple[float, int, int]] = []
            for score, x, y in candidates:
                box = (x, y, x + template_width, y + template_height)
                if any(self._pixel_iou(box, (ox, oy, ox + template_width, oy + template_height)) >= 0.30 for _, ox, oy in accepted):
                    continue
                accepted.append((score, x, y))
                if len(accepted) >= 100:
                    break
            detections = [Detection(class_id, (x + template_width / 2) / target_width, (y + template_height / 2) / target_height, template_width / target_width, template_height / target_height) for _, x, y in accepted]
            self.match_result_ready.emit((target_image, detections, len(accepted)))
        except Exception as exc:
            self.match_failed.emit(str(exc))
        finally:
            self.match_finished.emit()

    @staticmethod
    def _pixel_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = area_first + area_second - intersection
        return intersection / union if union else 0.0

    def _apply_match_result(self, payload: object) -> None:
        image, detections, count = payload  # type: ignore[misc]
        if self.index >= 0 and self.images[self.index] == image:
            self.canvas.append_annotations(detections)
        self.status_changed.emit(f"标注匹配完成，找到 {count} 个相似目标")

    def _match_done(self) -> None:
        self._match_busy = False
        self._match_run_button.setEnabled(True)

    def change_selected_class(self) -> None:
        class_id = self.class_combo.currentIndex()
        if self.canvas.set_selected_class(class_id):
            self.status_changed.emit("已更新选中标注的类别")
        else:
            QMessageBox.information(self, "标注编辑", "请先点击一个已有标注进行选中。")

    def _sam_point_clicked(self, _x: float, _y: float, label: int) -> None:
        point_type = "正点" if label == 1 else "负点"
        self.status_changed.emit(f"已添加 SAM {point_type}，正在自动标注")
        if not self._sam_busy:
            self.run_sam()

    def run_sam(self) -> None:
        if self.index < 0:
            QMessageBox.warning(self, "SAM 点选", "请先打开图片并点选目标。")
            return
        points = self.canvas.sam_points
        labels = self.canvas.sam_labels
        if not points:
            QMessageBox.warning(self, "SAM 点选", "请先在图片上点击目标前景点。")
            return
        model = self.sam_model.text().strip()
        if not model:
            QMessageBox.warning(self, "SAM 点选", "请填写 SAM 模型。")
            return
        image = self.images[self.index]
        output = self.sam_output.currentData()
        self._sam_busy = True
        self._sam_run_button.setEnabled(False)
        self.status_changed.emit("正在运行 SAM 分割")
        threading.Thread(target=self._sam_work, args=(image, points, labels, model, output), daemon=True).start()

    def _sam_work(self, image: Path, points: list[tuple[float, float]], labels: list[int], model: str, output: str) -> None:
        try:
            from PIL import Image

            width, height = Image.open(image).size
            pixel_points = [[x * width, y * height] for x, y in points]
            if self._sam_annotator is None or self._sam_model_path != model:
                self._sam_annotator = SamAnnotator(model)
                self._sam_model_path = model
            results = self._sam_annotator.segment(image, points=pixel_points, labels=labels)
            if not results or results[0].masks is None or not results[0].masks.xy:
                raise RuntimeError("模型没有返回有效掩膜")
            polygons = results[0].masks.xy
            polygon = max(polygons, key=self._polygon_area)
            normalized = tuple((float(x) / width, float(y) / height) for x, y in polygon)
            if len(normalized) < 3:
                raise RuntimeError("掩膜轮廓点不足")
            self.sam_result_ready.emit((normalized, output))
            result_name = "矩形框" if output == "detect" else "多边形"
            self.status_changed.emit(f"SAM 完成，已生成{result_name}")
        except Exception as exc:
            self.sam_failed.emit(str(exc))
        finally:
            self._sam_busy = False
            self.sam_finished.emit()

    @staticmethod
    def _polygon_area(points: object) -> float:
        values = [(float(x), float(y)) for x, y in points]  # type: ignore[misc]
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(values, values[1:] + values[:1]))) / 2


class AutoAnnotatePage(QWidget):
    status_changed = Signal(str)
    progress_changed = Signal(int, int, str, str)
    job_finished = Signal(object)

    def __init__(self, annotation_page: AnnotationPage) -> None:
        super().__init__()
        self.annotation_page = annotation_page
        self.detector = QLineEdit("yolo11n.pt")
        self.segmenter = QLineEdit("sam2_b.pt")
        self.task = QComboBox()
        self.task.addItem("检测框", "detect")
        self.task.addItem("分割掩膜（SAM）", "segment")
        self.class_filter = QComboBox()
        self.class_filter.addItem("全部类别", None)
        self.merge = QComboBox()
        self.merge.addItem("覆盖已有标注", "replace")
        self.merge.addItem("跳过已有标注", "skip")
        self.merge.addItem("追加到已有标注", "append")
        self.merge.addItem("智能去重", "smart_dedup")
        self.merge.addItem("手动保留，仅补新框", "append_new")
        self.merge_iou = QLineEdit("0.45")
        self.run_button = QPushButton("开始自动标注")
        self.run_button.setObjectName("primaryButton")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label = QLabel("未开始")
        self.progress_label.setObjectName("taskStatus")
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)
        title = QLabel("批量自动标注")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        config_card = QGroupBox("模型与批量设置")
        config_card.setObjectName("autoBatchCard")
        form = QFormLayout(config_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        detector_row = QHBoxLayout()
        detector_row.addWidget(self.detector, 1)
        detector_browse = QPushButton("浏览")
        detector_browse.clicked.connect(lambda: self.choose_model(self.detector, "选择检测模型"))
        detector_row.addWidget(detector_browse)
        sam_row = QHBoxLayout()
        sam_row.addWidget(self.segmenter, 1)
        sam_browse = QPushButton("浏览")
        sam_browse.clicked.connect(lambda: self.choose_model(self.segmenter, "选择 SAM 模型"))
        sam_row.addWidget(sam_browse)
        form.addRow("检测模型", detector_row)
        form.addRow("SAM 模型", sam_row)
        form.addRow("输出类型", self.task)
        form.addRow("类别过滤", self.class_filter)
        form.addRow("已有标注", self.merge)
        form.addRow("合并 IOU", self.merge_iou)
        form.addRow(self.run_button)
        layout.addWidget(config_card)
        progress_card = QGroupBox("批量进度")
        progress_card.setObjectName("autoProgressCard")
        progress_form = QFormLayout(progress_card)
        progress_form.setContentsMargins(14, 14, 14, 14)
        progress_form.setHorizontalSpacing(14)
        progress_form.setVerticalSpacing(9)
        controls = QHBoxLayout()
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        progress_form.addRow(controls)
        progress_form.addRow("批量进度", self.progress)
        progress_form.addRow("当前状态", self.progress_label)
        layout.addWidget(progress_card)
        layout.addStretch(1)
        self.run_button.clicked.connect(self.run)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop)
        self.progress_changed.connect(self.update_progress)
        self.job_finished.connect(self._job_done)

    def choose_model(self, target: QLineEdit, title: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, title, target.text(), "模型 (*.pt *.onnx *.engine);;所有文件 (*.*)")
        if selected:
            target.setText(selected)

    def sync_classes(self, names: list[str]) -> None:
        current = self.class_filter.currentData()
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("全部类别", None)
        for class_id, name in enumerate(names):
            self.class_filter.addItem(f"{class_id}: {name}", class_id)
        if current is not None:
            index = self.class_filter.findData(current)
            if index >= 0:
                self.class_filter.setCurrentIndex(index)
        self.class_filter.blockSignals(False)

    def run(self) -> None:
        root = self.annotation_page.project_root.text()
        if not root:
            QMessageBox.warning(self, "自动标注", "请先在“标注”页面打开项目目录。")
            return
        image_root = Path(root) / "images" if (Path(root) / "images").exists() else Path(root)
        labels = Path(root) / "labels"
        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self._stop_event.clear()
        self._pause_event.set()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_label.setText("正在准备模型")
        self.status_changed.emit("正在执行自动标注")
        try:
            merge_iou = float(self.merge_iou.text())
            if not 0 <= merge_iou <= 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "自动标注", "合并 IOU 必须是 0 到 1 之间的数字。")
            self.run_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            return
        class_id = self.class_filter.currentData()
        class_ids = None if class_id is None else (int(class_id),)
        args = (image_root, labels, self.detector.text().strip(), self.segmenter.text().strip(), self.task.currentData(), self.merge.currentData(), class_ids, merge_iou)
        threading.Thread(target=self._work, args=args, daemon=True).start()

    def toggle_pause(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_button.setText("继续")
            self.progress_label.setText("已暂停，等待继续")
        else:
            self._pause_event.set()
            self.pause_button.setText("暂停")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self.progress_label.setText("正在停止…")
        self.stop_button.setEnabled(False)

    def update_progress(self, current: int, total: int, name: str, state: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(min(current, total))
        self.progress_label.setText(f"{current}/{total}  {name}  {state}")

    def _job_done(self, stats: object) -> None:
        values = stats if isinstance(stats, dict) else {}
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.status_changed.emit(
            "批量标注结束：成功 {success}，跳过 {skipped}，失败 {failed}，生成 {labels} 条".format(
                success=values.get("success", 0), skipped=values.get("skipped", 0), failed=values.get("failed", 0), labels=values.get("labels", 0)
            )
        )

    def _work(self, images: Path, labels: Path, detector: str, segmenter: str, task: str, merge: str, class_ids: tuple[int, ...] | None, merge_iou: float) -> None:
        stats = {"success": 0, "skipped": 0, "failed": 0, "labels": 0}
        try:
            annotator = AutoAnnotator(detector, segmenter)
            annotator.annotate_folder(
                images,
                labels,
                task,
                merge=merge,
                progress_callback=lambda current, total, name, state: self.progress_changed.emit(current, total, name, state),
                stop_event=self._stop_event,
                pause_event=self._pause_event,
                class_ids=class_ids,
                merge_iou=merge_iou,
            )
            stats = dict(annotator.last_batch_stats)
        except Exception as exc:
            self.status_changed.emit(f"自动标注失败：{exc}")
            stats = {"success": 0, "skipped": 0, "failed": 1, "labels": 0}
        finally:
            self.job_finished.emit(stats)


class TrainingPage(QWidget):
    status_changed = Signal(str)
    job_finished = Signal()
    export_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = Event()
        self.model = QLineEdit("yolo11n.pt")
        self.data = QLineEdit("data.yaml")
        self.epochs = QLineEdit("100")
        self.imgsz = QLineEdit("640")
        self.batch = QLineEdit("-1")
        self.patience = QLineEdit("30")
        self.workers = QLineEdit("0")
        self.device = QLineEdit(detect_training_device())
        self.device.setToolTip("无 NVIDIA CUDA 显卡时填写 cpu；有可用 NVIDIA GPU 时可填写 0。")
        self.project = QLineEdit("runs/train")
        self.name = QLineEdit("experiment")
        self.start = QPushButton("开始训练")
        self.start.setObjectName("primaryButton")
        self.stop = QPushButton("停止训练")
        self.stop.setEnabled(False)
        self.export_format = QComboBox()
        self.export_format.addItem("ONNX", "onnx")
        self.export_format.addItem("OpenVINO", "openvino")
        self.export_format.addItem("TensorRT 引擎", "engine")
        self.export_format.addItem("NCNN", "ncnn")
        self.export_imgsz = QLineEdit("640")
        self.export_opset = QLineEdit("12")
        self.export_simplify = QComboBox()
        self.export_simplify.addItem("启用", True)
        self.export_simplify.addItem("关闭", False)
        self.export_button = QPushButton("导出模型")
        self.log = QTextEdit()
        self.log.setObjectName("taskLog")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.model_browse = QPushButton("浏览")
        self.data_browse = QPushButton("浏览")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)
        title = QLabel("模型训练与导出")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        training_card = QGroupBox("训练参数")
        training_card.setObjectName("trainingCard")
        training_form = QFormLayout(training_card)
        training_form.setContentsMargins(14, 14, 14, 14)
        training_form.setHorizontalSpacing(14)
        training_form.setVerticalSpacing(9)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.model_browse)
        data_row = QHBoxLayout()
        data_row.addWidget(self.data, 1)
        data_row.addWidget(self.data_browse)
        training_form.addRow("模型", model_row)
        training_form.addRow("数据集 YAML", data_row)
        training_form.addRow("训练轮数", self.epochs)
        training_form.addRow("图片尺寸", self.imgsz)
        training_form.addRow("批次大小", self.batch)
        training_form.addRow("早停耐心值", self.patience)
        training_form.addRow("数据线程数", self.workers)
        training_form.addRow("计算设备", self.device)
        training_form.addRow("输出目录", self.project)
        training_form.addRow("实验名称", self.name)
        training_buttons = QHBoxLayout()
        training_buttons.addWidget(self.start)
        training_buttons.addWidget(self.stop)
        training_form.addRow(training_buttons)
        layout.addWidget(training_card)
        export_card = QGroupBox("模型转换")
        export_card.setObjectName("trainingConvertCard")
        export_form = QFormLayout(export_card)
        export_form.setContentsMargins(14, 14, 14, 14)
        export_form.setHorizontalSpacing(14)
        export_form.setVerticalSpacing(9)
        export_form.addRow("导出格式", self.export_format)
        export_form.addRow("导出图片尺寸", self.export_imgsz)
        export_form.addRow("ONNX Opset", self.export_opset)
        export_form.addRow("简化 ONNX", self.export_simplify)
        export_form.addRow(self.export_button)
        layout.addWidget(export_card)
        log_card = QGroupBox("训练日志")
        log_card.setObjectName("trainingLogCard")
        self.log.setObjectName("trainingLogEdit")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(14, 14, 14, 14)
        log_layout.addWidget(self.log)
        layout.addWidget(log_card)
        self.start.clicked.connect(self.run)
        self.stop.clicked.connect(self.stop_training)
        self.export_button.clicked.connect(self.run_export)
        self.model_browse.clicked.connect(self.choose_model)
        self.data_browse.clicked.connect(self.choose_data)
        self.job_finished.connect(lambda: self.start.setEnabled(True))
        self.job_finished.connect(lambda: self.stop.setEnabled(False))
        self.export_finished.connect(lambda: self.export_button.setEnabled(True))
        self.status_changed.connect(self.log_message)

    def log_message(self, message: str) -> None:
        self.log.append(message)

    def choose_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model.text(), "模型 (*.pt *.onnx *.engine);;所有文件 (*.*)")
        if selected:
            self.model.setText(selected)

    def choose_data(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择数据集配置", self.data.text(), "YAML (*.yaml *.yml);;所有文件 (*.*)")
        if selected:
            self.use_dataset(selected)

    def use_dataset(self, path: str) -> None:
        self.data.setText(path)
        yaml_path = Path(path).expanduser()
        if yaml_path.suffix.lower() in {".yaml", ".yml"}:
            self.project.setText(str(yaml_path.resolve().parent / "runs" / "train"))

    def run(self) -> None:
        try:
            config = TrainingConfig(
                model=self.model.text(),
                data=self.data.text(),
                epochs=int(self.epochs.text()),
                imgsz=int(self.imgsz.text()),
                batch=float(self.batch.text()),
                patience=int(self.patience.text()),
                workers=int(self.workers.text()),
                device=self.device.text(),
                project=self.project.text(),
                name=self.name.text(),
            )
        except ValueError:
            QMessageBox.warning(self, "训练", "训练参数必须是有效数字。")
            return
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self._stop_event.clear()
        self.log.clear()
        self.status_changed.emit("正在训练模型")
        threading.Thread(target=self._work, args=(config,), daemon=True).start()

    def stop_training(self) -> None:
        if not self.stop.isEnabled():
            return
        self.stop.setEnabled(False)
        self.status_changed.emit("正在停止训练，请等待当前批次结束")
        self._stop_event.set()

    def run_export(self) -> None:
        try:
            config = ExportConfig(
                model=self.model.text(),
                format=self.export_format.currentData(),
                imgsz=int(self.export_imgsz.text()),
                opset=int(self.export_opset.text()),
                simplify=bool(self.export_simplify.currentData()),
                device=self.device.text().strip() or None,
            )
        except ValueError:
            QMessageBox.warning(self, "模型导出", "导出图片尺寸和 ONNX Opset 必须是整数。")
            return
        self.export_button.setEnabled(False)
        self.status_changed.emit("正在导出模型")
        threading.Thread(target=self._export_work, args=(config,), daemon=True).start()

    def _work(self, config: TrainingConfig) -> None:
        stream = _TrainingLogStream(self.status_changed.emit)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = train(config, stop_event=self._stop_event)
            stream.flush()
            output_dir = getattr(result, "save_dir", None)
            if output_dir:
                self.status_changed.emit(f"训练完成，结果目录：{output_dir}")
            else:
                self.status_changed.emit("训练完成")
        except TrainingStopped:
            stream.flush()
            self.status_changed.emit("训练已停止")
        except Exception as exc:
            stream.flush()
            self.status_changed.emit(f"训练失败：{exc}")
        finally:
            self.job_finished.emit()

    def _export_work(self, config: ExportConfig) -> None:
        try:
            output = export_model(config)
            self.status_changed.emit(f"模型导出完成：{output}")
        except Exception as exc:
            self.status_changed.emit(f"模型导出失败：{exc}")
        finally:
            self.export_finished.emit()


class ValidationPage(QWidget):
    """Run a trained model against an image, folder, or video source."""

    status_changed = Signal(str)
    preview_ready = Signal(str)
    job_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.model = QLineEdit("best.pt")
        self.image = QLineEdit()
        self.source_kind = QComboBox()
        self.source_kind.addItem("单张图片", "image")
        self.source_kind.addItem("图片目录", "folder")
        self.source_kind.addItem("视频文件", "video")
        self.output_dir = QLineEdit("runs/validation")
        self.confidence = QLineEdit("0.25")
        self.run_button = QPushButton("开始推理")
        self.run_button.setObjectName("primaryButton")
        self.preview = QLabel("请选择模型和图片")
        self.preview.setObjectName("previewPanel")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(420)
        config_card = QGroupBox("模型")
        config_card.setObjectName("validationCard")
        form = QFormLayout(config_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        model_browse = QPushButton("浏览")
        model_browse.clicked.connect(self.choose_model)
        model_row.addWidget(model_browse)
        form.addRow("模型", model_row)
        source_section = QLabel("输入源")
        source_section.setObjectName("formSection")
        form.addRow(source_section)
        form.addRow("输入来源", self.source_kind)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image, 1)
        browse = QPushButton("浏览")
        browse.clicked.connect(self.choose_image)
        image_row.addWidget(browse)
        form.addRow("来源路径", image_row)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_browse = QPushButton("浏览")
        output_browse.clicked.connect(self.choose_output_dir)
        output_row.addWidget(output_browse)
        form.addRow("输出目录", output_row)
        parameter_section = QLabel("推理参数")
        parameter_section.setObjectName("formSection")
        form.addRow(parameter_section)
        form.addRow("置信度", self.confidence)
        form.addRow(self.run_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(10)
        title = QLabel("模型验证")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(config_card)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 760])
        layout.addWidget(splitter, 1)
        self.run_button.clicked.connect(self.run)
        self.source_kind.currentIndexChanged.connect(lambda _index: self._update_source_placeholder())
        self.preview_ready.connect(self.show_preview)
        self.job_finished.connect(lambda: self.run_button.setEnabled(True))
        self._update_source_placeholder()

    def _update_source_placeholder(self) -> None:
        kind = self.source_kind.currentData()
        self.image.setPlaceholderText({"image": "选择一张图片", "folder": "选择图片目录", "video": "选择视频文件"}[kind])

    def choose_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model.text(), "模型 (*.pt *.onnx *.engine);;所有文件 (*.*)")
        if selected:
            self.model.setText(selected)

    def choose_image(self) -> None:
        kind = self.source_kind.currentData()
        if kind == "folder":
            selected = QFileDialog.getExistingDirectory(self, "选择图片目录", self.image.text())
        else:
            filters = "视频 (*.mp4 *.avi *.mov *.mkv)" if kind == "video" else "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
            selected, _ = QFileDialog.getOpenFileName(self, "选择输入来源", self.image.text(), filters)
        if selected:
            self.image.setText(selected)

    def choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择推理输出目录", self.output_dir.text())
        if selected:
            self.output_dir.setText(selected)

    def run(self) -> None:
        source = Path(self.image.text())
        kind = self.source_kind.currentData()
        source_ok = source.is_dir() if kind == "folder" else source.is_file()
        if not Path(self.model.text()).is_file() or not source_ok:
            QMessageBox.warning(self, "模型验证", "请选择存在的模型文件和输入来源。")
            return
        try:
            confidence = float(self.confidence.text())
        except ValueError:
            QMessageBox.warning(self, "模型验证", "置信度必须是数字。")
            return
        self.run_button.setEnabled(False)
        self.status_changed.emit("正在执行模型推理")
        threading.Thread(target=self._work, args=(self.model.text(), source, kind, confidence, Path(self.output_dir.text())), daemon=True).start()

    def _work(self, model_path: str, source: Path, kind: str, confidence: float, output_dir: Path) -> None:
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            if kind == "video":
                model.predict(str(source), conf=confidence, save=True, project=str(output_dir.parent), name=output_dir.name, exist_ok=True, verbose=False)
                self.status_changed.emit(f"视频推理完成，结果目录：{output_dir}")
                return
            sources = [source] if kind == "image" else sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            first_output = None
            for image_path in sources:
                result = model.predict(str(image_path), conf=confidence, verbose=False)[0]
                output = output_dir / image_path.name
                result.save(filename=str(output))
                first_output = first_output or output
            if first_output:
                self.preview_ready.emit(str(first_output))
            self.status_changed.emit(f"图片推理完成，共处理 {len(sources)} 张")
        except Exception as exc:
            self.status_changed.emit(f"推理失败：{exc}")
        finally:
            self.job_finished.emit()

    def show_preview(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class SettingsPage(QWidget):
    theme_changed = Signal(str)
    display_changed = Signal(object)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("configPage")
        self.theme = QComboBox()
        self.theme.addItems(["跟随系统", "深色", "浅色"])
        self.show_guide_lines = QCheckBox("显示辅助线（中心十字准线）")
        self.show_guide_lines.setChecked(True)
        self.zoom_step = QComboBox()
        for percent in (10, 15, 25, 35):
            self.zoom_step.addItem(f"{percent}%", percent)
        self.zoom_step.setCurrentIndex(self.zoom_step.findData(25))
        self.keep_zoom = QCheckBox("切换图片保持放大状态")
        self.annotation_line_width = QComboBox()
        for width in range(1, 5):
            self.annotation_line_width.addItem(f"{width} 像素", width)
        self.annotation_line_width.setCurrentIndex(self.annotation_line_width.findData(2))
        self.annotation_fill = QCheckBox("标注线 + 内部填充")
        self.config_path = QLabel("本地配置文件：未初始化")
        self.config_path.setObjectName("configHint")
        self._build()
        self.theme.currentTextChanged.connect(self.theme_changed)
        self.show_guide_lines.toggled.connect(self._emit_display_settings)
        self.zoom_step.currentIndexChanged.connect(self._emit_display_settings)
        self.keep_zoom.toggled.connect(self._emit_display_settings)
        self.annotation_line_width.currentIndexChanged.connect(self._emit_display_settings)
        self.annotation_fill.toggled.connect(self._emit_display_settings)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)

        title = QLabel("软件配置")
        title.setObjectName("configTitle")
        subtitle = QLabel("界面主题和本地使用设置会保存在当前工具目录。")
        subtitle.setObjectName("configSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        guide_card, guide_form = self._make_card("辅助线")
        guide_form.addRow(self.show_guide_lines)
        guide_form.addRow("作用范围", QLabel("仅在图片区域内显示"))
        columns.addWidget(guide_card, 1)

        view_card, view_form = self._make_card("视图")
        view_form.addRow("界面主题", self.theme)
        view_form.addRow("缩放步进", self.zoom_step)
        view_form.addRow(self.keep_zoom)
        columns.addWidget(view_card, 1)
        layout.addLayout(columns)

        display_columns = QHBoxLayout()
        display_columns.setSpacing(12)
        annotation_card, annotation_form = self._make_card("标注显示")
        annotation_form.addRow("标注线粗细", self.annotation_line_width)
        annotation_form.addRow(self.annotation_fill)
        display_columns.addWidget(annotation_card, 1)

        storage_card, storage_form = self._make_card("本地保存")
        storage_form.addRow("保存方式", QLabel("关闭软件时自动保存"))
        storage_form.addRow("项目记录", QLabel("记住最近打开的项目目录"))
        display_columns.addWidget(storage_card, 1)
        layout.addLayout(display_columns)

        local_card, local_form = self._make_card("配置文件")
        local_form.addRow(self.config_path)
        layout.addWidget(local_card)
        layout.addStretch(1)

        actions = QHBoxLayout()
        reset = QPushButton("恢复本页默认")
        reset.clicked.connect(self._reset_defaults)
        save = QPushButton("保存")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_requested)
        actions.addStretch(1)
        actions.addWidget(reset)
        actions.addWidget(save)
        layout.addLayout(actions)

    @staticmethod
    def _make_card(title_text: str) -> tuple[QFrame, QFormLayout]:
        card = QFrame()
        card.setObjectName("configCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 13, 14, 14)
        card_layout.setSpacing(9)
        title = QLabel(title_text)
        title.setObjectName("configSectionTitle")
        card_layout.addWidget(title)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        card_layout.addLayout(form)
        return card, form

    def set_config_path(self, path: str) -> None:
        self.config_path.setText(f"配置文件：{path}")
        self.config_path.setToolTip(path)

    def display_settings(self) -> dict[str, object]:
        return {
            "show_guide_lines": self.show_guide_lines.isChecked(),
            "zoom_step_percent": int(self.zoom_step.currentData()),
            "keep_zoom_on_image_switch": self.keep_zoom.isChecked(),
            "annotation_line_width": int(self.annotation_line_width.currentData()),
            "annotation_fill_enabled": self.annotation_fill.isChecked(),
        }

    def load_display_settings(self, settings: AppSettings) -> None:
        controls = (self.show_guide_lines, self.zoom_step, self.keep_zoom, self.annotation_line_width, self.annotation_fill)
        for control in controls:
            control.blockSignals(True)
        self.show_guide_lines.setChecked(settings.show_guide_lines)
        self.zoom_step.setCurrentIndex(max(0, self.zoom_step.findData(settings.zoom_step_percent)))
        self.keep_zoom.setChecked(settings.keep_zoom_on_image_switch)
        self.annotation_line_width.setCurrentIndex(max(0, self.annotation_line_width.findData(settings.annotation_line_width)))
        self.annotation_fill.setChecked(settings.annotation_fill_enabled)
        for control in controls:
            control.blockSignals(False)
        self._emit_display_settings()

    def _emit_display_settings(self, *_args) -> None:
        self.display_changed.emit(self.display_settings())

    def _reset_defaults(self) -> None:
        self.theme.setCurrentText("跟随系统")
        self.show_guide_lines.setChecked(True)
        self.zoom_step.setCurrentIndex(self.zoom_step.findData(25))
        self.keep_zoom.setChecked(False)
        self.annotation_line_width.setCurrentIndex(self.annotation_line_width.findData(2))
        self.annotation_fill.setChecked(False)
        self.save_requested.emit()


class ShortcutSettingsPage(QWidget):
    shortcuts_changed = Signal(object)
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("shortcutPage")
        self.previous_image = QKeySequenceEdit(QKeySequence("A"))
        self.next_image = QKeySequenceEdit(QKeySequence("D"))
        self.save_annotation = QKeySequenceEdit(QKeySequence.StandardKey.Save)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)
        title = QLabel("快捷键设置")
        title.setObjectName("shortcutTitle")
        subtitle = QLabel("点击输入框后直接按键录入。图片切换快捷键仅在画布区域生效。")
        subtitle.setObjectName("shortcutSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("shortcutCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 13, 14, 14)
        card_layout.setSpacing(9)
        section = QLabel("标注操作")
        section.setObjectName("shortcutSectionTitle")
        card_layout.addWidget(section)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.addRow("上一张图片", self.previous_image)
        form.addRow("下一张图片", self.next_image)
        form.addRow("保存标注", self.save_annotation)
        card_layout.addLayout(form)
        layout.addWidget(card)
        layout.addStretch(1)

        actions = QHBoxLayout()
        reset = QPushButton("恢复默认")
        reset.clicked.connect(self._reset_defaults)
        save = QPushButton("保存")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        actions.addStretch(1)
        actions.addWidget(reset)
        actions.addWidget(save)
        layout.addLayout(actions)

    @staticmethod
    def _sequence_text(editor: QKeySequenceEdit) -> str:
        return editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)

    def values(self) -> dict[str, str]:
        return {
            "previous_image_shortcut": self._sequence_text(self.previous_image) or "A",
            "next_image_shortcut": self._sequence_text(self.next_image) or "D",
            "save_annotation_shortcut": self._sequence_text(self.save_annotation) or "Ctrl+S",
        }

    def load_settings(self, settings: AppSettings) -> None:
        self.previous_image.setKeySequence(QKeySequence(settings.previous_image_shortcut))
        self.next_image.setKeySequence(QKeySequence(settings.next_image_shortcut))
        self.save_annotation.setKeySequence(QKeySequence(settings.save_annotation_shortcut))

    def _reset_defaults(self) -> None:
        self.previous_image.setKeySequence(QKeySequence("A"))
        self.next_image.setKeySequence(QKeySequence("D"))
        self.save_annotation.setKeySequence(QKeySequence.StandardKey.Save)

    def _save(self) -> None:
        values = self.values()
        duplicates = [value for value in values.values() if list(values.values()).count(value) > 1]
        if duplicates:
            QMessageBox.warning(self, "快捷键设置", "快捷键不能重复，请调整后再保存。")
            return
        self.shortcuts_changed.emit(values)
        self.save_requested.emit()


class AboutPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("aboutPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)

        title = QLabel("关于软件")
        title.setObjectName("aboutTitle")
        subtitle = QLabel("YOLO Annotation Tool Pro")
        subtitle.setObjectName("aboutSubTitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("aboutCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(12)
        identity = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(QIcon(str(Path(__file__).with_name("assets") / "app.svg")).pixmap(56, 56))
        icon.setFixedSize(60, 60)
        identity.addWidget(icon)
        name_column = QVBoxLayout()
        name = QLabel("YOLO Annotation Tool Pro")
        name.setObjectName("aboutName")
        mode = QLabel("本地运行模式")
        mode.setObjectName("aboutMode")
        name_column.addWidget(name)
        name_column.addWidget(mode)
        name_column.addStretch(1)
        identity.addLayout(name_column, 1)
        card_layout.addLayout(identity)

        section = QLabel("已恢复功能")
        section.setObjectName("aboutSectionTitle")
        card_layout.addWidget(section)
        capabilities = QLabel("图片项目管理、YOLO 矩形框与分割标注、SAM 点选、批量自动标注、模型训练、模型验证和快捷键配置")
        capabilities.setObjectName("aboutText")
        capabilities.setWordWrap(True)
        card_layout.addWidget(capabilities)

        config_path = QLabel(f"本地配置目录：{Path.cwd() / 'config'}")
        config_path.setObjectName("aboutHint")
        config_path.setWordWrap(True)
        card_layout.addWidget(config_path)
        open_config = QPushButton("打开配置目录")
        open_config.clicked.connect(self.open_config_folder)
        card_layout.addWidget(open_config, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(card)
        layout.addStretch(1)

    def open_config_folder(self) -> None:
        folder = Path.cwd() / "config"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Annotation Tool Pro")
        self.setWindowIcon(QIcon(str(Path(__file__).with_name("assets") / "app.svg")))
        self.resize(1200, 760)
        self.annotation = AnnotationPage()
        self.auto = AutoAnnotatePage(self.annotation)
        self.auto.job_finished.connect(lambda _stats: self.annotation.update_stats())
        self.annotation.classes_changed.connect(self.auto.sync_classes)
        self.auto.sync_classes(self.annotation.current_class_names())
        self.training = TrainingPage()
        self.validation = ValidationPage()
        self.settings = SettingsPage()
        self.shortcuts = ShortcutSettingsPage()
        self.about = AboutPage()
        self.settings_store = SettingsStore()
        self.settings.set_config_path(str(self.settings_store.path))
        self.pages = QStackedWidget()

        def scroll_page(page: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setObjectName("pageScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            return scroll

        self.pages.addWidget(self.annotation)
        self.pages.addWidget(scroll_page(self.auto))
        self.pages.addWidget(scroll_page(self.training))
        self.pages.addWidget(scroll_page(self.validation))
        self.pages.addWidget(scroll_page(self.settings))
        self.pages.addWidget(scroll_page(self.shortcuts))
        self.pages.addWidget(scroll_page(self.about))
        self._build_navigation()
        self._build_status_widgets()
        self.statusBar().showMessage("就绪：欢迎使用标注工具。")
        for page in (self.annotation, self.auto, self.training, self.validation):
            page.status_changed.connect(self.statusBar().showMessage)
        self.annotation.project_root.textChanged.connect(self._update_status_project)
        self.annotation.dataset_ready.connect(self._use_dataset_yaml)
        self.annotation.canvas.view_changed.connect(self._update_status_view)
        self.annotation.stats_changed.connect(self._update_status_stats)
        self.settings.theme_changed.connect(self._on_theme_changed)
        self.settings.display_changed.connect(self._apply_display_settings)
        self.settings.save_requested.connect(self._save_settings_and_report)
        self.shortcuts.shortcuts_changed.connect(self._apply_shortcuts)
        self.shortcuts.save_requested.connect(self._save_settings_and_report)
        self._restore_settings()

    def _restore_settings(self) -> None:
        settings = self.settings_store.load()
        self.annotation.project_root.setText(settings.last_project)
        self.annotation.sam_model.setText(settings.sam_model)
        self.auto.detector.setText(settings.detector_model)
        self.auto.segmenter.setText(settings.sam_model)
        self.training.model.setText(settings.training_model)
        self.training.use_dataset(settings.training_data)
        if settings.training_project and settings.training_project != "runs/train":
            self.training.project.setText(settings.training_project)
        self.validation.model.setText(settings.validation_model)
        self.validation.output_dir.setText(settings.validation_output_dir)
        self.settings.theme.setCurrentText(settings.theme)
        self.settings.load_display_settings(settings)
        self.shortcuts.load_settings(settings)
        self._apply_shortcuts(self.shortcuts.values())
        self.apply_theme(settings.theme)
        self._update_theme_button_text(settings.theme)

    def _save_settings(self) -> None:
        self.settings_store.save(
            AppSettings(
                theme=self.settings.theme.currentText(),
                **self.settings.display_settings(),
                **self.shortcuts.values(),
                last_project=self.annotation.project_root.text().strip(),
                detector_model=self.auto.detector.text().strip(),
                sam_model=self.annotation.sam_model.text().strip() or self.auto.segmenter.text().strip(),
                training_model=self.training.model.text().strip(),
                training_data=self.training.data.text().strip(),
                training_project=self.training.project.text().strip(),
                validation_model=self.validation.model.text().strip(),
                validation_output_dir=self.validation.output_dir.text().strip(),
            )
        )

    def _apply_display_settings(self, values: object) -> None:
        if isinstance(values, dict):
            self.annotation.canvas.set_display_settings(**values)

    def _apply_shortcuts(self, values: object) -> None:
        if not isinstance(values, dict):
            return
        self.annotation._save_shortcut.setKey(QKeySequence(str(values["save_annotation_shortcut"])))
        self.annotation._previous_shortcut.setKey(QKeySequence(str(values["previous_image_shortcut"])))
        self.annotation._next_shortcut.setKey(QKeySequence(str(values["next_image_shortcut"])))

    def _use_dataset_yaml(self, path: str) -> None:
        self.training.use_dataset(path)

    def _save_settings_and_report(self) -> None:
        self._save_settings()
        self.statusBar().showMessage("软件配置已保存")

    def _build_navigation(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(160)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)
        self.sidebar_toggle = QPushButton("☰ 折叠菜单")
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setToolTip("收起菜单")
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        sidebar_layout.addWidget(self.sidebar_toggle)
        self.navigation_list = QListWidget()
        self.navigation_list.setObjectName("menuList")
        self._page_titles = ("标注管理", "自动标注", "训练管理", "模型验证", "软件配置", "快捷键设置", "关于软件")
        self._page_icons = (
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            QStyle.StandardPixmap.SP_ComputerIcon,
            QStyle.StandardPixmap.SP_MediaPlay,
            QStyle.StandardPixmap.SP_DialogApplyButton,
            QStyle.StandardPixmap.SP_FileDialogContentsView,
            QStyle.StandardPixmap.SP_CommandLink,
            QStyle.StandardPixmap.SP_MessageBoxInformation,
        )
        for title, standard_icon in zip(self._page_titles, self._page_icons):
            item = QListWidgetItem(self.style().standardIcon(standard_icon), title)
            self.navigation_list.addItem(item)
        for index in range(self.navigation_list.count()):
            self.navigation_list.item(index).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.navigation_list.setCurrentRow(0)
        self.navigation_list.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.navigation_list, 1)
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("primaryButton")
        self.theme_button.clicked.connect(self._toggle_theme)
        sidebar_layout.addWidget(self.theme_button)

        main_content = QFrame()
        main_content.setObjectName("mainContent")
        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.pages)
        layout.addWidget(self.sidebar)
        layout.addWidget(main_content, 1)
        self.setCentralWidget(root)

    def _toggle_sidebar(self) -> None:
        expanded = self.sidebar.width() > 70
        self.sidebar.setFixedWidth(70 if expanded else 160)
        for index in range(len(self._page_titles)):
            self.navigation_list.item(index).setText("" if expanded else self._page_titles[index])
            alignment = Qt.AlignmentFlag.AlignCenter if expanded else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            self.navigation_list.item(index).setTextAlignment(alignment)
        self.sidebar_toggle.setText("☰" if expanded else "☰ 折叠菜单")
        self.sidebar_toggle.setToolTip("展开菜单" if expanded else "收起菜单")

    def _switch_page(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self._page_titles):
            self.statusBar().showMessage(f"当前页面：{self._page_titles[index]}")

    def _toggle_theme(self) -> None:
        target = "浅色" if self.settings.theme.currentText() == "深色" else "深色"
        self.settings.theme.setCurrentText(target)

    def _on_theme_changed(self, theme: str) -> None:
        self.apply_theme(theme)
        self._update_theme_button_text(theme)

    def _update_theme_button_text(self, theme: str) -> None:
        if hasattr(self, "theme_button"):
            self.theme_button.setText("切换为浅色主题" if theme == "深色" else "切换为深色主题")

    def _build_status_widgets(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.status_project_label = QLabel("当前项目：未打开")
        self.status_project_label.setObjectName("statusProjectLabel")
        self.status_project_label.setFixedWidth(260)
        self.status_project_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_view_label = QLabel("缩放: 100%  旋转: --")
        self.status_view_label.setObjectName("statusViewLabel")
        self.status_view_label.setMaximumWidth(150)
        self.status_mid_sep = QLabel("")
        self.status_mid_sep.setObjectName("statusMidSep")
        self.status_stats_label = QLabel("图片总数:0  已标注:0  未标注:0  负样本:0")
        self.status_stats_label.setObjectName("statusStatsLabel")
        self.status_stats_label.setMaximumWidth(290)
        self.status_right_panel = QWidget()
        self.status_right_panel.setObjectName("statusRightPanel")
        right_layout = QHBoxLayout(self.status_right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        for widget in (
            self.status_view_label,
            self.status_stats_label,
        ):
            right_layout.addWidget(widget)
        bar.addWidget(self.status_project_label)
        bar.addPermanentWidget(self.status_right_panel)

    def _update_status_project(self, root: str) -> None:
        root = root.strip()
        if not root:
            self.status_project_label.setText("当前项目：未打开")
            self.status_project_label.setToolTip("")
            return
        name = Path(root).name or root
        self.status_project_label.setText(f"当前项目：{name}")
        self.status_project_label.setToolTip(root)

    def _update_status_view(self, zoom: float) -> None:
        self.status_view_label.setText(f"缩放: {round(zoom * 100)}%  旋转: --")

    def _update_status_stats(self, total: int, labeled: int, unlabeled: int, negative: int) -> None:
        self.status_stats_label.setText(f"图片总数:{total}  已标注:{labeled}  未标注:{unlabeled}  负样本:{negative}")

    def apply_theme(self, theme: str) -> None:
        if theme == "深色":
            self.setStyleSheet(
                "QMainWindow, QWidget { background: #0d1117; color: #c9d1d9; }"
                "QToolBar { background: #161b22; border: 0; border-bottom: 1px solid #30363d; spacing: 4px; padding: 5px 8px; }"
                "QToolButton { color: #c9d1d9; padding: 7px 13px; border-radius: 4px; }"
                "QToolButton:hover { background: #21262d; }"
                "QToolButton:checked { background: #1f6feb; color: #ffffff; }"
                "QLabel#appBrand { color: #f0f6fc; font-size: 15px; font-weight: 600; padding: 0 8px 0 2px; }"
                "QLabel#localMode { color: #7ee787; background: #1b3a2a; border-radius: 9px; padding: 3px 8px; margin-left: 6px; }"
                "QFrame#sidebar { background: #131720; border: 0; border-right: 1px solid #30363d; }"
                "QFrame#mainContent { background: #1f232a; border: 0; }"
                "QLabel#sidebarBrand { color: #f0f6fc; font-size: 15px; font-weight: 600; padding: 4px 3px; }"
                "QPushButton#sidebarToggle { background: transparent; color: #8b949e; border: 0; padding: 4px; text-align: right; }"
                "QPushButton#sidebarToggle:hover { background: #21262d; color: #f0f6fc; }"
                "QListWidget#menuList { background: transparent; border: 0; padding: 0; }"
                "QListWidget#menuList::item { color: #d2d9e1; min-height: 28px; margin: 3px 0; padding: 10px 12px; border-radius: 8px; }"
                "QListWidget#menuList::item:hover { background: #202a39; color: #ffffff; }"
                "QListWidget#menuList::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #06b6d4); border-left: 3px solid #93c5fd; color: #ffffff; font-weight: 600; }"
                "QPushButton#sidebarThemeButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 7px; }"
                "QPushButton#sidebarThemeButton:hover { background: #30363d; border-color: #58a6ff; }"
                "QLineEdit, QComboBox, QListWidget, QTextEdit { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px; }"
                "QListWidget::item { padding: 5px 3px; border-radius: 3px; } QListWidget::item:selected { background: #1f6feb; color: #ffffff; }"
                "QListWidget#imageList::item { min-height: 21px; border-bottom: 1px solid #21262d; padding: 5px 5px; }"
                "QListWidget#classList::item { min-height: 20px; border-bottom: 1px solid #21262d; padding: 5px; }"
                "QListWidget#currentAnnotationList::item { min-height: 20px; border-bottom: 1px solid #21262d; padding: 5px; }"
                "QPushButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 7px 10px; }"
                "QPushButton:hover { background: #30363d; border-color: #58a6ff; }"
                "QPushButton#primaryButton { background: #1f6feb; color: #ffffff; border-color: #1f6feb; font-weight: 600; }"
                "QPushButton#primaryButton:hover { background: #388bfd; border-color: #388bfd; }"
                "QPushButton:disabled { color: #6e7681; }"
                "QFrame#sidePanel { background: #161b22; border: 1px solid #30363d; border-radius: 5px; }"
                "QScrollArea#sideScroll { background: transparent; border: 0; }"
                "QFrame#settingsPanel { background: #1f242d; border: 1px solid #30363d; border-radius: 8px; }"
                "QFrame#settingsPanel QLabel { font-size: 12px; }"
                "QFrame#settingsPanel QLineEdit { min-height: 18px; padding: 2px 6px; }"
                "QFrame#settingsPanel QComboBox { min-height: 18px; padding: 2px 24px 2px 6px; font-size: 12px; }"
                "QScrollArea#settingsScroll { background: transparent; border: 0; }"
                "QScrollArea#pageScroll { background: transparent; border: 0; }"
                "QLabel#sideSection { color: #8b949e; font-size: 12px; font-weight: 600; margin-top: 6px; padding: 6px 2px 4px; border-bottom: 1px solid #30363d; }"
                "QFrame#listHeader { background: #0d1117; border: 1px solid #30363d; border-radius: 3px; }"
                "QLabel#listHeaderLabel { color: #8b949e; font-size: 11px; font-weight: 600; }"
                "QLabel#pageTitle { color: #f0f6fc; font-size: 18px; font-weight: 600; padding: 6px 0 12px 0; }"
                "QLabel#configTitle { color: #f0f6fc; font-size: 18px; font-weight: 600; }"
                "QLabel#configSubTitle, QLabel#configHint { color: #8b949e; font-size: 12px; }"
                "QFrame#configCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
                "QLabel#configSectionTitle { color: #c9d1d9; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #30363d; }"
                "QFrame#configCard QLabel { color: #c9d1d9; font-size: 12px; }"
                "QFrame#configCard QComboBox { min-height: 20px; padding: 3px 24px 3px 6px; }"
                "QLabel#shortcutTitle { color: #f0f6fc; font-size: 18px; font-weight: 600; }"
                "QLabel#shortcutSubTitle { color: #8b949e; font-size: 12px; }"
                "QFrame#shortcutCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
                "QLabel#shortcutSectionTitle { color: #c9d1d9; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #30363d; }"
                "QFrame#shortcutCard QLabel { color: #c9d1d9; font-size: 12px; }"
                "QKeySequenceEdit { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; min-height: 20px; padding: 3px 6px; }"
                "QKeySequenceEdit:focus { border-color: #58a6ff; }"
                "QLabel#aboutTitle { color: #f0f6fc; font-size: 18px; font-weight: 600; }"
                "QLabel#aboutSubTitle, QLabel#aboutHint { color: #8b949e; font-size: 12px; }"
                "QFrame#aboutCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; }"
                "QLabel#aboutName { color: #f0f6fc; font-size: 16px; font-weight: 600; }"
                "QLabel#aboutMode { color: #7ee787; font-size: 12px; }"
                "QLabel#aboutSectionTitle { color: #c9d1d9; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #30363d; }"
                "QLabel#aboutText { color: #c9d1d9; font-size: 12px; line-height: 1.5; }"
                "QGroupBox#trainingCard, QGroupBox#trainingConvertCard, QGroupBox#trainingLogCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #c9d1d9; }"
                "QGroupBox#trainingCard::title, QGroupBox#trainingConvertCard::title, QGroupBox#trainingLogCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #8b949e; }"
                "QGroupBox#validationCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #c9d1d9; }"
                "QGroupBox#validationCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #8b949e; }"
                "QLabel#formSection { color: #8b949e; font-size: 12px; font-weight: 600; padding: 7px 0 3px; border-bottom: 1px solid #30363d; }"
                "QGroupBox#autoBatchCard, QGroupBox#autoProgressCard { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #c9d1d9; }"
                "QGroupBox#autoBatchCard::title, QGroupBox#autoProgressCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #8b949e; }"
                "QFrame#annotationHeader { background: #161b22; border: 1px solid #30363d; border-radius: 5px; }"
                "QLabel#annotationTitle { color: #f0f6fc; font-size: 16px; font-weight: 600; padding: 0 0 2px 0; }"
                "QLabel#statusStats { color: #8b949e; background: #161b22; border: 1px solid #30363d; border-radius: 9px; padding: 4px 8px; }"
                "QFrame#canvasPanel { background: #161b22; border: 1px solid #30363d; border-radius: 5px; }"
                "QWidget#canvasWorkspace { background: transparent; }"
                "QFrame#classifyBar { background: #161b22; border: 1px solid #30363d; border-radius: 5px; }"
                "QLabel#classifyBarLabel { color: #c9d1d9; font-weight: 600; }"
                "QLabel#canvasModeLabel { color: #8b949e; }"
                "QLabel#canvasStatus { color: #8b949e; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 6px 9px; }"
                "QSplitter::handle { background: #30363d; width: 5px; }"
                "QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }"
                "QScrollBar::handle:vertical { background: #484f58; border-radius: 4px; min-height: 24px; }"
                "QScrollBar::handle:vertical:hover { background: #6e7681; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
                "QProgressBar { background: #161b22; border: 1px solid #30363d; border-radius: 4px; text-align: center; }"
                "QProgressBar::chunk { background: #409eff; border-radius: 3px; }"
                "QLabel#taskStatus { color: #8b949e; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 7px; }"
                "QTextEdit#taskLog { font-family: Consolas; background: #0d1117; border-color: #30363d; }"
                "QTextEdit#trainingLogEdit { font-family: Consolas; background: #0d1117; border-color: #30363d; }"
                "QLabel#previewPanel { background: #0d1117; border: 1px solid #30363d; border-radius: 5px; color: #8b949e; }"
                "QStatusBar { background: #161b22; color: #8b949e; border-top: 1px solid #30363d; }"
                "QWidget#statusRightPanel { background: transparent; }"
                "QLabel#statusProjectLabel, QLabel#statusViewLabel, QLabel#statusStatsLabel { color: #8b949e; padding: 0 5px; }"
                "QLabel#statusProjectSep, QLabel#statusMidSep { color: #8b949e; padding: 0 2px; }"
            )
        else:
            self.setStyleSheet(
                "QMainWindow, QWidget { background: #f5f7fa; color: #303133; }"
                "QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #dcdfe6; spacing: 4px; padding: 5px 8px; }"
                "QToolButton { color: #303133; padding: 7px 13px; border-radius: 4px; }"
                "QToolButton:hover { background: #ecf5ff; color: #409eff; }"
                "QToolButton:checked { background: #ecf5ff; color: #409eff; font-weight: 600; }"
                "QLabel#appBrand { color: #303133; font-size: 15px; font-weight: 600; padding: 0 8px 0 2px; }"
                "QLabel#localMode { color: #67c23a; background: #f0f9eb; border-radius: 9px; padding: 3px 8px; margin-left: 6px; }"
                "QFrame#sidebar { background: #f8fbff; border: 0; border-right: 1px solid #d4dbe6; }"
                "QFrame#mainContent { background: #f5f7fb; border: 0; }"
                "QLabel#sidebarBrand { color: #303133; font-size: 15px; font-weight: 600; padding: 4px 3px; }"
                "QPushButton#sidebarToggle { background: transparent; color: #909399; border: 0; padding: 4px; text-align: right; }"
                "QPushButton#sidebarToggle:hover { background: #ecf5ff; color: #409eff; }"
                "QListWidget#menuList { background: transparent; border: 0; padding: 0; }"
                "QListWidget#menuList::item { color: #374151; min-height: 28px; margin: 3px 0; padding: 10px 12px; border-radius: 8px; }"
                "QListWidget#menuList::item:hover { background: #e9f2ff; color: #0f172a; }"
                "QListWidget#menuList::item:selected { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #06b6d4); border-left: 3px solid #93c5fd; color: #ffffff; font-weight: 600; }"
                "QPushButton#sidebarThemeButton { background: #ffffff; color: #606266; border: 1px solid #dcdfe6; border-radius: 4px; padding: 7px; }"
                "QPushButton#sidebarThemeButton:hover { color: #409eff; border-color: #409eff; background: #ecf5ff; }"
                "QLineEdit, QComboBox, QListWidget, QTextEdit { background: #ffffff; color: #303133; border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px; selection-background-color: #409eff; }"
                "QListWidget::item { padding: 5px 3px; border-radius: 3px; } QListWidget::item:selected { background: #ecf5ff; color: #409eff; }"
                "QListWidget#imageList::item { min-height: 21px; border-bottom: 1px solid #ebeef5; padding: 5px 5px; }"
                "QListWidget#classList::item { min-height: 20px; border-bottom: 1px solid #ebeef5; padding: 5px; }"
                "QListWidget#currentAnnotationList::item { min-height: 20px; border-bottom: 1px solid #ebeef5; padding: 5px; }"
                "QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border-color: #409eff; }"
                "QPushButton { background: #ffffff; color: #303133; border: 1px solid #dcdfe6; border-radius: 4px; padding: 7px 10px; }"
                "QPushButton:hover { color: #409eff; border-color: #409eff; background: #ecf5ff; }"
                "QPushButton#primaryButton { background: #409eff; color: #ffffff; border-color: #409eff; font-weight: 600; }"
                "QPushButton#primaryButton:hover { background: #66b1ff; border-color: #66b1ff; color: #ffffff; }"
                "QPushButton:disabled { color: #c0c4cc; }"
                "QFrame#sidePanel { background: #ffffff; border: 1px solid #dcdfe6; border-radius: 5px; }"
                "QScrollArea#sideScroll { background: transparent; border: 0; }"
                "QFrame#settingsPanel { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 8px; }"
                "QFrame#settingsPanel QLabel { font-size: 12px; }"
                "QFrame#settingsPanel QLineEdit { min-height: 18px; padding: 2px 6px; }"
                "QFrame#settingsPanel QComboBox { min-height: 18px; padding: 2px 24px 2px 6px; font-size: 12px; }"
                "QScrollArea#settingsScroll { background: transparent; border: 0; }"
                "QScrollArea#pageScroll { background: transparent; border: 0; }"
                "QLabel#sideSection { color: #909399; font-size: 12px; font-weight: 600; margin-top: 6px; padding: 6px 2px 4px; border-bottom: 1px solid #dcdfe6; }"
                "QFrame#listHeader { background: #f5f7fa; border: 1px solid #dcdfe6; border-radius: 3px; }"
                "QLabel#listHeaderLabel { color: #909399; font-size: 11px; font-weight: 600; }"
                "QLabel#pageTitle { color: #303133; font-size: 18px; font-weight: 600; padding: 6px 0 12px 0; }"
                "QLabel#configTitle { color: #303133; font-size: 18px; font-weight: 600; }"
                "QLabel#configSubTitle, QLabel#configHint { color: #909399; font-size: 12px; }"
                "QFrame#configCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; }"
                "QLabel#configSectionTitle { color: #303133; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #e5e7eb; }"
                "QFrame#configCard QLabel { color: #606266; font-size: 12px; }"
                "QFrame#configCard QComboBox { min-height: 20px; padding: 3px 24px 3px 6px; }"
                "QLabel#shortcutTitle { color: #303133; font-size: 18px; font-weight: 600; }"
                "QLabel#shortcutSubTitle { color: #909399; font-size: 12px; }"
                "QFrame#shortcutCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; }"
                "QLabel#shortcutSectionTitle { color: #303133; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #e5e7eb; }"
                "QFrame#shortcutCard QLabel { color: #606266; font-size: 12px; }"
                "QKeySequenceEdit { background: #ffffff; color: #303133; border: 1px solid #dcdfe6; border-radius: 4px; min-height: 20px; padding: 3px 6px; }"
                "QKeySequenceEdit:focus { border-color: #409eff; }"
                "QLabel#aboutTitle { color: #303133; font-size: 18px; font-weight: 600; }"
                "QLabel#aboutSubTitle, QLabel#aboutHint { color: #909399; font-size: 12px; }"
                "QFrame#aboutCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; }"
                "QLabel#aboutName { color: #303133; font-size: 16px; font-weight: 600; }"
                "QLabel#aboutMode { color: #67c23a; font-size: 12px; }"
                "QLabel#aboutSectionTitle { color: #303133; font-size: 13px; font-weight: 600; padding-bottom: 3px; border-bottom: 1px solid #e5e7eb; }"
                "QLabel#aboutText { color: #606266; font-size: 12px; line-height: 1.5; }"
                "QGroupBox#trainingCard, QGroupBox#trainingConvertCard, QGroupBox#trainingLogCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #303133; }"
                "QGroupBox#trainingCard::title, QGroupBox#trainingConvertCard::title, QGroupBox#trainingLogCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #606266; }"
                "QGroupBox#validationCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #303133; }"
                "QGroupBox#validationCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #606266; }"
                "QLabel#formSection { color: #606266; font-size: 12px; font-weight: 600; padding: 7px 0 3px; border-bottom: 1px solid #e5e7eb; }"
                "QGroupBox#autoBatchCard, QGroupBox#autoProgressCard { background: #ffffff; border: 1px solid #d4dbe6; border-radius: 6px; margin-top: 10px; font-weight: 600; color: #303133; }"
                "QGroupBox#autoBatchCard::title, QGroupBox#autoProgressCard::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #606266; }"
                "QFrame#annotationHeader { background: #ffffff; border: 1px solid #dcdfe6; border-radius: 5px; }"
                "QLabel#annotationTitle { color: #303133; font-size: 16px; font-weight: 600; padding: 0 0 2px 0; }"
                "QLabel#statusStats { color: #606266; background: #f4f4f5; border: 1px solid #e9e9eb; border-radius: 9px; padding: 4px 8px; }"
                "QFrame#canvasPanel { background: #1a1a1a; border: 1px solid #dcdfe6; border-radius: 5px; }"
                "QWidget#canvasWorkspace { background: transparent; }"
                "QFrame#classifyBar { background: #ffffff; border: 1px solid #dcdfe6; border-radius: 5px; }"
                "QLabel#classifyBarLabel { color: #303133; font-weight: 600; }"
                "QLabel#canvasModeLabel { color: #909399; }"
                "QLabel#canvasStatus { color: #606266; background: #ffffff; border: 1px solid #dcdfe6; border-radius: 4px; padding: 6px 9px; }"
                "QSplitter::handle { background: #dcdfe6; width: 5px; }"
                "QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }"
                "QScrollBar::handle:vertical { background: #c0c4cc; border-radius: 4px; min-height: 24px; }"
                "QScrollBar::handle:vertical:hover { background: #909399; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
                "QProgressBar { background: #ffffff; border: 1px solid #dcdfe6; border-radius: 4px; text-align: center; }"
                "QProgressBar::chunk { background: #409eff; border-radius: 3px; }"
                "QLabel#taskStatus { color: #606266; background: #f4f4f5; border: 1px solid #e9e9eb; border-radius: 4px; padding: 7px; }"
                "QTextEdit#taskLog { font-family: Consolas; background: #1a1a1a; color: #e5eaf3; border-color: #303133; }"
                "QTextEdit#trainingLogEdit { font-family: Consolas; background: #1a1a1a; color: #e5eaf3; border-color: #303133; }"
                "QLabel#previewPanel { background: #1a1a1a; border: 1px solid #303133; border-radius: 5px; color: #c0c4cc; }"
                "QStatusBar { background: #ffffff; color: #606266; border-top: 1px solid #dcdfe6; }"
                "QWidget#statusRightPanel { background: transparent; }"
                "QLabel#statusProjectLabel, QLabel#statusViewLabel, QLabel#statusStatsLabel { color: #606266; padding: 0 5px; }"
                "QLabel#statusProjectSep, QLabel#statusMidSep { color: #57606a; padding: 0 2px; }"
            )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        event.accept()


def main() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    _relaunch_with_cuda_runtime()
    app = QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    app.exec()


def _relaunch_with_cuda_runtime() -> None:
    """Restart the GUI with the installed CUDA environment when launched from .venv."""
    if getattr(sys, "frozen", False):
        return
    target = Path(r"E:\Anaconda3\envs\yolov11\python.exe")
    current = Path(sys.executable)
    if os.environ.get("YOLO_CUDA_RELAUNCHED") == "1" or not target.is_file():
        return
    try:
        if current.resolve() == target.resolve():
            return
    except OSError:
        return
    # Keep explicit system/conda launches untouched; this fallback is for the
    # repository's CPU virtual environment, which cannot provide CUDA.
    current_text = str(current).replace("/", "\\").lower()
    if "\\.venv\\scripts\\python.exe" not in current_text:
        return
    os.environ["YOLO_CUDA_RELAUNCHED"] = "1"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "config"))
    os.execv(str(target), [str(target), "-m", "yolo_annotation_tool.qt_app", *sys.argv[1:]])


if __name__ == "__main__":
    main()
