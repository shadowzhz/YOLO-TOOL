"""Small local settings store for the desktop application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class AppSettings:
    theme: str = "跟随系统"
    show_guide_lines: bool = True
    zoom_step_percent: int = 25
    keep_zoom_on_image_switch: bool = False
    annotation_line_width: int = 2
    annotation_fill_enabled: bool = False
    previous_image_shortcut: str = "A"
    next_image_shortcut: str = "D"
    save_annotation_shortcut: str = "Ctrl+S"
    last_project: str = ""
    detector_model: str = "yolo11n.pt"
    sam_model: str = "sam2_b.pt"
    training_model: str = "yolo11n.pt"
    training_data: str = "data.yaml"
    training_project: str = "runs/train"
    validation_model: str = "best.pt"
    validation_output_dir: str = "runs/validation"


class SettingsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path.cwd() / "config" / "app_settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppSettings()
        if not isinstance(raw, dict):
            return AppSettings()
        defaults = AppSettings()
        values = {}
        for field in fields(AppSettings):
            value = raw.get(field.name)
            default = getattr(defaults, field.name)
            if isinstance(default, bool) and isinstance(value, bool):
                values[field.name] = value
            elif isinstance(default, int) and not isinstance(default, bool) and isinstance(value, int) and not isinstance(value, bool):
                values[field.name] = value
            elif isinstance(default, str) and isinstance(value, str):
                values[field.name] = value
        return AppSettings(**values)

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
