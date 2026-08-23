"""Ultralytics model export wrapper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportConfig:
    model: str
    format: str = "onnx"
    imgsz: int = 640
    opset: int | None = 12
    simplify: bool = True
    device: str | None = None


def export_model(config: ExportConfig):
    """Export a PyTorch model using the format identifiers accepted by Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
    kwargs = {"format": config.format, "imgsz": config.imgsz, "simplify": config.simplify}
    if config.opset is not None and config.format == "onnx":
        kwargs["opset"] = config.opset
    if config.device and config.format in {"engine", "ncnn"}:
        kwargs["device"] = config.device
    return YOLO(config.model).export(**kwargs)
