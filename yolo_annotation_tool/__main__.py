from __future__ import annotations

import argparse

from .sam import AutoAnnotator
from .export import ExportConfig, export_model
from .training import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="yolo-annotation-tool")
    sub = parser.add_subparsers(dest="command", required=True)
    auto = sub.add_parser("auto-annotate", help="Generate YOLO labels with a detector")
    auto.add_argument("images")
    auto.add_argument("labels")
    auto.add_argument("--detector", default="yolo11n.pt")
    auto.add_argument("--segmenter", default="sam2_b.pt")
    auto.add_argument("--conf", type=float, default=0.25)
    auto.add_argument("--task", choices=("detect", "segment"), default="detect")
    fit = sub.add_parser("train", help="Train a YOLO model")
    fit.add_argument("--model", default="yolo11n.pt")
    fit.add_argument("--data", default="data.yaml")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--imgsz", type=int, default=640)
    fit.add_argument("--batch", type=float, default=-1)
    fit.add_argument("--patience", type=int, default=30)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--project", default="runs/train")
    fit.add_argument("--name", default="experiment")
    convert = sub.add_parser("export", help="Export a trained model")
    convert.add_argument("--model", required=True)
    convert.add_argument("--format", choices=("onnx", "openvino", "engine", "ncnn"), default="onnx")
    convert.add_argument("--imgsz", type=int, default=640)
    convert.add_argument("--opset", type=int, default=12)
    convert.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.command == "auto-annotate":
        count = AutoAnnotator(args.detector, args.segmenter, args.conf).annotate_folder(args.images, args.labels, args.task)
        print(f"Generated {count} labels")
    elif args.command == "train":
        train(TrainingConfig(model=args.model, data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience, device=args.device, workers=args.workers, project=args.project, name=args.name))
    else:
        result = export_model(ExportConfig(model=args.model, format=args.format, imgsz=args.imgsz, opset=args.opset, device=args.device))
        print(result)


if __name__ == "__main__":
    main()
