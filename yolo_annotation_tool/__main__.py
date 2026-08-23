from __future__ import annotations

import argparse
import runpy

from .dataset import make_dataset_yaml, split_annotated_dataset, validate_dataset
from .sam import AutoAnnotator
from .export import ExportConfig, export_model
from .training import TrainingConfig, detect_training_device, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="yolo-annotation-tool")
    sub = parser.add_subparsers(dest="command", required=True)

    auto = sub.add_parser("auto-annotate", help="Generate YOLO labels with a detector/SAM")
    auto.add_argument("images")
    auto.add_argument("labels")
    auto.add_argument("--detector", default="yolo11n.pt")
    auto.add_argument("--segmenter", default="sam2_b.pt")
    auto.add_argument("--conf", type=float, default=0.25)
    auto.add_argument("--task", choices=("detect", "segment"), default="detect")
    auto.add_argument("--merge", choices=("replace", "skip", "append", "smart_dedup", "append_new"), default="replace")
    auto.add_argument("--merge-iou", type=float, default=0.45)
    auto.add_argument("--classes", type=int, nargs="*", default=None)

    validate = sub.add_parser("validate", help="Validate a YOLO dataset")
    validate.add_argument("root")
    validate.add_argument("--classes", type=int, default=None, help="Number of classes; omit to skip class-range validation")

    split = sub.add_parser("split", help="Split annotated images into train/val and create data.yaml")
    split.add_argument("root")
    split.add_argument("names", nargs="+", help="Class names in class-id order")
    split.add_argument("--val-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)

    yaml = sub.add_parser("make-yaml", help="Create an Ultralytics data.yaml")
    yaml.add_argument("root")
    yaml.add_argument("names", nargs="+")
    yaml.add_argument("--output", default=None)

    fit = sub.add_parser("train", help="Train a YOLO model")
    fit.add_argument("--model", default="yolo11n.pt")
    fit.add_argument("--data", default="data.yaml")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--imgsz", type=int, default=640)
    fit.add_argument("--batch", type=float, default=-1)
    fit.add_argument("--patience", type=int, default=30)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--device", default=None, help="cpu, mps, or CUDA device such as 0; defaults to automatic CUDA/CPU detection")
    fit.add_argument("--project", default="runs/train")
    fit.add_argument("--name", default="experiment")

    convert = sub.add_parser("export", help="Export a trained model")
    convert.add_argument("--model", required=True)
    convert.add_argument("--format", choices=("onnx", "openvino", "engine", "ncnn"), default="onnx")
    convert.add_argument("--imgsz", type=int, default=640)
    convert.add_argument("--opset", type=int, default=12)
    convert.add_argument("--device", default=None)

    sub.add_parser("gui", help="Launch the lightweight Tkinter desktop UI")
    sub.add_parser("qt", help="Launch the PySide6 desktop UI module")

    args = parser.parse_args()

    if args.command == "auto-annotate":
        annotator = AutoAnnotator(args.detector, args.segmenter, args.conf)
        count = annotator.annotate_folder(
            args.images,
            args.labels,
            task=args.task,
            merge=args.merge,
            merge_iou=args.merge_iou,
            class_ids=args.classes,
        )
        print(f"Generated {count} labels; stats={annotator.last_batch_stats}")
    elif args.command == "validate":
        report = validate_dataset(args.root, args.classes)
        print(f"images={report.images} labels={report.labels} missing={len(report.missing_labels)} malformed={len(report.malformed_labels)} ok={report.ok}")
        if report.missing_labels:
            print("Missing labels:")
            print("\n".join(report.missing_labels))
        if report.malformed_labels:
            print("Malformed labels:")
            print("\n".join(report.malformed_labels))
        if not report.ok:
            raise SystemExit(1)
    elif args.command == "split":
        report = split_annotated_dataset(args.root, args.names, val_fraction=args.val_fraction, seed=args.seed)
        print(f"source={report.source_images} train={report.train_images} val={report.val_images} skipped_unlabeled={len(report.skipped_unlabeled)} yaml={report.yaml_path}")
    elif args.command == "make-yaml":
        print(make_dataset_yaml(args.root, args.names, args.output))
    elif args.command == "train":
        device = args.device or detect_training_device()
        train(
            TrainingConfig(
                model=args.model,
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                patience=args.patience,
                device=device,
                workers=args.workers,
                project=args.project,
                name=args.name,
            )
        )
    elif args.command == "export":
        result = export_model(ExportConfig(model=args.model, format=args.format, imgsz=args.imgsz, opset=args.opset, device=args.device))
        print(result)
    elif args.command == "gui":
        from .gui import main as gui_main

        gui_main()
    else:
        # Running with runpy keeps the Qt module's own startup code intact and
        # avoids duplicating its large UI construction code in this CLI entrypoint.
        runpy.run_module("yolo_annotation_tool.qt_app", run_name="__main__")


if __name__ == "__main__":
    main()
