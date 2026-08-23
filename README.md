# YOLO-TOOL

YOLO 自动标注、SAM 分割、数据集整理、YOLO 训练与模型导出工具。

> 当前仓库的核心实现基于 Ultralytics；桌面端主要使用 PySide6，仓库同时保留了一个较轻量的 Tkinter GUI。

## 功能概览

- YOLO 自动检测标注：将检测框写成标准 YOLO `class x_center y_center width height` 标签。
- YOLO + SAM 自动分割：先由 YOLO 定位目标，再由 SAM 根据检测框生成 polygon 标签。
- 批量自动标注：递归处理图片目录，并支持 `replace`、`skip`、`append`、`smart_dedup`、`append_new` 等合并模式。
- 数据集检查：检查图片、缺失标签、标签格式及 class id。
- 数据集划分：把 `images/` 与 `labels/` 中已标注数据复制到 `train/` 和 `val/`，并生成 `data.yaml`。
- YOLO 训练：封装 Ultralytics `YOLO(...).train(...)`。
- 模型导出：支持 ONNX、OpenVINO、TensorRT Engine、NCNN。
- 桌面 GUI：PySide6 完整工作流，以及 Tkinter 简化界面。

## 运行环境

建议使用 Python 3.10+ 的独立虚拟环境。

### 安装基础依赖

```bash
python -m pip install -U pip
python -m pip install ultralytics PySide6
```

如果只使用 Tkinter 简化 GUI，一般不需要额外安装 Tkinter；某些 Linux 发行版需要单独安装系统包。

训练 GPU 是否可用取决于本机的 PyTorch/CUDA 安装。项目在训练时会检查 CUDA；没有可用 CUDA 时可以使用 CPU。

> 当前仓库没有提供 `requirements.txt` 或 `pyproject.toml`，因此依赖需要按上述方式手动安装。

## 一、启动程序

### PySide6 主界面

当前推荐从项目根目录执行：

```bash
python -m yolo_annotation_tool.qt_app
```

`qt_app.py` 是当前较完整的桌面实现，包含标注、数据集、训练、验证、导出和设置相关工作流。

### Tkinter 简化界面

```bash
python -m yolo_annotation_tool.gui
```

该界面主要提供：图片目录、标签目录、检测模型、SAM 模型、数据集 YAML，以及矩形框自动标注、SAM 分割自动标注和训练按钮。

### 命令行

模块入口还提供三个命令：

```bash
python -m yolo_annotation_tool auto-annotate ...
python -m yolo_annotation_tool train ...
python -m yolo_annotation_tool export ...
```

## 二、导入/准备图片

建议先创建如下项目结构：

```text
my_dataset/
├── images/
└── labels/
```

把待标注图片放入 `images/`。支持递归读取：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

标签放在 `labels/`，并与图片保持相同相对路径，仅把扩展名改为 `.txt`。

例如：

```text
my_dataset/
├── images/
│   ├── 001.jpg
│   └── subdir/002.jpg
└── labels/
    ├── 001.txt
    └── subdir/002.txt
```

## 三、自动标注

### 1. 矩形框检测

GUI 中选择：

- 图片目录：`my_dataset/images`
- 标签目录：`my_dataset/labels`
- 检测模型：例如 `yolo11n.pt`

然后执行“自动标注（矩形框）”。

CLI 等价操作：

```bash
python -m yolo_annotation_tool auto-annotate my_dataset/images my_dataset/labels
```

可指定模型和置信度：

```bash
python -m yolo_annotation_tool auto-annotate \
  my_dataset/images my_dataset/labels \
  --detector yolo11n.pt \
  --conf 0.25 \
  --task detect
```

生成的标签格式为：

```text
class_id x_center y_center width height
```

所有坐标均为 0~1 的归一化值。

### 2. YOLO + SAM 分割

GUI 中执行“自动标注（SAM 分割）”，或 CLI：

```bash
python -m yolo_annotation_tool auto-annotate \
  my_dataset/images my_dataset/labels \
  --detector yolo11n.pt \
  --segmenter sam2_b.pt \
  --task segment
```

工作过程是：

```text
图片
  ↓
YOLO 检测框
  ↓
把检测框作为 SAM prompt
  ↓
SAM mask
  ↓
polygon 归一化
  ↓
YOLO segmentation label
```

## 四、人工修正

### 当前代码状态：部分完成

核心数据结构已经支持：

- `Detection`：YOLO 矩形框
- `Polygon`：YOLO 分割 polygon
- `load_annotations()`：读取两种标签
- `save_annotations()`：写回两种标签

因此底层“标签读写”已经具备人工编辑所需的数据基础。

但需要注意：**当前仓库并没有在 README 所描述的意义上提供一个完整、独立、稳定的标注画布工作流**。`qt_app.py` 是主要桌面实现，但在继续使用前应以实际界面和当前代码为准检查具体编辑按钮是否已接通。

如果目标是生产级人工标注，建议后续重点确认并补齐：

1. 鼠标创建/拖拽矩形框。
2. polygon 节点编辑。
3. 删除/复制/修改 class。
4. 撤销/重做。
5. 自动保存与切图时保存确认。
6. 当前图片与 label 的一致性提示。

## 五、数据集划分

核心数据集工具提供 `split_annotated_dataset()`：

```text
my_dataset/
├── images/
├── labels/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

默认验证集比例为 20%，随机种子为 42。

只有存在对应 `.txt` 标签的图片才会进入 train/val；没有标签的图片会被跳过，而不会被误认为负样本。空标签文件则会被视为合法的负样本。

划分后会自动生成：

```text
my_dataset/data.yaml
```

内容类似：

```yaml
path: /absolute/path/to/my_dataset
train: train/images
val: val/images
nc: 2
names:
  0: cat
  1: dog
```

### 当前 CLI 状态

`__main__.py` 当前**没有直接暴露数据集划分命令**。数据集划分主要由 Qt 工作流调用，或者通过 Python API 调用：

```python
from yolo_annotation_tool.dataset import split_annotated_dataset

report = split_annotated_dataset(
    "my_dataset",
    ["cat", "dog"],
    val_fraction=0.2,
    seed=42,
)
print(report)
```

## 六、训练

### GUI

在 Qt 主界面中配置模型、数据集、epochs、image size、batch、device 等参数，然后开始训练。

训练核心最终调用 Ultralytics：

```python
YOLO(model).train(...)
```

### CLI

最简单：

```bash
python -m yolo_annotation_tool train
```

默认参数：

```text
model: yolo11n.pt
data: data.yaml
epochs: 100
imgsz: 640
batch: -1
patience: 30
workers: 0
device: cpu
project: runs/train
name: experiment
```

例如使用 GPU：

```bash
python -m yolo_annotation_tool train \
  --model yolo11n.pt \
  --data my_dataset/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch -1 \
  --device 0 \
  --project runs/train \
  --name my_model
```

如果指定 CUDA 设备但 PyTorch 没有检测到可用 CUDA，训练包装器会直接报错；此时改用 `--device cpu` 或正确安装 CUDA 版 PyTorch。

训练输出默认位于：

```text
runs/train/experiment/
```

其中通常会产生 Ultralytics 的训练结果及权重，例如 `weights/best.pt`。

## 七、模型导出

训练完成后，可以把 `best.pt` 导出为部署格式。

例如导出 ONNX：

```bash
python -m yolo_annotation_tool export \
  --model runs/train/experiment/weights/best.pt \
  --format onnx
```

支持：

```text
onnx
openvino
engine
ncnn
```

可指定图片尺寸：

```bash
python -m yolo_annotation_tool export \
  --model runs/train/experiment/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --opset 12
```

## 八、推荐的完整工作流

```text
1. 安装 Python + Ultralytics + PySide6
        ↓
2. python -m yolo_annotation_tool.qt_app
        ↓
3. 创建/选择数据集项目
        ↓
4. 将原始图片放入 images/
        ↓
5. 选择检测模型
        ↓
6. 自动标注（矩形框）
        │
        └── 如果需要分割 → YOLO + SAM
        ↓
7. 人工检查/修正标签
        ↓
8. 验证数据集
        ↓
9. 按比例划分 train / val
        ↓
10. 生成 data.yaml
        ↓
11. 开始 YOLO 训练
        ↓
12. 查看 runs/train/.../weights/best.pt
        ↓
13. 导出 ONNX / OpenVINO / Engine / NCNN
```

## 九、当前功能接通情况

| 功能 | 当前状态 | 说明 |
|---|---|---|
| YOLO 自动检测 | 已接通 | `AutoAnnotator.annotate_detection()` |
| YOLO + SAM 分割 | 已接通 | YOLO 框作为 SAM boxes |
| 批量自动标注 | 已接通 | 支持递归图片目录 |
| 自动标注进度/停止/暂停 | 已接通（API 层） | GUI 是否暴露全部控制取决于具体界面 |
| 标注 merge | 已接通（API 层） | replace/skip/append/smart_dedup/append_new |
| YOLO label 读写 | 已接通 | box + polygon |
| 数据集验证 | 已接通 | 图片、缺失标签、格式、class id |
| train/val 划分 | 已接通（API/Qt 工作流） | 默认 val 20%，seed 42 |
| data.yaml 生成 | 已接通 | 自动生成 Ultralytics YAML |
| YOLO 训练 | 已接通 | Ultralytics wrapper |
| CUDA 检查 | 已接通 | 无 CUDA 时可退回 CPU |
| 模型导出 | 已接通 | ONNX/OpenVINO/Engine/NCNN |
| Tkinter GUI | 已接通 | 简化功能 |
| PySide6 GUI | 已接通 | 当前主力桌面实现 |
| 完整人工标注画布 | 需进一步确认/完善 | 底层 annotation API 已存在，但不要假设所有编辑交互都已完整接通 |
| CLI 数据集划分 | 未提供 | 可使用 Python API 或 Qt 工作流 |
| CLI 验证命令 | 未提供 | `__main__.py` 当前没有 validate 子命令 |
| requirements.txt | 未提供 | 当前需手动安装依赖 |
| pyproject.toml | 未提供 | 当前不是标准 Python package 安装流程 |

## 十、常见问题

### 找不到 `ultralytics`

```bash
python -m pip install ultralytics
```

### 找不到 `PySide6`

```bash
python -m pip install PySide6
```

### GPU 不可用

检查 PyTorch 是否能看到 CUDA。项目训练包装器会拒绝在 CUDA 不可用时使用 CUDA device。

### 自动标注没有结果

优先检查：

1. 图片路径是否正确。
2. 模型是否成功加载。
3. `--conf` 是否过高。
4. 图片中的目标是否属于模型能够识别的类别。
5. SAM 分割时是否正确安装并加载 SAM 模型。

### 图片没有标签

数据集划分会主动跳过没有 `.txt` 的图片，避免把“尚未标注”错误当成“负样本”。

## 十一、开发者入口

核心 API：

```python
from yolo_annotation_tool import (
    AutoAnnotator,
    SamAnnotator,
    TrainingConfig,
    train,
    ExportConfig,
    export_model,
    validate_dataset,
    load_annotations,
    save_annotations,
)
```

主要模块：

- `annotations.py`：Detection / Polygon 与 YOLO label 序列化。
- `dataset.py`：数据集结构、验证、划分、YAML。
- `sam.py`：YOLO 自动标注与 SAM 分割。
- `training.py`：Ultralytics 训练封装。
- `export.py`：模型导出封装。
- `qt_app.py`：PySide6 桌面应用。
- `gui.py`：Tkinter 简化桌面应用。
- `settings.py`：应用设置。

## License

当前仓库未提供明确的 LICENSE 文件；在对外发布或商用前，请先确定项目许可证。