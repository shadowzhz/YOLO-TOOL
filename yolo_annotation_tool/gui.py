"""Small desktop shell around the recovered core workflow."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .sam import AutoAnnotator
from .training import TrainingConfig, train


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLO 标注工具")
        self.geometry("720x420")
        self._events: queue.Queue[str] = queue.Queue()
        self.images = tk.StringVar()
        self.labels = tk.StringVar()
        self.detector = tk.StringVar(value="yolo11n.pt")
        self.segmenter = tk.StringVar(value="sam2_b.pt")
        self.data = tk.StringVar(value="data.yaml")
        self._build()
        self.after(100, self._drain_events)

    def _build(self) -> None:
        panel = ttk.Frame(self, padding=16)
        panel.pack(fill="both", expand=True)
        fields = [("图片目录", self.images), ("标签目录", self.labels), ("检测模型", self.detector), ("SAM 模型", self.segmenter), ("数据集 YAML", self.data)]
        for row, (title, variable) in enumerate(fields):
            ttk.Label(panel, text=title).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(panel, textvariable=variable, width=72).grid(row=row, column=1, sticky="ew", pady=6)
            if title in {"图片目录", "标签目录", "数据集 YAML"}:
                ttk.Button(panel, text="浏览", command=lambda v=variable, t=title: self._browse(v, t)).grid(row=row, column=2, padx=(8, 0))
        panel.columnconfigure(1, weight=1)
        actions = ttk.Frame(panel)
        actions.grid(row=len(fields), column=0, columnspan=3, sticky="ew", pady=(18, 8))
        ttk.Button(actions, text="自动标注（矩形框）", command=lambda: self._run(self._annotate, "detect")).pack(side="left")
        ttk.Button(actions, text="自动标注（SAM 分割）", command=lambda: self._run(self._annotate, "segment")).pack(side="left", padx=8)
        ttk.Button(actions, text="开始训练", command=lambda: self._run(self._train)).pack(side="left")
        self.status = tk.StringVar(value="就绪")
        ttk.Label(panel, textvariable=self.status).grid(row=len(fields) + 1, column=0, columnspan=3, sticky="w", pady=12)

    def _browse(self, variable: tk.StringVar, title: str) -> None:
        if title == "数据集 YAML":
            selected = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        else:
            selected = filedialog.askdirectory()
        if selected:
            variable.set(selected)

    def _run(self, function, *args) -> None:
        threading.Thread(target=self._worker, args=(function, *args), daemon=True).start()

    def _worker(self, function, *args) -> None:
        try:
            result = function(*args)
            self._events.put(f"完成：{result}")
        except Exception as exc:
            self._events.put(f"错误：{exc}")

    def _annotate(self, task: str) -> int:
        if not self.images.get() or not self.labels.get():
            raise ValueError("请先选择图片目录和标签目录")
        return AutoAnnotator(self.detector.get(), self.segmenter.get()).annotate_folder(self.images.get(), self.labels.get(), task)

    def _train(self) -> str:
        if not self.data.get():
            raise ValueError("请先选择数据集 YAML")
        train(TrainingConfig(model=self.detector.get(), data=self.data.get()))
        return "训练已启动"

    def _drain_events(self) -> None:
        try:
            while True:
                message = self._events.get_nowait()
                self.status.set(message)
                if message.startswith("错误："):
                    messagebox.showerror("YOLO 标注工具", message[3:].strip())
        except queue.Empty:
            pass
        self.after(100, self._drain_events)


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
