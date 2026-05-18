#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2

from vio_frontend import PipelineConfig, VioFrontendPipeline, load_pipeline_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mp4_frontend.json"

BG = "#f5f7f8"
PANEL = "#ffffff"
TEXT = "#172026"
MUTED = "#5c6970"
LINE = "#d7dee2"
ACCENT = "#0f766e"
ACCENT_DARK = "#115e59"
DANGER = "#b42318"


class TkFrontendGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Underwater VIO Frontend Lab")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.configure(bg=BG)
        self.preview_image = None

        videos = sorted((PROJECT_ROOT / "video_src").glob("*.mp4"))
        default_video = str(videos[0]) if videos else ""

        self.input_var = tk.StringVar(value=default_video)
        self.output_var = tk.StringVar(value=str(PROJECT_ROOT / "outputs" / "tk_runs"))
        self.max_frames_var = tk.StringVar(value="200")
        self.geometry_model_var = tk.StringVar(value="auto")
        self.max_features_var = tk.StringVar(value="200")
        self.min_features_var = tk.StringVar(value="80")
        self.min_distance_var = tk.StringVar(value="30")
        self.quality_var = tk.StringVar(value="0.01")
        self.fb_threshold_var = tk.StringVar(value="0.8")
        self.ransac_threshold_var = tk.StringVar(value="0.75")
        self.assumed_focal_scale_var = tk.StringVar(value="1.2")
        self.motion_scale_var = tk.StringVar(value="0.05")
        self.pose_min_inliers_var = tk.StringVar(value="20")
        self.pose_ransac_threshold_var = tk.StringVar(value="1.0")
        self.clahe_var = tk.BooleanVar(value=True)
        self.draw_rejected_var = tk.BooleanVar(value=True)
        self.write_video_var = tk.BooleanVar(value=True)
        self.odometry_var = tk.BooleanVar(value=True)

        self.sweep_max_features_var = tk.StringVar(value="150,200,250")
        self.sweep_min_distance_var = tk.StringVar(value="30,40")
        self.sweep_ransac_var = tk.StringVar(value="0.5,0.75,1.0")

        self._build_layout(videos)
        self._set_status("Ready")

    def _build_layout(self, videos: list[Path]) -> None:
        header = tk.Frame(self.root, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            header,
            text="Underwater VIO Frontend Lab",
            bg=PANEL,
            fg=TEXT,
            font=("Helvetica", 18, "bold"),
        ).pack(side=tk.LEFT, padx=16, pady=12)
        tk.Label(
            header,
            text="Tkinter parameter runner",
            bg=PANEL,
            fg=MUTED,
            font=("Helvetica", 12),
        ).pack(side=tk.LEFT, padx=4, pady=12)

        body = tk.Frame(self.root, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)

        left_shell = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        left_shell.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        canvas = tk.Canvas(left_shell, width=410, bg=PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_shell, orient=tk.VERTICAL, command=canvas.yview)
        self.controls = tk.Frame(canvas, bg=PANEL)
        controls_window = canvas.create_window((0, 0), window=self.controls, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def on_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            canvas.itemconfigure(controls_window, width=event.width)

        self.controls.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))

        right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_controls(videos)
        self._build_result_panel(right)

    def _build_controls(self, videos: list[Path]) -> None:
        row = 0
        self._label(row, "Input MP4")
        row += 1
        self.video_menu = tk.OptionMenu(
            self.controls,
            self.input_var,
            *([str(path) for path in videos] or [""]),
        )
        self._style_option_menu(self.video_menu)
        self.video_menu.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12)
        self._button(row, 2, "Browse", self._browse_input, bg="#334155")
        row += 1

        self._label(row, "Output Dir")
        row += 1
        self._entry(row, self.output_var, col=0, span=2)
        self._button(row, 2, "Browse", self._browse_output, bg="#334155")
        row += 1

        row = self._entry_pair(row, "max_frames", self.max_frames_var, "geometry_model", None)
        self.geometry_menu = tk.OptionMenu(
            self.controls,
            self.geometry_model_var,
            "auto",
            "fundamental",
            "essential",
            "none",
        )
        self._style_option_menu(self.geometry_menu)
        self.geometry_menu.grid(row=row - 1, column=1, sticky="ew", padx=(6, 12), pady=(0, 8))

        row = self._entry_pair(row, "max_features", self.max_features_var, "min_features", self.min_features_var)
        row = self._entry_pair(row, "min_distance", self.min_distance_var, "quality", self.quality_var)
        row = self._entry_pair(
            row,
            "forward_backward_threshold",
            self.fb_threshold_var,
            "ransac_threshold",
            self.ransac_threshold_var,
        )
        row = self._entry_pair(
            row,
            "assumed_focal_scale",
            self.assumed_focal_scale_var,
            "motion_scale",
            self.motion_scale_var,
        )
        row = self._entry_pair(
            row,
            "pose_min_inliers",
            self.pose_min_inliers_var,
            "pose_ransac_threshold",
            self.pose_ransac_threshold_var,
        )

        checks = tk.Frame(self.controls, bg=PANEL)
        checks.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(4, 8))
        self._check(checks, "CLAHE", self.clahe_var, 0, 0)
        self._check(checks, "Draw rejected", self.draw_rejected_var, 0, 1)
        self._check(checks, "Write video", self.write_video_var, 1, 0)
        self._check(checks, "Odometry CSV", self.odometry_var, 1, 1)
        row += 1

        self.run_button = self._button(row, 0, "Run Selected", self._run_selected, span=3, bg=ACCENT)
        row += 1
        self.sweep_button = self._button(row, 0, "Run Sweep", self._run_sweep, span=3, bg="#334155")
        row += 1

        tk.Frame(self.controls, height=1, bg=LINE).grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=14)
        row += 1
        self._section_title(row, "Sweep lists, comma separated")
        row += 1
        row = self._entry_single(row, "sweep max_features", self.sweep_max_features_var)
        row = self._entry_single(row, "sweep min_distance", self.sweep_min_distance_var)
        row = self._entry_single(row, "sweep ransac_threshold", self.sweep_ransac_var)

        self.controls.grid_columnconfigure(0, weight=1)
        self.controls.grid_columnconfigure(1, weight=1)

    def _build_result_panel(self, parent: tk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        self.status_label = tk.Label(
            parent,
            text="Ready",
            bg="#ecfdf5",
            fg=ACCENT_DARK,
            anchor="w",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=8,
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

        self.preview_label = tk.Label(
            parent,
            text="Run을 누르면 preview가 표시됩니다.",
            bg="#101820",
            fg="#cbd5e1",
            anchor="center",
            font=("Helvetica", 14),
            width=80,
            height=22,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)

        self.result_text = tk.Text(
            parent,
            height=14,
            bg="#fbfcfc",
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            highlightbackground=LINE,
            highlightthickness=1,
            font=("Menlo", 12),
            wrap=tk.WORD,
        )
        self.result_text.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))

    def _label(self, row: int, text: str) -> None:
        tk.Label(
            self.controls,
            text=text,
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            font=("Helvetica", 11, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 3))

    def _section_title(self, row: int, text: str) -> None:
        tk.Label(
            self.controls,
            text=text,
            bg=PANEL,
            fg=TEXT,
            anchor="w",
            font=("Helvetica", 12, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=12)

    def _entry(self, row: int, variable: tk.StringVar, col: int = 0, span: int = 1) -> tk.Entry:
        entry = tk.Entry(
            self.controls,
            textvariable=variable,
            bg="#ffffff",
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.SOLID,
            highlightthickness=0,
            borderwidth=1,
        )
        entry.grid(row=row, column=col, columnspan=span, sticky="ew", padx=(12 if col == 0 else 6, 12), pady=(0, 8))
        return entry

    def _entry_pair(
        self,
        row: int,
        left_label: str,
        left_var: tk.StringVar,
        right_label: str,
        right_var: tk.StringVar | None,
    ) -> int:
        tk.Label(self.controls, text=left_label, bg=PANEL, fg=MUTED, anchor="w").grid(
            row=row,
            column=0,
            sticky="ew",
            padx=(12, 6),
            pady=(8, 3),
        )
        tk.Label(self.controls, text=right_label, bg=PANEL, fg=MUTED, anchor="w").grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(6, 12),
            pady=(8, 3),
        )
        row += 1
        self._entry(row, left_var, col=0)
        if right_var is not None:
            self._entry(row, right_var, col=1)
        return row + 1

    def _entry_single(self, row: int, label: str, variable: tk.StringVar) -> int:
        self._label(row, label)
        row += 1
        self._entry(row, variable, col=0, span=3)
        return row + 1

    def _button(
        self,
        row: int,
        col: int,
        text: str,
        command,
        span: int = 1,
        bg: str = ACCENT,
    ) -> tk.Button:
        button = tk.Button(
            self.controls,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=bg,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=8,
            font=("Helvetica", 11, "bold"),
        )
        button.grid(row=row, column=col, columnspan=span, sticky="ew", padx=12, pady=(4, 8))
        return button

    def _check(self, parent: tk.Frame, text: str, variable: tk.BooleanVar, row: int, col: int) -> None:
        tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL,
            activeforeground=TEXT,
            selectcolor=PANEL,
            anchor="w",
        ).grid(row=row, column=col, sticky="ew", pady=2)

    def _style_option_menu(self, menu: tk.OptionMenu) -> None:
        menu.configure(
            bg="#ffffff",
            fg=TEXT,
            activebackground="#ffffff",
            activeforeground=TEXT,
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=0,
        )
        menu["menu"].configure(bg="#ffffff", fg=TEXT)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(PROJECT_ROOT / "video_src"),
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*")],
        )
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(initialdir=str(PROJECT_ROOT / "outputs"))
        if path:
            self.output_var.set(path)

    def _run_selected(self) -> None:
        self._start_worker(self._selected_worker)

    def _run_sweep(self) -> None:
        self._start_worker(self._sweep_worker)

    def _start_worker(self, target) -> None:
        self.run_button.configure(state=tk.DISABLED)
        self.sweep_button.configure(state=tk.DISABLED)
        self._set_status("Running...")
        threading.Thread(target=target, daemon=True).start()

    def _selected_worker(self) -> None:
        try:
            output_dir = Path(self.output_var.get()) / time.strftime("run_%Y%m%d_%H%M%S")
            config = self._build_config(output_dir)
            summary = VioFrontendPipeline(config).run()
            preview = self._make_preview(summary)
            self.root.after(0, lambda: self._show_summary(summary, preview))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._show_error(exc))

    def _sweep_worker(self) -> None:
        try:
            output_root = Path(self.output_var.get()) / time.strftime("sweep_%Y%m%d_%H%M%S")
            output_root.mkdir(parents=True, exist_ok=True)
            rows = []
            for max_features, min_distance, ransac_threshold in itertools.product(
                [int(v) for v in _parse_list(self.sweep_max_features_var.get())],
                [int(v) for v in _parse_list(self.sweep_min_distance_var.get())],
                [float(v) for v in _parse_list(self.sweep_ransac_var.get())],
            ):
                run_dir = output_root / f"mf{max_features}_md{min_distance}_rt{ransac_threshold:g}"
                config = self._build_config(run_dir)
                config.features.max_features = max_features
                config.features.min_distance = min_distance
                config.geometry.ransac_threshold = ransac_threshold
                config.output.write_video = False
                summary = VioFrontendPipeline(config).run()
                rows.append(
                    {
                        "max_features": max_features,
                        "min_distance": min_distance,
                        "ransac_threshold": ransac_threshold,
                        "mean_active_tracks": summary["mean_active_tracks"],
                        "mean_geometry_inliers": summary["mean_geometry_inliers"],
                        "mean_geometry_inlier_ratio": summary["mean_geometry_inlier_ratio"],
                        "pose_success_ratio": summary["pose_success_ratio"],
                        "summary": summary["output_summary"],
                    }
                )
            result_path = output_root / "sweep_results.csv"
            with result_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self.root.after(0, lambda: self._show_sweep(result_path, rows))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._show_error(exc))

    def _build_config(self, output_dir: Path) -> PipelineConfig:
        config = load_pipeline_config(DEFAULT_CONFIG)
        config.dataset.input_path = Path(self.input_var.get())
        config.dataset.max_frames = int(self.max_frames_var.get())
        config.geometry.model = self.geometry_model_var.get()
        config.features.max_features = int(self.max_features_var.get())
        config.features.min_features = int(self.min_features_var.get())
        config.features.min_distance = int(self.min_distance_var.get())
        config.features.quality = float(self.quality_var.get())
        config.optical_flow.forward_backward_threshold = float(self.fb_threshold_var.get())
        config.geometry.ransac_threshold = float(self.ransac_threshold_var.get())
        config.odometry.assumed_focal_scale = float(self.assumed_focal_scale_var.get())
        config.odometry.motion_scale = float(self.motion_scale_var.get())
        config.odometry.min_pose_inliers = int(self.pose_min_inliers_var.get())
        config.odometry.pose_ransac_threshold = float(self.pose_ransac_threshold_var.get())
        config.preprocess.clahe = bool(self.clahe_var.get())
        config.output.draw_rejected = bool(self.draw_rejected_var.get())
        config.output.write_video = bool(self.write_video_var.get())
        config.odometry.enabled = bool(self.odometry_var.get())
        config.output.output_dir = output_dir
        return config

    def _make_preview(self, summary: dict[str, object]) -> Path | None:
        video_path = Path(str(summary.get("output_video", "")))
        if not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, 20)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        max_width = 820
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (max_width, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        preview_path = video_path.with_name(video_path.stem + "_tk_preview.ppm")
        cv2.imwrite(str(preview_path), frame)
        return preview_path

    def _show_summary(self, summary: dict[str, object], preview: Path | None) -> None:
        self._unlock()
        self._set_status("Done")
        if preview is not None:
            self.preview_image = tk.PhotoImage(file=str(preview))
            self.preview_label.configure(image=self.preview_image, text="", bg="#101820")
        else:
            self.preview_label.configure(
                image="",
                text="No preview. Write video가 꺼져 있습니다.",
                bg="#101820",
                fg="#cbd5e1",
            )
        lines = [
            f"frames_processed: {summary['frames_processed']}",
            f"mean_active_tracks: {summary['mean_active_tracks']:.2f}",
            f"mean_geometry_inliers: {summary['mean_geometry_inliers']:.2f}",
            f"mean_geometry_inlier_ratio: {summary['mean_geometry_inlier_ratio']:.3f}",
            f"pose_success_ratio: {summary['pose_success_ratio']:.3f}",
            f"trajectory_scale_mode: {summary['trajectory_scale_mode']}",
            "",
            f"tracked_video: {summary['output_video']}",
            f"tracks_csv: {summary['output_csv']}",
            f"odometry_csv: {summary['output_odometry_csv']}",
            f"summary_json: {summary['output_summary']}",
        ]
        self._set_result("\n".join(lines))

    def _show_sweep(self, result_path: Path, rows: list[dict[str, object]]) -> None:
        self._unlock()
        self._set_status("Sweep done")
        lines = [f"sweep_results: {result_path}", ""]
        for row in rows:
            lines.append(
                "mf={max_features} md={min_distance} rt={ransac_threshold} "
                "active={mean_active_tracks:.1f} inliers={mean_geometry_inliers:.1f} "
                "ratio={mean_geometry_inlier_ratio:.3f} pose={pose_success_ratio:.3f}".format(**row)
            )
        self._set_result("\n".join(lines))

    def _show_error(self, exc: Exception) -> None:
        self._unlock()
        self._set_status("Failed", error=True)
        messagebox.showerror("Run failed", str(exc))
        self._set_result(str(exc))

    def _unlock(self) -> None:
        self.run_button.configure(state=tk.NORMAL)
        self.sweep_button.configure(state=tk.NORMAL)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(
            text=text,
            bg="#fef3f2" if error else "#ecfdf5",
            fg=DANGER if error else ACCENT_DARK,
        )

    def _set_result(self, text: str) -> None:
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)


def _parse_list(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Sweep list cannot be empty")
    return values


def main() -> None:
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
    root = tk.Tk()
    TkFrontendGui(root)
    root.lift()
    root.update_idletasks()
    root.mainloop()


if __name__ == "__main__":
    main()
