#!/usr/bin/env python3
"""
Retro Tools Main Menu
- Choose images folder
- Choose DB folder
- Launch tile_scribe.py (folder navigation, fullscreen default)
- Launch tile_editor.py (mask painting)
- Launch tile_annotator.py (name/category/min_score)
- Build atlas with compose_atlas.py (magenta background + JSON)

Place this file next to: tile_scribe.py, tile_editor.py, tile_annotator.py, compose_atlas.py
"""

import os
import sys
import json
import subprocess
import threading
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Retro Tools – Main Menu"
CFG_FILE = "retro_tools_gui_config.json"

# ---------- Helpers ----------

def script_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))

def find_script(name: str) -> str:
    p = os.path.join(script_dir(), name)
    return p if os.path.isfile(p) else name  # fallback to PATH

def safe_mkdir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def normpath_or_empty(p: str) -> str:
    p = (p or "").strip()
    return os.path.normpath(p) if p else ""

# ---------- Config Model ----------

@dataclass
class Config:
    images_dir: str = ""
    db_dir: str = "tile_db"

    # Common display / resize
    original_res: str = "320x200"     # optional; leave empty to disable
    windowed: bool = False            # Tile Scribe: fullscreen by default, windowed if true
    window_scale: float = 5.0         # Tile Scribe scale when not fullscreen

    # Tile Scribe matching
    match_method: str = "sqdiff"      # sqdiff | ccorr
    match_space: str = "gray"         # gray | bgr | hsv-hs
    edge_weight: float = 0.25
    score_thr: str = "0.04"           # string so empty is allowed
    iou_thr: float = 0.2
    max_matches_per_tile: int = 12
    suppress_overlap: bool = True
    alpha: float = 0.35
    zoom_factor: int = 8
    zoom_win: int = 240

    # Atlas options
    atlas_out_image: str = "atlas.png"
    atlas_out_json: str = "atlas.json"
    atlas_out_mask: str = ""          # empty = not generated
    atlas_padding: int = 2
    atlas_max_width: int = 2048
    atlas_sort: str = "height"        # height | width | area | none | category... (compose_atlas supports more)
    atlas_power_of_two: bool = True

    # Annotator options
    annotator_scale: int = 24
    annotator_categories: str = "bg,fg,ui,player,enemy,collectible"

    # Paths to scripts (auto found; you can override)
    tile_scribe_path: str = "tile_scribe.py"
    tile_editor_path: str = "tile_editor.py"
    tile_annotator_path: str = "tile_annotator.py"
    compose_atlas_path: str = "compose_atlas.py"

    def normalize(self):
        # normalize user-chosen paths
        self.images_dir = normpath_or_empty(self.images_dir)
        self.db_dir = normpath_or_empty(self.db_dir)
        self.atlas_out_image = self.atlas_out_image.strip()
        self.atlas_out_json = self.atlas_out_json.strip()
        self.atlas_out_mask = self.atlas_out_mask.strip()
        # resolve scripts relative to this file if present
        self.tile_scribe_path = find_script(self.tile_scribe_path)
        self.tile_editor_path = find_script(self.tile_editor_path)
        self.tile_annotator_path = find_script(self.tile_annotator_path)
        self.compose_atlas_path = find_script(self.compose_atlas_path)

def load_config() -> Config:
    try:
        with open(os.path.join(script_dir(), CFG_FILE), "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = Config(**data)
        cfg.normalize()
        return cfg
    except Exception:
        cfg = Config()
        cfg.normalize()
        return cfg

def save_config(cfg: Config):
    try:
        data = asdict(cfg)
        with open(os.path.join(script_dir(), CFG_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        messagebox.showwarning("Save config", f"Could not save config:\n{e}")

# ---------- GUI ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x720")
        self.minsize(860, 660)

        self.cfg = load_config()
        self._build_ui()
        self._load_values()

        # Save on close to persist any changes
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # UI layout
    def _build_ui(self):
        pad = 8

        # Top: Folders
        frm_paths = ttk.LabelFrame(self, text="Folders", padding=pad)
        frm_paths.pack(fill="x", padx=pad, pady=(pad, 0))

        self.var_images_dir = tk.StringVar()
        self.var_db_dir = tk.StringVar()

        row = 0
        ttk.Label(frm_paths, text="Images folder:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.var_images_dir, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm_paths, text="Browse…", command=self._choose_images_dir).grid(row=row, column=2, padx=2)
        row += 1
        ttk.Label(frm_paths, text="DB folder:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.var_db_dir, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm_paths, text="Browse…", command=self._choose_db_dir).grid(row=row, column=2, padx=2)
        frm_paths.columnconfigure(1, weight=1)

        # Middle: Tile Scribe options
        frm_ts = ttk.LabelFrame(self, text="Tile Scribe Options", padding=pad)
        frm_ts.pack(fill="x", padx=pad, pady=(pad, 0))

        self.var_original_res = tk.StringVar()
        self.var_windowed = tk.BooleanVar()
        self.var_window_scale = tk.StringVar()

        self.var_match_method = tk.StringVar()
        self.var_match_space = tk.StringVar()
        self.var_edge_weight = tk.StringVar()
        self.var_score_thr = tk.StringVar()
        self.var_iou_thr = tk.StringVar()
        self.var_max_matches = tk.StringVar()
        self.var_suppress = tk.BooleanVar()
        self.var_alpha = tk.StringVar()
        self.var_zoom_factor = tk.StringVar()
        self.var_zoom_win = tk.StringVar()

        # Row 0
        ttk.Label(frm_ts, text="Original res (e.g., 320x200):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_ts, textvariable=self.var_original_res, width=12).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Checkbutton(frm_ts, text="Windowed (opt-out fullscreen)", variable=self.var_windowed).grid(row=0, column=2, sticky="w", padx=10)
        ttk.Label(frm_ts, text="Window scale:").grid(row=0, column=3, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_window_scale, width=6).grid(row=0, column=4, sticky="w", padx=5)

        # Row 1
        ttk.Label(frm_ts, text="Match method:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(frm_ts, textvariable=self.var_match_method, width=8, values=["sqdiff", "ccorr"], state="readonly").grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(frm_ts, text="Space:").grid(row=1, column=2, sticky="e")
        ttk.Combobox(frm_ts, textvariable=self.var_match_space, width=8, values=["gray", "bgr", "hsv-hs"], state="readonly").grid(row=1, column=3, sticky="w", padx=5)
        ttk.Label(frm_ts, text="Edge weight:").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_edge_weight, width=6).grid(row=1, column=5, sticky="w", padx=5)

        # Row 2
        ttk.Label(frm_ts, text="Score thr:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_ts, textvariable=self.var_score_thr, width=8).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(frm_ts, text="IoU thr:").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_iou_thr, width=6).grid(row=2, column=3, sticky="w", padx=5)
        ttk.Label(frm_ts, text="Max matches/tile:").grid(row=2, column=4, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_max_matches, width=6).grid(row=2, column=5, sticky="w", padx=5)

        # Row 3
        ttk.Checkbutton(frm_ts, text="Suppress cross-tile overlaps", variable=self.var_suppress).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(frm_ts, text="Overlay alpha:").grid(row=3, column=2, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_alpha, width=6).grid(row=3, column=3, sticky="w", padx=5)
        ttk.Label(frm_ts, text="Zoom factor:").grid(row=3, column=4, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_zoom_factor, width=6).grid(row=3, column=5, sticky="w", padx=5)
        ttk.Label(frm_ts, text="Zoom win:").grid(row=3, column=6, sticky="e")
        ttk.Entry(frm_ts, textvariable=self.var_zoom_win, width=6).grid(row=3, column=7, sticky="w", padx=5)

        # Annotator options
        frm_ann = ttk.LabelFrame(self, text="Tile Annotator Options", padding=pad)
        frm_ann.pack(fill="x", padx=pad, pady=(pad, 0))

        self.var_annotator_scale = tk.StringVar()
        self.var_annotator_categories = tk.StringVar()

        ttk.Label(frm_ann, text="Scale (px per cell):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_ann, textvariable=self.var_annotator_scale, width=6).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(frm_ann, text="Category presets (comma-separated):").grid(row=0, column=2, sticky="e")
        ttk.Entry(frm_ann, textvariable=self.var_annotator_categories).grid(row=0, column=3, sticky="we", padx=5)
        frm_ann.columnconfigure(3, weight=1)

        # Atlas options
        frm_atlas = ttk.LabelFrame(self, text="Atlas Builder Options", padding=pad)
        frm_atlas.pack(fill="x", padx=pad, pady=(pad, 0))

        self.var_atlas_out_image = tk.StringVar()
        self.var_atlas_out_json = tk.StringVar()
        self.var_atlas_out_mask = tk.StringVar()
        self.var_atlas_padding = tk.StringVar()
        self.var_atlas_max_width = tk.StringVar()
        self.var_atlas_sort = tk.StringVar()
        self.var_atlas_power2 = tk.BooleanVar()

        ttk.Label(frm_atlas, text="Output image:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_atlas, textvariable=self.var_atlas_out_image, width=36).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Label(frm_atlas, text="Output JSON:").grid(row=0, column=2, sticky="e")
        ttk.Entry(frm_atlas, textvariable=self.var_atlas_out_json, width=24).grid(row=0, column=3, sticky="we", padx=5)
        ttk.Label(frm_atlas, text="Output mask (optional):").grid(row=0, column=4, sticky="e")
        ttk.Entry(frm_atlas, textvariable=self.var_atlas_out_mask, width=24).grid(row=0, column=5, sticky="we", padx=5)

        ttk.Label(frm_atlas, text="Padding:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_atlas, textvariable=self.var_atlas_padding, width=6).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(frm_atlas, text="Max width:").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm_atlas, textvariable=self.var_atlas_max_width, width=8).grid(row=1, column=3, sticky="w", padx=5)
        ttk.Label(frm_atlas, text="Sort:").grid(row=1, column=4, sticky="e")
        ttk.Combobox(frm_atlas, textvariable=self.var_atlas_sort, width=14,
                     values=["height","width","area","none","category","name","category-name","category-area","name-area"],
                     state="readonly").grid(row=1, column=5, sticky="w", padx=5)
        ttk.Checkbutton(frm_atlas, text="Power of two", variable=self.var_atlas_power2).grid(row=1, column=6, sticky="w")

        for col in (1,3,5):
            frm_atlas.columnconfigure(col, weight=1)

        # Bottom: Actions
        frm_actions = ttk.LabelFrame(self, text="Actions", padding=pad)
        frm_actions.pack(fill="x", padx=pad, pady=(pad, pad))

        ttk.Button(frm_actions, text="Launch Tile Scribe", command=self._run_tile_scribe).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(frm_actions, text="Launch Tile Editor (mask)", command=self._run_tile_editor).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(frm_actions, text="Launch Tile Annotator", command=self._run_tile_annotator).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Button(frm_actions, text="Build Atlas", command=self._run_compose_atlas).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        ttk.Button(frm_actions, text="Save Settings", command=self._save_ui_to_cfg).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(frm_actions, text="Load Settings", command=self._reload_config).grid(row=0, column=5, padx=5, pady=5)

        # Console/log
        self.txt_log = tk.Text(self, height=10)
        self.txt_log.pack(fill="both", expand=True, padx=pad, pady=(0, pad))
        self._log("Ready.")

    def _load_values(self):
        cfg = self.cfg
        # folders
        self.var_images_dir.set(cfg.images_dir)
        self.var_db_dir.set(cfg.db_dir)
        # tile scribe
        self.var_original_res.set(cfg.original_res)
        self.var_windowed.set(cfg.windowed)
        self.var_window_scale.set(str(cfg.window_scale))
        self.var_match_method.set(cfg.match_method)
        self.var_match_space.set(cfg.match_space)
        self.var_edge_weight.set(str(cfg.edge_weight))
        self.var_score_thr.set(cfg.score_thr)
        self.var_iou_thr.set(str(cfg.iou_thr))
        self.var_max_matches.set(str(cfg.max_matches_per_tile))
        self.var_suppress.set(cfg.suppress_overlap)
        self.var_alpha.set(str(cfg.alpha))
        self.var_zoom_factor.set(str(cfg.zoom_factor))
        self.var_zoom_win.set(str(cfg.zoom_win))
        # annotator
        self.var_annotator_scale.set(str(cfg.annotator_scale))
        self.var_annotator_categories.set(cfg.annotator_categories)
        # atlas
        self.var_atlas_out_image.set(cfg.atlas_out_image)
        self.var_atlas_out_json.set(cfg.atlas_out_json)
        self.var_atlas_out_mask.set(cfg.atlas_out_mask)
        self.var_atlas_padding.set(str(cfg.atlas_padding))
        self.var_atlas_max_width.set(str(cfg.atlas_max_width))
        self.var_atlas_sort.set(cfg.atlas_sort)
        self.var_atlas_power2.set(cfg.atlas_power_of_two)

    def _save_ui_to_cfg(self, also_persist=True):
        c = self.cfg
        c.images_dir = normpath_or_empty(self.var_images_dir.get())
        c.db_dir = normpath_or_empty(self.var_db_dir.get())
        c.original_res = self.var_original_res.get().strip()
        c.windowed = bool(self.var_windowed.get())
        try: c.window_scale = float(self.var_window_scale.get())
        except: c.window_scale = 5.0
        c.match_method = self.var_match_method.get()
        c.match_space = self.var_match_space.get()
        try: c.edge_weight = float(self.var_edge_weight.get())
        except: c.edge_weight = 0.25
        c.score_thr = self.var_score_thr.get().strip()
        try: c.iou_thr = float(self.var_iou_thr.get())
        except: c.iou_thr = 0.2
        try: c.max_matches_per_tile = int(self.var_max_matches.get())
        except: c.max_matches_per_tile = 12
        c.suppress_overlap = bool(self.var_suppress.get())
        try: c.alpha = float(self.var_alpha.get())
        except: c.alpha = 0.35
        try: c.zoom_factor = int(self.var_zoom_factor.get())
        except: c.zoom_factor = 8
        try: c.zoom_win = int(self.var_zoom_win.get())
        except: c.zoom_win = 240
        # annotator
        try: c.annotator_scale = int(self.var_annotator_scale.get())
        except: c.annotator_scale = 24
        c.annotator_categories = self.var_annotator_categories.get().strip()
        # atlas
        c.atlas_out_image = self.var_atlas_out_image.get().strip()
        c.atlas_out_json = self.var_atlas_out_json.get().strip()
        c.atlas_out_mask = self.var_atlas_out_mask.get().strip()
        try: c.atlas_padding = int(self.var_atlas_padding.get())
        except: c.atlas_padding = 2
        try: c.atlas_max_width = int(self.var_atlas_max_width.get())
        except: c.atlas_max_width = 2048
        c.atlas_sort = self.var_atlas_sort.get()
        c.atlas_power_of_two = bool(self.var_atlas_power2.get())

        c.normalize()
        if also_persist:
            save_config(c)
            self._log("Settings saved.")

    def _reload_config(self):
        self.cfg = load_config()
        self._load_values()
        self._log("Settings reloaded.")

    # --- File pickers ---
    def _choose_images_dir(self):
        d = filedialog.askdirectory(
            title="Choose Images Folder",
            mustexist=True,
            initialdir=self.var_images_dir.get() or os.getcwd()
        )
        if d:
            self.var_images_dir.set(normpath_or_empty(d))
            self._save_ui_to_cfg(also_persist=True)  # persist immediately

    def _choose_db_dir(self):
        d = filedialog.askdirectory(
            title="Choose DB Folder",
            mustexist=False,
            initialdir=self.var_db_dir.get() or os.getcwd()
        )
        if d:
            self.var_db_dir.set(normpath_or_empty(d))
            self._save_ui_to_cfg(also_persist=True)  # persist immediately

    # --- Actions ---
    def _run_tile_scribe(self):
        self._save_ui_to_cfg(also_persist=True)
        cfg = self.cfg

        if not cfg.images_dir or not os.path.isdir(cfg.images_dir):
            messagebox.showerror("Tile Scribe", "Please choose a valid Images folder.")
            return
        if not cfg.db_dir:
            messagebox.showerror("Tile Scribe", "Please choose a DB folder.")
            return
        safe_mkdir(cfg.db_dir)

        cmd = [sys.executable, cfg.tile_scribe_path, "--images-dir", cfg.images_dir, "--db-dir", cfg.db_dir]
        if cfg.windowed:
            cmd.append("--windowed")
        if cfg.original_res:
            cmd += ["--original-res", cfg.original_res]
        cmd += ["--window-scale", str(cfg.window_scale)]
        cmd += ["--match-method", cfg.match_method,
                "--match-space", cfg.match_space,
                "--edge-weight", str(cfg.edge_weight),
                "--iou-thr", str(cfg.iou_thr),
                "--max-matches-per-tile", str(cfg.max_matches_per_tile),
                "--alpha", str(cfg.alpha),
                "--zoom-factor", str(cfg.zoom_factor),
                "--zoom-win", str(cfg.zoom_win)]
        if cfg.suppress_overlap:
            cmd.append("--suppress-overlap")
        if cfg.score_thr:
            cmd += ["--score-thr", cfg.score_thr]

        self._spawn(cmd, "Tile Scribe")

    def _run_tile_editor(self):
        self._save_ui_to_cfg(also_persist=True)
        cfg = self.cfg
        if not cfg.db_dir:
            messagebox.showerror("Tile Editor", "Please choose a DB folder.")
            return
        safe_mkdir(cfg.db_dir)

        cmd = [sys.executable, cfg.tile_editor_path, "--db-dir", cfg.db_dir]
        self._spawn(cmd, "Tile Editor")

    def _run_tile_annotator(self):
        self._save_ui_to_cfg(also_persist=True)
        cfg = self.cfg
        if not cfg.db_dir:
            messagebox.showerror("Tile Annotator", "Please choose a DB folder.")
            return
        safe_mkdir(cfg.db_dir)

        cmd = [sys.executable, cfg.tile_annotator_path,
               "--db-dir", cfg.db_dir,
               "--scale", str(cfg.annotator_scale)]
        if cfg.annotator_categories:
            cmd += ["--categories", cfg.annotator_categories]
        self._spawn(cmd, "Tile Annotator")

    def _run_compose_atlas(self):
        self._save_ui_to_cfg(also_persist=True)
        cfg = self.cfg
        if not cfg.db_dir:
            messagebox.showerror("Build Atlas", "Please choose a DB folder.")
            return
        safe_mkdir(cfg.db_dir)

        def out_path(p):
            if not p:
                return ""
            return p if os.path.isabs(p) else os.path.join(cfg.db_dir, p)

        out_img = out_path(cfg.atlas_out_image)
        out_json = out_path(cfg.atlas_out_json)
        out_mask = out_path(cfg.atlas_out_mask) if cfg.atlas_out_mask else ""

        safe_mkdir(os.path.dirname(out_img) or ".")
        safe_mkdir(os.path.dirname(out_json) or ".")
        if out_mask:
            safe_mkdir(os.path.dirname(out_mask) or ".")

        cmd = [sys.executable, cfg.compose_atlas_path,
               "--db-dir", cfg.db_dir,
               "--out-image", out_img,
               "--out-json", out_json,
               "--padding", str(cfg.atlas_padding),
               "--max-width", str(cfg.atlas_max_width),
               "--sort", cfg.atlas_sort]
        if cfg.atlas_power_of_two:
            cmd.append("--power-of-two")
        if out_mask:
            cmd += ["--out-mask", out_mask]

        self._spawn(cmd, "Compose Atlas")

    # process launcher + live log capture
    def _spawn(self, cmd, title):
        """Spawn a child process and stream its stdout/stderr into the GUI log."""
        try:
            self._log(f"[{title}] Launching:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            proc = subprocess.Popen(
                cmd,
                cwd=script_dir(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self._log(f"[{title}] Started. (output below)")
            # Stream output on a background thread so UI stays responsive
            def _pump():
                try:
                    for line in iter(proc.stdout.readline, ''):
                        if not line:
                            break
                        self._log(f"[{title}] {line.rstrip()}")
                except Exception as e:
                    self._log(f"[{title}] [reader error] {e}")
                finally:
                    try:
                        rc = proc.wait()
                    except Exception:
                        rc = None
                    self._log(f"[{title}] Exit code: {rc}")
            threading.Thread(target=_pump, daemon=True).start()
        except Exception as e:
            messagebox.showerror(title, f"Failed to launch:\n{e}")

    def _log(self, text: str):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def _on_close(self):
        # Persist whatever is in the UI before closing
        self._save_ui_to_cfg(also_persist=True)
        self.destroy()

# ---------- Run ----------

if __name__ == "__main__":
    app = App()
    app.mainloop()
