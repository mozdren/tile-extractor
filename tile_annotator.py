#!/usr/bin/env python3
import argparse
import os
import json
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

PURPLE_BGR = (255, 0, 255)
GRID_COLOR = (255, 255, 255)
GRID_THICK = 1

# -------------- util --------------

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def to_gray_mask(mask_img: Optional[np.ndarray], shape_hw: Tuple[int,int]) -> np.ndarray:
    """Return uint8 mask (h,w) with values 0/255. If missing, all white; resize as needed."""
    h, w = shape_hw
    if mask_img is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if mask_img.ndim == 3:
        g = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        g = mask_img.copy()
    if g.shape != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
    _, g = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    return g

# -------------- DB --------------

class TileDB:
    def __init__(self, root: str):
        self.root = root
        ensure_dir(root)
        self.db_path = os.path.join(root, "tiles_db.json")
        data = load_json(self.db_path)
        if not data:
            raise SystemExit(f"Cannot find {self.db_path}")
        self.data = data
        self.tiles_dir = os.path.join(root, data.get("tiles_dir", "tiles"))
        self.tiles: List[dict] = data.get("tiles", [])
        # backfill keys we care about
        changed = False
        for t in self.tiles:
            if "name" not in t:
                t["name"] = ""
                changed = True
            if "category" not in t:
                t["category"] = ""
                changed = True
            if "mask" not in t or not t["mask"]:
                t["mask"] = os.path.join(data.get("tiles_dir", "tiles"), f"{t['id']}_mask.png")
                changed = True
        if changed:
            self.save()
        self.by_id: Dict[str, dict] = {t["id"]: t for t in self.tiles}

    def _path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def load_tile(self, t: dict) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(self._path(t["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read tile image: {t['file']}")
        h, w = img.shape[:2]
        mp = t.get("mask", "")
        mimg = cv2.imread(self._path(mp), cv2.IMREAD_UNCHANGED) if mp else None
        mask = to_gray_mask(mimg, (h, w))
        return img, mask

    def save(self):
        save_json(self.db_path, self.data)

# -------------- Annotator --------------

class Annotator:
    def __init__(self, db: TileDB, windowed: bool, scale: int, start_index: int, presets: List[str]):
        self.db = db
        self.tiles = db.tiles
        if not self.tiles:
            raise SystemExit("No tiles found in DB.")
        self.index = max(0, min(start_index, len(self.tiles)-1))

        # Window/init
        self.win = "Tile Annotator"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self.fullscreen = not windowed
        if self.fullscreen:
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.scale = max(2, int(scale))
        self.grid = True

        # Active tile data
        self.tile_img: Optional[np.ndarray] = None
        self.tile_mask: Optional[np.ndarray] = None
        self.h = self.w = self.S = 0

        # Canvas
        self.canvas: Optional[np.ndarray] = None
        self.left_roi = (0,0,0,0)

        # Editing state
        self.edit_field: Optional[str] = None  # None | "name" | "category" | "min_score"
        self.edit_buffer: str = ""
        self.cursor_visible = True
        self.cursor_tick = 0
        self.presets = presets
        self.preset_idx = 0

        self._load_tile(self.index)

    # ---------- load & layout ----------

    def _load_tile(self, idx: int):
        t = self.tiles[idx]
        img, mask = self.db.load_tile(t)
        self.tile_img = img
        self.tile_mask = mask
        self.h, self.w = img.shape[:2]
        self.S = max(self.h, self.w)
        self._build_canvas()

    def _compose_preview(self) -> np.ndarray:
        """Purple background + masked tile (SxS)."""
        out = np.zeros((self.S, self.S, 3), dtype=np.uint8)
        out[:] = PURPLE_BGR
        h, w = self.h, self.w
        m = (self.tile_mask[:h, :w] > 0)[:, :, None]
        out[:h, :w] = np.where(m, self.tile_img[:h, :w], out[:h, :w])
        return out

    def _build_canvas(self):
        pane = cv2.resize(self._compose_preview(), (self.S*self.scale, self.S*self.scale), interpolation=cv2.INTER_NEAREST)
        self.canvas = np.zeros_like(pane)
        self.canvas[:] = pane
        self.left_roi = (0, 0, pane.shape[1], pane.shape[0])
        if self.grid:
            self._draw_grid(self.canvas, self.left_roi, self.S, self.scale)

        # metadata panel text
        t = self.tiles[self.index]
        name = t.get("name", "")
        cat = t.get("category", "")
        msc = t.get("min_score", "")

        # Title line
        head = f"[{self.index+1}/{len(self.tiles)}] id={t['id']}  size={self.w}x{self.h}"
        self._put_text(head, y=18)

        # Fields
        def fmt_field(label, val, active):
            if active and self.edit_field:
                buf = self.edit_buffer
                # blinking caret
                caret = "_" if self.cursor_visible else " "
                return f"{label}: {buf}{caret}"
            else:
                return f"{label}: {val}"

        self._put_text(fmt_field("Name", name, self.edit_field=="name"), y=40)
        self._put_text(fmt_field("Category", cat, self.edit_field=="category"), y=62)
        self._put_text(fmt_field("min_score", msc, self.edit_field=="min_score"), y=84)

        # Footer / help
        help1 = "←/→ prev/next   N: name   C: category   M: min_score   G: grid   F: fullscreen   ?/H: help   Q: quit"
        help2 = "Editing: ENTER=save (auto)  ESC=cancel  (Category) TAB/Shift+TAB cycle presets"
        self._put_text(help1, y=self.canvas.shape[0]-24)
        self._put_text(help2, y=self.canvas.shape[0]-8)

        # cursor blink
        self.cursor_tick = (self.cursor_tick + 1) % 30
        self.cursor_visible = self.cursor_tick < 15

    def _draw_grid(self, img: np.ndarray, roi: Tuple[int,int,int,int], S: int, scale: int):
        x0, y0, w, h = roi
        for cx in range(S+1):
            px = x0 + cx*scale
            cv2.line(img, (px, y0), (px, y0 + h - 1), GRID_COLOR, GRID_THICK)
        for cy in range(S+1):
            py = y0 + cy*scale
            cv2.line(img, (x0, py), (x0 + w - 1, py), GRID_COLOR, GRID_THICK)

    def _put_text(self, text: str, y: int, x: int = 8, shadow=True):
        color_fg = (230, 230, 230)
        color_bg = (30, 30, 30)
        if shadow:
            cv2.putText(self.canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bg, 3, cv2.LINE_AA)
        cv2.putText(self.canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_fg, 1, cv2.LINE_AA)

    # ---------- navigation & edit ----------

    def _begin_edit(self, field: str):
        t = self.tiles[self.index]
        self.edit_field = field
        if field == "name":
            self.edit_buffer = t.get("name", "")
        elif field == "category":
            self.edit_buffer = t.get("category", "")
            # preset pointer to current value if present
            if self.presets:
                try:
                    self.preset_idx = self.presets.index(self.edit_buffer)
                except ValueError:
                    self.preset_idx = 0
        elif field == "min_score":
            val = t.get("min_score", "")
            self.edit_buffer = str(val) if val != "" else ""
        else:
            self.edit_buffer = ""

    def _commit_edit(self):
        t = self.tiles[self.index]
        changed = False
        if self.edit_field == "name":
            newv = self.edit_buffer.strip()
            if t.get("name","") != newv:
                t["name"] = newv
                changed = True
        elif self.edit_field == "category":
            newv = self.edit_buffer.strip()
            if t.get("category","") != newv:
                t["category"] = newv
                changed = True
        elif self.edit_field == "min_score":
            raw = self.edit_buffer.strip()
            if raw == "":
                if "min_score" in t:
                    del t["min_score"]
                    changed = True
            else:
                try:
                    newf = float(raw)
                    if t.get("min_score", None) != newf:
                        t["min_score"] = newf
                        changed = True
                except ValueError:
                    pass  # ignore invalid
        self.edit_field = None
        self.edit_buffer = ""
        if changed:
            self.db.save()
            print("[ok] Saved changes to DB.")

    def _cancel_edit(self):
        self.edit_field = None
        self.edit_buffer = ""

    def _cycle_preset(self, direction: int):
        if not self.presets or self.edit_field != "category":
            return
        self.preset_idx = (self.preset_idx + direction) % len(self.presets)
        self.edit_buffer = self.presets[self.preset_idx]

    def next_tile(self, delta: int):
        self.index = int(np.clip(self.index + delta, 0, len(self.tiles)-1))
        self._load_tile(self.index)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    # ---------- main loop ----------

    def loop(self):
        while True:
            self._build_canvas()
            cv2.imshow(self.win, self.canvas)

            k = cv2.waitKeyEx(33) & 0xFFFFFFFF
            if k == 0xFFFFFFFF:
                continue

            # Help (avoid 0x70 which collides with 'p' on Windows)
            if k in (ord('?'), ord('h'), ord('H'), 65470):
                self._print_help()
                continue

            # Quit
            if k in (ord('q'), 27) and self.edit_field is None:
                break

            # Fullscreen
            if k in (ord('f'), ord('F')) and self.edit_field is None:
                self.toggle_fullscreen(); continue

            # Grid
            if k in (ord('g'), ord('G')) and self.edit_field is None:
                self.grid = not self.grid; continue

            # (Optional) Manual Save still available, but not required
            if k in (ord('s'), ord('S')) and self.edit_field is None:
                self.db.save()
                print("[ok] DB saved.")
                continue

            # Navigation
            if self.edit_field is None:
                if k in (83, 2555904):   # Right
                    self.next_tile(+1); continue
                if k in (81, 2424832):   # Left
                    self.next_tile(-1); continue

            # Begin edits
            if self.edit_field is None:
                if k in (ord('n'), ord('N')):
                    self._begin_edit("name"); continue
                if k in (ord('c'), ord('C')):
                    self._begin_edit("category"); continue
                if k in (ord('m'), ord('M')):
                    self._begin_edit("min_score"); continue

            # While editing
            if self.edit_field is not None:
                # Confirm / cancel
                if k in (10, 13):          # Enter -> autosave commit
                    self._commit_edit(); continue
                if k in (27,):             # Esc
                    self._cancel_edit(); continue
                # Preset cycling (TAB / Shift+TAB)
                if k in (9,):              # Tab
                    self._cycle_preset(+1); continue
                if k == 353:               # Shift+Tab (common on some systems)
                    self._cycle_preset(-1); continue
                # Backspace / Delete
                if k in (8, 255, 3014656):  # Backspace on various platforms
                    self.edit_buffer = self.edit_buffer[:-1]; continue
                # Add character if printable
                ch = _key_to_char(k)
                if ch is not None:
                    # For min_score, restrict to valid float characters
                    if self.edit_field == "min_score":
                        if ch in "0123456789.-":
                            self.edit_buffer += ch
                    else:
                        self.edit_buffer += ch

        cv2.destroyAllWindows()

    def _print_help(self):
        print("""
Tile Annotator
--------------
- View each tile on purple background with mask applied.
- Edit metadata: name, category, and optional per-tile min_score.
- Changes are AUTOSAVED on ENTER (no need to press S).

Controls:
  ← / →        Previous / Next tile
  N            Edit name
  C            Edit category  (Tab / Shift+Tab to cycle presets if provided)
  M            Edit min_score (float; leave empty to remove per-tile override)
  G            Toggle grid
  F            Toggle fullscreen
  ENTER        Save current edit (autosave)
  ESC          Cancel current edit
  ? / H        Help
  Q            Quit
""")

# ---------- key mapping ----------

def _key_to_char(k: int) -> Optional[str]:
    """
    Convert cv2.waitKeyEx code to a printable ASCII char (basic).
    Handles A-Z, a-z, 0-9, space, underscore, dash, period, comma, slash, colon, semicolon.
    """
    # standard ASCII
    if 32 <= k <= 126:
        ch = chr(k)
        # allow typical filename-ish chars
        if ch.isalnum() or ch in " _-.,/:;+=!@#$%^&()[]{}'\"":
            return ch
        return None
    # numpad dot on some systems
    if k == 46:
        return "."
    return None

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser(description="Annotate tiles: name, category, and optional per-tile min_score. Fullscreen by default, autosaves on ENTER.")
    ap.add_argument("--db-dir", default="tile_db", help="DB directory containing tiles_db.json")
    ap.add_argument("--index", type=int, default=0, help="Start at tile index (0-based)")
    ap.add_argument("--windowed", action="store_true", help="Run windowed (default is fullscreen)")
    ap.add_argument("--scale", type=int, default=24, help="Integer pixel scale for the tile view")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated preset categories to cycle (e.g. 'bg,fg,ui,player,enemy')")
    args = ap.parse_args()

    presets = [s.strip() for s in args.categories.split(",") if s.strip()] if args.categories else []

    db = TileDB(args.db_dir)
    app = Annotator(db, windowed=args.windowed, scale=args.scale, start_index=args.index, presets=presets)
    app.loop()

if __name__ == "__main__":
    main()
