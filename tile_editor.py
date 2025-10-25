#!/usr/bin/env python3
import argparse
import os
import json
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

PURPLE_BGR = (255, 0, 255)  # SDL colorkey preview
GRID_COLOR = (255, 255, 255)
GRID_THICK = 1

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

    def save_mask(self, t: dict, mask_u8: np.ndarray):
        """Save mask to t['mask'] path (ensure size matches t's w,h)."""
        h, w = int(t["h"]), int(t["w"])
        if mask_u8.shape != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        p = self._path(t["mask"])
        ensure_dir(os.path.dirname(p))
        cv2.imwrite(p, mask_u8)

# ----------------------------
# Editor
# ----------------------------

class Editor:
    def __init__(self, db: TileDB, windowed: bool, scale: int, start_index: int = 0):
        self.db = db
        self.tiles = db.tiles
        if not self.tiles:
            raise SystemExit("No tiles found in DB.")
        self.index = max(0, min(start_index, len(self.tiles)-1))

        self.win = "Tile Editor"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self.fullscreen = not windowed
        if self.fullscreen:
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # view scaling
        self.scale = max(2, int(scale))          # integer pixel scale on each pane
        self.grid = True
        self.overlay_left = True                 # show purple overlay on left pane (live mask)

        # brush & mode
        self.brush = 1                           # px brush size (square)
        self.mode_keep = False                   # False=ERASE (purple), True=KEEP (show)
        self.left_button_down = False
        self.right_button_down = False

        # current tile data
        self.tile_img: Optional[np.ndarray] = None   # (h,w,3)
        self.tile_mask: Optional[np.ndarray] = None  # (h,w) 0/255
        self.h = self.w = self.S = 0

        # square working canvases (display only)
        self.sq_img_left: Optional[np.ndarray] = None     # (S,S,3)
        self.sq_mask_work: Optional[np.ndarray] = None    # (S,S) for display edits; we crop to (h,w) on save

        # layout bookkeeping
        self.canvas: Optional[np.ndarray] = None
        self.left_roi = (0,0,0,0)     # x,y,w,h on full canvas for left pane
        self.right_roi = (0,0,0,0)    # for right pane

        # load first tile
        self._load_tile(self.index)

        # mouse
        cv2.setMouseCallback(self.win, self._on_mouse)

    # --------- loading & layout ---------

    def _load_tile(self, idx: int):
        t = self.tiles[idx]
        img, mask = self.db.load_tile(t)
        h, w = img.shape[:2]
        self.tile_img = img
        self.tile_mask = mask
        self.h, self.w = h, w
        self.S = max(h, w)  # square side for display

        # square left image
        self.sq_img_left = np.zeros((self.S, self.S, 3), dtype=np.uint8)
        self.sq_img_left[:h, :w] = img

        # square working mask
        self.sq_mask_work = np.zeros((self.S, self.S), dtype=np.uint8)
        self.sq_mask_work[:h, :w] = mask  # keep original mask in real area

        # (re)build layout canvas
        self._build_canvas()

    def _compose_preview(self) -> np.ndarray:
        """Return (S,S,3) purple background + masked tile (top-left aligned)."""
        out = np.zeros((self.S, self.S, 3), dtype=np.uint8)
        out[:] = PURPLE_BGR
        if self.tile_img is None or self.sq_mask_work is None:
            return out
        h, w = self.h, self.w
        roi = out[:h, :w]
        m = (self.sq_mask_work[:h, :w] > 0)[:, :, None]
        roi[:] = np.where(m, self.tile_img[:h, :w], roi)
        return out

    def _build_canvas(self):
        # left pane base (scaled)
        pane_left = cv2.resize(self.sq_img_left, (self.S*self.scale, self.S*self.scale), interpolation=cv2.INTER_NEAREST)

        # optional live purple overlay on left showing masked-out pixels
        if self.overlay_left and self.sq_mask_work is not None:
            mask_big = cv2.resize(self.sq_mask_work, (self.S*self.scale, self.S*self.scale), interpolation=cv2.INTER_NEAREST)
            purple = np.zeros_like(pane_left); purple[:] = PURPLE_BGR
            a = 0.35
            blended = (a*purple + (1.0-a)*pane_left).astype(np.uint8)
            pane_left = np.where(mask_big[:, :, None] == 0, blended, pane_left)

        # right preview
        prev = self._compose_preview()
        pane_right = cv2.resize(prev, (self.S*self.scale, self.S*self.scale), interpolation=cv2.INTER_NEAREST)

        # stack with margin
        margin = max(8, self.scale*2)
        H = pane_left.shape[0]
        W = pane_left.shape[1]*2 + margin
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        # left
        canvas[:, :pane_left.shape[1]] = pane_left
        # right
        canvas[:, pane_left.shape[1]+margin: pane_left.shape[1]+margin+pane_right.shape[1]] = pane_right

        self.canvas = canvas
        self.left_roi = (0, 0, pane_left.shape[1], pane_left.shape[0])
        self.right_roi = (pane_left.shape[1]+margin, 0, pane_right.shape[1], pane_right.shape[0])

        # grid on left
        if self.grid:
            self._draw_grid(self.canvas, self.left_roi, self.S, self.scale)

        # footer
        mode_text = "KEEP (show tile)" if self.mode_keep else "ERASE → purple"
        footer = (
            f"[{self.index+1}/{len(self.tiles)}] id={self.tiles[self.index]['id']}  "
            f"size={self.w}x{self.h}  view={self.S}x{self.S}  "
            f"brush={self.brush}px  MODE: {mode_text}  "
            "(ENTER=Save  ←/→=Prev/Next  M/SPACE=Toggle mode  L/R mouse=paint  "
            "Z/X=Brush±  G=Grid  O=Left overlay  R=All keep  0=All bg  F=Fullscreen  ?=Help)"
        )
        self._put_footer(footer)

    def _draw_grid(self, img: np.ndarray, roi: Tuple[int,int,int,int], S: int, scale: int):
        x0, y0, w, h = roi
        for cx in range(S+1):
            px = x0 + cx*scale
            cv2.line(img, (px, y0), (px, y0 + h - 1), GRID_COLOR, GRID_THICK)
        for cy in range(S+1):
            py = y0 + cy*scale
            cv2.line(img, (x0, py), (x0 + w - 1, py), GRID_COLOR, GRID_THICK)

    def _put_footer(self, text: str):
        if self.canvas is None:
            return
        y = self.canvas.shape[0] - 8
        cv2.putText(self.canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 3, cv2.LINE_AA)
        cv2.putText(self.canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

    # --------- mouse & painting ---------

    def _pane_pixel_at(self, x: int, y: int, roi: Tuple[int,int,int,int]) -> Optional[Tuple[int,int]]:
        rx, ry, rw, rh = roi
        if not (rx <= x < rx+rw and ry <= y < ry+rh):
            return None
        px = (x - rx) // self.scale
        py = (y - ry) // self.scale
        if 0 <= px < self.S and 0 <= py < self.S:
            return (px, py)
        return None

    def _paint_at(self, px: int, py: int, keep: bool):
        """Paint a square brush on the working square mask; clamp to real tile area for edits."""
        x0 = max(0, px - (self.brush//2))
        y0 = max(0, py - (self.brush//2))
        x1 = min(self.w, x0 + self.brush)
        y1 = min(self.h, y0 + self.brush)
        if x1 <= x0 or y1 <= y0:
            return
        self.sq_mask_work[y0:y1, x0:x1] = 255 if keep else 0

    def _on_mouse(self, event, x, y, flags, userdata):
        # Paint only in LEFT pane
        p = self._pane_pixel_at(x, y, self.left_roi)
        if p is None:
            if event == cv2.EVENT_LBUTTONUP:
                self.left_button_down = False
            if event == cv2.EVENT_RBUTTONUP:
                self.right_button_down = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_down = True
            # left paints current mode
            self._paint_at(p[0], p[1], keep=self.mode_keep)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_button_down = True
            # right paints opposite of current mode
            self._paint_at(p[0], p[1], keep=not self.mode_keep)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.left_button_down:
                self._paint_at(p[0], p[1], keep=self.mode_keep)
            elif self.right_button_down:
                self._paint_at(p[0], p[1], keep=not self.mode_keep)
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_button_down = False
        elif event == cv2.EVENT_RBUTTONUP:
            self.right_button_down = False

    # --------- actions ---------

    def save_mask(self):
        if self.sq_mask_work is None:
            return
        cropped = self.sq_mask_work[:self.h, :self.w]
        self.db.save_mask(self.tiles[self.index], cropped)
        print(f"[ok] Saved mask -> {self.tiles[self.index]['mask']}")

    def next_tile(self, delta: int):
        self.index = int(np.clip(self.index + delta, 0, len(self.tiles)-1))
        self._load_tile(self.index)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    def loop(self):
        while True:
            self._build_canvas()                 # rebuild every frame for live preview/overlay
            cv2.imshow(self.win, self.canvas)

            k = cv2.waitKeyEx(16) & 0xFFFFFFFF
            if k == 0xFFFFFFFF:
                continue

            if k in (27, ord('q')):              # ESC / q
                break
            if k in (10, 13):                    # ENTER: save
                self.save_mask()
            if k in (83, 2555904):               # Right arrow
                self.next_tile(+1)
            if k in (81, 2424832):               # Left arrow
                self.next_tile(-1)
            if k in (ord('f'), ord('F')):        # Fullscreen toggle
                self.toggle_fullscreen()
            if k in (ord('g'), ord('G')):        # Grid toggle
                self.grid = not self.grid
            if k in (ord('o'), ord('O')):        # Left overlay toggle
                self.overlay_left = not self.overlay_left
            if k in (ord('z'), ord('Z')):        # brush +
                self.brush = min(32, self.brush + 1)
            if k in (ord('x'), ord('X')):        # brush -
                self.brush = max(1, self.brush - 1)
            if k in (ord('r'), ord('R')):        # reset to all white (keep)
                if self.sq_mask_work is not None:
                    self.sq_mask_work[:self.h, :self.w] = 255
            if k == ord('0'):                    # set all to background (erase)
                if self.sq_mask_work is not None:
                    self.sq_mask_work[:self.h, :self.w] = 0
            if k in (ord('m'), ord('M'), ord(' ')):  # toggle paint mode
                self.mode_keep = not self.mode_keep
            # Help
            if k in (ord('?'), 0x70, 65470):  # '?' or F1
                self._show_help()

        cv2.destroyAllWindows()

    def _show_help(self):
        print("""
Tile Editor (Square View)
-------------------------
- ALWAYS square in the editor (display-only). Saved mask stays at the original WxH.
- Left pane: source tile with optional purple overlay showing masked-out pixels.
- Right pane: purple preview using the mask (what SDL will blit).
- Mouse:
    Left-click / drag  -> paint CURRENT MODE (default = ERASE → purple)
    Right-click / drag -> paint OPPOSITE of current mode
- Keyboard:
    ENTER     -> Save mask (cropped to original size)
    ← / →     -> Previous / Next tile
    M / SPACE -> Toggle paint mode (ERASE ↔ KEEP)
    Z / X     -> Brush size + / -
    G         -> Toggle grid on left
    O         -> Toggle purple overlay on left
    R         -> Reset mask to all KEEP (white)
    0         -> Set mask to all background (erase = purple)
    F         -> Toggle fullscreen
    ? or F1   -> This help
    ESC / q   -> Quit
""")

# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Manual mask/tile editor (ALWAYS square view; saves original-size masks).")
    ap.add_argument("--db-dir", default="tile_db", help="DB directory containing tiles_db.json")
    ap.add_argument("--index", type=int, default=0, help="Start at tile index (0-based).")
    ap.add_argument("--windowed", action="store_true", help="Run windowed (default is fullscreen).")
    ap.add_argument("--scale", type=int, default=24, help="Integer pixel scale per pane (visual).")
    args = ap.parse_args()

    db = TileDB(args.db_dir)
    ed = Editor(db, windowed=args.windowed, scale=args.scale, start_index=args.index)
    ed.loop()

if __name__ == "__main__":
    main()
