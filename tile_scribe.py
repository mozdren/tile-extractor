#!/usr/bin/env python3
import argparse
import cv2
import json
import os
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional

# ---------------------------
# Utilities / I/O
# ---------------------------

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p)

def load_json(p: str) -> Optional[dict]:
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: str, data: dict):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def tile_md5(bgr: np.ndarray) -> str:
    return hashlib.md5(bgr.tobytes()).hexdigest()

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def parse_size(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().replace("x", " ").split()
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Resolution must be like 320x200 or 256x240.")

def to_gray_mask(mask_img: Optional[np.ndarray], shape_hw: Tuple[int,int]) -> np.ndarray:
    """Return uint8 mask (h,w) with values 0/255. If missing, return all white; resize as needed."""
    h, w = shape_hw
    if mask_img is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if mask_img.ndim == 3:
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_img.copy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask

def list_images_in_dir(d: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for name in sorted(os.listdir(d)):
        lower = name.lower()
        if os.path.splitext(lower)[1] in exts:
            files.append(os.path.join(d, name))
    return files

# ---------------------------
# Match space / edges
# ---------------------------

def to_match_space(img_bgr: np.ndarray, space: str) -> np.ndarray:
    if space == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if space == "hsv-hs":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        return cv2.merge([h, s])  # 2-channel
    return img_bgr  # "bgr"

def edge_map(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if img_bgr_or_gray.ndim == 3:
        g = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        g = img_bgr_or_gray
    return cv2.Canny(g, 40, 120)

# ---------------------------
# NMS helpers
# ---------------------------

def iou(a, b):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter + 1e-9
    return inter/union

def nms_rects(rects, scores, iou_thr, keep_max=None):
    idxs = np.argsort(-np.array(scores))  # sort high->low
    keep = []
    for i in idxs:
        ri = rects[i]
        if any(iou(ri, rects[j]) >= iou_thr for j in keep):
            continue
        keep.append(i)
        if keep_max and len(keep) >= keep_max:
            break
    return keep

# ---------------------------
# DB schema (with mask, name, category)
# ---------------------------

class TileDB:
    def __init__(self, root: str):
        self.root = root
        ensure_dir(root)
        self.db_path = os.path.join(root, "tiles_db.json")
        self.tiles_dir = os.path.join(root, "tiles")
        ensure_dir(self.tiles_dir)
        self.data = load_json(self.db_path) or {"version": 2, "tiles_dir": "tiles", "tiles": []}
        if "version" not in self.data:
            self.data["version"] = 2
        if "tiles_dir" not in self.data:
            self.data["tiles_dir"] = "tiles"
        if "tiles" not in self.data:
            self.data["tiles"] = []
        # backfill mask/name/category
        changed = False
        for t in self.data["tiles"]:
            if "mask" not in t or not t["mask"]:
                t["mask"] = os.path.join(self.data["tiles_dir"], f"{t['id']}_mask.png")
                abs_mask = self._path(t["mask"])
                if not os.path.exists(abs_mask):
                    self._write_default_mask_for_tile(t)
                changed = True
            if "name" not in t:
                t["name"] = ""
                changed = True
            if "category" not in t:
                t["category"] = ""
                changed = True
        if changed:
            self.save()
        self._reindex()

    def _reindex(self):
        self.by_hash: Dict[Tuple[int,int,str], str] = {}
        self.by_id: Dict[str, dict] = {}
        for t in self.data["tiles"]:
            self.by_id[t["id"]] = t
            self.by_hash[(t["w"], t["h"], t["hash"])] = t["id"]

    def _next_id(self) -> str:
        n = len(self.data["tiles"]) + 1
        return f"t{n:05d}"

    def _path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def _default_mask_path_for_id(self, tid: str) -> str:
        return os.path.join(self.data["tiles_dir"], f"{tid}_mask.png")

    def _write_default_mask_for_tile(self, entry: dict):
        h, w = int(entry["h"]), int(entry["w"])
        mask = np.full((h, w), 255, dtype=np.uint8)
        abs_mask = self._path(entry["mask"])
        ensure_dir(os.path.dirname(abs_mask))
        cv2.imwrite(abs_mask, mask)

    def add_tile(self, tile_img: np.ndarray, label: Optional[str] = None) -> dict:
        h, w = tile_img.shape[:2]
        hsh = tile_md5(tile_img)
        key = (w, h, hsh)
        if key in self.by_hash:
            return self.by_id[self.by_hash[key]]

        tid = self._next_id()
        rel_file = os.path.join(self.data["tiles_dir"], f"{tid}.png")
        rel_mask = self._default_mask_path_for_id(tid)
        abs_file = self._path(rel_file)
        abs_mask = self._path(rel_mask)
        ensure_dir(os.path.dirname(abs_file))
        cv2.imwrite(abs_file, tile_img)
        ensure_dir(os.path.dirname(abs_mask))
        cv2.imwrite(abs_mask, np.full((h, w), 255, dtype=np.uint8))  # default all white

        entry = {
            "id": tid, "w": w, "h": h, "hash": hsh,
            "file": rel_file,
            "mask": rel_mask,
            "name": label or "",
            "category": ""
        }
        self.data["tiles"].append(entry)
        self._reindex()
        self.save()
        return entry

    def delete_tile(self, tile_id: str) -> bool:
        idx = None
        for i, t in enumerate(self.data["tiles"]):
            if t["id"] == tile_id:
                idx = i
                break
        if idx is None:
            return False
        t = self.data["tiles"].pop(idx)
        for key in ("file", "mask"):
            p = self._path(t.get(key, ""))
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self._reindex()
        self.save()
        return True

    def save(self):
        save_json(self.db_path, self.data)

    @property
    def tiles(self) -> List[dict]:
        return self.data["tiles"]

    def load_tile_image(self, t: dict) -> Optional[np.ndarray]:
        return cv2.imread(self._path(t["file"]), cv2.IMREAD_COLOR)

    def load_tile_mask(self, t: dict) -> np.ndarray:
        mp = self._path(t.get("mask", ""))
        mask_img = cv2.imread(mp, cv2.IMREAD_UNCHANGED) if mp and os.path.exists(mp) else None
        return to_gray_mask(mask_img, (int(t["h"]), int(t["w"])))

# ---------------------------
# Interactive session
# ---------------------------

class RectEditor:
    def __init__(self, img: np.ndarray):
        self.set_base(img)
        self.rect = [self.w//4, self.h//4, self.w//2, self.h//2]  # x,y,w,h
        self.dragging = False
        self.drag_anchor = (0, 0)
        self.mode = "idle"
        self.overlay_matches = True  # default coverage ON
        self.matches_cache: Optional[List[Tuple[int,int,int,int,str,float]]] = None
        self.coverage_mask: Optional[np.ndarray] = None  # union of placed tile masks
        self.alpha = 0.35
        self.mouse_pos = (self.w//2, self.h//2)

        # Zoom
        self.zoom_enabled = False
        self.zoom_factor = 8
        self.zoom_win = 240
        self.zoom_grid = True

        # Help overlay
        self.show_help = False

        # Delete-on-RMB in coverage
        self.delete_request_tid: Optional[str] = None

        # Drawing state
        self.drawing_new = False
        self.start_pt = None

    def set_base(self, img: np.ndarray):
        self.base = img
        self.h, self.w = img.shape[:2]
        self.mouse_pos = (self.w//2, self.h//2)

    def set_rect(self, x, y, w, h):
        x = clamp(x, 0, self.w-1)
        y = clamp(y, 0, self.h-1)
        w = clamp(w, 1, self.w - x)
        h = clamp(h, 1, self.h - y)
        self.rect = [x, y, w, h]

    def clamp_rect_to_image(self):
        x,y,w,h = self.rect
        x = clamp(x, 0, self.w-1)
        y = clamp(y, 0, self.h-1)
        w = clamp(w, 1, self.w - x)
        h = clamp(h, 1, self.h - y)
        self.rect = [x,y,w,h]

    def _match_at_point(self, px: int, py: int) -> Optional[str]:
        if not self.matches_cache:
            return None
        for rx, ry, rw, rh, tid, score in sorted(self.matches_cache, key=lambda r: -r[5]):
            if rx <= px < rx + rw and ry <= py < ry + rh:
                return tid
        return None

    def mouse_cb_image_coords(self, event, mx_img, my_img, flags, userdata):
        self.mouse_pos = (clamp(mx_img, 0, self.w-1), clamp(my_img, 0, self.h-1))

        if self.overlay_matches and event == cv2.EVENT_RBUTTONDOWN:
            tid = self._match_at_point(mx_img, my_img)
            if tid:
                self.delete_request_tid = tid
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_new = True
            self.start_pt = (mx_img, my_img)
            self.set_rect(mx_img, my_img, 1, 1)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_new:
            x0, y0 = self.start_pt
            x1, y1 = mx_img, my_img
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0) + 1
            h = abs(y1 - y0) + 1
            self.set_rect(x, y, w, h)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing_new:
            self.drawing_new = False

        elif event == cv2.EVENT_RBUTTONDOWN and not self.overlay_matches:
            self.mode = "move"
            self.dragging = True
            self.drag_anchor = (mx_img, my_img)
            self._rect_start = list(self.rect)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.mode == "move":
            dx = mx_img - self.drag_anchor[0]
            dy = my_img - self.drag_anchor[1]
            x0, y0, w0, h0 = self._rect_start
            self.set_rect(x0 + dx, y0 + dy, w0, h0)

        elif event == cv2.EVENT_RBUTTONUP and self.dragging:
            self.dragging = False
            self.mode = "idle"

    def nudge(self, dx=0, dy=0, dw=0, dh=0):
        x,y,w,h = self.rect
        self.set_rect(x+dx, y+dy, w+dw, h+dh)

    def current_tile(self) -> np.ndarray:
        x,y,w,h = self.rect
        return self.base[y:y+h, x:x+w].copy()

    def draw_ui(self, canvas: np.ndarray, show_matches: bool,
                matches: Optional[List[Tuple[int,int,int,int,str,float]]],
                footer: str = ""):
        disp = canvas.copy()
        x,y,w,h = self.rect
        cv2.rectangle(disp, (x,y), (x+w-1,y+h-1), (0,255,255), 1)

        # --- Masked coverage overlay (purple) ---
        if show_matches:
            if self.coverage_mask is not None:
                # purple overlay where mask>0
                purple = np.zeros_like(disp); purple[:] = (255,0,255)
                a = float(self.alpha)
                blended = (a*purple + (1.0-a)*disp).astype(np.uint8)
                m = (self.coverage_mask > 0)
                if m.any():
                    disp = np.where(m[:, :, None], blended, disp)
            elif matches:
                # fallback: solid purple rects (when mask not available)
                overlay = disp.copy()
                for rx, ry, rw, rh, tid, score in matches:
                    cv2.rectangle(overlay, (rx,ry), (rx+rw-1, ry+rh-1), (255,0,255), -1)
                cv2.addWeighted(overlay, self.alpha, disp, 1.0 - self.alpha, 0, disp)

        if self.show_help:
            hud = [
                "Left-drag: draw   Right-drag: move   WASD/X: move rect   [/]: w -/+   {/}: h -/+   ENTER: add tile",
                "Coverage ON by default (purple = known)   C: toggle   S: save DB   F: fullscreen   Z: zoom (,/.)",
                "Coverage mode: Right-click a purple area = DELETE that tile from DB",
                "Images: Left/Right = previous/next frame   F1/H/?: toggle help",
                "Matching: --match-method [sqdiff|ccorr]  --match-space [gray|bgr|hsv-hs]  --edge-weight 0..1",
                "--score-thr (sqdiff lower better; ccorr higher better), --iou-thr, --max-matches-per-tile, --suppress-overlap",
            ]
            y0 = 18
            for line in hud:
                cv2.putText(disp, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 3, cv2.LINE_AA)
                cv2.putText(disp, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)
                y0 += 18

        if footer:
            cv2.putText(disp, footer, (8, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

        return disp

    def render_zoom(self):
        if not self.zoom_enabled:
            return
        mx, my = self.mouse_pos
        patch = max(4, self.zoom_win // max(1, self.zoom_factor))
        half = patch // 2
        x0 = clamp(mx - half, 0, self.w - patch)
        y0 = clamp(my - half, 0, self.h - patch)
        roi = self.base[y0:y0+patch, x0:x0+patch]
        zoom = cv2.resize(roi, (self.zoom_win, self.zoom_win), interpolation=cv2.INTER_NEAREST)
        if self.zoom_grid:
            mid = self.zoom_win // 2
            cv2.line(zoom, (mid, 0), (mid, self.zoom_win-1), (50, 50, 50), 1)
            cv2.line(zoom, (0, mid), (self.zoom_win-1, mid), (50, 50, 50), 1)
            cv2.rectangle(zoom, (0,0), (self.zoom_win-1, self.zoom_win-1), (120,120,120), 1)
        cv2.imshow("Zoom", zoom)

# ---------------------------
# Matching core (strict + tunable)
# ---------------------------

def match_tile_all_positions(
    image_bgr: np.ndarray,
    tile_bgr: np.ndarray,
    mask_u8: Optional[np.ndarray],
    method: str,
    space: str,
    edge_weight: float,
    score_thr: Optional[float],
    iou_thr: float,
    keep_max: int
):
    """
    Returns list of (x,y,w,h,score) after thresholding and IoU-NMS.
    For 'sqdiff': lower is better; For 'ccorr': higher is better.
    With edge_weight>0, we fuse edge CCORR into a similarity (higher better).
    """
    if score_thr is None:
        score_thr = 0.045 if method == "sqdiff" else 0.93

    imgM = to_match_space(image_bgr, space)
    tplM = to_match_space(tile_bgr, space)
    use_mask = mask_u8 is not None and mask_u8.shape[:2] == tplM.shape[:2]

    if method == "sqdiff":
        m = cv2.TM_SQDIFF_NORMED
        try:
            res_main = cv2.matchTemplate(imgM, tplM, m, mask=mask_u8 if use_mask else None)
        except TypeError:
            res_main = cv2.matchTemplate(imgM, tplM, m)
        good_y, good_x = np.where(res_main <= score_thr)
        invert_for_sort = True   # lower is better
        primary_scores = res_main
    else:
        m = cv2.TM_CCORR_NORMED
        try:
            res_main = cv2.matchTemplate(imgM, tplM, m, mask=mask_u8 if use_mask else None)
        except TypeError:
            res_main = cv2.matchTemplate(imgM, tplM, m)
        good_y, good_x = np.where(res_main >= score_thr)
        invert_for_sort = False  # higher is better
        primary_scores = res_main

    th, tw = tile_bgr.shape[:2]
    rects = []
    vals = []

    if edge_weight > 0:
        imgE = edge_map(image_bgr)
        tplE = edge_map(tile_bgr)
        res_edge = cv2.matchTemplate(imgE, tplE, cv2.TM_CCORR_NORMED, mask=mask_u8 if use_mask else None)
        for (y, x) in zip(good_y, good_x):
            s_main = float(primary_scores[y, x])
            if method == "sqdiff":
                s_main_sim = 1.0 - s_main  # convert to similarity
                s_edge = float(res_edge[y, x])
                s = (1.0 - edge_weight) * s_main_sim + edge_weight * s_edge
            else:
                s = (1.0 - edge_weight) * float(primary_scores[y, x]) + edge_weight * float(res_edge[y, x])
            rects.append((int(x), int(y), tw, th))
            vals.append(s)
        invert_for_sort = False  # similarity => higher better
    else:
        for (y, x) in zip(good_y, good_x):
            rects.append((int(x), int(y), tw, th))
            vals.append(float(primary_scores[y, x]))

    if not rects:
        return []

    sort_scores = [-v for v in vals] if invert_for_sort else vals
    keep_idx = nms_rects(rects, sort_scores, iou_thr=iou_thr, keep_max=keep_max)
    out = []
    for i in keep_idx:
        x,y,w,h = rects[i]
        s = vals[i]
        out.append((x,y,w,h,s))
    return out

def compute_coverage(
    image_bgr: np.ndarray,
    tiles: List[dict],
    db: TileDB,
    method: str,
    space: str,
    edge_weight: float,
    score_thr: Optional[float],
    iou_thr: float,
    max_matches_per_tile: int,
    suppress_overlap: bool
) -> List[Tuple[int,int,int,int,str,float]]:
    H, W = image_bgr.shape[:2]
    claimed = np.zeros((H, W), dtype=np.uint8) if suppress_overlap else None
    all_rects: List[Tuple[int,int,int,int,str,float]] = []

    for t in tiles:
        tile_img = db.load_tile_image(t)
        if tile_img is None:
            continue
        mask = db.load_tile_mask(t)

        per_thr = t.get("min_score", None)
        hits = match_tile_all_positions(
            image_bgr, tile_img, mask,
            method=method,
            space=space,
            edge_weight=edge_weight,
            score_thr=per_thr if per_thr is not None else score_thr,
            iou_thr=iou_thr,
            keep_max=max_matches_per_tile
        )

        for (x,y,w,h,s) in hits:
            if claimed is not None:
                roi = claimed[y:y+h, x:x+w]
                if roi.size > 0 and (roi.mean() > 64):  # ~25% claimed
                    continue
                roi[:] = 255
            all_rects.append((x,y,w,h,t["id"],float(s)))

    return all_rects

def build_coverage_mask(
    image_shape_hw: Tuple[int, int],
    matches: List[Tuple[int,int,int,int,str,float]],
    db: "TileDB"
) -> np.ndarray:
    """Union of all matched tile masks, placed at their (x,y) positions."""
    H, W = image_shape_hw
    cov = np.zeros((H, W), dtype=np.uint8)
    for (x,y,w,h,tid,score) in matches:
        t = db.by_id.get(tid)
        if not t:
            continue
        mask = db.load_tile_mask(t)  # 0/255, (h,w)
        # clip to image bounds
        x2 = min(W, x + w); y2 = min(H, y + h)
        if x >= W or y >= H or x2 <= x or y2 <= y:
            continue
        mw = x2 - x; mh = y2 - y
        m_crop = mask[0:mh, 0:mw]
        cov[y:y+mh, x:x+mw] = np.maximum(cov[y:y+mh, x:x+mw], m_crop)
    return cov

# ---------------------------
# Main application
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive tile/sprite scribe with folder nav, masks, purple masked coverage overlay (ON by default), strict matching, original-res, zoom, fullscreen-by-default, F1 help, and RMB delete in coverage mode.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Path to a single image OR a directory of images.")
    g.add_argument("--images-dir", help="Directory containing input images.")
    ap.add_argument("--db-dir", default="tile_db", help="Directory to store the database and tiles.")
    # Matching controls
    ap.add_argument("--match-method", choices=["sqdiff", "ccorr"], default="sqdiff",
                    help="sqdiff = TM_SQDIFF_NORMED (lower better), ccorr = TM_CCORR_NORMED (higher better).")
    ap.add_argument("--match-space", choices=["gray", "bgr", "hsv-hs"], default="gray",
                    help="Color space for matching. gray is strict; hsv-hs robust to brightness.")
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="0..1: blend in an edge match for extra strictness (e.g., 0.25).")
    ap.add_argument("--score-thr", type=float, default=None,
                    help="Override global threshold. For sqdiff use ~0.03–0.06; for ccorr use ~0.95–0.99.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Deprecated alias for --score-thr.")
    ap.add_argument("--max-matches-per-tile", type=int, default=64,
                    help="Cap matches per tile before NMS.")
    ap.add_argument("--iou-thr", type=float, default=0.15,
                    help="IoU threshold for NMS (higher => fewer kept).")
    ap.add_argument("--suppress-overlap", action="store_true",
                    help="Suppress overlaps across different tiles using a claimed-area mask.")
    # Overlay / display
    ap.add_argument("--alpha", type=float, default=0.35, help="Overlay transparency.")
    ap.add_argument("--original-res", type=parse_size, help="If provided (e.g., 320x200), resize each frame to this resolution (nearest).")
    ap.add_argument("--window-scale", type=float, default=3.0, help="Scale the displayed image (keeps edit coords native).")
    ap.add_argument("--window-width", type=int, help="Force window width (overrides --window-scale).")
    ap.add_argument("--window-height", type=int, help="Force window height (overrides --window-scale).")
    ap.add_argument("--zoom-win", type=int, default=240, help="Zoom window size (square pixels).")
    ap.add_argument("--zoom-factor", type=int, default=8, help="Initial zoom factor (integer).")
    ap.add_argument("--windowed", action="store_true", help="Run windowed (opt-out). Default is fullscreen.")
    args = ap.parse_args()

    if args.score_thr is None and args.threshold is not None:
        args.score_thr = args.threshold

    # Build image list
    if args.images_dir:
        img_dir = args.images_dir
    else:
        img_dir = args.image if (args.image and os.path.isdir(args.image)) else None

    image_list: List[str] = []
    if img_dir:
        if not os.path.isdir(img_dir):
            raise SystemExit(f"Not a directory: {img_dir}")
        image_list = list_images_in_dir(img_dir)
        if not image_list:
            raise SystemExit(f"No images found in {img_dir}")
        current_index = 0
        current_path = image_list[current_index]
    else:
        current_path = args.image
        if not os.path.isfile(current_path):
            raise SystemExit(f"Cannot read image: {current_path}")
        image_list = [current_path]
        current_index = 0

    def read_and_resize(pth: str) -> np.ndarray:
        src = cv2.imread(pth, cv2.IMREAD_COLOR)
        if src is None:
            raise SystemExit(f"Cannot read image: {pth}")
        if args.original_res:
            tw, th = args.original_res
            ih, iw = src.shape[:2]
            src_aspect, dst_aspect = iw/ih, tw/th
            if abs(src_aspect - dst_aspect) > 0.01:
                print(f"[warn] Aspect mismatch: source {iw}x{ih} ~{src_aspect:.3f}, target {tw}x{th} ~{dst_aspect:.3f}")
            return cv2.resize(src, (tw, th), interpolation=cv2.INTER_NEAREST)
        return src

    img = read_and_resize(current_path)

    # init DB and editor
    db = TileDB(args.db_dir)
    ed = RectEditor(img)
    ed.alpha = args.alpha
    ed.zoom_win = args.zoom_win
    ed.zoom_factor = max(1, args.zoom_factor)

    # ---- Window setup (resizable + scaling) ----
    win = "Tile Scribe"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def compute_display_scale_for_current(base_w, base_h):
        nonlocal disp_w, disp_h, display_scale
        if args.window_width and args.window_height:
            dw, dh = args.window_width, args.window_height
            scale_x = dw / base_w
            scale_y = dh / base_h
            display_scale = min(scale_x, scale_y)
        elif args.window_width:
            display_scale = args.window_width / base_w
        elif args.window_height:
            display_scale = args.window_height / base_h
        else:
            display_scale = max(1e-6, args.window_scale)
        disp_w = int(base_w * display_scale)
        disp_h = int(base_h * display_scale)

    disp_w = disp_h = 0
    display_scale = 1.0
    compute_display_scale_for_current(ed.w, ed.h)
    cv2.resizeWindow(win, disp_w, disp_h)

    def _mouse_cb(event, mx, my, flags, userdata):
        ix = int(round(mx / display_scale))
        iy = int(round(my / display_scale))
        ed.mouse_cb_image_coords(event, ix, iy, flags, userdata)
    cv2.setMouseCallback(win, _mouse_cb)

    zoom_win_name = "Zoom"
    fullscreen = not args.windowed  # FULLSCREEN DEFAULT
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def refresh_coverage():
        return compute_coverage(
            ed.base, db.tiles, db,
            method=args.match_method,
            space=args.match_space,
            edge_weight=args.edge_weight,
            score_thr=args.score_thr,
            iou_thr=args.iou_thr,
            max_matches_per_tile=args.max_matches_per_tile,
            suppress_overlap=args.suppress_overlap
        )

    def recompute_overlay():
        if ed.overlay_matches:
            ed.matches_cache = refresh_coverage()
            ed.coverage_mask = build_coverage_mask((ed.h, ed.w), ed.matches_cache, db)
        else:
            ed.matches_cache = None
            ed.coverage_mask = None

    def switch_image(delta: int):
        """Switch current image by delta (e.g., -1 for prev, +1 for next). Keep fullscreen if enabled."""
        nonlocal current_index, current_path, image_list, ed, disp_w, disp_h, display_scale, fullscreen
        if len(image_list) <= 1:
            return
        new_index = int(np.clip(current_index + delta, 0, len(image_list)-1))
        if new_index == current_index:
            return
        current_index = new_index
        current_path = image_list[current_index]
        new_img = read_and_resize(current_path)
        ed.set_base(new_img)
        ed.clamp_rect_to_image()
        compute_display_scale_for_current(ed.w, ed.h)
        if not fullscreen:
            cv2.resizeWindow(win, disp_w, disp_h)
        else:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        recompute_overlay()

    def footer_text():
        if len(image_list) > 1:
            fname = os.path.basename(current_path)
            return f"[{current_index+1}/{len(image_list)}] {fname}"
        else:
            return os.path.basename(current_path)

    # ---- Auto-run coverage on first image ----
    if ed.overlay_matches:
        print("[info] Computing initial coverage (purple masked overlay)...")
        recompute_overlay()
        if ed.matches_cache is not None:
            print(f"[ok] {len(ed.matches_cache)} matches found.")

    # main loop
    while True:
        # Process pending delete request
        if ed.delete_request_tid:
            tid = ed.delete_request_tid
            ed.delete_request_tid = None
            if db.delete_tile(tid):
                print(f"[ok] Deleted tile {tid} from DB.")
            else:
                print(f"[warn] Tile {tid} not found.")
            recompute_overlay()

        base_canvas = ed.draw_ui(ed.base, ed.overlay_matches, ed.matches_cache, footer=footer_text())
        canvas = cv2.resize(base_canvas, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST) if display_scale != 1.0 else base_canvas
        cv2.imshow(win, canvas)

        if ed.zoom_enabled:
            ed.render_zoom()

        k = cv2.waitKeyEx(16) & 0xFFFFFFFF
        if k == 0xFFFFFFFF:
            continue

        # Quit
        if k in (ord('q'), 27):
            break

        # Fullscreen toggle (F)
        if k in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if not fullscreen:
                cv2.resizeWindow(win, disp_w, disp_h)

        # Help toggle: F1 (Windows=0x70, X11 often 65470), fallback H / ?
        if k in (0x70, 65470) or k in (ord('h'), ord('H'), ord('?')):
            ed.show_help = not ed.show_help

        # Save DB
        if k in (ord('s'), ord('S')):
            db.save()
            print("[ok] DB saved.")

        # Toggle coverage (C)
        if k in (ord('c'), ord('C')):
            ed.overlay_matches = not ed.overlay_matches
            if ed.overlay_matches:
                print("[info] Computing coverage (purple masked overlay)...")
            recompute_overlay()
            if ed.overlay_matches and ed.matches_cache is not None:
                print(f"[ok] {len(ed.matches_cache)} matches.")

        # Image navigation: Left/Right arrows
        if k in (81, 2424832):   # Left arrow
            switch_image(-1)
            continue
        if k in (83, 2555904):   # Right arrow
            switch_image(+1)
            continue

        # Zoom toggle (Z) and zoom factor (, .)
        if k in (ord('z'), ord('Z')):
            ed.zoom_enabled = not ed.zoom_enabled
            if ed.zoom_enabled:
                cv2.namedWindow(zoom_win_name, cv2.WINDOW_AUTOSIZE)
                print(f"[info] Zoom ON x{ed.zoom_factor}")
            else:
                try:
                    cv2.destroyWindow(zoom_win_name)
                except:
                    pass
        if k in (ord(','),):
            ed.zoom_factor = max(1, ed.zoom_factor - 1); print(f"[info] Zoom x{ed.zoom_factor}")
        if k in (ord('.'),):
            ed.zoom_factor = min(64, ed.zoom_factor + 1); print(f"[info] Zoom x{ed.zoom_factor}")

        # Nudge (use WASD/X to avoid arrow conflicts with image navigation)
        if k in (ord('a'), ord('A')): ed.nudge(dx=-1)
        if k in (ord('d'), ord('D')): ed.nudge(dx=+1)
        if k in (ord('w'), ord('W')): ed.nudge(dy=-1)
        if k in (ord('x'), ord('X')): ed.nudge(dy=+1)

        # Resize rect: width with [ ] , height with { }
        if k == ord('['): ed.nudge(dw=-1)
        if k == ord(']'): ed.nudge(dw=+1)
        if k == ord('{'): ed.nudge(dh=-1)
        if k == ord('}'): ed.nudge(dh=+1)

        # ENTER: accept tile (add to DB)
        if k in (10, 13):
            tile = ed.current_tile()
            if tile.size == 0:
                print("[warn] Empty selection.")
                continue
            entry = db.add_tile(tile)
            print(f"[ok] Added tile {entry['id']}  {entry['w']}x{entry['h']}  -> {entry['file']} (mask: {entry['mask']})")
            print("     Tip: edit mask externally (white=match, black=ignore).")
            recompute_overlay()

    try:
        cv2.destroyWindow(zoom_win_name)
    except:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
