#!/usr/bin/env python3
import argparse
import os
import json
from typing import List, Tuple, Optional, Dict, Set
import cv2
import numpy as np

PURPLE = (255, 0, 255)
SKIP_FILE = "dedupe_skip.json"  # stored in DB dir

# ---------------------------
# I/O + DB helpers
# ---------------------------

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Optional[dict]:
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: str, data: dict):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def to_gray_mask(mask_img: Optional[np.ndarray], shape_hw: Tuple[int,int]) -> np.ndarray:
    """Return single-channel uint8 of (h,w), 0=bg, 255=keep. If missing, all white."""
    h, w = shape_hw
    if mask_img is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if mask_img.ndim == 3:
        g = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        g = mask_img
    if g.shape != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
    _, g = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    return g

def mask_is_trivial(mask: np.ndarray) -> bool:
    return np.all(mask == 255)

class TileDB:
    def __init__(self, root: str):
        self.root = root
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
            raise RuntimeError(f"Failed to read tile image: {t['file']}")
        h, w = img.shape[:2]
        mimg = None
        mp = t.get("mask", "")
        if mp:
            mp_abs = self._path(mp)
            if os.path.exists(mp_abs):
                mimg = cv2.imread(mp_abs, cv2.IMREAD_UNCHANGED)
        mask = to_gray_mask(mimg, (h, w))
        return img, mask

    def delete_tile(self, tile_id: str) -> bool:
        idx = None
        for i, t in enumerate(self.tiles):
            if t["id"] == tile_id:
                idx = i; break
        if idx is None:
            return False
        t = self.tiles.pop(idx)
        for key in ("file", "mask"):
            p = t.get(key, "")
            if p:
                abs_p = self._path(p)
                try:
                    if os.path.exists(abs_p):
                        os.remove(abs_p)
                except Exception:
                    pass
        self.by_id = {t["id"]: t for t in self.tiles}
        save_json(self.db_path, self.data)
        return True

    def save(self):
        save_json(self.db_path, self.data)

# ---------------------------
# Persistent skip pairs
# ---------------------------

class SkipPairs:
    def __init__(self, db_dir: str):
        self.path = os.path.join(db_dir, SKIP_FILE)
        self.pairs: Set[Tuple[str, str]] = set()
        self._load()

    @staticmethod
    def _norm(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _load(self):
        data = load_json(self.path) or {}
        items = data.get("skip_pairs", [])
        for pair in items:
            if isinstance(pair, list) and len(pair) == 2:
                a, b = str(pair[0]), str(pair[1])
                self.pairs.add(self._norm(a, b))

    def add(self, a: str, b: str):
        self.pairs.add(self._norm(a, b))
        self._save()

    def contains(self, a: str, b: str) -> bool:
        return self._norm(a, b) in self.pairs

    def _save(self):
        out = {"skip_pairs": [list(p) for p in sorted(self.pairs)]}
        save_json(self.path, out)

# ---------------------------
# Similarity (mask-aware, mirror-aware)
# ---------------------------

def normalized_l1(a: np.ndarray, b: np.ndarray, mask_bool: np.ndarray) -> float:
    if not mask_bool.any():
        return 1.0
    diff = np.abs(a[mask_bool].astype(np.int16) - b[mask_bool].astype(np.int16))
    mae = diff.mean() / 255.0
    return float(mae)

def best_similarity(a_img: np.ndarray, a_mask: np.ndarray,
                    b_img: np.ndarray, b_mask: np.ndarray,
                    allow_mirror: bool = True) -> Tuple[float, bool]:
    assert a_img.shape == b_img.shape, "Size mismatch not supported."
    a_non = not mask_is_trivial(a_mask)
    b_non = not mask_is_trivial(b_mask)
    if a_non and not b_non:
        region = a_mask > 0
    elif b_non and not a_non:
        region = b_mask > 0
    else:
        region = (a_mask > 0) & (b_mask > 0)

    s_direct = normalized_l1(a_img, b_img, region)
    best_s, mirrored = s_direct, False

    if allow_mirror:
        b_flip = cv2.flip(b_img, 1)
        if b_non and not a_non:
            region_m = cv2.flip(b_mask, 1) > 0
        elif a_non and not b_non:
            region_m = a_mask > 0
        else:
            region_m = (a_mask > 0) & (cv2.flip(b_mask,1) > 0)
        s_mir = normalized_l1(a_img, b_flip, region_m)
        if s_mir < best_s:
            best_s, mirrored = s_mir, True

    return best_s, mirrored

# ---------------------------
# Candidate search
# ---------------------------

def group_by_size(tiles: List[dict]) -> Dict[Tuple[int,int], List[dict]]:
    groups: Dict[Tuple[int,int], List[dict]] = {}
    for t in tiles:
        wh = (int(t["w"]), int(t["h"]))
        groups.setdefault(wh, []).append(t)
    return groups

def find_best_pair(db: TileDB,
                   remaining_ids: Set[str],
                   threshold: float,
                   allow_mirror: bool,
                   checked: Set[Tuple[str,str]],
                   skip_pairs: SkipPairs) -> Optional[Tuple[dict, dict, float, bool]]:
    tiles = [db.by_id[i] for i in remaining_ids if i in db.by_id]
    by_size = group_by_size(tiles)
    best = None  # (score, a, b, mirrored)

    for (w,h), lst in by_size.items():
        if len(lst) < 2:
            continue
        cache_img, cache_msk = {}, {}
        def get_im_m(t):
            tid = t["id"]
            if tid not in cache_img:
                img, msk = db.load_tile(t)
                cache_img[tid] = img; cache_msk[tid] = msk
            return cache_img[tid], cache_msk[tid]

        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                a = lst[i]; b = lst[j]
                key = (a["id"], b["id"])
                # Already checked in this session, or explicitly skipped forever?
                if key in checked or skip_pairs.contains(a["id"], b["id"]):
                    continue
                ai, am = get_im_m(a)
                bi, bm = get_im_m(b)
                try:
                    s, mirrored = best_similarity(ai, am, bi, bm, allow_mirror=allow_mirror)
                except AssertionError:
                    continue
                if s <= threshold:
                    if best is None or s < best[0]:
                        best = (s, a, b, mirrored)
    if best is None:
        return None
    # mark this candidate as checked for this search round so we don't re-evaluate it if user keeps both (we'll store it permanently then)
    checked.add((best[1]["id"], best[2]["id"]))
    return (best[1], best[2], best[0], best[3])

# ---------------------------
# UI
# ---------------------------

class DedupeUI:
    def __init__(self, scale: int = 12, windowed: bool = False):
        self.scale = max(2, int(scale))
        self.windowed = windowed
        self.win = "Tile Dedupe"
        self.choice: Optional[str] = None  # "keep_left" | "keep_right" | "keep_both" | "not_dup" | "exit"

    def _compose_panel(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        S = max(H, W)
        out = np.zeros((S, S, 3), dtype=np.uint8); out[:] = PURPLE
        m = (mask[:H, :W] > 0)[:, :, None]
        out[:H, :W] = np.where(m, img, out[:H, :W])
        return cv2.resize(out, (S*self.scale, S*self.scale), interpolation=cv2.INTER_NEAREST)

    def show_pair(self, a: dict, a_img: np.ndarray, a_mask: np.ndarray,
                        b: dict, b_img: np.ndarray, b_mask: np.ndarray,
                        score: float, mirrored: bool):
        left = self._compose_panel(a_img, a_mask)
        right = self._compose_panel(b_img, b_mask)
        H = max(left.shape[0], right.shape[0])
        W = left.shape[1] + right.shape[1] + 20
        canvas = np.zeros((H + 80, W, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        y0 = 30
        canvas[y0:y0+left.shape[0], 10:10+left.shape[1]] = left
        canvas[y0:y0+right.shape[0], 10+left.shape[1]+10:10+left.shape[1]+10+right.shape[1]] = right

        def put(text, y, x=10, fg=(240,240,240)):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)

        title = f"Possible duplicate  |  score={score:.4f}  |  mirror={'YES' if mirrored else 'no'}"
        put(title, 22)
        put(f"LEFT  id={a['id']}  {a.get('name','')}  [{a.get('category','')}]", H + 55, 10)
        put(f"RIGHT id={b['id']}  {b.get('name','')}  [{b.get('category','')}]", H + 55, 10 + left.shape[1] + 10)
        put("L-Click=keep LEFT  R-Click=keep RIGHT  ENTER=keep BOTH & skip pair  N=Not duplicate (skip)  ESC=exit", H + 30, 10)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        if not self.windowed:
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.win, canvas.shape[1], canvas.shape[0])

        left_rect = (10, y0, left.shape[1], left.shape[0])
        right_rect = (10 + left.shape[1] + 10, y0, right.shape[1], right.shape[0])

        def on_mouse(evt, x, y, flags, userdata):
            if evt == cv2.EVENT_LBUTTONDOWN:
                if (left_rect[0] <= x < left_rect[0]+left_rect[2] and
                    left_rect[1] <= y < left_rect[1]+left_rect[3]):
                    self.choice = "keep_left"
                elif (right_rect[0] <= x < right_rect[0]+right_rect[2] and
                      right_rect[1] <= y < right_rect[1]+right_rect[3]):
                    self.choice = "keep_right"
            if evt == cv2.EVENT_RBUTTONDOWN:
                if (left_rect[0] <= x < left_rect[0]+left_rect[2] and
                    left_rect[1] <= y < left_rect[1]+left_rect[3]):
                    self.choice = "keep_left"
                elif (right_rect[0] <= x < right_rect[0]+right_rect[2] and
                      right_rect[1] <= y < right_rect[1]+right_rect[3]):
                    self.choice = "keep_right"

        cv2.setMouseCallback(self.win, on_mouse)

        while True:
            cv2.imshow(self.win, canvas)
            k = cv2.waitKeyEx(16) & 0xFFFFFFFF
            if k == 0xFFFFFFFF:
                if self.choice is not None: break
                continue
            if k in (27, ord('q')):  # Esc/q
                self.choice = "exit"; break
            if k in (10, 13):  # Enter
                self.choice = "keep_both"; break
            if k in (ord('n'), ord('N')):
                self.choice = "not_dup"; break
            if self.choice is not None:
                break

        cv2.setMouseCallback(self.win, lambda *args: None)

    def show_message(self, text: str):
        W, H = 1000, 240
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if not self.windowed else cv2.WINDOW_NORMAL)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (W - tw) // 2
        y = (H + th) // 2
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240,240,240), 2, cv2.LINE_AA)
        cv2.imshow(self.win, canvas)
        # brief show; return to caller so it can try again or exit
        cv2.waitKeyEx(400)

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except:
            pass

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Find and resolve duplicate tiles/sprites. Mask-aware, optional mirror detection. Persists non-duplicate pairs.")
    ap.add_argument("--db-dir", default="tile_db", help="DB directory containing tiles_db.json")
    ap.add_argument("--threshold", type=float, default=0.03, help="Duplicate threshold (normalized L1; lower stricter).")
    ap.add_argument("--allow-mirror", action="store_true", help="Treat horizontal mirror as duplicate candidate.")
    ap.add_argument("--scale", type=int, default=12, help="Pixel scale for preview panels.")
    ap.add_argument("--windowed", action="store_true", help="Run windowed (default fullscreen).")
    args = ap.parse_args()

    db = TileDB(args.db_dir)
    ids: Set[str] = {t["id"] for t in db.tiles}
    checked: Set[Tuple[str,str]] = set()  # session-local “already examined” pairs
    skip_pairs = SkipPairs(args.db_dir)   # persistent “not a duplicate” pairs

    ui = DedupeUI(scale=args.scale, windowed=args.windowed)

    try:
        while True:
            cand = find_best_pair(db, ids, args.threshold, args.allow_mirror, checked, skip_pairs)
            if not cand:
                ui.show_message("No duplicates found under current threshold.")
                # Try once more in case user changed DB externally; otherwise allow Esc to exit
                # If nothing changes, loop will keep showing the message; exit on Esc from window close.
                # To break automatically, uncomment:
                # break
                # Instead, we just show and then break to end session
                break

            a, b, score, mirrored = cand
            a_img, a_m = db.load_tile(a)
            b_img, b_m = db.load_tile(b)

            ui.choice = None
            ui.show_pair(a, a_img, a_m, b, b_img, b_m, score, mirrored)

            if ui.choice == "exit":
                break
            elif ui.choice == "keep_left":
                # Delete right; remove it from ID set and purge checked pairs involving it
                if db.delete_tile(b["id"]):
                    print(f"[ok] Deleted duplicate {b['id']} (kept {a['id']})")
                    ids.discard(b["id"])
                    checked = {pair for pair in checked if b["id"] not in pair}
                else:
                    print(f"[warn] Failed to delete {b['id']}")
            elif ui.choice == "keep_right":
                if db.delete_tile(a["id"]):
                    print(f"[ok] Deleted duplicate {a['id']} (kept {b['id']})")
                    ids.discard(a["id"])
                    checked = {pair for pair in checked if a["id"] not in pair}
                else:
                    print(f"[warn] Failed to delete {a['id']}")
            elif ui.choice in ("keep_both", "not_dup"):
                # Persist this pair as not-a-duplicate so it won't resurface again (now or in future)
                skip_pairs.add(a["id"], b["id"])
                print(f"[info] Marked not-duplicate pair: {a['id']} <> {b['id']}")
                # Nothing else to change; continue searching
            else:
                # Fallback: treat as keep both & skip pair
                skip_pairs.add(a["id"], b["id"])

        ui.close()
        print("[done] Dedupe session ended.")
    except KeyboardInterrupt:
        ui.close()
        print("\n[aborted]")

if __name__ == "__main__":
    main()
