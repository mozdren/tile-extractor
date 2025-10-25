#!/usr/bin/env python3
import argparse, os, json
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from collections import defaultdict

MAGENTA_BGR = (255, 0, 255)  # SDL colorkey background

# ---------------------------
# I/O helpers
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
    """Return single-channel uint8 of (h,w), 0=masked (bg), 255=keep. If missing, all white."""
    h, w = shape_hw
    if mask_img is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if mask_img.ndim == 3:
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_img
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)

# ---------------------------
# DB access (compatible with tile_scribe / tile_editor / tile_annotator)
# ---------------------------

class TileDB:
    def __init__(self, root: str):
        self.root = root
        self.db_path = os.path.join(root, "tiles_db.json")
        data = load_json(self.db_path)
        if not data:
            raise SystemExit(f"Cannot find {self.db_path}")
        self.data = data
        self.tiles_dir = os.path.join(root, data.get("tiles_dir", "tiles"))
        self.tiles = data.get("tiles", [])

    def _path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def load_tile_and_mask(self, t: dict) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(self._path(t["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read tile image: {t['file']}")
        h, w = img.shape[:2]
        mask_img = None
        mp = t.get("mask", "")
        if mp:
            mask_abs = self._path(mp)
            if os.path.exists(mask_abs):
                mask_img = cv2.imread(mask_abs, cv2.IMREAD_UNCHANGED)
        mask = to_gray_mask(mask_img, (h, w))
        return img, mask

# ---------------------------
# Packing (shelf packer)
# ---------------------------

def sort_tiles(tiles: List[dict], key: str) -> List[dict]:
    """Return a NEW list sorted by the requested key."""
    def area(t): return int(t["w"]) * int(t["h"])
    def cat(t):  return (t.get("category") or "").lower()
    def name(t): return (t.get("name") or "").lower()
    if key == "none":
        return tiles[:]
    if key == "height":
        return sorted(tiles, key=lambda t: (-int(t["h"]), -int(t["w"])))
    if key == "width":
        return sorted(tiles, key=lambda t: (-int(t["w"]), -int(t["h"])))
    if key == "area":
        return sorted(tiles, key=lambda t: (-area(t), -int(t["h"])))
    if key == "category":
        return sorted(tiles, key=lambda t: (cat(t), -area(t)))
    if key == "name":
        return sorted(tiles, key=lambda t: (name(t), -area(t)))
    if key == "category-name":
        return sorted(tiles, key=lambda t: (cat(t), name(t), -area(t)))
    if key == "category-area":
        return sorted(tiles, key=lambda t: (cat(t), -area(t)))
    if key == "name-area":
        return sorted(tiles, key=lambda t: (name(t), -area(t)))
    return tiles[:]

def shelf_pack(tiles: List[dict], max_width: int, padding: int) -> Tuple[int, int, Dict[str, Tuple[int,int]]]:
    """
    Return (atlas_w, atlas_h, positions), where positions[tile_id] = (x, y).
    Simple shelf: fill rows left->right, wrap to new row when exceeding max_width.
    """
    widest = max((int(t["w"]) for t in tiles), default=0)
    if widest + 2*padding > max_width:
        max_width = widest + 2*padding

    x = padding
    y = padding
    row_h = 0
    positions = {}

    for t in tiles:
        w, h = int(t["w"]), int(t["h"])
        if x + w + padding > max_width:  # new row
            y += row_h + padding
            x = padding
            row_h = 0
        positions[t["id"]] = (x, y)
        x += w + padding
        row_h = max(row_h, h)

    atlas_w = max_width
    atlas_h = y + row_h + padding
    return atlas_w, atlas_h, positions

def make_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

# ---------------------------
# Compositing
# ---------------------------

def blit_tile_with_mask(dest_bgr: np.ndarray, x: int, y: int, tile_bgr: np.ndarray, mask_u8: np.ndarray):
    """
    Copy tile pixels where mask>0; leave dest otherwise (magenta background shows through).
    """
    h, w = tile_bgr.shape[:2]
    roi = dest_bgr[y:y+h, x:x+w]
    m = (mask_u8 > 0)[:, :, None]
    roi[:] = np.where(m, tile_bgr, roi)

def blit_mask(dest_mask: np.ndarray, x: int, y: int, mask_u8: np.ndarray):
    h, w = mask_u8.shape[:2]
    dest_mask[y:y+h, x:x+w] = mask_u8

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compose tiles + masks into a magenta sprite sheet with JSON including per-sprite coordinates.")
    ap.add_argument("--db-dir", default="tile_db", help="Directory containing tiles_db.json and tiles/")
    ap.add_argument("--out-image", default="atlas.png", help="Output sprite/tiles image (magenta background).")
    ap.add_argument("--out-json",  default="atlas.json", help="Output metadata JSON.")
    ap.add_argument("--out-mask",  default=None, help="Optional output grayscale mask atlas (same layout).")
    ap.add_argument("--padding", type=int, default=1, help="Pixels between sprites.")
    ap.add_argument("--max-width", type=int, default=2048, help="Maximum atlas width for shelf packing.")
    ap.add_argument("--sort", choices=["height","width","area","none","category","name","category-name","category-area","name-area"],
                    default="category-name", help="Packing sort heuristic, now including category/name options.")
    ap.add_argument("--power-of-two", action="store_true", help="Pad atlas width/height up to next power of two.")
    ap.add_argument("--emit-uv", action="store_true", help="Also write normalized UVs (u0,v0,u1,v1) for each sprite.")
    # New: category/name aware controls
    ap.add_argument("--filter-category", type=str, default="", help="Comma-separated list of categories to include (others skipped).")
    ap.add_argument("--skip-uncategorized", action="store_true", help="Exclude tiles with empty category.")
    args = ap.parse_args()

    db = TileDB(args.db_dir)
    tiles = db.tiles
    if not tiles:
        raise SystemExit("No tiles found in DB.")

    # Filter by category if requested
    allow_cats = None
    if args.filter_category.strip():
        allow_cats = {c.strip().lower() for c in args.filter_category.split(",") if c.strip()}
    filtered = []
    for t in tiles:
        cat = (t.get("category") or "").strip()
        if args.skip_uncategorized and not cat:
            continue
        if allow_cats is not None and cat.lower() not in allow_cats:
            continue
        filtered.append(t)
    tiles = filtered
    if not tiles:
        raise SystemExit("No tiles remain after filtering/skip options.")

    # Sort tiles for better packing / grouping
    tiles_sorted = sort_tiles(tiles, args.sort)

    # Compute layout
    atlas_w, atlas_h, positions = shelf_pack(tiles_sorted, args.max_width, args.padding)
    pot_w, pot_h = atlas_w, atlas_h
    if args.power_of_two:
        pot_w, pot_h = make_power_of_two(atlas_w), make_power_of_two(atlas_h)

    # Create canvases
    atlas = np.zeros((pot_h, pot_w, 3), dtype=np.uint8)
    atlas[:, :] = MAGENTA_BGR  # magenta background for SDL color key

    mask_atlas = None
    if args.out_mask:
        mask_atlas = np.zeros((pot_h, pot_w), dtype=np.uint8)  # 0 background by default

    # Blit every tile with its mask
    placed = []
    for t in tiles_sorted:
        tid = t["id"]
        img, mask = db.load_tile_and_mask(t)
        w, h = int(t["w"]), int(t["h"])
        x, y = positions[tid]
        if x + w > pot_w or y + h > pot_h:
            raise RuntimeError(f"Placement out of bounds for {tid} at {(x,y)} size {(w,h)} in {(pot_w,pot_h)}")
        blit_tile_with_mask(atlas, x, y, img, mask)
        if mask_atlas is not None:
            blit_mask(mask_atlas, x, y, mask)
        placed.append((t, x, y))

    # Write outputs
    ensure_dir(os.path.dirname(args.out_image) or ".")
    ensure_dir(os.path.dirname(args.out_json) or ".")
    cv2.imwrite(args.out_image, atlas)
    if mask_atlas is not None:
        ensure_dir(os.path.dirname(args.out_mask) or ".")
        cv2.imwrite(args.out_mask, mask_atlas)

    # Build metadata with coordinates + groups
    meta = {
        "version": 2,
        "image": os.path.basename(args.out_image),
        "size": {"w": int(pot_w), "h": int(pot_h)},
        "padding": int(args.padding),
        "power_of_two": bool(args.power_of_two),
        "sort": args.sort,
        "filters": {
            "filter_category": args.filter_category,
            "skip_uncategorized": bool(args.skip_uncategorized)
        },
        "sprites": [],
        "groups": {
            "by_category": {},   # category -> [ids in atlas order]
            "by_name": {}        # name -> [ids in atlas order]
        }
    }
    if args.out_mask:
        meta["mask_image"] = os.path.basename(args.out_mask)

    by_category = defaultdict(list)
    by_name = defaultdict(list)

    for idx, (t, x, y) in enumerate(placed):
        w, h = int(t["w"]), int(t["h"])
        name = t.get("name","")
        cat  = t.get("category","")
        entry = {
            "id": t["id"],
            "frame": idx,                         # order in the atlas
            "name": name,
            "category": cat,
            "src_file": t.get("file",""),
            "mask_file": t.get("mask",""),
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "rect": [int(x), int(y), int(w), int(h)]
        }
        if "min_score" in t:
            entry["min_score"] = t["min_score"]
        if args.emit_uv:
            u0 = x / float(pot_w); v0 = y / float(pot_h)
            u1 = (x + w) / float(pot_w); v1 = (y + h) / float(pot_h)
            entry["uv"] = [u0, v0, u1, v1]
        meta["sprites"].append(entry)

        # Grouping lists (ids in atlas order)
        if cat:
            by_category[cat].append(t["id"])
        else:
            by_category["_uncategorized_"].append(t["id"])
        if name:
            by_name[name].append(t["id"])

    # finalize groups (convert to normal dicts)
    meta["groups"]["by_category"] = dict(sorted(by_category.items(), key=lambda kv: kv[0].lower()))
    meta["groups"]["by_name"] = dict(sorted(by_name.items(), key=lambda kv: kv[0].lower()))

    save_json(args.out_json, meta)

    print(f"[ok] Wrote atlas: {args.out_image}  ({pot_w}x{pot_h})")
    if args.out_mask:
        print(f"[ok] Wrote mask atlas: {args.out_mask}")
    print(f"[ok] Wrote metadata: {args.out_json}")
    print(f"[count] {len(placed)} sprites")
    if args.filter_category:
        print(f"[info] Categories included: {args.filter_category}")
    if args.skip_uncategorized:
        print("[info] Skipped uncategorized tiles.")

if __name__ == "__main__":
    main()
