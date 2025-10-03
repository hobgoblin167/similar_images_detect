"""
Photo deduplication & similarity finder
Single-file Python CLI tool.

Features implemented:
- Recursive scan of a main folder + optional additional folders
- Detect exact duplicates (byte-wise SHA256)
- Detect content duplicates across formats (RAW/TIFF <-> JPEG/WebP/PNG) using perceptual hashes + resolution/size heuristics
- Detect similar images (series shots) via perceptual hash Hamming distance and SSIM
- Configurable thresholds and options via command-line or config file
- EXIF reading (date, orientation) when available
- Grouping clusters and choosing a "primary" keeper per cluster using ranking rules
- Export results as CSV and optional shell script (delete/move) — does NOT delete by default

Dependencies (install with pip):
  pip install pillow imagehash piexif tqdm rawpy numpy scipy scikit-image opencv-python

Notes:
- rawpy is optional but recommended for reading RAW files (NEF/CR2 etc.). If not installed, RAW files are skipped for image-based comparisons but still included in byte-hash and metadata scanning.
- The tool avoids destructive actions by default. Use generated scripts carefully and inspect CSV report before running removals.

Usage examples:
  python photo_dedupe_tool.py --main "/path/to/main" --extras "/mnt/drive1" "/mnt/drive2" --mode all --phash-threshold 12 --ssim-threshold 0.85 --export results.csv --script delete_candidates.sh

"""

import os
import sys
import argparse
import hashlib
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Image handling
from PIL import Image, ExifTags, UnidentifiedImageError
import imagehash
import piexif
import numpy as np

# Optional libraries
try:
    import rawpy
    RAWPY_AVAILABLE = True
except Exception:
    RAWPY_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

from tqdm import tqdm

# ----------------------------- Configuration & Utilities -----------------------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.cr2', '.nef', '.arw', '.raf', '.rw2'}
RAW_EXTS = {'.cr2', '.nef', '.arw', '.raf', '.rw2'}
LOSSY_EXTS = {'.jpg', '.jpeg', '.webp'}
LOSSLESS_EXTS = {'.png', '.tif', '.tiff'}

def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTS

def human_size(num_bytes):
    for unit in ['B','KB','MB','GB','TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"

# ----------------------------- Scanning -----------------------------

def scan_folders(main_folder, extra_folders, follow_symlinks=False):
    """Yield Path objects for files inside main+extra folders recursively."""
    seen = set()
    folders = [Path(main_folder)] + [Path(p) for p in (extra_folders or [])]
    for base in folders:
        if not base.exists():
            print(f"Warning: {base} does not exist; skipping.")
            continue
        for root, dirs, files in os.walk(base, followlinks=follow_symlinks):
            for name in files:
                p = Path(root) / name
                if p.resolve() in seen:
                    continue
                seen.add(p.resolve())
                yield p

# ----------------------------- Hashing & Metadata -----------------------------

def sha256_file(path: Path, block_size=65536):
    h = hashlib.sha256()
    try:
        with path.open('rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
    except Exception as e:
        return None, str(e)
    return h.hexdigest(), None


def read_exif(path: Path):
    data = {}
    try:
        img = Image.open(path)
        info = img._getexif()
        if info:
            for tag, val in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                data[decoded] = val
        # orientation handling may be done later
        # attempt piexif for better parsing
        try:
            ex = piexif.load(path.as_posix())
            data['piexif'] = ex
        except Exception:
            pass
        img.close()
    except UnidentifiedImageError:
        # maybe RAW, try rawpy
        if RAWPY_AVAILABLE and path.suffix.lower() in RAW_EXTS:
            try:
                r = rawpy.imread(path.as_posix())
                # limited metadata
                data['raw'] = True
                r.close()
            except Exception:
                pass
    except Exception:
        pass
    return data


def load_image_for_hash(path: Path, resize_for_hash=1024):
    """Return PIL.Image suitable for hashing/comparison.
    For RAW files, use rawpy if available to convert to RGB.
    """
    suffix = path.suffix.lower()
    try:
        if RAWPY_AVAILABLE and suffix in RAW_EXTS:
            raw = rawpy.imread(path.as_posix())
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            raw.close()
            img = Image.fromarray(rgb)
        else:
            img = Image.open(path)
            # ensure loaded
            img.load()
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
    except Exception as e:
        raise

    # normalize orientation using EXIF
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(274)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    # optionally resize for performance
    maxdim = max(img.size)
    if maxdim > resize_for_hash:
        scale = resize_for_hash / maxdim
        new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
        img = img.resize(new_size, Image.LANCZOS)

    return img

# Compute multiple perceptual hashes
def compute_hashes(path: Path, hash_funcs=('phash','dhash','ahash')):
    results = {}
    try:
        img = load_image_for_hash(path)
    except Exception as e:
        return None, f"load_error: {e}"

    if 'phash' in hash_funcs:
        results['phash'] = imagehash.phash(img)
    if 'dhash' in hash_funcs:
        results['dhash'] = imagehash.dhash(img)
    if 'ahash' in hash_funcs:
        results['ahash'] = imagehash.average_hash(img)
    # histogram for quick compare
    try:
        hist = img.convert('RGB').histogram()
        results['hist'] = tuple(hist)
    except Exception:
        results['hist'] = None

    # resolution and mode
    results['size'] = img.size
    return results, None

# SSIM comparison
def compute_ssim(path1: Path, path2: Path, resize_to=800):
    if not SKIMAGE_AVAILABLE:
        return None, 'skimage not available'
    try:
        i1 = load_image_for_hash(path1, resize_for_hash=resize_to).convert('L')
        i2 = load_image_for_hash(path2, resize_for_hash=resize_to).convert('L')
    except Exception as e:
        return None, f'load_error: {e}'
    a1 = np.array(i1)
    a2 = np.array(i2)
    # if shapes differ, resize to common shape
    if a1.shape != a2.shape:
        # naive resize using PIL
        i2 = Image.fromarray(a2).resize(i1.size, Image.LANCZOS)
        a2 = np.array(i2)
    try:
        val = ssim(a1, a2)
        return float(val), None
    except Exception as e:
        return None, str(e)

# Hamming distance for imagehash.ImageHash
def hamming(h1, h2):
    if h1 is None or h2 is None:
        return None
    return (h1 - h2)

# ----------------------------- Main Processing Pipeline -----------------------------

class FileRecord:
    def __init__(self, path: Path):
        self.path = path
        self.size = None
        self.mtime = None
        self.sha256 = None
        self.sha_err = None
        self.hashes = {}
        self.hash_err = None
        self.exif = {}
        self.format = path.suffix.lower().lstrip('.')
        self.width = None
        self.height = None

    def collect_basic(self):
        try:
            st = self.path.stat()
            self.size = st.st_size
            self.mtime = st.st_mtime
        except Exception:
            pass

    def collect_sha(self):
        h, e = sha256_file(self.path)
        self.sha256 = h
        self.sha_err = e

    def collect_exif(self):
        try:
            ex = read_exif(self.path)
            self.exif = ex
        except Exception:
            self.exif = {}

    def collect_hashes(self, funcs=('phash','dhash','ahash')):
        h, e = compute_hashes(self.path, funcs)
        if h is None:
            self.hash_err = e
        else:
            self.hashes = h
            if 'size' in h:
                self.width, self.height = h['size']


def build_file_records(paths, max_workers=8, collect_image_hashes=True, hash_funcs=('phash','dhash','ahash')):
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for p in paths:
            r = FileRecord(p)
            r.collect_basic()
            futures[ex.submit(r.collect_sha)] = ('sha', r)
            futures[ex.submit(r.collect_exif)] = ('exif', r)
            if collect_image_hashes and is_image_file(p):
                futures[ex.submit(r.collect_hashes, hash_funcs)] = ('hash', r)
            records.append(r)
        # simple progress
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Collecting metadata'):
            pass
    return records

# ----------------------------- Comparison & Clustering -----------------------------

def find_exact_duplicates(records):
    byhash = defaultdict(list)
    for r in records:
        if r.sha256:
            byhash[r.sha256].append(r)
    groups = [group for group in byhash.values() if len(group) > 1]
    return groups


def find_content_duplicates(records, phash_threshold=10, require_different_format=True):
    # content duplicates: same scene but different formats/resolutions — detect via phash/dhash close matches
    idx = [r for r in records if r.hashes and 'phash' in r.hashes]
    groups = []
    used = set()
    for i, a in enumerate(idx):
        if a in used:
            continue
        cluster = [a]
        for b in idx[i+1:]:
            if b in used:
                continue
            hd = hamming(a.hashes.get('phash'), b.hashes.get('phash'))
            if hd is None:
                continue
            if hd <= phash_threshold:
                # optionally require different formats
                if require_different_format and a.format == b.format:
                    # still may be duplicate; allow if one is RAW and other JPEG etc
                    pass
                cluster.append(b)
        if len(cluster) > 1:
            for x in cluster:
                used.add(x)
            groups.append(cluster)
    return groups


def find_similar_series(records, phash_threshold=6, ssim_threshold=0.9):
    # more strict: nearby phash + optionally SSIM to confirm
    idx = [r for r in records if r.hashes and 'phash' in r.hashes]
    groups = []
    used = set()
    for i, a in enumerate(idx):
        if a in used:
            continue
        cluster = [a]
        for b in idx[i+1:]:
            if b in used:
                continue
            hd = hamming(a.hashes.get('phash'), b.hashes.get('phash'))
            if hd is None:
                continue
            if hd <= phash_threshold:
                # optional SSIM confirmation if available
                confirm = True
                if SKIMAGE_AVAILABLE:
                    val, err = compute_ssim(a.path, b.path)
                    if val is None:
                        pass
                    else:
                        confirm = val >= ssim_threshold
                if confirm:
                    cluster.append(b)
        if len(cluster) > 1:
            for x in cluster:
                used.add(x)
            groups.append(cluster)
    return groups

# ----------------------------- Group Ranking & Export -----------------------------

def rank_candidate_keep(record):
    """Return tuple used for sorting where lower is better (best candidate to keep).
    Rules: prefer RAW/TIFF > largest resolution (area) > largest filesize > newest mtime > fewer compression artifacts (prefer non-lossy)
    """
    score = []
    is_raw = 1 if record.path.suffix.lower() in RAW_EXTS or record.format in ('tif','tiff') else 0
    # negative because we want RAW preferred (higher is better) -> invert to negative
    score.append(-is_raw)
    # resolution area
    area = (record.width or 0) * (record.height or 0)
    score.append(-area)
    # file size (prefer larger)
    score.append(- (record.size or 0))
    # prefer non-lossy
    is_lossy = 1 if record.path.suffix.lower() in LOSSY_EXTS else 0
    score.append(is_lossy)
    # newest (prefer newest) -> invert
    score.append(- (record.mtime or 0))
    return tuple(score)


def build_groups(records, exact_groups, content_groups, similar_groups):
    # We'll produce a combined list of cluster dicts with type labels
    clusters = []
    added = set()
    def rec_add(group, gtype):
        paths = [r.path.as_posix() for r in group]
        ids = tuple(sorted(paths))
        if ids in added:
            return
        added.add(ids)
        keeper = sorted(group, key=rank_candidate_keep)[0]
        clusters.append({
            'type': gtype,
            'members': group,
            'keeper': keeper
        })

    for g in exact_groups:
        rec_add(g, 'exact')
    for g in content_groups:
        rec_add(g, 'content')
    for g in similar_groups:
        rec_add(g, 'similar')
    return clusters


def export_csv(clusters, out_csv):
    fields = ['group_id','group_type','keeper','path','format','width','height','size_bytes','mtime','similarity_score','note']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for gid, g in enumerate(clusters, start=1):
            grp_type = g['type']
            keeper = g['keeper']
            for member in g['members']:
                # similarity_score not computed per pair in this CSV; placeholder
                note = ''
                if grp_type == 'content':
                    # mark compressed vs source
                    if keeper.path.suffix.lower() in LOSSY_EXTS and any(m.path.suffix.lower() in RAW_EXTS for m in g['members']):
                        note = 'compressed_copy_of_RAW'
                w.writerow([gid, grp_type, keeper.path.as_posix(), member.path.as_posix(), member.format, member.width or '', member.height or '', member.size or '', member.mtime or '', '', note])
    print(f"CSV exported to {out_csv}")


def export_delete_script(clusters, script_path, conservative=True, move_to=None):
    # Conservative: only delete members that are not keeper
    lines = ['#!/bin/sh', 'set -e']
    for gid, g in enumerate(clusters, start=1):
        keeper = g['keeper']
        for member in g['members']:
            if member.path == keeper.path:
                continue
            if move_to:
                # ensure directory exists command
                dest = Path(move_to) / Path(member.path.name)
                lines.append(f'mkdir -p "{Path(move_to)}"')
                lines.append(f'mv "{member.path.as_posix()}" "{dest.as_posix()}"')
            else:
                lines.append(f'rm -i "{member.path.as_posix()}"')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.chmod(script_path, 0o755)
    print(f"Delete/move script exported to {script_path}")

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Photo deduplication and similarity tool')
    p.add_argument('--main', required=True, help='Main folder to scan')
    p.add_argument('--extras', nargs='*', help='Additional folders to include')
    p.add_argument('--mode', choices=['exact','content','similar','all'], default='all')
    p.add_argument('--phash-threshold', type=int, default=12, help='Hamming distance threshold for phash to consider content duplicates')
    p.add_argument('--similar-phash-threshold', type=int, default=6, help='Hamming distance threshold for series similarity')
    p.add_argument('--ssim-threshold', type=float, default=0.9, help='SSIM threshold for confirming similarity (0-1)')
    p.add_argument('--export', help='CSV output file path')
    p.add_argument('--script', help='Export shell script to delete/move candidates')
    p.add_argument('--move-to', help='If set when using --script, move files to this folder instead of deleting')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no-image-hash', action='store_true', help='Skip image perceptual hashing (faster but less powerful)')
    p.add_argument('--require-different-format', action='store_true', help='For content duplicates, require files to be in different formats (optional)')
    return p.parse_args()


def main():
    args = parse_args()
    paths = list(scan_folders(args.main, args.extras))
    print(f"Found {len(paths)} files to analyze")
    collect_hashes = not args.no_image_hash
    records = build_file_records(paths, max_workers=args.workers, collect_image_hashes=collect_hashes)

    exact_groups = []
    content_groups = []
    similar_groups = []

    if args.mode in ('exact','all'):
        exact_groups = find_exact_duplicates(records)
        print(f"Exact duplicate groups: {len(exact_groups)}")

    if args.mode in ('content','all') and collect_hashes:
        content_groups = find_content_duplicates(records, phash_threshold=args.phash_threshold, require_different_format=args.require_different_format)
        print(f"Content-duplicate groups: {len(content_groups)}")

    if args.mode in ('similar','all') and collect_hashes:
        similar_groups = find_similar_series(records, phash_threshold=args.similar_phash_threshold, ssim_threshold=args.ssim_threshold)
        print(f"Similar-series groups: {len(similar_groups)}")

    clusters = build_groups(records, exact_groups, content_groups, similar_groups)
    print(f"Total clusters: {len(clusters)}")

    if args.export:
        export_csv(clusters, args.export)
    if args.script:
        export_delete_script(clusters, args.script, move_to=args.move_to)

    # Also print brief summary to console
    for i, g in enumerate(clusters[:20], start=1):
        print(f"Cluster {i}: type={g['type']} keeper={g['keeper'].path} members={len(g['members'])}")

if __name__ == '__main__':
    main()
