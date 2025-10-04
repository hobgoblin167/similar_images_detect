#!/usr/bin/env python3
"""
photo_dedupe_server.py
Single-file Flask + image-comparison program.

Features:
- Serve an HTML upload UI (embedded).
- Accept .zip uploads (streamed to disk).
- Safe unzip into temporary workdir.
- Scan recursively, compute SHA256 (exact duplicates), perceptual hashes (phash/dhash/ahash)
  and optional SSIM confirmation.
- Group exact/content/similar duplicates, choose 'keeper' per group by ranking rules.
- Export CSV report and optional shell delete/move script.
- Status polling via /status/<task_id> and file download endpoints.

Important notes:
- Flask's MAX_CONTENT_LENGTH is left None here; for production configure reverse proxy (nginx) limits.
- For RAW support install rawpy. For SSIM install scikit-image.
- This script attempts to avoid loading entire uploads to memory by streaming to disk.

REQUIREMENTS:
pip install flask pillow imagehash piexif tqdm numpy
# Optional (recommended for RAW and SSIM):
pip install rawpy scikit-image opencv-python
"""

import os
import io
import zipfile
import tempfile
import shutil
import threading
import json
import uuid
import hashlib
import csv
import math
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from flask import Flask, request, jsonify, send_file, abort, Response
from werkzeug.utils import secure_filename
# Image libraries
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

# ---------- Configuration ----------
UPLOAD_DIR = Path(tempfile.gettempdir()) / "photo_dedupe_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = UPLOAD_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.zip'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.cr2', '.nef', '.arw', '.raf', '.rw2'}
RAW_EXTS = {'.cr2', '.nef', '.arw', '.raf', '.rw2'}
LOSSY_EXTS = {'.jpg', '.jpeg', '.webp'}
LOSSLESS_EXTS = {'.png', '.tif', '.tiff'}

# Executor for background tasks
EXECUTOR = ThreadPoolExecutor(max_workers=2)

# In-memory task store: for demo only. Format:
# TASKS[task_id] = {
#   status: 'queued'|'extracting'|'scanning'|'clustering'|'done'|'error',
#   progress: 0-100,
#   msg: optional string,
#   summary: {...},
#   csv_path: str or None,
#   script_path: str or None,
#   workdir: str path
# }
TASKS = {}

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None  # rely on reverse proxy for limits in prod

# ---------- Helper / Safety utilities ----------

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def secure_stream_save(file_storage, dest_path: Path):
    """Save uploaded FileStorage to dest_path in streaming mode (no full-memory)."""
    with dest_path.open('wb') as dst:
        chunk_size = 64 * 1024
        stream = file_storage.stream
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)

def safe_extract_zip(zip_path: Path, extract_to: Path):
    """Safely extract zip: prevent path traversal; return list of extracted file Paths."""
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            # Skip directories
            if member.is_dir():
                continue
            member_name = member.filename
            # Skip absolute paths and path traversal
            if os.path.isabs(member_name) or '..' in Path(member_name).parts:
                # skip suspicious
                continue
            target = extract_to / member_name
            target_parent = target.parent
            target_parent.mkdir(parents=True, exist_ok=True)
            # Extract member to a safe temporary file then move (ZipFile supports extract)
            with z.open(member, 'r') as source, open(target, 'wb') as dest:
                shutil.copyfileobj(source, dest)
            extracted.append(target)
    return extracted

def human_size(num_bytes):
    for unit in ['B','KB','MB','GB','TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"

# ---------- Image reading & hashing ----------

def load_image_for_hash(path: Path, resize_for_hash=1024):
    """Load image suitable for imagehash. Convert RAW using rawpy if available."""
    suffix = path.suffix.lower()
    try:
        if RAWPY_AVAILABLE and suffix in RAW_EXTS:
            raw = rawpy.imread(path.as_posix())
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            raw.close()
            img = Image.fromarray(rgb)
        else:
            img = Image.open(path)
            img.load()
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
    except UnidentifiedImageError:
        raise
    except Exception as e:
        raise

    # normalize orientation using EXIF if possible
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

    maxdim = max(img.size)
    if maxdim > resize_for_hash:
        scale = resize_for_hash / float(maxdim)
        new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img

def compute_hashes(path: Path, hash_funcs=('phash','dhash','ahash')):
    """Return dict of hashes or raise."""
    results = {}
    img = load_image_for_hash(path)
    if 'phash' in hash_funcs:
        results['phash'] = imagehash.phash(img)
    if 'dhash' in hash_funcs:
        results['dhash'] = imagehash.dhash(img)
    if 'ahash' in hash_funcs:
        results['ahash'] = imagehash.average_hash(img)
    try:
        results['hist'] = tuple(img.convert('RGB').histogram())
    except Exception:
        results['hist'] = None
    results['size'] = img.size
    return results

def compute_ssim(path1: Path, path2: Path, resize_to=800):
    """Return SSIM float 0..1 or None and optional error message."""
    if not SKIMAGE_AVAILABLE:
        return None, 'skimage not available'
    try:
        i1 = load_image_for_hash(path1, resize_for_hash=resize_to).convert('L')
        i2 = load_image_for_hash(path2, resize_for_hash=resize_to).convert('L')
    except Exception as e:
        return None, f'load_error: {e}'
    a1 = np.array(i1)
    a2 = np.array(i2)
    if a1.shape != a2.shape:
        i2 = Image.fromarray(a2).resize(i1.size, Image.LANCZOS)
        a2 = np.array(i2)
    try:
        val = ssim(a1, a2)
        return float(val), None
    except Exception as e:
        return None, str(e)

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
        try:
            ex = piexif.load(path.as_posix())
            data['piexif'] = ex
        except Exception:
            pass
        img.close()
    except UnidentifiedImageError:
        if RAWPY_AVAILABLE and path.suffix.lower() in RAW_EXTS:
            try:
                r = rawpy.imread(path.as_posix())
                data['raw'] = True
                r.close()
            except Exception:
                pass
    except Exception:
        pass
    return data

# ---------- Record class & scanning pipeline ----------

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
        try:
            h = compute_hashes(self.path, funcs)
            self.hashes = h
            if 'size' in h:
                self.width, self.height = h['size']
        except Exception as e:
            self.hash_err = str(e)

def build_file_records(paths, task_id=None, max_workers=8, collect_image_hashes=True, hash_funcs=('phash','dhash','ahash')):
    records = []
    total = len(paths)
    if task_id:
        TASKS[task_id].setdefault('progress', 0)
    def worker(p):
        r = FileRecord(p)
        r.collect_basic()
        r.collect_sha()
        r.collect_exif()
        if collect_image_hashes and p.suffix.lower() in IMAGE_EXTS:
            r.collect_hashes(hash_funcs)
        return r

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, p): p for p in paths}
        completed = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
                records.append(r)
            except Exception as e:
                # skip failing files
                pass
            completed += 1
            if task_id:
                TASKS[task_id]['progress'] = int(10 + 70 * (completed / total))  # scanning progress chunk
    return records

# ---------- Comparison & clustering ----------

def hamming(h1, h2):
    if h1 is None or h2 is None:
        return None
    try:
        return int(h1 - h2)
    except Exception:
        return None

def find_exact_duplicates(records):
    byhash = defaultdict(list)
    for r in records:
        if r.sha256:
            byhash[r.sha256].append(r)
    groups = [group for group in byhash.values() if len(group) > 1]
    return groups

def find_content_duplicates(records, phash_threshold=10, require_different_format=True):
    idx = [r for r in records if r.hashes and 'phash' in r.hashes]
    groups = []
    used = set()
    for i, a in enumerate(idx):
        if a.path.as_posix() in used:
            continue
        cluster = [a]
        for b in idx[i+1:]:
            if b.path.as_posix() in used:
                continue
            hd = hamming(a.hashes.get('phash'), b.hashes.get('phash'))
            if hd is None:
                continue
            if hd <= phash_threshold:
                if require_different_format and a.format == b.format:
                    # allow if one is RAW and other JPEG, else still include (configurable)
                    if not (a.path.suffix.lower() in RAW_EXTS and b.path.suffix.lower() not in RAW_EXTS) and not (b.path.suffix.lower() in RAW_EXTS and a.path.suffix.lower() not in RAW_EXTS):
                        # same format and not RAW<->other -> still consider but this branch can be changed
                        pass
                cluster.append(b)
        if len(cluster) > 1:
            for x in cluster:
                used.add(x.path.as_posix())
            groups.append(cluster)
    return groups

def find_similar_series(records, phash_threshold=6, ssim_threshold=0.9):
    idx = [r for r in records if r.hashes and 'phash' in r.hashes]
    groups = []
    used = set()
    for i, a in enumerate(idx):
        if a.path.as_posix() in used:
            continue
        cluster = [a]
        for b in idx[i+1:]:
            if b.path.as_posix() in used:
                continue
            hd = hamming(a.hashes.get('phash'), b.hashes.get('phash'))
            if hd is None:
                continue
            if hd <= phash_threshold:
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
                used.add(x.path.as_posix())
            groups.append(cluster)
    return groups

def rank_candidate_keep(record):
    """Tuple with values where smaller is better."""
    is_raw = 1 if record.path.suffix.lower() in RAW_EXTS or record.format in ('tif','tiff') else 0
    area = (record.width or 0) * (record.height or 0)
    size = record.size or 0
    is_lossy = 1 if record.path.suffix.lower() in LOSSY_EXTS else 0
    # prefer raw (higher is better) -> invert to negative so smaller tuple preferred
    return (-is_raw, -area, -size, is_lossy, - (record.mtime or 0))

def build_groups(records, exact_groups, content_groups, similar_groups):
    clusters = []
    added = set()
    def rec_add(group, gtype):
        ids = tuple(sorted([r.path.as_posix() for r in group]))
        if ids in added:
            return
        added.add(ids)
        keeper = sorted(group, key=rank_candidate_keep)[0]
        clusters.append({'type': gtype, 'members': group, 'keeper': keeper})
    for g in exact_groups:
        rec_add(g, 'exact')
    for g in content_groups:
        rec_add(g, 'content')
    for g in similar_groups:
        rec_add(g, 'similar')
    return clusters

# ---------- Export helpers ----------

def export_csv(clusters, out_csv_path: Path):
    fields = ['group_id','group_type','keeper','path','format','width','height','size_bytes','mtime','similarity_score','note']
    with out_csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for gid, g in enumerate(clusters, start=1):
            grp_type = g['type']
            keeper = g['keeper']
            for member in g['members']:
                note = ''
                if grp_type == 'content':
                    if keeper.path.suffix.lower() in LOSSY_EXTS and any(m.path.suffix.lower() in RAW_EXTS for m in g['members']):
                        note = 'compressed_copy_of_RAW'
                w.writerow([gid, grp_type, keeper.path.as_posix(), member.path.as_posix(), member.format, member.width or '', member.height or '', member.size or '', member.mtime or '', '', note])
    return out_csv_path

def export_delete_script(clusters, script_path: Path, conservative=True, move_to: Path=None):
    lines = ['#!/bin/sh', 'set -euo pipefail']
    for gid, g in enumerate(clusters, start=1):
        keeper = g['keeper']
        for member in g['members']:
            if member.path == keeper.path:
                continue
            if move_to:
                dest_dir = move_to
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / member.path.name
                lines.append(f'mkdir -p "{dest_dir.as_posix()}"')
                lines.append(f'mv "{member.path.as_posix()}" "{dest.as_posix()}"')
            else:
                # interactive delete as safer default
                lines.append(f'rm -i "{member.path.as_posix()}"')
    script_path.write_text("\n".join(lines), encoding='utf-8')
    script_path.chmod(0o755)
    return script_path

# ---------- Task processing ----------

def process_zip_task(task_id: str, zip_path: Path, options: dict):
    TASKS[task_id] = {'status': 'extracting', 'progress': 1, 'msg': 'extracting', 'workdir': None}
    workdir = Path(tempfile.mkdtemp(prefix=f"photo_dedupe_{task_id}_"))
    try:
        # Safe extract
        extracted = safe_extract_zip(zip_path, workdir)
        TASKS[task_id]['progress'] = 5
        TASKS[task_id]['workdir'] = workdir.as_posix()
        TASKS[task_id]['status'] = 'scanning'
        TASKS[task_id]['msg'] = 'scanning files'

        # collect files recursively
        all_files = [p for p in workdir.rglob('*') if p.is_file()]
        TASKS[task_id]['summary'] = {'total_files': len(all_files)}
        # scanning and hashing
        collect_hashes = not options.get('no_image_hash', False)
        workers = int(options.get('workers', 8))
        records = build_file_records(all_files, task_id=task_id, max_workers=workers, collect_image_hashes=collect_hashes)

        TASKS[task_id]['progress'] = 80
        TASKS[task_id]['status'] = 'clustering'
        TASKS[task_id]['msg'] = 'finding duplicates'

        # find groups according to requested mode
        mode = options.get('mode', 'all')
        phash_thr = int(options.get('phash_threshold', 12))
        similar_phash_thr = int(options.get('similar_phash_threshold', 6))
        ssim_thr = float(options.get('ssim_threshold', 0.9))
        require_diff_format = bool(options.get('require_different_format', True))

        exact_groups = []
        content_groups = []
        similar_groups = []

        if mode in ('exact','all'):
            exact_groups = find_exact_duplicates(records)
        if mode in ('content','all') and collect_hashes:
            content_groups = find_content_duplicates(records, phash_threshold=phash_thr, require_different_format=require_diff_format)
        if mode in ('similar','all') and collect_hashes:
            similar_groups = find_similar_series(records, phash_threshold=similar_phash_thr, ssim_threshold=ssim_thr)

        clusters = build_groups(records, exact_groups, content_groups, similar_groups)
        TASKS[task_id]['progress'] = 95
        TASKS[task_id]['status'] = 'exporting'
        TASKS[task_id]['msg'] = 'exporting results'

        # export CSV and script
        ts = int(time.time())
        csv_path = RESULTS_DIR / f"photo_dedupe_{task_id}_{ts}.csv"
        script_path = RESULTS_DIR / f"photo_dedupe_{task_id}_{ts}.sh"
        export_csv(clusters, csv_path)
        export_delete_script(clusters, script_path, conservative=True, move_to=None)

        # minimal summary for UI
        TASKS[task_id].update({
            'status': 'done',
            'progress': 100,
            'msg': 'done',
            'summary': {
                'total_files': len(records),
                'exact_groups': len(exact_groups),
                'content_groups': len(content_groups),
                'similar_groups': len(similar_groups),
                'total_clusters': len(clusters),
            },
            'csv_path': csv_path.as_posix(),
            'script_path': script_path.as_posix(),
            'clusters_sample': [
                {
                    'type': c['type'],
                    'keeper': c['keeper'].path.as_posix(),
                    'members': [m.path.as_posix() for m in c['members']]
                } for c in clusters[:50]
            ]
        })
    except Exception as e:
        TASKS[task_id] = {'status': 'error', 'progress': 100, 'msg': str(e)}
    finally:
        # remove uploaded zip to save space
        try:
            zip_path.unlink()
        except Exception:
            pass

# ---------- Routes (simple UI embedded) ----------

INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Photo Deduper — Upload ZIP</title>
  <style>
    body { font-family: Inter, system-ui, -apple-system, Arial, sans-serif; max-width:1000px; margin:2rem auto; padding:1rem; color:#111; background:#f7fafc; }
    .card{ background:#fff; border:1px solid #e6e9ef; padding:1.2rem; border-radius:10px; box-shadow:0 6px 18px rgba(20,20,50,0.04); }
    form > * { display:block; margin:0.6rem 0; }
    label.inline { display:inline-block; margin-right:1rem; }
    button { padding:0.6rem 1rem; border-radius:8px; background:#2563eb; color:#fff; border:none; cursor:pointer;}
    button:disabled{ background:#9fb9ff; cursor:default;}
    progress{ width:100%; height:14px; border-radius:8px; }
    pre { background:#0f172a; color:#e6eef8; padding:0.8rem; border-radius:8px; overflow:auto; }
    .muted { color:#6b7280; font-size:0.9rem; }
    .small { font-size:0.9rem; }
  </style>
</head>
<body>
  <h2>Photo Deduper — загрузка ZIP архива</h2>
  <div class="card">
    <form id="uploadForm">
      <label>ZIP файл с фотоархивом:
        <input type="file" id="fileInput" name="file" accept=".zip" required>
      </label>

      <div>
        <label class="inline"><input type="checkbox" name="mode_exact" checked> точные дубликаты</label>
        <label class="inline"><input type="checkbox" name="mode_content" checked> содержательные дубли (phash)</label>
        <label class="inline"><input type="checkbox" name="mode_similar"> схожие серии</label>
      </div>

      <div>
        <label>phash threshold: <input type="number" id="phash" value="12" min="0" max="64"></label>
        <label>similar phash threshold: <input type="number" id="sim_phash" value="6" min="0" max="64"></label>
        <label>SSIM threshold: <input type="number" step="0.01" id="ssim" value="0.90" min="0" max="1"></label>
      </div>

      <div class="small muted">Файлы загружаются на сервер и обрабатываются фоново. По завершении вы сможете скачать CSV и скрипт удаления (скрипт не выполняется автоматически).</div>

      <p>
        <button type="submit" id="submitBtn">Загрузить и начать анализ</button>
      </p>
    </form>

    <div id="status" style="margin-top:1rem"></div>
    <progress id="prog" value="0" max="100" style="display:none"></progress>

    <div id="result" style="margin-top:1rem"></div>
  </div>

<script>
const form = document.getElementById('uploadForm');
const prog = document.getElementById('prog');
const statusDiv = document.getElementById('status');
const resultDiv = document.getElementById('result');
const submitBtn = document.getElementById('submitBtn');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) { alert('Выберите ZIP файл'); return; }
  const file = fileInput.files[0];

  const options = {
    phash_threshold: Number(document.getElementById('phash').value || 12),
    similar_phash_threshold: Number(document.getElementById('sim_phash').value || 6),
    ssim_threshold: Number(document.getElementById('ssim').value || 0.9),
    mode: (document.querySelector('input[name="mode_similar"]').checked ? 'all' : (document.querySelector('input[name="mode_content"]').checked && document.querySelector('input[name="mode_exact"]').checked ? 'all' : (document.querySelector('input[name="mode_content"]').checked ? 'content' : (document.querySelector('input[name="mode_exact"]').checked ? 'exact' : 'all'))))
  };

  const formData = new FormData();
  formData.append('file', file, file.name);
  formData.append('options', JSON.stringify(options));

  submitBtn.disabled = true;
  statusDiv.innerText = 'Начало загрузки...';
  prog.style.display = 'block'; prog.value = 0;

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload');
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) {
      prog.value = Math.round(ev.loaded/ev.total*100);
    }
  };
  xhr.onload = () => {
    submitBtn.disabled = false;
    if (xhr.status === 202) {
      const data = JSON.parse(xhr.responseText);
      statusDiv.innerText = 'Задача создана: ' + data.task_id;
      pollStatus(data.task_id);
    } else {
      statusDiv.innerText = 'Ошибка загрузки: ' + xhr.responseText;
    }
  };
  xhr.onerror = () => {
    submitBtn.disabled = false;
    statusDiv.innerText = 'Network error';
  };
  xhr.send(formData);
});

function pollStatus(taskId){
  statusDiv.innerText = 'Ожидание результата... (task: ' + taskId + ')';
  resultDiv.innerHTML = '';
  const iv = setInterval(async () => {
    try {
      const r = await fetch('/status/' + taskId);
      if (!r.ok) throw new Error('status fetch failed');
      const j = await r.json();
      statusDiv.innerText = `Статус: ${j.status} — ${j.msg || ''}`;
      prog.style.display = 'block';
      prog.value = j.progress || 0;
      if (j.status === 'done') {
        clearInterval(iv);
        prog.value = 100;
        let html = `<div class="small">Готово. Файлов: ${j.summary?.total_files || 'n/a'}. Кластеров: ${j.summary?.total_clusters || 'n/a'}.</div>`;
        if (j.csv_path) html += `<p><a href="/download_csv/${taskId}">Скачать CSV</a> &nbsp; <a href="/download_script/${taskId}">Скачать скрипт удаления</a></p>`;
        if (j.clusters_sample) {
          html += `<details><summary>Превью найденных групп (${j.clusters_sample.length})</summary><pre>${JSON.stringify(j.clusters_sample, null, 2)}</pre></details>`;
        }
        resultDiv.innerHTML = html;
      } else if (j.status === 'error') {
        clearInterval(iv);
        resultDiv.innerHTML = `<div class="small">Ошибка: ${j.msg || 'unknown'}</div>`;
      }
    } catch (err) {
      console.error(err);
    }
  }, 2000);
}
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return Response(INDEX_HTML, mimetype='text/html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'only .zip allowed'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    dest = UPLOAD_DIR / filename
    try:
        secure_stream_save(file, dest)
    except Exception as e:
        return jsonify({'error': f'failed saving upload: {e}'}), 500

    # parse options if present
    options = {}
    opts_raw = request.form.get('options')
    if opts_raw:
        try:
            options = json.loads(opts_raw)
        except Exception:
            options = {}

    task_id = uuid.uuid4().hex
    TASKS[task_id] = {'status': 'queued', 'progress': 0, 'msg': 'queued'}
    # schedule background processing
    EXECUTOR.submit(process_zip_task, task_id, dest, options)
    return jsonify({'task_id': task_id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def status(task_id):
    t = TASKS.get(task_id)
    if not t:
        return jsonify({'error': 'not found'}), 404
    # Return a copy with limited fields
    safe = {
        'status': t.get('status'),
        'progress': t.get('progress', 0),
        'msg': t.get('msg'),
        'summary': t.get('summary'),
        'csv_path': True if t.get('csv_path') else False,
        'script_path': True if t.get('script_path') else False,
        'clusters_sample': t.get('clusters_sample')
    }
    # For convenience include raw paths for download endpoints
    if t.get('csv_path'):
        safe['csv_path'] = True
    return jsonify(safe)

@app.route('/download_csv/<task_id>', methods=['GET'])
def download_csv(task_id):
    t = TASKS.get(task_id)
    if not t or not t.get('csv_path'):
        return jsonify({'error': 'csv not available'}), 404
    p = Path(t['csv_path'])
    if not p.exists():
        return jsonify({'error': 'csv missing'}), 404
    return send_file(p.as_posix(), as_attachment=True, download_name=p.name)

@app.route('/download_script/<task_id>', methods=['GET'])
def download_script(task_id):
    t = TASKS.get(task_id)
    if not t or not t.get('script_path'):
        return jsonify({'error': 'script not available'}), 404
    p = Path(t['script_path'])
    if not p.exists():
        return jsonify({'error': 'script missing'}), 404
    return send_file(p.as_posix(), as_attachment=True, download_name=p.name)

@app.route('/cleanup_old', methods=['POST','GET'])
def cleanup_old():
    """Very simple cleanup endpoint to remove old result files and uploads older than X seconds."""
    keep_seconds = int(request.args.get('keep_seconds', 60*60*24*2))  # default 2 days
    now = time.time()
    removed = 0
    for p in list(UPLOAD_DIR.iterdir()):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > keep_seconds:
                p.unlink()
                removed += 1
        except Exception:
            pass
    for p in list(RESULTS_DIR.iterdir()):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > keep_seconds:
                p.unlink()
                removed += 1
        except Exception:
            pass
    return jsonify({'removed': removed})

if __name__ == '__main__':
    print("Starting Photo Deduper server on http://0.0.0.0:5000")
    print("Note: For production, run behind nginx/gunicorn and configure upload limits appropriately.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
