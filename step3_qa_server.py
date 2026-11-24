"""
Step 3: Interactive QA & Manual Label Editor Web UI Server
- Fast multi-threaded HTTP server with in-memory index for 32K+ images
- Visual QA + Manual Bounding Box Editor:
  * Click & drag on canvas to draw new drone boxes
  * Click box to select -> Delete/Backspace to remove wrong box
  * Brightness/Contrast boost (B) for dark / tiny drone images
  * Save labels (S) -> writes YOLO .txt, backs up old label, logs manual_fixed
- Filters: Category / Error Audit Severity / Class / QA Status / Split / Search
- Before/After comparison (Side-by-side & Overlay)
- Shortcuts: Left/Right, A (Approve), F (Flag), S (Save), B (Brighten), Del (Delete box)
"""

import os
import sys
import json
import gzip
import csv
import shutil
import http.server
import urllib.parse
import mimetypes
import threading
from datetime import datetime
from pathlib import Path

import argparse

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--dir", default=r"e:\trend_AI\clean_dataset")
_p.add_argument("--port", type=int, default=8899)
_args, _ = _p.parse_known_args()

DATASET_DIR = Path(_args.dir)
AUDIT_LOG = DATASET_DIR / "relabel_audit_log.csv"
SUSPICIOUS_CSV = DATASET_DIR / "suspicious_labels_audit.csv"
BACKUP_DIR = DATASET_DIR / "labels_backup"
QA_STATUS_FILE = DATASET_DIR / "qa_review_status.json"
PORT = _args.port

STATUS_LOCK = threading.Lock()
IMAGES_INDEX = []
IMAGES_BY_NAME = {}
QA_STATUS = {}


def load_audit_log():
    audit = {}
    if AUDIT_LOG.exists():
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                audit[row["image"]] = row
    return audit


def load_suspicious_audit():
    susp = {}
    if SUSPICIOUS_CSV.exists():
        with open(SUSPICIOUS_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                susp[row["image"]] = {
                    "severity": row.get("severity", "ok"),
                    "reasons": row.get("reasons", ""),
                }
    return susp


def load_qa_status():
    if QA_STATUS_FILE.exists():
        try:
            with open(QA_STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_qa_status(status):
    with open(QA_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def read_yolo_boxes(lbl_path):
    boxes = []
    if lbl_path.exists():
        with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append({
                        "class": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                    })
    return boxes


def build_index():
    global IMAGES_INDEX, IMAGES_BY_NAME, QA_STATUS
    print("Building image index...")
    audit = load_audit_log()
    susp_map = load_suspicious_audit()
    QA_STATUS = load_qa_status()
    items = []
    by_name = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        if not img_dir.exists():
            continue
        names = sorted(
            entry.name for entry in os.scandir(img_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts
        )
        for name in names:
            audit_info = audit.get(name)
            susp_info = susp_map.get(name)
            category = "original"
            num_boxes = 1
            if audit_info:
                action = audit_info.get("action", "")
                if action == "auto_labeled":
                    category = "auto_labeled"
                elif action == "manual_fixed":
                    category = "manual_fixed"
                else:
                    category = "relabeled"
                try:
                    num_boxes = int(audit_info.get("final_boxes", 1))
                except ValueError:
                    num_boxes = 1

            classes_present = [0] if num_boxes > 0 else []
            entry = {
                "name": name,
                "split": split,
                "category": category,
                "qa_status": QA_STATUS.get(name, "pending"),
                "num_boxes": num_boxes,
                "classes": classes_present,
                "changes": audit_info.get("changes", "") if audit_info else "",
                "severity": susp_info["severity"] if susp_info else "ok",
                "reasons": susp_info["reasons"] if susp_info else "",
            }
            items.append(entry)
            by_name[name] = entry

    IMAGES_INDEX = items
    IMAGES_BY_NAME = by_name
    print(f"Indexed {len(IMAGES_INDEX)} images.")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔍 QA & Manual Label Editor - Drone Dataset</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; user-select: none; }

.header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 10px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #0f3460;
}
.header h1 { font-size: 1.2em; color: #00d2ff; }
.header .stats { font-size: 0.84em; color: #aaa; }
.header .stats span { color: #00d2ff; font-weight: bold; }

.toolbar {
    background: #1a1a1a;
    padding: 8px 22px;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid #333;
}
.toolbar label { font-size: 0.8em; color: #aaa; margin-right: 3px; }
.toolbar select, .toolbar input {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #fff;
    padding: 4px 9px;
    border-radius: 6px;
    font-size: 0.82em;
}
.toolbar select:focus, .toolbar input:focus { border-color: #00d2ff; outline: none; }

.main-container {
    display: flex;
    height: calc(100vh - 158px);
}

.sidebar-wrap {
    width: 295px;
    background: #141414;
    border-right: 1px solid #333;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}
.sidebar-pager {
    padding: 7px 12px;
    background: #1c1c1c;
    border-bottom: 1px solid #2b2b2b;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78em;
    color: #aaa;
}
.sidebar-pager button {
    background: #2c2c2c; border: 1px solid #444; color: #ddd;
    padding: 3px 8px; border-radius: 4px; cursor: pointer;
}
.sidebar-pager button:disabled { opacity: 0.4; cursor: default; }
.sidebar { flex: 1; overflow-y: auto; }
.sidebar-item {
    padding: 8px 12px;
    border-bottom: 1px solid #222;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-item:hover { background: #1f1f1f; }
.sidebar-item.active { background: #0f3460; border-left: 3px solid #00d2ff; }
.sidebar-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sidebar-item .dot.pending { background: #666; }
.sidebar-item .dot.approved { background: #00e676; }
.sidebar-item .dot.flagged { background: #ff5252; }
.sidebar-item .info { flex: 1; min-width: 0; }
.sidebar-item .info .name { font-size: 0.79em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sidebar-item .info .meta { font-size: 0.69em; color: #888; }
.sidebar-item .badge { font-size: 0.64em; padding: 2px 5px; border-radius: 3px; flex-shrink: 0; }
.badge.relabeled { background: #e65100; color: #fff; }
.badge.auto_labeled { background: #1565c0; color: #fff; }
.badge.manual_fixed { background: #2e7d32; color: #fff; }
.badge.original { background: #333; color: #aaa; }

.viewer {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px;
    overflow: auto;
}
.edit-hint {
    font-size: 0.78em;
    color: #90caf9;
    margin-bottom: 6px;
    background: #102030;
    border: 1px solid #1e3a5f;
    padding: 4px 12px;
    border-radius: 6px;
}
.canvas-container {
    display: flex;
    gap: 18px;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
}
.canvas-wrapper { text-align: center; }
.canvas-wrapper h3 { font-size: 0.86em; color: #aaa; margin-bottom: 5px; }
canvas {
    max-width: 100%;
    max-height: calc(100vh - 300px);
    border: 2px solid #333;
    border-radius: 8px;
    cursor: crosshair;
}

.controls {
    padding: 10px 20px;
    background: #1a1a1a;
    border-top: 1px solid #333;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.controls button {
    padding: 8px 16px;
    border: none;
    border-radius: 7px;
    font-size: 0.86em;
    font-weight: 600;
    cursor: pointer;
}
.btn-prev, .btn-next { background: #333; color: #fff; }
.btn-approve { background: #00c853; color: #fff; }
.btn-flag { background: #ff1744; color: #fff; }
.btn-save { background: #2979ff; color: #fff; }
.btn-save.dirty { background: #ff9100; animation: pulse 1.2s infinite; }
.btn-clear { background: #6d1b1b; color: #ffcdd2; }
.btn-bright { background: #37474f; color: #fff; }
.btn-bright.active { background: #f9a825; color: #000; }
.btn-active { box-shadow: 0 0 0 3px rgba(0,210,255,0.65); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.78; } }

.kbd {
    display: inline-block; background: #222; color: #ccc; padding: 1px 5px;
    border-radius: 3px; font-size: 0.72em; margin-left: 3px; font-family: monospace;
}
.image-info { margin-top: 6px; text-align: center; font-size: 0.82em; color: #aaa; }
.image-info .changes { margin-top: 3px; color: #ff9800; font-size: 0.78em; max-width: 900px; }

.toggle-group { display: flex; gap: 4px; }
.toggle-btn {
    padding: 4px 10px; background: #2a2a2a; border: 1px solid #444;
    color: #aaa; border-radius: 6px; cursor: pointer; font-size: 0.79em;
}
.toggle-btn.active { background: #0f3460; border-color: #00d2ff; color: #00d2ff; }
.progress-bar { width: 140px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
.progress-bar .fill { height: 100%; background: linear-gradient(90deg, #00d2ff, #00e676); }
.empty-state { text-align: center; color: #666; padding: 40px; }
</style>
</head>
<body>

<div class="header">
    <h1>🔍 QA & Manual Label Editor — Drone Dataset</h1>
    <div class="stats">
        Đang xem: <span id="stat-current">0</span> / <span id="stat-filtered">0</span> (Tổng: <span id="stat-total">0</span>) |
        ✅ <span id="stat-approved">0</span>
        🚩 <span id="stat-flagged">0</span>
        ⏳ <span id="stat-pending">0</span>
        <div class="progress-bar" style="display:inline-block; vertical-align:middle; margin-left:8px;">
            <div class="fill" id="progress-fill" style="width:0%"></div>
        </div>
    </div>
</div>

<div class="toolbar">
    <div>
        <label>Loại:</label>
        <select id="filter-category">
            <option value="all">Tất cả</option>
            <option value="manual_fixed">✏️ Đã sửa thủ công</option>
            <option value="relabeled">🔧 Đã vẽ lại / sửa nhãn</option>
            <option value="auto_labeled">🆕 Mới thêm (Auto-label)</option>
            <option value="original">📋 Nhãn gốc</option>
        </select>
    </div>
    <div>
        <label>Kiểm toán lỗi:</label>
        <select id="filter-severity">
            <option value="all">Tất cả</option>
            <option value="critical">🔴 CRITICAL (Sai tâm ngắm / Tòa nhà)</option>
            <option value="high">🟠 HIGH (Độ tin cậy thấp / Nghi sai)</option>
            <option value="medium">🟡 MEDIUM (Nhỏ / Cần xem)</option>
            <option value="any_susp">⚠️ Tất cả ảnh nghi vấn</option>
            <option value="ok">✅ Ảnh Sạch</option>
        </select>
    </div>
    <div>
        <label>Class:</label>
        <select id="filter-class">
            <option value="all">Tất cả</option>
            <option value="0">0: drone</option>
            <option value="empty">0 box</option>
        </select>
    </div>
    <div>
        <label>Trạng thái:</label>
        <select id="filter-status">
            <option value="all">Tất cả</option>
            <option value="pending">⏳ Chưa duyệt</option>
            <option value="approved">✅ Đã duyệt</option>
            <option value="flagged">🚩 Cần sửa</option>
        </select>
    </div>
    <div>
        <label>Split:</label>
        <select id="filter-split">
            <option value="all">Tất cả</option>
            <option value="train">Train</option>
            <option value="val">Val</option>
            <option value="test">Test</option>
        </select>
    </div>
    <div>
        <input type="text" id="search-box" placeholder="Tìm tên ảnh..." style="width:130px;" />
    </div>
    <div class="toggle-group">
        <button class="toggle-btn active" id="view-current" onclick="setViewMode('current')">Vẽ / Sửa nhãn</button>
        <button class="toggle-btn" id="view-compare" onclick="setViewMode('compare')">Song song Trước/Sau</button>
        <button class="toggle-btn" id="view-overlay" onclick="setViewMode('overlay')">Đè Nhãn cũ</button>
    </div>
</div>

<div class="main-container">
    <div class="sidebar-wrap">
        <div class="sidebar-pager">
            <button id="page-prev" onclick="changePage(-1)">◀</button>
            <span id="page-info">Trang 1/1</span>
            <button id="page-next" onclick="changePage(1)">▶</button>
        </div>
        <div class="sidebar" id="sidebar"></div>
    </div>
    <div class="viewer" id="viewer">
        <div class="edit-hint">
            ✏️ <strong>Chỉnh sửa thủ công:</strong> Kéo chuột trên ảnh để vẽ khung <code>drone</code> mới | Click vào khung để chọn rồi nhấn <span class="kbd">Del</span> để xóa | Nhấn <span class="kbd">B</span> tăng sáng ảnh tối | Nhấn <span class="kbd">S</span> để lưu nhãn!
        </div>
        <div class="empty-state" id="empty-state">
            <h2>📂 Đang tải danh sách ảnh...</h2>
        </div>
        <div class="canvas-container" id="canvas-container" style="display:none;"></div>
        <div class="image-info" id="image-info" style="display:none;"></div>
    </div>
</div>

<div class="controls">
    <button class="btn-prev" onclick="navigate(-1)">← Trước <span class="kbd">←</span></button>
    <button class="btn-bright" id="btn-bright" onclick="toggleBright()">☀️ Tăng sáng <span class="kbd">B</span></button>
    <button class="btn-clear" onclick="clearAllBoxes()">🗑️ Xóa hết box <span class="kbd">C</span></button>
    <button class="btn-save" id="btn-save" onclick="saveCurrentLabels()">💾 Lưu nhãn <span class="kbd">S</span></button>
    <button class="btn-approve" id="btn-approve" onclick="setStatus('approved')">✅ Đạt <span class="kbd">A</span></button>
    <button class="btn-flag" id="btn-flag" onclick="setStatus('flagged')">🚩 Cần sửa <span class="kbd">F</span></button>
    <button class="btn-next" onclick="navigate(1)">Sau → <span class="kbd">→</span></button>
</div>

<script>
const CLASS_NAMES = {0: "drone"};
const BACKUP_COLOR = "#ff4444";
const PAGE_SIZE = 200;

let allImages = [];
let filteredImages = [];
let currentIndex = -1;
let currentPage = 0;
let viewMode = "current";
let detailCache = {};

// Manual editor state
let currentImageObj = null;
let editableBoxes = [];
let backupBoxes = [];
let selectedBoxIdx = -1;
let isDirty = false;
let brightenOn = false;
let isDrawing = false;
let dragStart = null;
let dragCurrent = null;
let mainCanvas = null;

async function loadData() {
    const resp = await fetch("/api/images");
    allImages = await resp.json();
    applyFilters(true);
    updateStats();
}

function applyFilters(resetSelection = false) {
    const cat = document.getElementById("filter-category").value;
    const sev = document.getElementById("filter-severity").value;
    const cls = document.getElementById("filter-class").value;
    const status = document.getElementById("filter-status").value;
    const split = document.getElementById("filter-split").value;
    const search = document.getElementById("search-box").value.toLowerCase();

    filteredImages = allImages.filter(img => {
        if (cat !== "all" && img.category !== cat) return false;
        if (sev === "any_susp" && img.severity === "ok") return false;
        if (sev !== "all" && sev !== "any_susp" && img.severity !== sev) return false;
        if (cls === "empty" && img.num_boxes !== 0) return false;
        if (cls !== "all" && cls !== "empty" && !img.classes.includes(parseInt(cls))) return false;
        if (status !== "all" && img.qa_status !== status) return false;
        if (split !== "all" && img.split !== split) return false;
        if (search && !img.name.toLowerCase().includes(search)) return false;
        return true;
    });

    document.getElementById("stat-filtered").textContent = filteredImages.length;

    if (resetSelection || currentIndex < 0 || currentIndex >= filteredImages.length) {
        currentIndex = filteredImages.length > 0 ? 0 : -1;
        currentPage = 0;
    } else {
        currentPage = Math.floor(currentIndex / PAGE_SIZE);
    }

    renderSidebar();
    if (currentIndex >= 0) {
        selectImage(currentIndex);
    } else {
        document.getElementById("empty-state").innerHTML = "<h2>Không tìm thấy ảnh phù hợp bộ lọc</h2>";
        document.getElementById("empty-state").style.display = "block";
        document.getElementById("canvas-container").style.display = "none";
        document.getElementById("image-info").style.display = "none";
        document.getElementById("stat-current").textContent = 0;
    }
}

function changePage(delta) {
    const totalPages = Math.max(1, Math.ceil(filteredImages.length / PAGE_SIZE));
    const next = currentPage + delta;
    if (next >= 0 && next < totalPages) {
        currentPage = next;
        renderSidebar();
    }
}

function renderSidebar() {
    const sidebar = document.getElementById("sidebar");
    const totalPages = Math.max(1, Math.ceil(filteredImages.length / PAGE_SIZE));
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    document.getElementById("page-info").textContent = `Trang ${currentPage + 1}/${totalPages}`;
    document.getElementById("page-prev").disabled = currentPage <= 0;
    document.getElementById("page-next").disabled = currentPage >= totalPages - 1;

    const start = currentPage * PAGE_SIZE;
    const end = Math.min(start + PAGE_SIZE, filteredImages.length);
    let html = "";
    for (let idx = start; idx < end; idx++) {
        const img = filteredImages[idx];
        const active = idx === currentIndex ? "active" : "";
        const statusClass = img.qa_status || "pending";
        const badgeText = img.category === "manual_fixed" ? "Tay" :
                          img.category === "relabeled" ? "Sửa" :
                          img.category === "auto_labeled" ? "Mới" : "Gốc";
        const sevIcon = img.severity === "critical" ? "🔴" :
                        img.severity === "high" ? "🟠" :
                        img.severity === "medium" ? "🟡" : "";
        html += `<div class="sidebar-item ${active}" onclick="selectImage(${idx})" id="sb-${idx}">
            <div class="dot ${statusClass}"></div>
            <div class="info">
                <div class="name">${sevIcon} ${img.name}</div>
                <div class="meta">${img.split} | ${img.num_boxes} box</div>
            </div>
            <span class="badge ${img.category}">${badgeText}</span>
        </div>`;
    }
    sidebar.innerHTML = html || '<div class="empty-state"><p>Trống</p></div>';
}

function updateStats() {
    const total = allImages.length;
    let approved = 0, flagged = 0;
    for (let i = 0; i < total; i++) {
        if (allImages[i].qa_status === "approved") approved++;
        else if (allImages[i].qa_status === "flagged") flagged++;
    }
    const pending = total - approved - flagged;
    document.getElementById("stat-total").textContent = total;
    document.getElementById("stat-approved").textContent = approved;
    document.getElementById("stat-flagged").textContent = flagged;
    document.getElementById("stat-pending").textContent = pending;
    const progress = total > 0 ? ((approved + flagged) / total * 100) : 0;
    document.getElementById("progress-fill").style.width = progress + "%";
}

function markDirty(val) {
    isDirty = val;
    document.getElementById("btn-save").classList.toggle("dirty", isDirty);
}

async function getDetail(img, force = false) {
    const key = `${img.split}/${img.name}`;
    if (!force && detailCache[key]) return detailCache[key];
    const resp = await fetch(`/api/detail/${img.split}/${encodeURIComponent(img.name)}`);
    const data = await resp.json();
    detailCache[key] = data;
    return data;
}

async function selectImage(idx) {
    if (idx < 0 || idx >= filteredImages.length) return;
    currentIndex = idx;
    selectedBoxIdx = -1;
    markDirty(false);

    const targetPage = Math.floor(idx / PAGE_SIZE);
    if (targetPage !== currentPage) {
        currentPage = targetPage;
        renderSidebar();
    } else {
        document.querySelectorAll(".sidebar-item").forEach(el => el.classList.remove("active"));
        const sbItem = document.getElementById(`sb-${idx}`);
        if (sbItem) {
            sbItem.classList.add("active");
            sbItem.scrollIntoView({ block: "nearest" });
        }
    }

    const img = filteredImages[idx];
    document.getElementById("empty-state").style.display = "none";
    document.getElementById("canvas-container").style.display = "flex";
    document.getElementById("image-info").style.display = "block";
    document.getElementById("stat-current").textContent = idx + 1;

    const detail = await getDetail(img);
    editableBoxes = (detail.labels || []).map(b => ({...b}));
    backupBoxes = (detail.backup_labels || []).map(b => ({...b}));

    const image = new window.Image();
    image.onload = () => {
        if (currentIndex === idx) {
            currentImageObj = image;
            renderViewer(img);
        }
    };
    image.src = `/image/${img.split}/${encodeURIComponent(img.name)}`;
    updateImageInfo(img);
}

function updateImageInfo(img) {
    let infoHtml = `<strong>${img.name}</strong> | ${img.split} | <strong>${editableBoxes.length}</strong> box(es) | `;
    infoHtml += `Trạng thái: <span style="color:${
        img.qa_status === 'approved' ? '#00e676' :
        img.qa_status === 'flagged' ? '#ff5252' : '#aaa'
    }">${img.qa_status === 'approved' ? '✅ Đạt' :
         img.qa_status === 'flagged' ? '🚩 Cần sửa' : '⏳ Chưa duyệt'}</span>`;
    if (img.changes) {
        infoHtml += `<div class="changes">Lịch sử nhãn: ${img.changes}</div>`;
    }
    if (img.reasons) {
        infoHtml += `<div class="changes" style="color:#ff5252;">⚠️ Cảnh báo (${img.severity.toUpperCase()}): ${img.reasons}</div>`;
    }
    document.getElementById("image-info").innerHTML = infoHtml;
    document.getElementById("btn-approve").classList.toggle("btn-active", img.qa_status === "approved");
    document.getElementById("btn-flag").classList.toggle("btn-active", img.qa_status === "flagged");
}

function renderViewer(imgData) {
    if (!currentImageObj) return;
    const container = document.getElementById("canvas-container");
    container.innerHTML = "";
    const maxH = window.innerHeight - 300;
    const maxW = container.clientWidth - 30;
    const hasBackup = backupBoxes.length > 0;
    const isAuto = imgData.category === "auto_labeled";

    if (viewMode === "compare" && (hasBackup || isAuto)) {
        const halfW = (maxW - 20) / 2;
        const oldTitle = isAuto ? "🔴 Nhãn Cũ (Trống - 0 box)" : "🔴 Nhãn Cũ (Trước)";
        container.appendChild(createStaticCanvas(oldTitle, currentImageObj, backupBoxes, BACKUP_COLOR, halfW, maxH));
        container.appendChild(createEditableCanvas("🟢 Nhãn Mới (Kéo để vẽ / Click để xóa)", currentImageObj, halfW, maxH, false));
    } else if (viewMode === "overlay" && hasBackup) {
        container.appendChild(createEditableCanvas("Chồng nhãn: 🔴 Cũ (đứt) vs 🟢 Mới (chỉnh sửa được)", currentImageObj, maxW, maxH, true));
    } else {
        container.appendChild(createEditableCanvas("🟢 Nhãn hiện tại (Kéo chuột vẽ box mới / Click box → Del xóa)", currentImageObj, maxW, maxH, false));
    }
}

function createStaticCanvas(title, image, labels, color, maxW, maxH) {
    const wrapper = document.createElement("div");
    wrapper.className = "canvas-wrapper";
    const h3 = document.createElement("h3");
    h3.textContent = title;
    wrapper.appendChild(h3);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.min(maxW / image.width, maxH / image.height, 1);
    canvas.width = Math.round(image.width * scale);
    canvas.height = Math.round(image.height * scale);
    if (brightenOn) ctx.filter = "brightness(2.2) contrast(1.4)";
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
    drawBoxes(ctx, labels, color, canvas.width, canvas.height, false, -1);
    wrapper.appendChild(canvas);
    return wrapper;
}

function createEditableCanvas(title, image, maxW, maxH, showOverlayOld) {
    const wrapper = document.createElement("div");
    wrapper.className = "canvas-wrapper";
    const h3 = document.createElement("h3");
    h3.textContent = title;
    wrapper.appendChild(h3);
    const canvas = document.createElement("canvas");
    const scale = Math.min(maxW / image.width, maxH / image.height, 1);
    canvas.width = Math.round(image.width * scale);
    canvas.height = Math.round(image.height * scale);
    mainCanvas = canvas;
    canvas.dataset.overlay = showOverlayOld ? "1" : "0";

    attachCanvasEvents(canvas);
    redrawMainCanvas();
    wrapper.appendChild(canvas);
    return wrapper;
}

function redrawMainCanvas() {
    if (!mainCanvas || !currentImageObj) return;
    const ctx = mainCanvas.getContext("2d");
    const cw = mainCanvas.width, ch = mainCanvas.height;
    ctx.clearRect(0, 0, cw, ch);
    if (brightenOn) ctx.filter = "brightness(2.2) contrast(1.4)";
    ctx.drawImage(currentImageObj, 0, 0, cw, ch);
    ctx.filter = "none";

    if (mainCanvas.dataset.overlay === "1" && backupBoxes.length > 0) {
        drawBoxes(ctx, backupBoxes, "#ff4444", cw, ch, true, -1);
    }
    drawBoxes(ctx, editableBoxes, "#00ff88", cw, ch, false, selectedBoxIdx);

    if (isDrawing && dragStart && dragCurrent) {
        const x = Math.min(dragStart.x, dragCurrent.x);
        const y = Math.min(dragStart.y, dragCurrent.y);
        const w = Math.abs(dragCurrent.x - dragStart.x);
        const h = Math.abs(dragCurrent.y - dragStart.y);
        ctx.strokeStyle = "#00d2ff";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }
}

function attachCanvasEvents(canvas) {
    const getPos = (e) => {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (canvas.width / rect.width),
            y: (e.clientY - rect.top) * (canvas.height / rect.height),
        };
    };

    canvas.addEventListener("mousedown", (e) => {
        const pos = getPos(e);
        dragStart = pos;
        dragCurrent = pos;
        isDrawing = true;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!isDrawing) return;
        dragCurrent = getPos(e);
        redrawMainCanvas();
    });

    canvas.addEventListener("mouseup", (e) => {
        if (!isDrawing) return;
        isDrawing = false;
        const end = getPos(e);
        const dx = Math.abs(end.x - dragStart.x);
        const dy = Math.abs(end.y - dragStart.y);

        if (dx < 5 && dy < 5) {
            // Click -> select box under cursor
            const cw = canvas.width, ch = canvas.height;
            selectedBoxIdx = -1;
            for (let i = editableBoxes.length - 1; i >= 0; i--) {
                const b = editableBoxes[i];
                const x1 = (b.cx - b.w / 2) * cw, y1 = (b.cy - b.h / 2) * ch;
                const x2 = (b.cx + b.w / 2) * cw, y2 = (b.cy + b.h / 2) * ch;
                if (end.x >= x1 && end.x <= x2 && end.y >= y1 && end.y <= y2) {
                    selectedBoxIdx = i;
                    break;
                }
            }
        } else {
            // Drag -> add new drone box
            const cw = canvas.width, ch = canvas.height;
            const x1 = Math.max(0, Math.min(dragStart.x, end.x));
            const y1 = Math.max(0, Math.min(dragStart.y, end.y));
            const x2 = Math.min(cw, Math.max(dragStart.x, end.x));
            const y2 = Math.min(ch, Math.max(dragStart.y, end.y));
            const bw = (x2 - x1) / cw;
            const bh = (y2 - y1) / ch;
            if (bw > 0.004 && bh > 0.004) {
                editableBoxes.push({
                    class: 0,
                    cx: (x1 + x2) / 2 / cw,
                    cy: (y1 + y2) / 2 / ch,
                    w: bw,
                    h: bh,
                });
                selectedBoxIdx = editableBoxes.length - 1;
                markDirty(true);
                if (currentIndex >= 0) updateImageInfo(filteredImages[currentIndex]);
            }
        }
        dragStart = null;
        dragCurrent = null;
        redrawMainCanvas();
    });
}

function drawBoxes(ctx, labels, color, cw, ch, dashed, selectedIdx) {
    labels.forEach((lbl, idx) => {
        const w = lbl.w * cw;
        const h = lbl.h * ch;
        const x = lbl.cx * cw - w / 2;
        const y = lbl.cy * ch - h / 2;
        const isSel = idx === selectedIdx;
        const strokeCol = isSel ? "#ffea00" : color;
        ctx.strokeStyle = strokeCol;
        ctx.lineWidth = isSel ? 3.5 : 2.2;
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.strokeRect(x, y, w, h);

        const labelText = (CLASS_NAMES[lbl.class] || `class_${lbl.class}`) + (isSel ? " [Đang chọn]" : "");
        ctx.font = "bold 12px Segoe UI, sans-serif";
        const textW = ctx.measureText(labelText).width + 8;
        const textH = 17;
        const labelY = Math.max(textH, y);
        ctx.fillStyle = strokeCol;
        ctx.fillRect(x, labelY - textH, textW, textH);
        ctx.fillStyle = "#000";
        ctx.fillText(labelText, x + 4, labelY - 4);
    });
    ctx.setLineDash([]);
}

function deleteSelectedBox() {
    if (selectedBoxIdx >= 0 && selectedBoxIdx < editableBoxes.length) {
        editableBoxes.splice(selectedBoxIdx, 1);
        selectedBoxIdx = -1;
        markDirty(true);
        redrawMainCanvas();
        if (currentIndex >= 0) updateImageInfo(filteredImages[currentIndex]);
    }
}

function clearAllBoxes() {
    if (editableBoxes.length === 0) return;
    editableBoxes = [];
    selectedBoxIdx = -1;
    markDirty(true);
    redrawMainCanvas();
    if (currentIndex >= 0) updateImageInfo(filteredImages[currentIndex]);
}

function toggleBright() {
    brightenOn = !brightenOn;
    document.getElementById("btn-bright").classList.toggle("active", brightenOn);
    if (currentIndex >= 0) renderViewer(filteredImages[currentIndex]);
}

async function saveCurrentLabels() {
    if (currentIndex < 0 || currentIndex >= filteredImages.length) return;
    const img = filteredImages[currentIndex];
    const resp = await fetch("/api/save_labels", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            image: img.name,
            split: img.split,
            labels: editableBoxes,
        })
    });
    const res = await resp.json();
    if (res.ok) {
        img.num_boxes = editableBoxes.length;
        img.classes = editableBoxes.length > 0 ? [0] : [];
        img.category = "manual_fixed";
        img.qa_status = "approved";
        img.severity = "ok";
        img.reasons = "";
        img.changes = `Manual edit: ${editableBoxes.length} box(es)`;
        delete detailCache[`${img.split}/${img.name}`];
        markDirty(false);
        renderSidebar();
        updateStats();
        updateImageInfo(img);
        if (currentIndex < filteredImages.length - 1) {
            setTimeout(() => navigate(1), 150);
        }
    }
}

function setViewMode(mode) {
    viewMode = mode;
    document.querySelectorAll(".toggle-btn").forEach(el => el.classList.remove("active"));
    document.getElementById(`view-${mode}`).classList.add("active");
    if (currentIndex >= 0) renderViewer(filteredImages[currentIndex]);
}

function navigate(direction) {
    const newIdx = currentIndex + direction;
    if (newIdx >= 0 && newIdx < filteredImages.length) {
        selectImage(newIdx);
    }
}

async function setStatus(status) {
    if (currentIndex < 0 || currentIndex >= filteredImages.length) return;
    const img = filteredImages[currentIndex];
    const newStatus = img.qa_status === status ? "pending" : status;
    img.qa_status = newStatus;

    await fetch("/api/review", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: img.name, status: newStatus})
    });

    renderSidebar();
    updateStats();
    updateImageInfo(img);

    if (newStatus === "approved" && currentIndex < filteredImages.length - 1) {
        setTimeout(() => navigate(1), 120);
    }
}

document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    switch(e.key) {
        case "ArrowLeft": e.preventDefault(); navigate(-1); break;
        case "ArrowRight": e.preventDefault(); navigate(1); break;
        case "a": case "A": e.preventDefault(); setStatus("approved"); break;
        case "f": case "F": e.preventDefault(); setStatus("flagged"); break;
        case "s": case "S": e.preventDefault(); saveCurrentLabels(); break;
        case "b": case "B": e.preventDefault(); toggleBright(); break;
        case "c": case "C": e.preventDefault(); clearAllBoxes(); break;
        case "Delete": case "Backspace": e.preventDefault(); deleteSelectedBox(); break;
    }
});

["filter-category", "filter-severity", "filter-class", "filter-status", "filter-split"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => applyFilters(true));
});
document.getElementById("search-box").addEventListener("input", () => applyFilters(true));

loadData();
</script>
</body>
</html>"""


class QAHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path in ("/", ""):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif path == "/api/images":
            raw = json.dumps(IMAGES_INDEX, ensure_ascii=False).encode("utf-8")
            accept_enc = self.headers.get("Accept-Encoding", "")
            if "gzip" in accept_enc:
                payload = gzip.compress(raw, compresslevel=5)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        elif path.startswith("/api/detail/"):
            parts = path.split("/")
            if len(parts) >= 5:
                split = parts[3]
                filename = "/".join(parts[4:])
                stem = Path(filename).stem
                lbl_file = DATASET_DIR / "labels" / split / f"{stem}.txt"
                backup_file = BACKUP_DIR / split / f"{stem}.txt"
                detail = {
                    "labels": read_yolo_boxes(lbl_file),
                    "backup_labels": read_yolo_boxes(backup_file),
                }
                payload = json.dumps(detail).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        elif path.startswith("/image/"):
            parts = path.split("/")
            if len(parts) >= 4:
                split = parts[2]
                filename = "/".join(parts[3:])
                filepath = DATASET_DIR / "images" / split / filename
                if filepath.exists():
                    mime = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
                    data = filepath.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_response(404)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        if self.path == "/api/review":
            img_name = data["image"]
            new_status = data["status"]
            with STATUS_LOCK:
                QA_STATUS[img_name] = new_status
                if img_name in IMAGES_BY_NAME:
                    IMAGES_BY_NAME[img_name]["qa_status"] = new_status
                save_qa_status(QA_STATUS)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        elif self.path == "/api/save_labels":
            img_name = data["image"]
            split = data["split"]
            labels = data.get("labels", [])
            stem = Path(img_name).stem
            lbl_file = DATASET_DIR / "labels" / split / f"{stem}.txt"
            backup_file = BACKUP_DIR / split / f"{stem}.txt"

            with STATUS_LOCK:
                orig_boxes = len(read_yolo_boxes(lbl_file))
                # Backup original label once
                if lbl_file.exists() and not backup_file.exists():
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(lbl_file, backup_file)

                # Write new YOLO labels
                lines = []
                for b in labels:
                    cx = max(0.0, min(1.0, float(b["cx"])))
                    cy = max(0.0, min(1.0, float(b["cy"])))
                    bw = max(0.0, min(1.0, float(b["w"])))
                    bh = max(0.0, min(1.0, float(b["h"])))
                    cls = int(b.get("class", 0))
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                # Append to audit log
                file_exists = AUDIT_LOG.exists()
                with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["image", "split", "action", "changes", "original_boxes", "final_boxes", "timestamp"],
                    )
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({
                        "image": img_name,
                        "split": split,
                        "action": "manual_fixed",
                        "changes": f"Manual edit: {orig_boxes} -> {len(lines)} box(es)",
                        "original_boxes": orig_boxes,
                        "final_boxes": len(lines),
                        "timestamp": datetime.now().isoformat(),
                    })

                # Mark approved in QA status & memory index
                QA_STATUS[img_name] = "approved"
                save_qa_status(QA_STATUS)
                if img_name in IMAGES_BY_NAME:
                    entry = IMAGES_BY_NAME[img_name]
                    entry["num_boxes"] = len(lines)
                    entry["classes"] = [0] if lines else []
                    entry["category"] = "manual_fixed"
                    entry["qa_status"] = "approved"
                    entry["severity"] = "ok"
                    entry["reasons"] = ""
                    entry["changes"] = f"Manual edit: {orig_boxes} -> {len(lines)} box(es)"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        else:
            self.send_response(404)
            self.end_headers()


def main():
    if not (DATASET_DIR / "images").exists():
        print("ERROR: Dataset not found.")
        sys.exit(1)

    build_index()
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), QAHandler)
    print(f"QA & Manual Editor running at http://localhost:{PORT} ({len(IMAGES_INDEX)} images)")
    server.serve_forever()


if __name__ == "__main__":
    main()
