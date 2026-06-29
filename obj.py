"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MISSING OBJECT DETECTION v2  ─  Smart AI + Voice + Interactive Training      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  INSTALL (run once):                                                           ║
║    pip install ultralytics opencv-python pillow                               ║
║    pip install pyaudio pyttsx3 numpy SpeechRecognition                        ║
║    pip install fpdf2 lapx                                                      ║
║                                                                                ║
║  VOICE COMMANDS:                                                               ║
║   "start" / "stop"     → toggle camera                                        ║
║   "find bottle"        → track bottle, alert when missing                     ║
║   "where is my phone"  → instant status with last-seen location               ║
║   "what do you see"    → announce all objects with positions                  ║
║   "remove bottle"      → stop tracking bottle                                 ║
║   "clear"              → remove all targets                                   ║
║   "snapshot"           → save current frame as JPG                            ║
║   "train mug"          → interactive selection-based object training          ║
║   "heatmap"            → toggle detection heatmap overlay                     ║
║   "summary"            → session detection summary                            ║
║   "export pdf"         → save PDF report                                      ║
║   "camera one"         → switch to camera index 1                             ║
║   "mute" / "unmute"    → toggle speech output                                 ║
║   "quit"               → exit app                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STDLIB
# ═══════════════════════════════════════════════════════════════════════════════
import tkinter as tk
from tkinter import simpledialog, filedialog
import threading
import traceback
import queue
import time
import csv
import os
import io
import json
import wave
import struct
import shutil
import sqlite3
import hashlib
import random
import subprocess
import sys
from datetime import datetime
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# THIRD-PARTY
# ═══════════════════════════════════════════════════════════════════════════════
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
from ultralytics import YOLO
import pyttsx3
import pyaudio
import speech_recognition as sr

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import torch
    HAS_TORCH = True
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if _DEVICE.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INIT] CPU mode (install torch+CUDA for GPU acceleration)")
except ImportError:
    HAS_TORCH = False
    _DEVICE = "cpu"

try:
    import whisper as _whisper_lib
    _whisper_model = _whisper_lib.load_model("base")
    HAS_WHISPER = True
    print("[INIT] Whisper STT loaded — offline, high accuracy")
except Exception:
    HAS_WHISPER = False
    print("[INIT] Whisper not found — using Google STT  (pip install openai-whisper)")

try:
    import clip as _clip_lib
    _clip_model, _clip_preprocess = _clip_lib.load("ViT-B/32", device="cpu")
    HAS_CLIP = True
    print("[INIT] CLIP loaded — semantic false-detection elimination enabled")
except Exception:
    HAS_CLIP = False
    print("[INIT] CLIP not found (pip install git+https://github.com/openai/CLIP.git)")

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM COLOUR PALETTE  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
BG       = "#0A0C14"
PANEL    = "#0E1120"
CARD     = "#131629"
CARD2    = "#181D35"
BDR      = "#1E2340"
BDR2     = "#252B48"

GOLD     = "#C9A96E"
GOLD2    = "#E8C98A"
GOLD_DIM = "#6B5530"
TEAL     = "#3ECFCF"
GREEN    = "#2ECC8F"
RED      = "#E05C6F"
ORANGE   = "#E8944A"
PURPLE   = "#9B7CE8"
WHITE    = "#EDF0FF"
MUTED    = "#5A6080"
DIM      = "#070910"
GLOW_G   = "#0A2A1A"
GLOW_R   = "#2A0A12"

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR     = os.path.join(ROOT_DIR, "snapshots")
TRAINING_DIR     = os.path.join(ROOT_DIR, "custom_training")
CUSTOM_MODEL_F   = os.path.join(ROOT_DIR, "custom_model.pt")
CUSTOM_LABELS_F  = os.path.join(ROOT_DIR, "custom_labels.json")
VIDEO_OUTPUT_DIR = os.path.join(ROOT_DIR, "video_output")
DB_PATH          = os.path.join(ROOT_DIR, "detection_log.db")
REPORTS_DIR      = os.path.join(ROOT_DIR, "reports")
CLIP_EMBED_F     = os.path.join(ROOT_DIR, "clip_embeddings.npz")

os.makedirs(SNAPSHOT_DIR,     exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR,      exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CLIP SEMANTIC VERIFICATION
# ── Async worker so CLIP never blocks inference (200ms CPU latency)
# ═══════════════════════════════════════════════════════════════════════════════
_clip_embeddings: dict = {}   # {obj_name: np.ndarray shape(512,)}
_clip_valid:      dict = {}   # {obj_name: bool}  — continuously updated
_clip_verify_q: queue.Queue = queue.Queue(maxsize=8)
_CLIP_EMBED_SIM_THRESH = 0.22   # min cosine sim for stored-embedding match
_CLIP_TEXT_SIM_THRESH  = 0.17   # min cosine sim for text-label match


def _load_clip_embeddings():
    global _clip_embeddings
    if os.path.exists(CLIP_EMBED_F):
        try:
            data = np.load(CLIP_EMBED_F, allow_pickle=True)
            _clip_embeddings = {k: data[k] for k in data.files}
            print(f"[CLIP] Embeddings loaded: {list(_clip_embeddings.keys())}")
        except Exception as e:
            print(f"[CLIP] Embed load error: {e}")

_load_clip_embeddings()


def _clip_text_sim(crop_bgr: np.ndarray, label: str) -> float:
    if not HAS_CLIP or crop_bgr is None or crop_bgr.size == 0:
        return 1.0
    try:
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_t = _clip_preprocess(pil).unsqueeze(0)
        txt_t = _clip_lib.tokenize([f"a {label}", f"a photo of {label}"]).to("cpu")
        with torch.no_grad():
            img_f = _clip_model.encode_image(img_t)
            txt_f = _clip_model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            return float((img_f @ txt_f.T).mean().item())
    except Exception:
        return 1.0


def _clip_embed_sim(crop_bgr: np.ndarray, embedding: np.ndarray) -> float:
    if not HAS_CLIP or crop_bgr is None or crop_bgr.size == 0:
        return 1.0
    try:
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        img_t = _clip_preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            img_f = _clip_model.encode_image(img_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            img_np = img_f.cpu().numpy().flatten()
        return float(np.dot(img_np, embedding.flatten()))
    except Exception:
        return 1.0


def _clip_worker():
    while True:
        item = _clip_verify_q.get()
        if item is None:
            break
        name, crop = item
        try:
            if name in _clip_embeddings:
                sim = _clip_embed_sim(crop, _clip_embeddings[name])
                _clip_valid[name] = sim > _CLIP_EMBED_SIM_THRESH
            else:
                sim = _clip_text_sim(crop, name)
                _clip_valid[name] = sim > _CLIP_TEXT_SIM_THRESH
        except Exception:
            _clip_valid[name] = True


if HAS_CLIP:
    threading.Thread(target=_clip_worker, daemon=True, name="clip").start()

# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE  — async write-behind so inference thread never blocks on I/O
# ═══════════════════════════════════════════════════════════════════════════════
def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS detections (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        TEXT,
        object    TEXT,
        status    TEXT,
        conf      REAL,
        pos_x     REAL,
        pos_y     REAL,
        direction TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        start_ts     TEXT,
        end_ts       TEXT,
        total_frames INTEGER DEFAULT 0
    )""")
    con.commit()
    con.close()

_init_db()
_db_session_id = [None]
_db_q: queue.Queue = queue.Queue()

def _db_worker():
    con = sqlite3.connect(DB_PATH)
    while True:
        row = _db_q.get()
        if row is None:
            break
        try:
            con.execute(
                "INSERT INTO detections(ts,object,status,conf,pos_x,pos_y,direction) "
                "VALUES(?,?,?,?,?,?,?)", row)
            con.commit()
        except Exception:
            pass

threading.Thread(target=_db_worker, daemon=True, name="db").start()

def _db_log(obj, status, conf=None, px=None, py=None, direction=None):
    _db_q.put((datetime.now().isoformat(), obj, status, conf, px, py, direction))

def _db_start_session():
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("INSERT INTO sessions(start_ts) VALUES(?)",
                      (datetime.now().isoformat(),))
    _db_session_id[0] = cur.lastrowid
    con.commit()
    con.close()

def _db_end_session(total_frames):
    sid = _db_session_id[0]
    if sid is None:
        return
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE sessions SET end_ts=?,total_frames=? WHERE id=?",
                    (datetime.now().isoformat(), total_frames, sid))
        con.commit()
        con.close()
    except Exception:
        pass
    _db_session_id[0] = None

# ═══════════════════════════════════════════════════════════════════════════════
# SCENE MEMORY  — tracks last-seen position/time per object
# ═══════════════════════════════════════════════════════════════════════════════
class SceneMemory:
    def __init__(self):
        self._d = {}

    def update(self, obj, conf, px, py, direction):
        now = time.time()
        if obj not in self._d:
            self._d[obj] = {"time": now, "pos_x": px, "pos_y": py,
                            "conf": conf, "direction": direction,
                            "history": deque(maxlen=60),
                            "vx": 0.0, "vy": 0.0}
        e = self._d[obj]
        # Velocity from last position
        dt = now - e["time"]
        if dt > 0.01:
            alpha = 0.3
            e["vx"] = alpha * (px - e["pos_x"]) / dt + (1 - alpha) * e["vx"]
            e["vy"] = alpha * (py - e["pos_y"]) / dt + (1 - alpha) * e["vy"]
        e["history"].append((now, px, py))
        e.update(time=now, pos_x=px, pos_y=py, conf=conf, direction=direction)

    def get(self, obj):
        return self._d.get(obj)

    def ago(self, obj) -> float:
        e = self._d.get(obj)
        return -1.0 if e is None else time.time() - e["time"]

    def direction(self, obj) -> str:
        e = self._d.get(obj)
        return e["direction"] if e else "somewhere"

    def predicted_direction(self, obj) -> str:
        """Where the object was heading when last seen."""
        e = self._d.get(obj)
        if e is None:
            return ""
        vx, vy = e.get("vx", 0.0), e.get("vy", 0.0)
        speed = math.hypot(vx, vy)
        if speed < 0.005:
            return ""
        angle = math.degrees(math.atan2(vy, vx))
        for lo, hi, lbl in [
            (-157.5, -112.5, "upper left"), (-112.5, -67.5, "upward"),
            ( -67.5,  -22.5, "upper right"), (-22.5,  22.5, "right"),
            (  22.5,   67.5, "lower right"), ( 67.5, 112.5, "downward"),
            ( 112.5,  157.5, "lower left"),
        ]:
            if lo <= angle < hi:
                return lbl
        return "left"

    def clear(self):
        self._d.clear()

_scene_mem = SceneMemory()

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SMOOTHER  — requires N of last M frames to confirm detection
# ═══════════════════════════════════════════════════════════════════════════════
class TemporalSmoother:
    def __init__(self, window=5, min_hits=3):
        self._w = window
        self._m = min_hits
        self._h = {}

    def update(self, obj, seen) -> bool:
        if obj not in self._h:
            self._h[obj] = deque(maxlen=self._w)
        self._h[obj].append(seen)
        return sum(self._h[obj]) >= self._m

    def reset(self, obj):
        self._h.pop(obj, None)

    def clear(self):
        self._h.clear()

_smoother = TemporalSmoother(window=7, min_hits=4)

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _direction_from_norm(cx, cy) -> str:
    h = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    v = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "middle")
    if h == "center" and v == "middle":
        return "center of frame"
    if h == "center":
        return f"{v} area"
    if v == "middle":
        return f"{h} side"
    return f"{v} {h}"

# ═══════════════════════════════════════════════════════════════════════════════
# NATURAL VOICE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════
_FOUND_T = [
    "I found your {obj}. It is on the {dir}.",
    "Your {obj} is visible now on the {dir} at {pct} percent confidence.",
    "Found it! Your {obj} is on the {dir}.",
    "Good news — I can see your {obj} on the {dir}.",
]
_REAPPEAR_T = [
    "Your {obj} is back! It reappeared on the {dir}.",
    "I found your {obj} again — it is on the {dir}.",
    "Your {obj} has returned to view on the {dir}.",
    "Alert — your {obj} just came back into view on the {dir}.",
]
_MISSING_T = [
    "Alert! Your {obj} is missing. It has been gone for {dur} seconds.",
    "Warning — I cannot see your {obj}. Missing for {dur} seconds. Try checking {suggest}.",
    "Your {obj} is not in view. It has been {dur} seconds. Please search {suggest}.",
    "Missing alert! Your {obj} has been out of sight for {dur} seconds. Check {suggest}.",
]
_SUGGEST = ["to the left", "to the right", "nearby areas", "behind you"]

def _say_found(obj, direction, conf):
    t = random.choice(_FOUND_T)
    return t.format(obj=obj.replace("_", " "), dir=direction, pct=int(conf * 100))

def _say_reappear(obj, direction):
    t = random.choice(_REAPPEAR_T)
    return t.format(obj=obj.replace("_", " "), dir=direction)

def _say_missing(obj, dur):
    t = random.choice(_MISSING_T)
    return t.format(obj=obj.replace("_", " "), dur=dur,
                    suggest=random.choice(_SUGGEST))

# ═══════════════════════════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════════════════════════
_tts_q    = queue.Queue()
_tts_mute = threading.Event()

# ── Windows PowerShell SAPI — reliable fallback that needs NO extra packages ──
def _ps_speak(txt: str) -> bool:
    """Speak via Windows built-in SAPI. Works even when pyttsx3 COM fails."""
    if sys.platform != "win32":
        return False
    safe = txt.replace("'", " ").replace('"', ' ').replace('`', ' ')[:500]
    try:
        subprocess.run(
            ["powershell", "-WindowStyle", "Hidden", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             "$s.Rate = 1; $s.Volume = 100; "
             "try { $s.SelectVoiceByHints('Female') } catch {}; "
             f"$s.Speak('{safe}')"],
            timeout=25, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return True
    except Exception as e:
        print(f"[TTS-PS] {e}")
        return False

def _tts_worker():
    # Try pyttsx3 first (lower latency when it works)
    eng = None
    try:
        eng = pyttsx3.init()
        eng.setProperty("rate", 155)
        eng.setProperty("volume", 1.0)
        for v in eng.getProperty("voices"):
            if any(k in v.name.lower() for k in ("zira", "hazel", "female", "english")):
                eng.setProperty("voice", v.id)
                break
        print("[TTS] pyttsx3 engine initialised OK")
    except Exception as e:
        print(f"[TTS] pyttsx3 init failed ({e}) — PowerShell SAPI will handle all speech")
        eng = None

    while True:
        txt = _tts_q.get()
        if txt is None:
            break
        if not _tts_mute.is_set():
            spoken = False
            if eng is not None:
                try:
                    eng.say(txt)
                    eng.runAndWait()
                    spoken = True
                except Exception as e:
                    print(f"[TTS] pyttsx3 error: {e}  → falling back to PowerShell")
                    try:
                        eng = pyttsx3.init()   # attempt re-init for next time
                    except Exception:
                        eng = None
            if not spoken:
                _ps_speak(txt)
        _tts_q.task_done()

threading.Thread(target=_tts_worker, daemon=True, name="tts").start()

def speak(txt: str, force: bool = False):
    """Queue text for TTS. force=True clears pending queue so this plays next."""
    if not txt:
        return
    if force:
        # Drain pending items so this urgent message plays as soon as possible
        try:
            while True:
                _tts_q.get_nowait()
        except queue.Empty:
            pass
    print(f"[SPEAK] {txt}")
    _tts_q.put(txt)

# ═══════════════════════════════════════════════════════════════════════════════
# YOLO  —  try YOLO11 (latest) then fall back to YOLOv8
# ═══════════════════════════════════════════════════════════════════════════════
print("[INIT] Missing Object Detection v3 — Loading AI model …")
_base = None
# yolov8x-oiv7.pt = Open Images V7 — 601 classes including game controllers, joysticks etc.
for _mn in ("yolo11x.pt", "yolov8x-oiv7.pt", "yolov8x.pt", "yolo11n.pt", "yolov8n.pt"):
    try:
        _base = YOLO(_mn)
        print(f"[INIT] Loaded {_mn}")
        break
    except Exception:
        pass
if _base is None:
    raise RuntimeError("No YOLO model available. Run: pip install ultralytics")

# ── CLAHE for low-light / overexposed frame enhancement ──────────────────────
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _enhance_frame(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ── EMA Object Tracker — smooth jitter + estimate velocity ───────────────────
class ObjectTracker:
    _PA = 0.35    # position EMA alpha
    _VA = 0.25    # velocity EMA alpha
    _TTL = 14     # frames before stale track dropped

    def __init__(self):
        self._t: dict = {}

    def update(self, raw: dict) -> dict:
        smoothed = {}
        for name, (conf, cx, cy) in raw.items():
            if name in self._t:
                t = self._t[name]
                nvx = cx - t["cx"]
                nvy = cy - t["cy"]
                t["vx"] = self._VA * nvx + (1 - self._VA) * t["vx"]
                t["vy"] = self._VA * nvy + (1 - self._VA) * t["vy"]
                t["cx"] = self._PA * cx  + (1 - self._PA) * t["cx"]
                t["cy"] = self._PA * cy  + (1 - self._PA) * t["cy"]
                t["conf"] = conf
                t["ttl"]  = self._TTL
            else:
                self._t[name] = {"cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0,
                                  "conf": conf, "ttl": self._TTL}
            t = self._t[name]
            smoothed[name] = (conf, t["cx"], t["cy"])
        for name in list(self._t):
            if name not in raw:
                self._t[name]["ttl"] -= 1
                if self._t[name]["ttl"] <= 0:
                    del self._t[name]
        return smoothed

    def velocity_desc(self, name: str) -> str:
        t = self._t.get(name)
        if t is None:
            return ""
        vx, vy = t["vx"], t["vy"]
        speed = math.hypot(vx, vy)
        if speed < 0.004:
            return ""
        angle = math.degrees(math.atan2(vy, vx))
        for lo, hi, lbl in [
            (-157.5, -112.5, "upper left"), (-112.5, -67.5, "up"),
            ( -67.5,  -22.5, "upper right"), (-22.5,  22.5, "right"),
            (  22.5,   67.5, "lower right"), ( 67.5, 112.5, "down"),
            ( 112.5,  157.5, "lower left"),
        ]:
            if lo <= angle < hi:
                return f"moving {lbl}"
        return "moving left"

    def arrow_pts(self, name: str, cx_px: int, cy_px: int):
        t = self._t.get(name)
        if t is None:
            return None
        vx, vy = t["vx"], t["vy"]
        speed = math.hypot(vx, vy)
        if speed < 0.004:
            return None
        scale = min(70, int(speed * 800))
        ex = int(cx_px + (vx / speed) * scale)
        ey = int(cy_px + (vy / speed) * scale)
        return (cx_px, cy_px, ex, ey)

    def clear(self):
        self._t.clear()

_obj_tracker = ObjectTracker()

_custom_model  = None
_custom_labels = {}

def _load_custom_model():
    global _custom_model, _custom_labels
    if os.path.exists(CUSTOM_MODEL_F) and os.path.exists(CUSTOM_LABELS_F):
        try:
            _custom_model  = YOLO(CUSTOM_MODEL_F)
            _custom_labels = json.load(open(CUSTOM_LABELS_F))
            print(f"[CUSTOM] Loaded: {list(_custom_labels.keys())}")
        except Exception as e:
            print(f"[CUSTOM] {e}")

_load_custom_model()

def _all_objects() -> set:
    names = {n.lower() for n in _base.names.values()}
    names |= set(_custom_labels.keys())
    return names

# Per-category confidence thresholds — lower = more sensitive detection
_CAT_CONF = {
    "person":     0.45, "car": 0.40, "truck": 0.40, "bus": 0.40,
    "cell phone": 0.25, "phone": 0.25, "remote": 0.22,
    "bottle":     0.25, "cup": 0.25, "mug": 0.22,
    "laptop":     0.35, "keyboard": 0.30, "mouse": 0.22,
    "book":       0.22, "keys": 0.18, "wallet": 0.18, "watch": 0.22,
    "backpack":   0.28, "bag": 0.25,
    "cat":        0.40, "dog": 0.40,
    "scissors":   0.20, "fork": 0.20, "spoon": 0.20, "knife": 0.22,
    "glasses":    0.20, "handbag": 0.25, "suitcase": 0.30,
    "chair":      0.30, "couch": 0.30, "tv": 0.35,
}

def _cat_thresh(obj: str) -> float:
    # Per-object threshold, fallback to 80% of slider conf for better sensitivity
    return _CAT_CONF.get(obj.lower(), max(0.20, S.conf * 0.80))

_USE_BYTETRACK = True

# ── Object aliases: map user-spoken names → YOLO COCO class names ────────────
# Joystick is not a COCO class; it maps to "remote" visually in YOLO COCO.
# These aliases allow tracking by user-friendly name while still matching YOLO output.
_OBJ_ALIASES: dict = {
    "joystick":      ["remote", "cell phone", "mouse"],
    "controller":    ["remote", "cell phone", "mouse"],
    "gamepad":       ["remote", "mouse"],
    "joypad":        ["remote"],
    "specs":         ["glasses"],
    "spectacles":    ["glasses"],
    "goggles":       ["glasses"],
    "earphones":     ["remote"],
    "earbuds":       ["remote"],
    "headphones":    ["remote"],
    "pen":           ["remote", "scissors"],
    "pencil":        ["remote", "scissors"],
    "charger":       ["remote", "cell phone"],
    "mug":           ["cup", "wine glass"],
    "phone":         ["cell phone"],
    "mobile":        ["cell phone"],
    "tv remote":     ["remote"],
    "game remote":   ["remote"],
    "torch":         ["bottle"],
    "flashlight":    ["bottle"],
    "marker":        ["remote", "scissors"],
    "highlighter":   ["remote"],
    "lipstick":      ["remote"],
    "comb":          ["fork", "remote"],
}

# ── Reverse alias lookup: YOLO class → all user targets that alias it ────────
def _aliases_for_target(target: str) -> list:
    t = target.lower().replace(" ", "_")
    result = list(_OBJ_ALIASES.get(target.lower(), []))
    result += list(_OBJ_ALIASES.get(t, []))
    return list(dict.fromkeys(result))   # deduped

# Alert timing constants
ALERT_FIRST_S  = 5    # seconds missing before first alert
ALERT_REPEAT_S = 10   # seconds between subsequent alerts

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
RATE         = 16000
CHANNELS     = 1
SAMPLE_W     = 2
PA_FORMAT    = pyaudio.paInt16
FRAME_MS     = 30
FRAME_SAMP   = int(RATE * FRAME_MS / 1000)
FRAME_BYTES  = FRAME_SAMP * SAMPLE_W
START_THRESH = 3
END_THRESH   = 20
PRE_ROLL     = 8
MAX_REC_SECS = 8
MAX_FRAMES   = int(MAX_REC_SECS * 1000 / FRAME_MS)

_pa           = pyaudio.PyAudio()
_voice_active = threading.Event()
_voice_active.set()

def _frame_rms(data: bytes) -> float:
    count  = len(data) // 2
    shorts = struct.unpack(f"<{count}h", data)
    return (sum(s * s for s in shorts) / count) ** 0.5

def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_W)
        wf.setframerate(RATE)
        wf.writeframes(pcm)
    return buf.getvalue()

def _stt(pcm: bytes) -> str:
    if len(pcm) < FRAME_BYTES * 3:
        return ""
    if HAS_WHISPER:
        return _stt_whisper(pcm)
    return _stt_google(pcm)

def _stt_whisper(pcm: bytes) -> str:
    import tempfile
    wav = _pcm_to_wav(pcm)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            tmp = f.name
        result = _whisper_model.transcribe(
            tmp, language="en", fp16=False,
            condition_on_previous_text=False, verbose=False)
        return result.get("text", "").lower().strip()
    except Exception as e:
        print(f"[WHISPER] {e}")
        return _stt_google(pcm)
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass

def _stt_google(pcm: bytes) -> str:
    wav  = _pcm_to_wav(pcm)
    recg = sr.Recognizer()
    recg.energy_threshold         = 300
    recg.dynamic_energy_threshold = True
    audio = sr.AudioData(wav, RATE, SAMPLE_W)
    try:
        return recg.recognize_google(audio).lower().strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"[STT] Network error: {e}")
        _Q.put(lambda: _ui_set_mic("NET ERROR", RED))
        time.sleep(1)
        return ""

def _measure_ambient(stream, seconds: float = 1.0) -> float:
    frames_needed = int(seconds * 1000 / FRAME_MS)
    rms_vals = []
    for _ in range(frames_needed):
        try:
            data = stream.read(FRAME_SAMP, exception_on_overflow=False)
            rms_vals.append(_frame_rms(data))
        except Exception:
            pass
    return float(np.percentile(rms_vals, 85)) if rms_vals else 500.0

def _voice_loop():
    try:
        stream = _pa.open(
            format=PA_FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=FRAME_SAMP)
    except Exception as e:
        print(f"[MIC] Failed to open: {e}")
        _Q.put(lambda: _ui_set_mic("NO MIC", RED))
        speak("No microphone detected. Please use the buttons to control the app.")
        return

    _Q.put(lambda: _ui_set_mic("CALIBRATING", ORANGE))
    print("[MIC] Measuring ambient noise (1 sec) …")
    ambient   = _measure_ambient(stream, seconds=1.0)
    threshold = max(300.0, ambient * 2.0)
    print(f"[MIC] Ambient={ambient:.1f}  Threshold={threshold:.1f}")

    _Q.put(lambda: _ui_set_mic("LISTENING", GREEN))
    _Q.put(lambda: _ui_log(f"Mic ready — ambient {ambient:.0f}, threshold {threshold:.0f}"))
    speak("Ready. Say find bottle or start to begin.")

    pre_roll = deque(maxlen=PRE_ROLL)

    while _voice_active.is_set():
        try:
            loud_streak = 0
            while _voice_active.is_set():
                data = stream.read(FRAME_SAMP, exception_on_overflow=False)
                pre_roll.append(data)
                if _frame_rms(data) >= threshold:
                    loud_streak += 1
                    if loud_streak >= START_THRESH:
                        break
                else:
                    loud_streak = 0

            if not _voice_active.is_set():
                break

            _Q.put(lambda: _ui_set_mic("RECORDING", GREEN))
            captured     = list(pre_roll)
            quiet_streak = 0

            for _ in range(MAX_FRAMES):
                data = stream.read(FRAME_SAMP, exception_on_overflow=False)
                captured.append(data)
                if _frame_rms(data) < threshold:
                    quiet_streak += 1
                    if quiet_streak >= END_THRESH:
                        break
                else:
                    quiet_streak = 0

            _Q.put(lambda: _ui_set_mic("PROCESSING", PURPLE))
            pcm = b"".join(captured)

            def _run_stt(audio_pcm=pcm):
                text = _stt(audio_pcm)
                if text:
                    print(f"[VOICE] heard: '{text}'")
                    dispatch(text)
                else:
                    _Q.put(lambda: _ui_set_mic("LISTENING", GREEN))

            threading.Thread(target=_run_stt, daemon=True, name="stt").start()

        except OSError as e:
            print(f"[MIC] OSError: {e}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[MIC] {type(e).__name__}: {e}")
            time.sleep(0.2)

    stream.stop_stream()
    stream.close()
    print("[MIC] Stream closed.")

# ═══════════════════════════════════════════════════════════════════════════════
# APP STATE
# ═══════════════════════════════════════════════════════════════════════════════
class S:
    detecting   = False
    cap         = None
    conf        = 0.40
    targets     = []
    miss_start  = {}
    last_alert  = {}
    found_count = {}
    log         = []
    prev_t      = 0.0
    last_cmd_t  = 0.0
    frame_count = 0
    last_detected = {}

    # training
    training_mode     = False
    training_name     = ""
    training_count    = 0
    training_class_id = 0
    training_labels   = {}
    TRAIN_NEEDED      = 120

    # interactive selection-based training
    selection_mode = False   # mouse drag active on canvas
    selection_box  = None    # (x1n, y1n, x2n, y2n) normalised 0-1

    # shared inference results
    latest_annotated = None
    latest_detected  = {}
    result_lock      = threading.Lock()

    # video file mode
    video_mode          = False
    video_total_frames  = 0
    video_current_frame = 0
    video_output_path   = ""

    # multi-camera
    camera_index = 0

    # heatmap
    show_heatmap = False
    heatmap_data = None   # np.float32 (360,640)

    # analytics
    session_detections = {}
    session_start      = time.time()

_Q: queue.Queue = queue.Queue()

def _pump():
    while not _Q.empty():
        try:
            _Q.get_nowait()()
        except queue.Empty:
            break
        except Exception as _e:
            import traceback as _tb
            print(f"[PUMP] error in queued command: {_e}")
            _tb.print_exc()
    root.after(33, _pump)

# ═══════════════════════════════════════════════════════════════════════════════
# TARGET MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def add_target(obj: str):
    obj = obj.strip().lower()
    if not obj:
        return
    if obj not in S.targets:
        S.targets.append(obj)
        S.miss_start[obj]  = None
        S.last_alert[obj]  = None
        S.found_count[obj] = 0
        # Clear stale state so alert fires fresh
        if hasattr(S, "obj_state"):
            S.obj_state.pop(obj, None)
        _smoother.reset(obj)
        _ui_refresh_chips()
        _ui_log(f"Tracking: {obj}")
        speak(f"Tracking {obj.replace('_', ' ')}.")
    # Auto-start camera if not running — so alerts fire without needing "start" command
    if not S.detecting:
        start_detection()

def remove_target(obj: str):
    obj = obj.strip().lower()
    if obj in S.targets:
        S.targets.remove(obj)
        for d in (S.miss_start, S.last_alert, S.found_count):
            d.pop(obj, None)
        if hasattr(S, "obj_state"):
            S.obj_state.pop(obj, None)
        _smoother.reset(obj)
        _ui_refresh_chips()
        _ui_log(f"Removed: {obj}")
        speak(f"{obj.replace('_', ' ')} removed.")

def clear_targets():
    S.targets.clear()
    S.miss_start.clear()
    S.last_alert.clear()
    S.found_count.clear()
    _smoother.clear()
    _obj_tracker.clear()
    _ui_refresh_chips()
    _ui_log("All targets cleared")
    if hasattr(S, "obj_state"):
      S.obj_state.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_objects(text: str) -> list:
    found, lower = [], text.lower()
    for obj in sorted(_all_objects(), key=len, reverse=True):
        obj_spaced = obj.replace("_", " ")
        padded = f" {lower} "
        if (f" {obj} " in padded or f" {obj_spaced} " in padded) and obj not in found:
            found.append(obj)
            lower = lower.replace(obj_spaced, " ", 1).replace(obj, " ", 1)
    return found

def _h(text: str, *kws) -> bool:
    return any(k in text for k in kws)

def dispatch(text: str):
    now = time.time()
    if now - S.last_cmd_t < 0.5:
        return
    S.last_cmd_t = now
    t = text.lower().strip()
    if not t:
        return

    _Q.put(lambda tx=t: _ui_set_heard(tx))

    if _h(t, "quit", "exit", "goodbye", "shutdown", "close app"):
        speak("Goodbye! Stay safe.", force=True)
        _Q.put(lambda: root.after(1200, _safe_quit))
        return

    if _h(t, "unmute", "un mute", "voice on", "sound on"):
        _tts_mute.clear()
        speak("Voice is back on. How can I help you?")
        return
    if _h(t, "mute", "be quiet", "silence", "quiet"):
        _tts_mute.set()
        _Q.put(lambda: _ui_log("Muted"))
        return

    if _h(t, "start", "begin", "go", "turn on", "open camera",
           "switch on", "launch", "activate", "camera on", "run"):
        speak("Starting.")
        _Q.put(start_detection)
        return

    if _h(t, "stop", "halt", "pause", "turn off", "close camera",
           "switch off", "camera off", "disable", "end"):
        _Q.put(stop_detection)
        return

    if _h(t, "snapshot", "capture", "photo", "picture", "save image"):
        _Q.put(take_snapshot)
        return

    if _h(t, "export pdf", "save pdf", "pdf report"):
        _Q.put(export_pdf_report)
        return

    if _h(t, "export", "save log", "download log"):
        _Q.put(export_log)
        return

    if _h(t, "heatmap", "heat map"):
        _Q.put(_toggle_heatmap)
        return

    if _h(t, "summary", "summarize", "report", "analytics", "statistics"):
        _Q.put(_speak_summary)
        return

    if _h(t, "camera zero", "camera one", "camera two", "camera three",
           "switch camera", "change camera", "next camera"):
        idx = 0
        if "one" in t or " 1" in t:     idx = 1
        elif "two" in t or " 2" in t:   idx = 2
        elif "three" in t or " 3" in t: idx = 3
        elif "next" in t or "switch" in t or "change" in t:
            idx = (S.camera_index + 1) % 4
        _Q.put(lambda i=idx: _switch_camera(i))
        return

    if _h(t, "help", "what can you do", "commands", "how do i", "instructions"):
        speak("Here are my voice commands. "
              "Say start or stop to control the camera. "
              "Say find bottle to track an object and get missing alerts. "
              "Say where is my phone for instant location. "
              "Say what do you see for a full scene description. "
              "Say train joystick to teach me a new object. "
              "Say stop training when done. "
              "Say remove bottle to stop tracking it. "
              "Say clear to remove all targets. "
              "Say snapshot to take a photo. "
              "Say summary for detection statistics. "
              "Say mute or unmute to control my voice.")
        return

    if _h(t, "stop training", "cancel training", "end training",
          "finish training", "done training", "training stop"):
        if S.training_mode:
            nm = S.training_name.replace("_", " ")
            cnt = S.training_count
            S.training_mode  = False
            S.selection_mode = False
            speak(f"Training for {nm} stopped. I captured {cnt} samples. "
                  f"{'Enough to train — say train ' + nm + ' again to retrain.' if cnt >= 30 else 'Try again with more samples.'}")
            _Q.put(lambda: train_lbl.config(text="TRAINING STOPPED", fg=ORANGE))
        else:
            speak("No training session is currently running.")
        return

    if _h(t, "training status", "how many photos", "how many samples",
          "training count", "training progress"):
        if S.training_mode:
            nm  = S.training_name.replace("_", " ")
            cnt = S.training_count
            need = S.TRAIN_NEEDED
            speak(f"Training {nm}. Captured {cnt} of {need} samples. "
                  f"{'Almost done!' if cnt > need * 0.8 else 'Keep the object visible in the selected box.'}")
        else:
            speak("No training is running right now. "
                  "Say train followed by the object name to start.")
        return

    if _h(t, "louder", "speak louder", "increase volume", "volume up"):
        _Q.put(lambda: _ui_log("Volume: max"))
        speak("OK, speaking louder now.")
        return

    if _h(t, "softer", "quieter", "speak softer", "decrease volume", "volume down"):
        _Q.put(lambda: _ui_log("Volume: normal"))
        speak("OK.")
        return

    if _h(t, "train", "teach", "remember this", "learn"):
        name = t
        for kw in ("train", "teach", "remember this as", "remember this",
                   "learn to find", "learn"):
            name = name.replace(kw, "").strip()
        name = (name.strip(" .,!?") or "custom_object").replace(" ", "_")
        _Q.put(lambda n=name: start_training_interactive(n))
        return

    if _h(t, "what do you see", "what can you see", "what's there",
           "look around", "describe", "announce", "tell me"):
        _Q.put(_announce_all)
        return

    if _h(t, "where is", "where's", "can you find", "is there a",
           "is the", "find my", "do you see", "can you see", "have you seen"):
        objs = _extract_objects(t)
        if objs:
            def _instant_check(obs=objs):
                for o in obs:
                    label = o.replace("_", " ")
                    if o in S.last_detected:
                        mem = _scene_mem.get(o)
                        d   = mem["direction"] if mem else "the frame"
                        pct = int(S.last_detected[o] * 100)
                        speak(f"Yes! Your {label} is visible on the {d} "
                              f"at {pct} percent confidence.")
                    else:
                        ago = _scene_mem.ago(o)
                        if 0 < ago < 120:
                            d    = _scene_mem.direction(o)
                            pred = _scene_mem.predicted_direction(o)
                            loc  = (f"It was last moving {pred} from the {d}."
                                    if pred else f"last spotted on the {d}.")
                            speak(f"I cannot see your {label} right now. "
                                  f"About {int(ago)} seconds ago it was on the {d}. "
                                  f"{loc}")
                        else:
                            speak(f"I cannot detect your {label} in the frame. "
                                  f"Try moving the camera around slowly.")
                        if o not in S.targets:
                            add_target(o)
                            speak("I will keep watching and alert you when it appears.")
            _Q.put(_instant_check)
        else:
            speak("I did not catch the object name. Could you say that again?")
        return

    if _h(t, "confidence", "threshold", "sensitivity"):
        for w in t.split():
            if w.isdigit():
                val = max(10, min(95, int(w))) / 100
                speak(f"Detection confidence is now {int(val * 100)} percent.")
                _Q.put(lambda v=val: _set_conf(v))
                break
        return

    if _h(t, "remove", "untrack", "stop tracking", "forget", "delete"):
        objs = _extract_objects(t)
        if objs:
            for o in objs:
                _Q.put(lambda x=o: remove_target(x))
            names = ", ".join(o.replace("_", " ") for o in objs)
            speak(f"No longer tracking {names}.")
        else:
            speak("Which object would you like me to stop tracking?")
        return

    if _h(t, "clear", "reset", "remove all", "clear all", "clear targets"):
        _Q.put(clear_targets)
        speak("All tracking targets have been cleared. Ready for new ones.")
        return

    if _h(t, "find", "look for", "search for", "track", "watch for",
           "detect", "locate", "monitor", "spot"):
        objs = _extract_objects(t)
        if objs:
            for o in objs:
                _Q.put(lambda x=o: add_target(x))   # add_target() speaks the confirmation
        else:
            speak("I did not recognise that object. "
                  "Try saying: find bottle, or find phone.")
        return

    objs = _extract_objects(t)
    if objs:
        for o in objs:
            _Q.put(lambda x=o: add_target(x))   # add_target() speaks the confirmation
        return

    _Q.put(lambda: _ui_log(f"Not understood: '{t}'"))
    speak("I am not sure I understood that. "
          "Try saying: find bottle, start, or what do you see.")

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION — ByteTrack + temporal smoothing + directional + heatmap
# ═══════════════════════════════════════════════════════════════════════════════
IMG_W, IMG_H = 880, 510

_infer_stop   = threading.Event()
_infer_thread = None
_video_stop   = threading.Event()
_video_thread = None

_raw_frame_lock  = threading.Lock()
_raw_frame_store = [None]

# ── canvas selection overlay state ────────────────────────────────────────────
_sel_start   = [None, None]
_sel_rect_id = [None]
_sel_raw     = [None]   # (x1,y1,x2,y2) canvas pixels

def _canvas_press(event):
    if not S.selection_mode:
        return
    _sel_start[0] = event.x
    _sel_start[1] = event.y
    if _sel_rect_id[0]:
        try:
            cam_canvas.delete(_sel_rect_id[0])
        except Exception:
            pass
    _sel_rect_id[0] = cam_canvas.create_rectangle(
        event.x, event.y, event.x, event.y,
        outline=GOLD, width=2, dash=(6, 3))

def _canvas_drag(event):
    if not S.selection_mode or _sel_start[0] is None:
        return
    cam_canvas.coords(_sel_rect_id[0],
                      _sel_start[0], _sel_start[1], event.x, event.y)

def _canvas_release(event):
    if not S.selection_mode or _sel_start[0] is None:
        return
    cw = cam_canvas.winfo_width()  or IMG_W
    ch = cam_canvas.winfo_height() or IMG_H
    x1 = min(_sel_start[0], event.x)
    y1 = min(_sel_start[1], event.y)
    x2 = max(_sel_start[0], event.x)
    y2 = max(_sel_start[1], event.y)
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        speak("The selection is too small. Please draw a larger box around the object.")
        return
    S.selection_box = (x1 / cw, y1 / ch, x2 / cw, y2 / ch)
    _sel_raw[0]     = (x1, y1, x2, y2)
    name = S.training_name.replace("_", " ")
    speak(f"Selection confirmed. Capturing {name} from the selected region now. "
          f"Please keep the object in frame.")
    _ui_log(f"Selection set for '{S.training_name}' — capturing…")
    _Q.put(lambda: train_lbl.config(text=f"CAPTURING: {S.training_name.upper()}", fg=GOLD))

def _redraw_sel():
    if _sel_raw[0] and S.selection_mode:
        x1, y1, x2, y2 = _sel_raw[0]
        if _sel_rect_id[0]:
            try:
                cam_canvas.delete(_sel_rect_id[0])
            except Exception:
                pass
        _sel_rect_id[0] = cam_canvas.create_rectangle(
            x1, y1, x2, y2, outline=GOLD2, width=2, dash=(6, 3))


def _inference_worker():
    global _USE_BYTETRACK
    while not _infer_stop.is_set():
        if not S.detecting or S.cap is None:
            time.sleep(0.02)
            continue
        try:
            ok, frame = S.cap.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            with _raw_frame_lock:
                _raw_frame_store[0] = frame.copy()

            S.frame_count += 1
            now    = time.time()
            ts_str = datetime.now().strftime("%H:%M:%S")

            infer_frame = cv2.resize(frame, (640, 360))
            # Low-light + overexposure enhancement
            enhanced    = _enhance_frame(infer_frame)
            frame_rgb   = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            fh, fw      = infer_frame.shape[:2]

            # Adaptive confidence: loosen for established tracks
            _base_conf = max(0.25, S.conf * 0.85)

            # ── YOLO (ByteTrack with automatic fallback) ──────────────────
            if _USE_BYTETRACK:
                try:
                    results = _base.track(
                        frame_rgb, conf=_base_conf, iou=0.40,
                        imgsz=640, persist=True, tracker="bytetrack.yaml",
                        verbose=False, device=_DEVICE, agnostic_nms=True)
                except Exception as e:
                    print(f"[INFER] ByteTrack unavailable ({e}), using predict.")
                    _USE_BYTETRACK = False
                    results = _base.predict(
                        frame_rgb, conf=_base_conf, iou=0.40,
                        imgsz=640, verbose=False, device=_DEVICE)
            else:
                results = _base.predict(
                    frame_rgb, conf=_base_conf, iou=0.40,
                    imgsz=640, verbose=False, device=_DEVICE)

            # ── raw detections with positions ─────────────────────────────
            raw:      dict = {}   # {name: (conf, cx_norm, cy_norm)}
            raw_boxes: dict = {}  # {name: (bx1,by1,bx2,by2) pixel on infer_frame}
            for r in results:
                for box in r.boxes:
                    name = _base.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if conf < _cat_thresh(name):
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cx = ((x1 + x2) / 2) / fw
                    cy = ((y1 + y2) / 2) / fh
                    if conf > raw.get(name, (0,))[0]:
                        raw[name]       = (conf, cx, cy)
                        raw_boxes[name] = (x1, y1, x2, y2)

            # ── CLIP async semantic verification (every 6 frames) ─────────
            if HAS_CLIP and S.frame_count % 6 == 0:
                for obj_n, (bx1, by1, bx2, by2) in raw_boxes.items():
                    bx1c = max(0, bx1); by1c = max(0, by1)
                    bx2c = min(fw, bx2); by2c = min(fh, by2)
                    if bx2c > bx1c + 8 and by2c > by1c + 8:
                        crop = infer_frame[by1c:by2c, bx1c:bx2c].copy()
                        try:
                            _clip_verify_q.put_nowait((obj_n, crop))
                        except queue.Full:
                            pass

            # ── Reject CLIP-invalidated detections ────────────────────────
            if HAS_CLIP:
                raw = {n: v for n, v in raw.items()
                       if _clip_valid.get(n, True)}

            # ── EMA tracker smoothing ─────────────────────────────────────
            raw = _obj_tracker.update(raw)

            # ── custom model — OVERRIDES base YOLO for trained objects ────
            if _custom_model:
                try:
                    cres = _custom_model.predict(
                        infer_frame, conf=max(0.15, S.conf - 0.15),
                        iou=0.45, imgsz=640, verbose=False)
                    for r in cres:
                        for box in r.boxes:
                            name = _custom_model.names[int(box.cls[0])].lower()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx = ((x1 + x2) / 2) / fw
                            cy = ((y1 + y2) / 2) / fh
                            # Custom model always wins for its trained classes
                            raw[name] = (conf, cx, cy)
                except Exception:
                    pass

            # ── Alias resolution: map user-tracked names → YOLO detections ─
            # e.g. user tracks "joystick", YOLO detects "remote" →
            # raw["joystick"] = raw["remote"] so alert fires correctly
            for target in list(S.targets):
                if target in raw:
                    continue
                for alias in _aliases_for_target(target):
                    alias_n = alias.lower().replace("_", " ")
                    alias_u = alias.lower().replace(" ", "_")
                    hit = raw.get(alias_n) or raw.get(alias_u)
                    if hit:
                        raw[target] = hit   # alias satisfies the target
                        break

            # ── temporal smoothing ────────────────────────────────────────
            all_cands = set(raw.keys()) | set(S.targets)
            detected: dict = {}
            for obj in all_cands:
                if _smoother.update(obj, obj in raw) and obj in raw:
                    detected[obj] = raw[obj][0]

            # ── update scene memory & heatmap ─────────────────────────────
            for obj, (conf, cx, cy) in raw.items():
                direction = _direction_from_norm(cx, cy)
                _scene_mem.update(obj, conf, cx, cy, direction)
                S.session_detections[obj] = S.session_detections.get(obj, 0) + 1
                _db_log(obj, "DETECTED", conf, cx, cy, direction)

            if S.show_heatmap and S.heatmap_data is not None:
                for obj, (conf, cx, cy) in raw.items():
                    px = int(cx * fw)
                    py = int(cy * fh)
                    cv2.circle(S.heatmap_data, (px, py), 20, float(conf), -1)
                S.heatmap_data = np.clip(
                    cv2.GaussianBlur(S.heatmap_data, (0, 0), 5), 0, 1)

            # ── training frame capture ────────────────────────────────────
            if S.training_mode:
                _handle_training_frame(frame)

            # ── target alert logic ────────────────────────────────────────
            if not hasattr(S, "obj_state"):
                S.obj_state = {}

            # Timing constants
            FOUND_CONFIRM_FRAMES = 3    # consecutive detections before announcing found
            MISSING_CONFIRM_SEC  = 2.0  # seconds missing before first alert
            REPEAT_ALERT_SEC     = 5.0  # repeat missing alert every 5 seconds
            status_rows = []

            for obj in list(S.targets):
                label = obj.replace("_", " ")
                if obj not in S.obj_state:
                    S.obj_state[obj] = {
                        "state":         "UNKNOWN",
                        "found_frames":  0,
                        "missing_since": None,
                        "last_voice":    0.0,
                    }
                st = S.obj_state[obj]

                # ── OBJECT IS VISIBLE ─────────────────────────────────────
                if obj in detected:
                    conf_val  = detected[obj]
                    mem       = _scene_mem.get(obj)
                    direction = mem["direction"] if mem and "direction" in mem else "center"
                    st["found_frames"] += 1
                    st["missing_since"] = None
                    status_rows.append(("FOUND", obj, int(conf_val * 100)))

                    # Confirm found only after FOUND_CONFIRM_FRAMES in a row
                    if (st["found_frames"] >= FOUND_CONFIRM_FRAMES
                            and st["state"] != "FOUND"):
                        prev = st["state"]
                        st["state"]      = "FOUND"
                        st["last_voice"] = now
                        if prev == "MISSING":
                            # Was missing — now back: short clear announcement
                            speak(f"{label} found! It is on the {direction}.",
                                  force=True)
                            print(f"[VOICE] REAPPEARED: {label} @ {direction}")
                        else:
                            # First time seen this session
                            speak(f"{label} found. {direction}.", force=True)
                            print(f"[VOICE] FOUND: {label} @ {direction}")

                # ── OBJECT IS MISSING ─────────────────────────────────────
                else:
                    st["found_frames"] = 0
                    if st["missing_since"] is None:
                        st["missing_since"] = now
                    dur = now - st["missing_since"]
                    status_rows.append(("MISSING", obj, int(dur)))

                    if dur >= MISSING_CONFIRM_SEC:
                        if st["state"] != "MISSING":
                            # First missing alert — say it clearly, interrupt anything
                            st["state"]      = "MISSING"
                            st["last_voice"] = now
                            speak(f"{label} missing! {label} missing!", force=True)
                            print(f"[VOICE] MISSING (first): {label}")

                        elif now - st["last_voice"] >= REPEAT_ALERT_SEC:
                            # Repeating alert — do NOT use force so we don't cut off
                            # a currently-playing alert mid-sentence
                            st["last_voice"] = now
                            speak(f"{label} missing.")
                            print(f"[VOICE] MISSING (repeat): {label} — {int(dur)}s")

            # ── annotated frame ───────────────────────────────────────────
            annotated_bgr = results[0].plot(line_width=2, labels=True, conf=True)
            ah, aw        = annotated_bgr.shape[:2]

            # ── heatmap overlay ───────────────────────────────────────────
            if S.show_heatmap and S.heatmap_data is not None:
                hm_norm  = cv2.normalize(S.heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
                hm_u8    = hm_norm.astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
                hm_color = cv2.resize(hm_color, (aw, ah))
                mask     = cv2.resize(
                    (hm_u8 > 20).astype(np.uint8), (aw, ah)).astype(bool)
                blended  = cv2.addWeighted(annotated_bgr, 0.55, hm_color, 0.45, 0)
                annotated_bgr[mask] = blended[mask]

            # ── motion arrows ─────────────────────────────────────────────
            for obj_n, (_, cx_n, cy_n) in raw.items():
                pts = _obj_tracker.arrow_pts(obj_n, int(cx_n * aw), int(cy_n * ah))
                if pts:
                    cv2.arrowedLine(annotated_bgr,
                                    (pts[0], pts[1]), (pts[2], pts[3]),
                                    (0, 210, 255), 2, tipLength=0.35)

            # ── training overlay ──────────────────────────────────────────
            if S.training_mode:
                pct   = S.training_count / S.TRAIN_NEEDED
                bar_w = int(aw * pct)
                if S.selection_box:
                    x1n, y1n, x2n, y2n = S.selection_box
                    bx1 = int(x1n * aw); by1 = int(y1n * ah)
                    bx2 = int(x2n * aw); by2 = int(y2n * ah)
                    cv2.rectangle(annotated_bgr, (bx1, by1), (bx2, by2),
                                  (201, 169, 110), 3)
                    cv2.putText(annotated_bgr, "SELECTED REGION — CAPTURING",
                                (bx1 + 4, max(by1 - 6, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (201, 169, 110), 2, cv2.LINE_AA)
                else:
                    rx1, rx2 = int(aw * 0.20), int(aw * 0.80)
                    ry1, ry2 = int(ah * 0.20), int(ah * 0.80)
                    cv2.rectangle(annotated_bgr, (rx1, ry1), (rx2, ry2),
                                  (0, 215, 255), 3)
                    cv2.putText(annotated_bgr,
                                "DRAW A SELECTION BOX ON THE CAMERA VIEW",
                                (rx1 + 8, ry1 + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                                (0, 215, 255), 2, cv2.LINE_AA)
                overlay = annotated_bgr.copy()
                cv2.rectangle(overlay, (0, 0), (aw, 38), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_bgr, 0.4, 0, annotated_bgr)
                cv2.rectangle(annotated_bgr, (0, 0), (bar_w, 38),
                              (201, 169, 110), -1)
                tlbl = (f"TRAINING '{S.training_name.upper()}'"
                        f"  {S.training_count}/{S.TRAIN_NEEDED}")
                cv2.putText(annotated_bgr, tlbl, (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
                            cv2.LINE_AA)

            fps = 1.0 / (now - S.prev_t) if S.prev_t else 0
            S.prev_t = now

            with S.result_lock:
                S.latest_annotated = annotated_bgr
                S.latest_detected  = detected
                S._status_rows     = status_rows
                S._fps             = fps

        except Exception as e:
            print(f"[INFER] {e}")
            traceback.print_exc()
            time.sleep(0.05)

    print("[INFER] Worker stopped.")


_fl_last_ts = [0.0]   # track last frame-loop render time for skip logic

def _frame_loop():
    if not S.detecting and not S.video_mode:
        return

    with S.result_lock:
        annotated_bgr = S.latest_annotated
        detected      = S.latest_detected
        status_rows   = getattr(S, "_status_rows", [])
        fps           = getattr(S, "_fps", 0.0)
    S.last_detected = detected

    if annotated_bgr is not None:
        now_fl = time.time()
        # Skip rendering if less than 33ms since last render (cap at ~30fps UI)
        if now_fl - _fl_last_ts[0] < 0.033:
            root.after(10, _frame_loop)
            return
        _fl_last_ts[0] = now_fl

        cw, ch  = _get_canvas_size()
        img_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        # Use NEAREST for speed when fps < 12, BILINEAR otherwise
        resample = Image.BILINEAR if fps > 12 else Image.NEAREST
        pil_img = Image.fromarray(img_rgb).resize((cw, ch), resample)
        itk     = ImageTk.PhotoImage(pil_img)
        cam_canvas.imgtk = itk
        cam_canvas.delete("all")
        cam_canvas.create_image(0, 0, anchor="nw", image=itk)

        _redraw_sel()

        if S.video_mode and S.video_total_frames > 0:
            pct   = S.video_current_frame / S.video_total_frames
            bar_w = int(cw * pct)
            cam_canvas.create_rectangle(0, ch - 6, cw, ch,
                                        fill="#0A0C14", outline="")
            cam_canvas.create_rectangle(0, ch - 6, bar_w, ch,
                                        fill=GOLD, outline="")
            prog_txt = (f"Processing  {S.video_current_frame}/{S.video_total_frames}"
                        f"  ({int(pct*100)}%)")
            cam_canvas.create_text(cw // 2, ch - 3, anchor="center",
                                   text=prog_txt,
                                   font=("Courier", 7, "bold"), fill=WHITE)

        if S.detecting:
            _ui_update_fps(fps)
            _ui_update_det_bar(detected)
            _ui_update_status_bars(status_rows)

    root.after(33, _frame_loop)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
def _switch_camera(idx: int):
    was = S.detecting
    if was:
        stop_detection()
        time.sleep(0.4)
    S.camera_index = idx
    speak(f"Switching to camera {idx}.")
    _ui_log(f"Camera index → {idx}")
    if was:
        start_detection()

def _camera_is_live(cap) -> bool:
    """Return True only if camera produces real live video (not a static placeholder)."""
    frames = []
    for _ in range(4):
        ret, f = cap.read()
        if not ret or f is None:
            return False
        frames.append(f.astype(np.float32))
        time.sleep(0.04)
    # Static virtual cameras (DroidCam logo etc.) produce zero diff between frames.
    diff = sum(cv2.absdiff(frames[i], frames[i + 1]).mean()
               for i in range(len(frames) - 1))
    return diff > 0.05

def start_detection():
    global _infer_thread
    if S.detecting:
        return
    cam = None
    indices_to_try = list(dict.fromkeys([0, 1, 2, 3, S.camera_index]))
    backends_to_try = [cv2.CAP_MSMF, cv2.CAP_ANY, cv2.CAP_DSHOW]
    opened_idx = S.camera_index
    for idx in indices_to_try:
        for backend in backends_to_try:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c and c.isOpened():
                    if _camera_is_live(c):
                        cam = c
                        opened_idx = idx
                        break
                    else:
                        _ui_log(f"Skipped camera {idx} (static/virtual)")
                        c.release()
            except Exception:
                pass
        if cam and cam.isOpened():
            break
    S.camera_index = opened_idx
    if not cam or not cam.isOpened():
        speak("I cannot open the camera. Please check it is connected.")
        _ui_set_cam("ERROR", RED)
        return
    # Use 640×480 — universally supported by all built-in webcams.
    # Inference resizes internally; forcing 1280×720 breaks some drivers.
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    S.cap           = cam
    S.detecting     = True
    S.prev_t        = time.time()
    S.heatmap_data  = np.zeros((360, 640), dtype=np.float32)
    S.session_start = time.time()

    for obj in S.targets:
        S.miss_start[obj] = None
        S.last_alert[obj] = None
    # Reset per-object alert state so every restart begins clean
    S.obj_state = {}
    _smoother.clear()

    _infer_stop.clear()
    _infer_thread = threading.Thread(
        target=_inference_worker, daemon=True, name="infer")
    _infer_thread.start()

    _db_start_session()
    _ui_set_cam("LIVE", TEAL)
    btn_start.config(state="disabled", bg=CARD2)
    btn_stop.config(state="normal",   bg=RED)
    speak("Camera on.")
    _ui_log("Detection started")
    _frame_loop()

def stop_detection():
    if not S.detecting:
        return
    S.detecting      = False
    S.selection_mode = False
    S.selection_box  = None
    _sel_raw[0]      = None
    _obj_tracker.clear()
    _infer_stop.set()
    _db_end_session(S.frame_count)
    if S.cap:
        S.cap.release()
        S.cap = None
    _ui_set_cam("STANDBY", ORANGE)
    btn_start.config(state="normal",  bg=GREEN)
    btn_stop.config(state="disabled", bg=CARD2)
    _draw_standby()
    speak("Camera off.", force=True)
    _ui_log("Detection stopped")

def take_snapshot():
    with _raw_frame_lock:
        frame = _raw_frame_store[0]
    if frame is None or not S.detecting:
        speak("The camera is not active. Please start detection first.")
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
    cv2.imwrite(path, frame)
    speak("Snapshot saved successfully.")
    _ui_log(f"Snapshot saved: snap_{ts}.jpg")

def export_log():
    if not S.log:
        speak("The detection log is empty. Start detection to generate data.")
        return
    fname = os.path.join(ROOT_DIR,
                         f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Object", "Status", "Confidence", "Direction"])
        w.writerows(S.log)
    speak("Detection log has been saved as a CSV file.")
    _ui_log(f"Log exported: {os.path.basename(fname)}")

def export_pdf_report():
    if not HAS_FPDF:
        speak("PDF export requires fpdf2. Run:  pip install fpdf2")
        _ui_log("Install fpdf2: pip install fpdf2")
        return
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Missing Object Detection — Session Report",
                 ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8,
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ln=True)
        dur = int(time.time() - S.session_start)
        pdf.cell(0, 8,
                 f"Session: {dur}s  |  Frames: {S.frame_count}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detection Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for obj, count in sorted(S.session_detections.items(),
                                  key=lambda x: -x[1]):
            pdf.cell(0, 7, f"  {obj}: detected {count} times", ln=True)
        pdf.ln(4)
        if S.log:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Detection Log (last 100 entries)", ln=True)
            pdf.set_font("Courier", "", 8)
            for hdr, w_ in [("Time", 20), ("Object", 40), ("Status", 22),
                             ("Conf", 22), ("Direction", 50)]:
                pdf.cell(w_, 6, hdr, border=1)
            pdf.ln()
            for row in S.log[-100:]:
                row_p = list(row) + [""] * (5 - len(row))
                for cell, w_ in zip(row_p, [20, 40, 22, 22, 50]):
                    pdf.cell(w_, 5, str(cell)[:24], border=1)
                pdf.ln()
        fname = os.path.join(REPORTS_DIR,
                             f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(fname)
        speak("PDF report saved to the reports folder.")
        _ui_log(f"PDF saved: {os.path.basename(fname)}")
    except Exception as e:
        speak("Sorry, I could not generate the PDF report.")
        _ui_log(f"PDF error: {e}")

def _toggle_heatmap():
    S.show_heatmap = not S.show_heatmap
    if S.show_heatmap and S.heatmap_data is None:
        S.heatmap_data = np.zeros((360, 640), dtype=np.float32)
    state = "enabled" if S.show_heatmap else "disabled"
    speak(f"Detection heatmap {state}.")
    _ui_log(f"Heatmap {state}")

def _speak_summary():
    if not S.session_detections:
        speak("No detections recorded yet in this session.")
        return
    top  = sorted(S.session_detections.items(), key=lambda x: -x[1])[:5]
    dur  = int(time.time() - S.session_start)
    parts = ", ".join(f"{o.replace('_', ' ')} {c} times" for o, c in top)
    speak(f"In this {dur} second session I detected: {parts}.")

def _set_conf(v: float):
    S.conf = v
    conf_lbl.config(text=f"{int(v * 100)}%")
    conf_slider.set(int(v * 100))
    _ui_log(f"Confidence: {int(v * 100)}%")

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def _video_worker(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        def _err():
            _ui_log("ERROR: Cannot open video file.")
            speak("I cannot open that video file.")
            btn_load_video.config(state="normal", text="📁  LOAD VIDEO", bg=CARD)
            S.video_mode = False
        _Q.put(_err)
        return

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_name = os.path.splitext(os.path.basename(path))[0] + "_annotated.mp4"
    out_path = os.path.join(VIDEO_OUTPUT_DIR, out_name)
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               src_fps, (vid_w, vid_h))

    S.video_total_frames  = total
    S.video_current_frame = 0
    S.video_output_path   = out_path
    frame_idx = 0

    while not _video_stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        S.video_current_frame = frame_idx
        infer     = cv2.resize(frame, (640, 360))
        rgb       = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        res       = _base.predict(rgb, conf=max(0.40, S.conf),
                                  iou=0.45, imgsz=640, verbose=False)
        ann_small = res[0].plot(line_width=2, labels=True, conf=True)
        ann_full  = cv2.resize(ann_small, (vid_w, vid_h),
                               interpolation=cv2.INTER_LINEAR)
        writer.write(ann_full)
        with S.result_lock:
            S.latest_annotated = ann_full
        time.sleep(max(0.0, 1.0 / src_fps - 0.005))

    cap.release()
    writer.release()
    S.video_mode = False

    def _done():
        if _video_stop.is_set() and frame_idx < total:
            _ui_log("Video processing cancelled.")
            speak("Video processing cancelled.")
        else:
            _ui_log(f"Annotated video saved → video_output/{out_name}")
            speak("Video processing complete. Annotated output saved.")
        topbar_lbl.config(text="Live Camera Feed")
        btn_load_video.config(state="normal", text="📁  LOAD VIDEO", bg=CARD)
        btn_stop_video.config(state="disabled", bg=CARD2, fg=MUTED)
        _draw_standby()
    _Q.put(_done)

def load_video_file():
    global _video_thread
    if S.detecting:
        speak("Please stop the camera first before loading a video.")
        return
    if S.video_mode:
        speak("A video is already being processed.")
        return
    path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
            ("All files",   "*.*"),
        ])
    if not path:
        return
    S.video_mode       = True
    S.latest_annotated = None
    _video_stop.clear()
    btn_load_video.config(state="disabled", text="Processing…", bg=CARD2)
    btn_stop_video.config(state="normal",   bg=RED, fg=WHITE)
    topbar_lbl.config(text=f"Annotated Output Video — {os.path.basename(path)}")
    _ui_log(f"Loading video: {os.path.basename(path)}")
    speak("Video loaded. Processing has begun.")
    _video_thread = threading.Thread(
        target=_video_worker, args=(path,), daemon=True, name="video_infer")
    _video_thread.start()
    _frame_loop()

def stop_video_processing():
    if not S.video_mode:
        return
    _video_stop.set()

def _announce_all():
    if not S.detecting:
        speak("The camera is not running. Please say start to begin detection.")
        return
    if S.last_detected:
        top   = sorted(S.last_detected, key=lambda x: -S.last_detected[x])[:6]
        parts = []
        for obj in top:
            mem = _scene_mem.get(obj)
            d   = f" on the {mem['direction']}" if mem else ""
            pct = int(S.last_detected[obj] * 100)
            parts.append(f"{obj.replace('_', ' ')}{d} at {pct} percent confidence")
        speak("I can see: " + ". ".join(parts) + ".")
    else:
        speak("I cannot see anything clearly in the frame right now. "
              "Try moving the camera slowly to scan the area.")

def _get_canvas_size():
    root.update_idletasks()
    w = cam_canvas.winfo_width()
    h = cam_canvas.winfo_height()
    return (w if w >= 100 else IMG_W), (h if h >= 100 else IMG_H)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _img_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def _phash(img: np.ndarray) -> int:
    """Perceptual hash — detects near-duplicate frames, not just byte-identical."""
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    mean  = float(gray.mean())
    bits  = gray.flatten() > mean
    return int(sum(int(b) << i for i, b in enumerate(bits)))

def _phash_dist(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _grabcut_clean(frame: np.ndarray,
                   x1n: float, y1n: float, x2n: float, y2n: float) -> np.ndarray:
    """Apply GrabCut to isolate the selected object from background.
    Returns the frame with background pixels darkened (YOLO still needs full frame)."""
    fh, fw = frame.shape[:2]
    x1 = max(4, int(x1n * fw)); y1 = max(4, int(y1n * fh))
    x2 = min(fw - 4, int(x2n * fw)); y2 = min(fh - 4, int(y2n * fh))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return frame
    try:
        mask  = np.zeros(frame.shape[:2], np.uint8)
        bgdm  = np.zeros((1, 65), np.float64)
        fgdm  = np.zeros((1, 65), np.float64)
        rect  = (x1, y1, x2 - x1, y2 - y1)
        cv2.grabCut(frame, mask, rect, bgdm, fgdm, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                           255, 0).astype(np.uint8)
        # Soften the mask edges
        fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
        result  = frame.copy()
        bg_mask = fg_mask == 0
        result[bg_mask] = (result[bg_mask] * 0.3).astype(np.uint8)
        return result
    except Exception:
        return frame


def _augment(img: np.ndarray) -> list:
    h_, w_ = img.shape[:2]
    cx_, cy_ = w_ // 2, h_ // 2
    results = []
    # Horizontal flip
    results.append(cv2.flip(img, 1))
    # Brightness variants
    results.append(np.clip(img.astype(np.float32) * 1.25, 0, 255).astype(np.uint8))
    results.append(np.clip(img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8))
    # Rotation
    for angle in (10, -10, 5, -5):
        M = cv2.getRotationMatrix2D((cx_, cy_), angle, 1.0)
        results.append(cv2.warpAffine(img, M, (w_, h_)))
    # Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    results.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    # Contrast stretch
    results.append(np.clip((img.astype(np.float32) - 127) * 1.20 + 127, 0, 255).astype(np.uint8))
    # Saturation boost
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
        results.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    except Exception:
        pass
    # Slight perspective warp
    try:
        pts1 = np.float32([[0, 0], [w_, 0], [0, h_], [w_, h_]])
        pts2 = np.float32([[int(w_*0.03), int(h_*0.03)], [int(w_*0.97), 0],
                            [0, h_], [w_, h_]])
        Mper = cv2.getPerspectiveTransform(pts1, pts2)
        results.append(cv2.warpPerspective(img, Mper, (w_, h_)))
    except Exception:
        pass
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM OBJECT TRAINING — interactive selection-based
# ═══════════════════════════════════════════════════════════════════════════════
def start_training_interactive(name: str):
    if S.training_mode:
        speak("Training is already running. Please wait for it to finish.")
        return
    if not S.detecting:
        speak("Starting the camera first. Then draw a box on screen to select your object.")
        start_detection()
        root.after(1600, lambda: start_training_interactive(name))
        return

    name = name.strip().lower().replace(" ", "_") or "custom_object"
    labels = (json.load(open(CUSTOM_LABELS_F))
              if os.path.exists(CUSTOM_LABELS_F) else {})
    if name not in labels:
        labels[name] = len(labels)
    S.training_labels   = labels
    S.training_class_id = labels[name]

    img_dir = os.path.join(TRAINING_DIR, "images", name)
    lbl_dir = os.path.join(TRAINING_DIR, "labels", name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        try: os.remove(os.path.join(img_dir, f))
        except Exception: pass
    for f in os.listdir(lbl_dir):
        try: os.remove(os.path.join(lbl_dir, f))
        except Exception: pass

    S.selection_box  = None
    S.selection_mode = True
    _sel_raw[0]      = None
    if _sel_rect_id[0]:
        try: cam_canvas.delete(_sel_rect_id[0])
        except Exception: pass
    _sel_rect_id[0] = None

    S.training_mode  = True
    S.training_name  = name
    S.training_count = 0

    speak(f"Training mode started for {name.replace('_', ' ')}. "
          f"Click and drag on the camera view to draw a box around the object. "
          f"I will capture {S.TRAIN_NEEDED} photos automatically from that region.")
    _ui_log(f"INTERACTIVE TRAINING: Draw box on camera for '{name}'")
    _Q.put(lambda: train_lbl.config(text=f"DRAW BOX: {name.upper()}", fg=GOLD))


def _handle_training_frame(frame: np.ndarray):
    if S.training_count >= S.TRAIN_NEEDED:
        if S.training_count == S.TRAIN_NEEDED:
            S.training_count += 1
            name   = S.training_name
            labels = S.training_labels.copy()
            threading.Thread(
                target=_run_training, args=(name, labels),
                daemon=True, name="trainer").start()
        return

    if not S.selection_box:
        return
    if S.frame_count % 3 != 0:
        return

    x1n, y1n, x2n, y2n = S.selection_box
    fh, fw = frame.shape[:2]
    x1 = max(0, int(x1n * fw));  y1 = max(0, int(y1n * fh))
    x2 = min(fw, int(x2n * fw)); y2 = min(fh, int(y2n * fh))

    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return

    # Skip blurry frames (Laplacian variance)
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 25:
        return

    ts      = datetime.now().strftime("%H%M%S_%f")
    img_dir = os.path.join(TRAINING_DIR, "images", S.training_name)
    lbl_dir = os.path.join(TRAINING_DIR, "labels", S.training_name)

    # GrabCut: darken background so model focuses on selected object
    save_frame = _grabcut_clean(frame, x1n, y1n, x2n, y2n)
    cv2.imwrite(os.path.join(img_dir, f"{ts}.jpg"), save_frame)

    # Accurate YOLO label from exact selection coordinates
    cx = (x1n + x2n) / 2
    cy = (y1n + y2n) / 2
    bw = x2n - x1n
    bh = y2n - y1n
    with open(os.path.join(lbl_dir, f"{ts}.txt"), "w") as lf:
        lf.write(f"{S.training_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    S.training_count += 1
    if S.training_count == 10:
        speak(f"10 photos captured. Keep the object in the box.")
    elif S.training_count % 20 == 0:
        remain = S.TRAIN_NEEDED - S.training_count
        speak(f"{S.training_count} of {S.TRAIN_NEEDED} done. "
              f"{'Almost there!' if remain <= 20 else str(remain) + ' more to go.'}")


def _run_training(name: str, labels: dict):
    try:
        speak("All images captured! Starting model training now. "
              "This takes a few minutes. The camera will keep running.")
        _Q.put(lambda: train_lbl.config(text="AUGMENTING DATASET…", fg=ORANGE))

        # ── deduplicate + augment ─────────────────────────────────────────
        for obj_name in labels.keys():
            img_dir = os.path.join(TRAINING_DIR, "images", obj_name)
            lbl_dir = os.path.join(TRAINING_DIR, "labels", obj_name)
            if not os.path.exists(img_dir):
                continue

            orig_imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

            # Remove near-duplicates: MD5 (exact) + perceptual hash (similar frames)
            seen_md5:   set  = set()
            seen_ph:    list = []   # list of perceptual hashes
            PH_THRESH        = 12  # hamming distance — lower = stricter
            for fname in list(orig_imgs):
                fpath = os.path.join(img_dir, fname)
                md5   = _img_hash(fpath)
                img_c = cv2.imread(fpath)
                ph    = _phash(img_c) if img_c is not None else 0
                is_dup = (md5 in seen_md5 or
                          any(_phash_dist(ph, s) < PH_THRESH for s in seen_ph))
                if is_dup:
                    try:
                        os.remove(fpath)
                        lp = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
                        if os.path.exists(lp):
                            os.remove(lp)
                        orig_imgs.remove(fname)
                    except Exception:
                        pass
                else:
                    seen_md5.add(md5)
                    seen_ph.append(ph)

            # Augment if fewer than 60 clean images
            if len(orig_imgs) < 60:
                for fname in orig_imgs:
                    src_img = os.path.join(img_dir, fname)
                    src_lbl = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
                    if not (os.path.exists(src_img) and os.path.exists(src_lbl)):
                        continue
                    img = cv2.imread(src_img)
                    if img is None:
                        continue
                    parts = open(src_lbl).read().strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = parts[0]
                    cx_o, cy_o, bw_o, bh_o = (float(p) for p in parts[1:5])

                    for i, aug_img in enumerate(_augment(img), 1):
                        aug_fname = fname.replace(".jpg", f"_aug{i}.jpg")
                        cv2.imwrite(os.path.join(img_dir, aug_fname), aug_img)
                        aug_cx = (1.0 - cx_o) if i == 1 else cx_o  # mirror for hflip
                        with open(os.path.join(lbl_dir,
                                  aug_fname.replace(".jpg", ".txt")), "w") as lf:
                            lf.write(f"{cls_id} {aug_cx:.6f} {cy_o:.6f} "
                                     f"{bw_o:.6f} {bh_o:.6f}\n")

        _Q.put(lambda: train_lbl.config(text="TRAINING MODEL…", fg=ORANGE))

        # ── build YAML ────────────────────────────────────────────────────
        names_inv  = {v: k for k, v in labels.items()}
        train_dirs = []
        for obj_name in labels.keys():
            oi = os.path.join(TRAINING_DIR, "images", obj_name)
            ol = os.path.join(TRAINING_DIR, "labels", obj_name)
            if (os.path.exists(oi) and os.listdir(oi) and
                    os.path.exists(ol) and os.listdir(ol)):
                train_dirs.append(obj_name)

        if not train_dirs:
            speak("No training images were found. Please try again.")
            _Q.put(lambda: train_lbl.config(text="NO IMAGES", fg=RED))
            S.training_mode  = False
            S.selection_mode = False
            return

        nc         = len(labels)
        names_list = [names_inv[i] for i in range(nc)]
        td_fwd     = TRAINING_DIR.replace("\\", "/")

        yaml_path = os.path.join(TRAINING_DIR, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(f"path: {td_fwd}\n")
            if len(train_dirs) == 1:
                yf.write(f"train: images/{train_dirs[0]}\n")
                yf.write(f"val:   images/{train_dirs[0]}\n")
            else:
                yf.write("train:\n")
                for d in train_dirs:
                    yf.write(f"  - images/{d}\n")
                yf.write("val:\n")
                yf.write(f"  - images/{train_dirs[-1]}\n")
            yf.write(f"nc: {nc}\n")
            yf.write("names:\n")
            for nm in names_list:
                yf.write(f"  - {nm}\n")

        # Try YOLO11n first (latest), fall back to YOLOv8n
        _train_base = "yolo11n.pt"
        try:
            YOLO(_train_base)
        except Exception:
            _train_base = "yolov8n.pt"
        trainer = YOLO(_train_base)
        use_gpu = _DEVICE.startswith("cuda")
        trainer.train(
            data     = yaml_path,
            epochs   = 60,
            imgsz    = 640,
            batch    = 16 if use_gpu else 8,
            patience = 20,
            project  = TRAINING_DIR,
            name     = "run",
            exist_ok = True,
            verbose  = False,
            workers  = 0,
            amp      = use_gpu,
            device   = _DEVICE,
            lr0      = 0.005,
            lrf      = 0.01,
            mosaic   = 1.0,
            mixup    = 0.1,
            copy_paste = 0.1,
            degrees  = 10.0,
            flipud   = 0.1,
            fliplr   = 0.5,
        )

        best_pt = os.path.join(TRAINING_DIR, "run", "weights", "best.pt")
        if os.path.exists(best_pt):
            shutil.copy(best_pt, CUSTOM_MODEL_F)
            json.dump(labels, open(CUSTOM_LABELS_F, "w"))
            _load_custom_model()

            # ── Generate CLIP embeddings for trained object ────────────────
            if HAS_CLIP:
                _Q.put(lambda: train_lbl.config(text="CLIP EMBEDDING…", fg=TEAL))
                for obj_nm in labels.keys():
                    img_dir_e = os.path.join(TRAINING_DIR, "images", obj_nm)
                    if not os.path.exists(img_dir_e):
                        continue
                    feats = []
                    for fn in os.listdir(img_dir_e)[:40]:
                        if not fn.endswith(".jpg"):
                            continue
                        im = cv2.imread(os.path.join(img_dir_e, fn))
                        if im is None:
                            continue
                        try:
                            pil_e = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                            t_e   = _clip_preprocess(pil_e).unsqueeze(0)
                            with torch.no_grad():
                                f_e = _clip_model.encode_image(t_e)
                                f_e = f_e / f_e.norm(dim=-1, keepdim=True)
                                feats.append(f_e.cpu().numpy())
                        except Exception:
                            pass
                    if feats:
                        avg = np.mean(feats, axis=0).flatten()
                        avg /= np.linalg.norm(avg)
                        _clip_embeddings[obj_nm] = avg
                try:
                    np.savez(CLIP_EMBED_F, **_clip_embeddings)
                    print(f"[CLIP] Embeddings saved for: {list(_clip_embeddings.keys())}")
                except Exception as e:
                    print(f"[CLIP] Embed save error: {e}")

            _Q.put(lambda n=name: add_target(n))
            speak(f"Excellent! Training is complete. "
                  f"I am now tracking your {name.replace('_', ' ')}. "
                  f"I will alert you as soon as I detect it.")
            _Q.put(lambda n=name: (
                train_lbl.config(text=f"TRAINED: {n.upper()}  ✓", fg=GREEN),
                _ui_log(f"Custom model ready: {n}")))
        else:
            speak("Training finished but no model file was saved. Please try again.")
            _Q.put(lambda: train_lbl.config(text="TRAINING FAILED", fg=RED))

    except Exception as e:
        traceback.print_exc()
        print(f"[TRAIN] {e}")
        speak(f"Training encountered an error. {str(e)[:60]}")
        _Q.put(lambda: train_lbl.config(text="ERROR", fg=RED))
    finally:
        S.training_mode  = False
        S.selection_mode = False
        S.selection_box  = None
        _sel_raw[0]      = None

# ═══════════════════════════════════════════════════════════════════════════════
# GUI — ROOT WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("Missing Object Detection v3  —  Enterprise AI Surveillance Platform")
root.state("zoomed")
root.minsize(1100, 700)
root.resizable(True, True)
root.configure(bg=BG)

F_BRAND    = ("Georgia", 22, "bold")
F_BRAND_SM = ("Helvetica", 9, "italic")
F_CAP      = ("Helvetica", 7, "bold")
F_MONO     = ("Courier", 9)
F_MONO_B   = ("Courier", 10, "bold")
F_BODY     = ("Helvetica", 9)
F_BODY_B   = ("Helvetica", 9, "bold")
F_NUM      = ("Georgia", 16, "bold")
F_BTN      = ("Helvetica", 10, "bold")
F_CMD      = ("Courier", 8)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
SB_W = 362

sb_outer = tk.Frame(root, bg=PANEL, width=SB_W)
sb_outer.pack(side="left", fill="y")
sb_outer.pack_propagate(False)

tk.Frame(sb_outer, bg=GOLD, height=2).pack(fill="x", side="top")

sb_canvas  = tk.Canvas(sb_outer, bg=PANEL, highlightthickness=0, bd=0)
sb_vscroll = tk.Scrollbar(sb_outer, orient="vertical",
                           command=sb_canvas.yview,
                           bg=PANEL, troughcolor=BDR2, relief="flat")
sb_canvas.configure(yscrollcommand=sb_vscroll.set)

sb_vscroll.pack(side="right", fill="y")
sb_canvas.pack(side="left", fill="both", expand=True)

sb    = tk.Frame(sb_canvas, bg=PANEL)
_sb_w = sb_canvas.create_window((0, 0), window=sb, anchor="nw")

def _on_sb_frame_configure(event):
    sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))

def _on_sb_canvas_configure(event):
    sb_canvas.itemconfig(_sb_w, width=event.width)

sb.bind("<Configure>", _on_sb_frame_configure)
sb_canvas.bind("<Configure>", _on_sb_canvas_configure)

def _mw(event):
    sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
def _mw_linux(event):
    sb_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

root.bind_all("<MouseWheel>", _mw)
root.bind_all("<Button-4>",   _mw_linux)
root.bind_all("<Button-5>",   _mw_linux)

def _sec_label(parent, text):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(fill="x", padx=16, pady=(12, 3))
    tk.Label(f, text=text, font=F_CAP, bg=PANEL, fg=GOLD_DIM).pack(side="left")
    tk.Frame(f, bg=BDR2, height=1).pack(
        side="right", fill="x", expand=True, padx=(6, 0))

def _card(parent, height=None, bg=CARD):
    kw = {"height": height} if height else {}
    c  = tk.Frame(parent, bg=bg, **kw)
    c.pack(fill="x", padx=16, pady=2)
    if height:
        c.pack_propagate(False)
    return c

# ═══ BRAND ════════════════════════════════════════════════════════════════════
bh = tk.Frame(sb, bg=PANEL)
bh.pack(fill="x", padx=16, pady=(16, 4))
brow = tk.Frame(bh, bg=PANEL)
brow.pack(fill="x")
bcol = tk.Frame(brow, bg=PANEL)
bcol.pack(side="left")
tk.Label(bcol, text="Missing Object", font=F_BRAND, bg=PANEL, fg=WHITE).pack(anchor="w")
tk.Label(bcol, text="Detection",      font=F_BRAND, bg=PANEL, fg=GOLD).pack(anchor="w")
tk.Label(brow, text="  v3", font=("Helvetica", 11), bg=PANEL, fg=MUTED).pack(
    side="left", pady=(8, 0))
tk.Label(bh, text="Smart AI  ·  Voice Control  ·  Interactive Training",
         font=F_BRAND_SM, bg=PANEL, fg=MUTED).pack(anchor="w", pady=(2, 0))
tk.Frame(sb, bg=GOLD_DIM, height=1).pack(fill="x", padx=16, pady=(8, 0))

# ═══ SYSTEM STATUS ════════════════════════════════════════════════════════════
_sec_label(sb, "SYSTEM STATUS")
sc = _card(sb, height=96)

tk.Label(sc, text="CAMERA",     font=F_CAP, bg=CARD, fg=MUTED).place(x=12, y=8)
cam_status_lbl = tk.Label(sc, text="● STANDBY",
                           font=("Courier", 10, "bold"), bg=CARD, fg=ORANGE)
cam_status_lbl.place(x=12, y=24)

tk.Label(sc, text="MICROPHONE", font=F_CAP, bg=CARD, fg=MUTED).place(x=182, y=8)
mic_status_lbl = tk.Label(sc, text="● STARTING",
                           font=("Courier", 10, "bold"), bg=CARD, fg=ORANGE)
mic_status_lbl.place(x=182, y=24)

tk.Frame(sc, bg=BDR, height=1).place(x=12, y=58, width=316)
tk.Label(sc, text="TRAINING", font=F_CAP, bg=CARD, fg=MUTED).place(x=12, y=68)
train_lbl = tk.Label(sc, text="IDLE", font=F_MONO_B, bg=CARD, fg=MUTED)
train_lbl.place(x=86, y=68)

# ═══ LAST HEARD ═══════════════════════════════════════════════════════════════
_sec_label(sb, "VOICE — LAST HEARD")
hc = _card(sb, height=52)
tk.Label(hc, text="HEARD", font=F_CAP, bg=CARD, fg=MUTED).place(x=12, y=6)
heard_lbl = tk.Label(hc, text="waiting for voice input …",
                     font=("Helvetica", 9, "italic"),
                     bg=CARD, fg=GOLD, wraplength=306, anchor="w", justify="left")
heard_lbl.place(x=12, y=22)

# ═══ CONFIDENCE ═══════════════════════════════════════════════════════════════
_sec_label(sb, "DETECTION CONFIDENCE")
cc = _card(sb, height=58)
tk.Label(cc, text="Adjust detection sensitivity",
         font=F_CAP, bg=CARD, fg=MUTED).place(x=12, y=6)

def _on_conf(val):
    S.conf = float(val) / 100
    conf_lbl.config(text=f"{int(float(val))}%")

conf_slider = tk.Scale(
    cc, from_=10, to=95, orient="horizontal",
    command=_on_conf, bg=CARD, fg=WHITE, troughcolor=BDR2,
    highlightthickness=0, bd=0, showvalue=False,
    sliderrelief="flat", sliderlength=14, length=252,
    activebackground=GOLD)
conf_slider.set(40)
conf_slider.place(x=12, y=26)
conf_lbl = tk.Label(cc, text="40%", font=F_MONO_B, bg=CARD, fg=GOLD)
conf_lbl.place(x=272, y=28)

# ═══ TRACKED TARGETS ══════════════════════════════════════════════════════════
_sec_label(sb, "TRACKED TARGETS")
tc    = _card(sb, height=68)
chips = tk.Frame(tc, bg=CARD)
chips.place(x=8, y=6, width=322)

def _ui_refresh_chips():
    for w in chips.winfo_children():
        w.destroy()
    if not S.targets:
        tk.Label(chips,
                 text='Say  "find bottle"  or  "find person"  to begin',
                 font=("Helvetica", 8, "italic"), bg=CARD, fg=MUTED).pack(
                     anchor="w", pady=4)
        return
    row = tk.Frame(chips, bg=CARD)
    row.pack(fill="x")
    for obj in S.targets:
        chip = tk.Frame(row, bg=CARD2, padx=6, pady=3)
        chip.pack(side="left", padx=2, pady=2)
        tk.Label(chip, text="◆", font=("Helvetica", 7),
                 bg=CARD2, fg=GOLD).pack(side="left")
        tk.Label(chip, text=f" {obj}", font=("Helvetica", 8, "bold"),
                 bg=CARD2, fg=WHITE).pack(side="left")
        def _make_rm(o=obj):
            return lambda e: (remove_target(o), speak(f"{o} removed."))
        xl = tk.Label(chip, text="  ×", font=("Helvetica", 9, "bold"),
                      bg=CARD2, fg=MUTED, cursor="hand2")
        xl.pack(side="left")
        xl.bind("<Button-1>", _make_rm())

_ui_refresh_chips()

# ═══ TARGET STATUS ════════════════════════════════════════════════════════════
_sec_label(sb, "TARGET STATUS")
bars_outer = tk.Frame(sb, bg=CARD)
bars_outer.pack(fill="x", padx=16, pady=2)
bars_cv = tk.Canvas(bars_outer, bg=CARD, height=60, highlightthickness=0, bd=0)
bars_cv.pack(fill="x")

# ═══ CUSTOM TRAINING ══════════════════════════════════════════════════════════
_sec_label(sb, "CUSTOM TRAINING")
trc = _card(sb, height=86)

tk.Label(trc,
         text='Click TRAIN, then draw a box on the camera around\n'
              'your object, or say  "train mug"  to begin:',
         font=("Helvetica", 8), bg=CARD, fg=MUTED, justify="left").place(x=12, y=4)

def _prompt_train():
    n = simpledialog.askstring(
        "Train New Object",
        "Enter a name for the object to train\n"
        "(e.g.  mug, keys, wallet, headset):",
        parent=root)
    if not n:
        return
    start_training_interactive(n)

train_btn = tk.Button(
    trc, text="⊕  TRAIN NEW OBJECT",
    bg=CARD2, fg=GOLD, font=("Helvetica", 9, "bold"),
    relief="flat", cursor="hand2", pady=4, bd=0,
    activebackground=BDR2, activeforeground=GOLD2,
    command=_prompt_train)
train_btn.place(x=12, y=54, width=184)

trained_str = ", ".join(_custom_labels.keys()) or "none"
tk.Label(trc, text=f"Stored:  {trained_str}",
         font=("Courier", 8), bg=CARD, fg=TEAL).place(x=204, y=60)

# ═══ CONTROLS ═════════════════════════════════════════════════════════════════
_sec_label(sb, "CONTROLS")

r1 = tk.Frame(sb, bg=PANEL)
r1.pack(fill="x", padx=16, pady=2)
btn_start = tk.Button(r1, text="▶   START", bg=GREEN, fg="#000000",
                      font=F_BTN, relief="flat", cursor="hand2", pady=9, bd=0,
                      activebackground=GREEN, activeforeground="#000000",
                      command=start_detection)
btn_start.pack(side="left", fill="x", expand=True, padx=(0, 3))
btn_stop = tk.Button(r1, text="■   STOP", bg=CARD2, fg=MUTED,
                     font=F_BTN, relief="flat", cursor="hand2", pady=9, bd=0,
                     activebackground=RED, activeforeground=WHITE,
                     command=stop_detection, state="disabled")
btn_stop.pack(side="left", fill="x", expand=True)

r2 = tk.Frame(sb, bg=PANEL)
r2.pack(fill="x", padx=16, pady=2)
tk.Button(r2, text="◎   SNAPSHOT", bg=CARD, fg=WHITE, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=GOLD,
          command=take_snapshot).pack(
              side="left", fill="x", expand=True, padx=(0, 3))
tk.Button(r2, text="↓   EXPORT LOG", bg=CARD, fg=WHITE, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=TEAL,
          command=export_log).pack(side="left", fill="x", expand=True)

r3 = tk.Frame(sb, bg=PANEL)
r3.pack(fill="x", padx=16, pady=2)
tk.Button(r3, text="◉   WHAT DO YOU SEE?", bg=CARD, fg=TEAL, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=TEAL,
          command=lambda: _Q.put(_announce_all)).pack(fill="x")

r4 = tk.Frame(sb, bg=PANEL)
r4.pack(fill="x", padx=16, pady=2)
tk.Button(r4, text="🔥  HEATMAP", bg=CARD, fg=ORANGE, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=ORANGE,
          command=_toggle_heatmap).pack(
              side="left", fill="x", expand=True, padx=(0, 3))
tk.Button(r4, text="📄  PDF REPORT", bg=CARD, fg=PURPLE, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=PURPLE,
          command=export_pdf_report).pack(side="left", fill="x", expand=True)

r5 = tk.Frame(sb, bg=PANEL)
r5.pack(fill="x", padx=16, pady=2)
btn_load_video = tk.Button(
    r5, text="📁  LOAD VIDEO", bg=CARD, fg=WHITE, font=F_BTN,
    relief="flat", cursor="hand2", pady=7, bd=0,
    activebackground=CARD2, activeforeground=GOLD,
    command=load_video_file)
btn_load_video.pack(side="left", fill="x", expand=True, padx=(0, 3))
btn_stop_video = tk.Button(
    r5, text="■  STOP VIDEO", bg=CARD2, fg=MUTED, font=F_BTN,
    relief="flat", cursor="hand2", pady=7, bd=0,
    activebackground=RED, activeforeground=WHITE,
    command=stop_video_processing, state="disabled")
btn_stop_video.pack(side="left", fill="x", expand=True)

r6 = tk.Frame(sb, bg=PANEL)
r6.pack(fill="x", padx=16, pady=2)
tk.Button(r6, text="◈   SUMMARY", bg=CARD, fg=GOLD, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=GOLD2,
          command=_speak_summary).pack(
              side="left", fill="x", expand=True, padx=(0, 3))
tk.Button(r6, text="⇄   NEXT CAMERA", bg=CARD, fg=MUTED, font=F_BTN,
          relief="flat", cursor="hand2", pady=7, bd=0,
          activebackground=CARD2, activeforeground=WHITE,
          command=lambda: _switch_camera(
              (S.camera_index + 1) % 4)).pack(
              side="left", fill="x", expand=True)

# ═══ ACTIVITY LOG ═════════════════════════════════════════════════════════════
_sec_label(sb, "ACTIVITY LOG")
log_outer = tk.Frame(sb, bg=DIM)
log_outer.pack(fill="x", padx=16, pady=(0, 4))
log_box = tk.Text(log_outer, height=5, bg=DIM, fg=WHITE,
                  font=("Courier", 8), relief="flat", bd=0,
                  state="disabled", wrap="word",
                  insertbackground=GOLD, selectbackground=BDR2)
log_box.pack(fill="x", padx=8, pady=6)

# ═══ VOICE COMMANDS ═══════════════════════════════════════════════════════════
_sec_label(sb, "VOICE COMMANDS")
cf = tk.Frame(sb, bg=PANEL)
cf.pack(fill="x", padx=16, pady=(0, 18))

voice_cmds = [
    ('"start" / "stop"',    "Toggle camera"),
    ('"find bottle"',        "Track & alert"),
    ('"where is my phone"',  "Instant status + last seen"),
    ('"what do you see"',    "Announce all with position"),
    ('"remove bottle"',      "Stop tracking"),
    ('"clear"',              "Clear all targets"),
    ('"train mug"',          "Interactive selection training"),
    ('"snapshot"',           "Save frame as JPG"),
    ('"confidence 60"',      "Set threshold"),
    ('"heatmap"',            "Toggle heatmap overlay"),
    ('"summary"',            "Session analytics"),
    ('"export pdf"',         "Save PDF report"),
    ('"camera one"',         "Switch to camera 1"),
    ('"mute" / "unmute"',    "Toggle speech"),
]
for cmd, desc in voice_cmds:
    row = tk.Frame(cf, bg=PANEL)
    row.pack(fill="x", pady=1)
    tk.Label(row, text=cmd,  font=F_CMD, bg=PANEL, fg=GOLD,
             width=22, anchor="w").pack(side="left")
    tk.Label(row, text="—",  font=F_CMD, bg=PANEL, fg=MUTED).pack(
        side="left", padx=3)
    tk.Label(row, text=desc, font=("Helvetica", 8),
             bg=PANEL, fg=MUTED, anchor="w").pack(side="left")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
main = tk.Frame(root, bg=BG)
main.pack(side="right", expand=True, fill="both")

tk.Frame(main, bg=BDR2, width=1).pack(side="left", fill="y")

main_inner = tk.Frame(main, bg=BG)
main_inner.pack(expand=True, fill="both")

tb = tk.Frame(main_inner, bg=BG, height=50)
tb.pack(fill="x", padx=20, pady=(12, 4))
tb.pack_propagate(False)

topbar_lbl = tk.Label(tb, text="Live Camera Feed",
                      font=("Georgia", 14, "bold"), bg=BG, fg=WHITE)
topbar_lbl.pack(side="left", anchor="center")

stats_f = tk.Frame(tb, bg=BG)
stats_f.pack(side="right", anchor="center")

fps_badge = tk.Frame(stats_f, bg=CARD2, padx=10, pady=3)
fps_badge.pack(side="left", padx=4)
tk.Label(fps_badge, text="FPS", font=F_CAP, bg=CARD2, fg=MUTED).pack()
fps_lbl = tk.Label(fps_badge, text="—",
                   font=("Georgia", 15, "bold"), bg=CARD2, fg=GOLD)
fps_lbl.pack()

seen_badge = tk.Frame(stats_f, bg=CARD2, padx=10, pady=3)
seen_badge.pack(side="left", padx=4)
tk.Label(seen_badge, text="OBJECTS", font=F_CAP, bg=CARD2, fg=MUTED).pack()
seen_lbl = tk.Label(seen_badge, text="0",
                    font=("Georgia", 15, "bold"), bg=CARD2, fg=TEAL)
seen_lbl.pack()

cam_frame = tk.Frame(main_inner, bg=BDR2, padx=1, pady=1)
cam_frame.pack(expand=True, fill="both", padx=20, pady=2)

cam_canvas = tk.Canvas(cam_frame, bg="#02030A", highlightthickness=0)
cam_canvas.pack(expand=True, fill="both")

# ── canvas mouse bindings for interactive selection ───────────────────────────
cam_canvas.bind("<Button-1>",        _canvas_press)
cam_canvas.bind("<B1-Motion>",       _canvas_drag)
cam_canvas.bind("<ButtonRelease-1>", _canvas_release)

db = tk.Frame(main_inner, bg=CARD, height=46)
db.pack(fill="x", padx=20, pady=(4, 8))
db.pack_propagate(False)

tk.Frame(db, bg=GOLD, width=3).pack(side="left", fill="y")
inner_db = tk.Frame(db, bg=CARD)
inner_db.pack(side="left", fill="both", expand=True, padx=10, pady=7)
tk.Label(inner_db, text="DETECTED", font=F_CAP, bg=CARD, fg=MUTED).pack(
    side="left", padx=(0, 8))
all_det_lbl = tk.Label(inner_db, text="Start detection to see results.",
                        font=("Courier", 9), bg=CARD, fg=MUTED, anchor="w")
all_det_lbl.pack(side="left", fill="x", expand=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI UPDATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _ui_set_cam(text, color):
    cam_status_lbl.config(text=f"● {text}", fg=color)

def _ui_set_mic(text, color):
    mic_status_lbl.config(text=f"● {text}", fg=color)

def _ui_set_heard(text):
    heard_lbl.config(text=f'"{text}"')
    _ui_set_mic("LISTENING", GREEN)

def _ui_update_fps(fps):
    fps_lbl.config(text=f"{fps:.0f}")

def _ui_update_det_bar(detected: dict):
    if detected:
        parts = []
        for n, c in sorted(detected.items(), key=lambda x: -x[1])[:8]:
            mem = _scene_mem.get(n)
            d   = f" [{mem['direction']}]" if mem else ""
            parts.append(f"{n}{d}  {int(c*100)}%")
        all_det_lbl.config(text="   ·   ".join(parts), fg=WHITE)
        seen_lbl.config(text=f"{len(detected)}", fg=TEAL)
    else:
        all_det_lbl.config(text="Nothing detected in frame.", fg=MUTED)
        seen_lbl.config(text="0", fg=MUTED)

def _ui_update_status_bars(rows):
    bars_cv.delete("all")
    if not rows:
        bars_cv.create_text(12, 20, anchor="w",
                            text='No targets. Say "find bottle" to start.',
                            font=("Helvetica", 8, "italic"), fill=MUTED)
        bars_cv.config(height=40)
        return
    y, row_h, gap = 4, 26, 3
    w = bars_cv.winfo_width() or 320
    for kind, obj, val in rows:
        is_found   = kind == "FOUND"
        bg_c       = GLOW_G if is_found else GLOW_R
        dot        = GREEN  if is_found else RED
        bar_border = "#1A3D28" if is_found else "#3D1A22"
        bars_cv.create_rectangle(0, y, w, y + row_h, fill=bg_c, outline=bar_border)
        bars_cv.create_oval(8, y + 7, 17, y + 17, fill=dot, outline="")
        bars_cv.create_text(26, y + row_h // 2, anchor="w",
                            text=obj.upper(), font=("Helvetica", 8, "bold"), fill=WHITE)
        mem    = _scene_mem.get(obj)
        dir_s  = f"  {mem['direction']}" if mem and is_found else ""
        detail = (f"{val}% conf{dir_s}" if is_found else f"missing  {val}s")
        bars_cv.create_text(w - 8, y + row_h // 2, anchor="e",
                            text=detail, font=("Courier", 7, "bold"), fill=dot)
        bars_cv.create_line(0, y + row_h, w, y + row_h, fill=BDR, width=1)
        y += row_h + gap
    bars_cv.config(height=max(40, y + 4))

def _ui_log(msg: str):
    log_box.config(state="normal")
    ts = datetime.now().strftime("%H:%M:%S")
    log_box.insert("end", f"  {ts}  {msg}\n")
    log_box.see("end")
    log_box.config(state="disabled")

# ═══════════════════════════════════════════════════════════════════════════════
# STANDBY SCREEN  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_standby():
    root.update_idletasks()
    W = cam_canvas.winfo_width()
    H = cam_canvas.winfo_height()
    if W < 100: W = IMG_W
    if H < 100: H = IMG_H

    cam_canvas.delete("all")
    cx, cy = W // 2, H // 2

    cam_canvas.create_rectangle(0, 0, W, H, fill="#02030A", outline="")
    for x in range(0, W, 60):
        cam_canvas.create_line(x, 0, x, H, fill="#060815", width=1)
    for y in range(0, H, 60):
        cam_canvas.create_line(0, y, W, y, fill="#060815", width=1)

    bx1, by1, bx2, by2 = cx - 130, cy - 75, cx + 130, cy + 75
    dash, gap2 = 10, 5
    for ex in range(bx1, bx2, dash + gap2):
        x2 = min(ex + dash, bx2)
        cam_canvas.create_line(ex, by1, x2, by1, fill=GOLD_DIM, width=1)
        cam_canvas.create_line(ex, by2, x2, by2, fill=GOLD_DIM, width=1)
    for ey in range(by1, by2, dash + gap2):
        y2 = min(ey + dash, by2)
        cam_canvas.create_line(bx1, ey, bx1, y2, fill=GOLD_DIM, width=1)
        cam_canvas.create_line(bx2, ey, bx2, y2, fill=GOLD_DIM, width=1)

    cs = 16
    for hx, hy, sx, sy in [(bx1, by1, 1, 1), (bx2, by1, -1, 1),
                             (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
        cam_canvas.create_line(hx, hy, hx + cs * sx, hy, fill=GOLD, width=2)
        cam_canvas.create_line(hx, hy, hx, hy + cs * sy, fill=GOLD, width=2)

    cam_canvas.create_rectangle(bx1, by1 - 18, bx1 + 108, by1, fill=GOLD, outline="")
    cam_canvas.create_text(bx1 + 54, by1 - 9, anchor="center",
                            text="AI SCANNING…", font=("Courier", 7, "bold"),
                            fill="#000000")

    bar_y = by2 - 14
    cam_canvas.create_rectangle(bx1 + 8, bar_y, bx2 - 8, bar_y + 7,
                                 fill="#080F18", outline=GOLD_DIM)
    fw = int((bx2 - bx1 - 16) * 0.74)
    cam_canvas.create_rectangle(bx1 + 8, bar_y, bx1 + 8 + fw, bar_y + 7,
                                 fill=GOLD, outline="")
    cam_canvas.create_text(bx2 - 10, bar_y + 3, anchor="e",
                            text="74%", font=("Courier", 6, "bold"), fill=WHITE)

    lx1, ly1, lx2, ly2 = cx - 220, cy - 26, cx - 148, cy + 30
    for hx, hy, sx, sy in [(lx1, ly1, 1, 1), (lx2, ly1, -1, 1),
                             (lx1, ly2, 1, -1), (lx2, ly2, -1, -1)]:
        cam_canvas.create_line(hx, hy, hx + 10 * sx, hy, fill=GREEN, width=1)
        cam_canvas.create_line(hx, hy, hx, hy + 10 * sy, fill=GREEN, width=1)
    cam_canvas.create_rectangle(lx1, ly1 - 14, lx2, ly1, fill=GREEN, outline="")
    cam_canvas.create_text((lx1 + lx2) // 2, ly1 - 7, anchor="center",
                            text="FOUND  91%", font=("Courier", 6, "bold"),
                            fill="#000000")
    cam_canvas.create_line(lx2, (ly1 + ly2) // 2, bx1, (by1 + by2) // 2,
                            fill="#1A2A18", width=1, dash=(4, 4))

    rx1, ry1, rx2, ry2 = cx + 148, cy - 26, cx + 220, cy + 30
    for hx, hy, sx, sy in [(rx1, ry1, 1, 1), (rx2, ry1, -1, 1),
                             (rx1, ry2, 1, -1), (rx2, ry2, -1, -1)]:
        cam_canvas.create_line(hx, hy, hx + 10 * sx, hy, fill=RED, width=1)
        cam_canvas.create_line(hx, hy, hx, hy + 10 * sy, fill=RED, width=1)
    cam_canvas.create_rectangle(rx1, ry1 - 14, rx2, ry1, fill=RED, outline="")
    cam_canvas.create_text((rx1 + rx2) // 2, ry1 - 7, anchor="center",
                            text="MISSING", font=("Courier", 6, "bold"), fill=WHITE)
    cam_canvas.create_line(rx1, (ry1 + ry2) // 2, bx2, (by1 + by2) // 2,
                            fill="#2A1018", width=1, dash=(4, 4))

    cam_canvas.create_text(cx, cy - 140, anchor="center",
                            text="MISSING OBJECT DETECTION",
                            font=("Georgia", 20, "bold"), fill=GOLD)
    cam_canvas.create_text(cx, cy - 116, anchor="center",
                            text="AI-Powered  ·  Voice-Controlled  ·  Real-Time Tracking",
                            font=("Helvetica", 9), fill=MUTED)
    cam_canvas.create_text(cx, cy + 136, anchor="center",
                            text='Say  "start"  or click  ▶ START',
                            font=("Helvetica", 11), fill=MUTED)

    trained = ", ".join(_custom_labels.keys()) or "none"
    cam_canvas.create_text(12, H - 10, anchor="sw",
                            text=f"Trained objects:  {trained}",
                            font=("Courier", 7), fill=GOLD_DIM)
    cam_canvas.create_text(W - 10, 10, anchor="ne",
                            text="v3.0", font=("Courier", 7), fill=GOLD_DIM)

_resize_after_id = None
def _on_resize(event):
    global _resize_after_id
    if not S.detecting:
        if _resize_after_id:
            root.after_cancel(_resize_after_id)
        _resize_after_id = root.after(120, _draw_standby)

root.bind("<Configure>", _on_resize)
root.after(250, _draw_standby)

# ═══════════════════════════════════════════════════════════════════════════════
# SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_quit():
    _voice_active.clear()
    S.detecting = False
    _infer_stop.set()
    _db_q.put(None)
    _db_end_session(S.frame_count)
    if S.cap:
        S.cap.release()
    _tts_q.put(None)
    try:
        _pa.terminate()
    except Exception:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", _safe_quit)

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
root.after(33, _pump)
_ui_log("Missing Object Detection v3 — Enterprise AI ready.")
_ui_log("Calibrating microphone — please stay quiet …")
_ui_log('Say "start" or click ▶ START to begin.')

threading.Thread(target=_voice_loop, daemon=True, name="voice").start()

root.mainloop()
