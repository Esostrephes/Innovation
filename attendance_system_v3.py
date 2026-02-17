#!/usr/bin/env python3
# ============================================================
# UNIVERSITY ATTENDANCE SYSTEM  v3
# FastAPI + Supabase + Redis
# InsightFace facial embeddings (server-side, buffalo_l model)
# MediaPipe liveness (browser, head-movement challenge)
# JWT Admin Auth  •  Batch Attendance  •  Mobile Enrolment
# ============================================================
#
# NEW in v3
# ─────────
#  • /enroll           — student self-enrolment from phone
#                        captures face photo + insightface embedding
#                        stores admissions details (name, program, year…)
#  • /mark_attendance  — now ALSO does face-match against stored embedding
#                        (liveness challenge + face verification = dual proof)
#  • New DB columns    — full_name, email, program, year, embedding (base64 text)
#  • /admin/students   — shows admissions details + enrolment status
#
# REQUIREMENTS (pip)
# ──────────────────
#  fastapi uvicorn[standard] supabase redis passlib[bcrypt]
#  python-jose[cryptography] python-multipart
#  insightface onnxruntime numpy opencv-python-headless Pillow
#
# SUPABASE SETUP — run schema.sql in Supabase SQL Editor first
#
# ENV VARS
# ────────
#  SUPABASE_URL          — https://xxxx.supabase.co
#  SUPABASE_KEY          — service_role secret key (NOT the anon key)
#  REDIS_HOST  REDIS_PORT
#  JWT_SECRET
#  FACE_THRESHOLD        — cosine similarity cutoff (default 0.40)
#  INSIGHTFACE_MODEL_DIR — optional; defaults to ~/.insightface
# ============================================================

from __future__ import annotations

import base64
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np
import redis
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from supabase import create_client, Client

# ── InsightFace ──────────────────────────────────────────────
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# CONFIG
# ================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")   # use service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY environment variables are required.\n"
        "Set them before starting the server:\n"
        "  export SUPABASE_URL=https://xxxx.supabase.co\n"
        "  export SUPABASE_KEY=your_service_role_key"
    )

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
JWT_SECRET  = os.getenv("JWT_SECRET", "change_me_in_production_2026")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# InsightFace cosine similarity threshold (0–1; higher = stricter)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.40"))

# ================================================================
# SUPABASE CLIENT
# ================================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase client initialised → %s", SUPABASE_URL)

# ================================================================
# INSIGHTFACE — load once at startup
# ================================================================

logger.info("Loading InsightFace model (buffalo_l)…")
_face_app = FaceAnalysis(
    name="buffalo_l",
    root=os.getenv("INSIGHTFACE_MODEL_DIR", os.path.expanduser("~/.insightface")),
    providers=["CPUExecutionProvider"],   # swap to CUDAExecutionProvider for GPU
)
_face_app.prepare(ctx_id=0, det_size=(640, 640))
logger.info("InsightFace ready.")


def extract_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Return 512-d float32 unit-norm embedding for the largest face, or None."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = _face_app.get(img)
    if not faces:
        return None
    # Largest face by bounding-box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding   # shape (512,), already L2-normalised


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two unit-normalised vectors == cosine similarity."""
    return float(np.dot(a, b))


def embedding_to_b64(emb: np.ndarray) -> str:
    """Encode float32 array to base64 string for Supabase TEXT storage."""
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii")


def b64_to_embedding(b64: str) -> np.ndarray:
    """Decode base64 string back to float32 numpy array."""
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


# ================================================================
# APP
# ================================================================

app = FastAPI(title="UniAttend v3 — Supabase + InsightFace + Liveness")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# REDIS / IN-MEMORY CHALLENGE STORE
# (Redis optional — falls back to in-process dict for single-worker dev)
# ================================================================

_redis: Optional[redis.Redis] = None
_store: Dict[str, dict] = {}


def get_redis() -> Optional[redis.Redis]:
    global _redis
    if _redis is None:
        try:
            c = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            c.ping()
            _redis = c
            logger.info("Redis connected.")
        except redis.RedisError as e:
            logger.warning("Redis unavailable (%s). Using in-memory challenge store.", e)
    return _redis


def ch_set(key: str, val: str, ex: int) -> None:
    r = get_redis()
    if r:
        r.set(key, val, ex=ex)
    else:
        _store[key] = {"v": val, "exp": time.time() + ex}


def ch_get(key: str) -> Optional[str]:
    r = get_redis()
    if r:
        return r.get(key)
    e = _store.get(key)
    if e and time.time() < e["exp"]:
        return e["v"]
    _store.pop(key, None)
    return None


def ch_del(key: str) -> None:
    r = get_redis()
    if r:
        r.delete(key)
    else:
        _store.pop(key, None)


# ================================================================
# AUTH
# ================================================================

pwd_ctx       = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")


def make_token(sub: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": sub, "exp": exp}, JWT_SECRET, algorithm=ALGORITHM)


def require_admin(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload  = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(
            401, "Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


# ================================================================
# ADMIN ENDPOINTS
# ================================================================

@app.post("/admin/create")
def create_admin(username: str = Form(...), password: str = Form(...)):
    if len(username) < 3 or len(password) < 6:
        raise HTTPException(400, "Username ≥3 chars, password ≥6 chars")
    hashed = pwd_ctx.hash(password)
    try:
        supabase.table("admins").insert({
            "username":      username,
            "password_hash": hashed,
        }).execute()
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg or "unique" in msg or "23505" in msg:
            raise HTTPException(400, "Admin already exists")
        raise HTTPException(500, f"Database error: {e}")
    return {"status": "admin_created"}


@app.post("/admin/login")
def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    result = (
        supabase.table("admins")
        .select("password_hash")
        .eq("username", form_data.username)
        .maybe_single()
        .execute()
    )
    row = result.data
    if not row or not pwd_ctx.verify(form_data.password, row["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": make_token(form_data.username), "token_type": "bearer"}


@app.post("/admin/verify-token")
def verify_token(admin: str = Depends(require_admin)):
    return {"valid": True, "username": admin}


@app.get("/analytics")
def analytics(admin: str = Depends(require_admin)):
    today_start = (
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat()
    )

    total_res = supabase.table("attendance").select("id", count="exact").execute()
    total     = total_res.count or 0

    # Unique students — deduplicated in Python (PostgREST free tier lacks COUNT DISTINCT)
    unique_res = supabase.table("attendance").select("student_id").execute()
    unique     = len({r["student_id"] for r in (unique_res.data or [])})

    today_res  = (
        supabase.table("attendance")
        .select("id", count="exact")
        .gte("timestamp", today_start)
        .execute()
    )
    today_n = today_res.count or 0

    enrolled_res = (
        supabase.table("students")
        .select("id", count="exact")
        .not_.is_("embedding", "null")
        .execute()
    )
    enrolled = enrolled_res.count or 0

    reg_res    = supabase.table("students").select("id", count="exact").execute()
    registered = reg_res.count or 0

    return {
        "total_attendance_records": total,
        "unique_students_all_time": unique,
        "today_count": today_n,
        "enrolled_with_face": enrolled,
        "total_registered": registered,
        "admin": admin,
    }


@app.get("/students")
def list_students(admin: str = Depends(require_admin)):
    result = (
        supabase.table("students")
        .select(
            "student_id,full_name,email,program,year,"
            "phone,enrolled_at,created_at,embedding"
        )
        .order("created_at", desc=True)
        .execute()
    )
    rows = result.data or []
    # Don't send the raw embedding to the frontend — just a boolean flag
    for row in rows:
        row["has_face"] = row.pop("embedding") is not None
    return rows


# ================================================================
# STUDENT REGISTRATION (ID only — quick add before enrolment)
# ================================================================

@app.post("/register")
def register(student_id: str = Form(...)):
    sid = student_id.strip()
    if not sid or len(sid) > 100:
        raise HTTPException(400, "Invalid student ID")
    try:
        supabase.table("students").insert({
            "student_id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg or "unique" in msg or "23505" in msg:
            raise HTTPException(400, "Student already registered")
        raise HTTPException(500, f"Database error: {e}")
    return {"status": "registered", "student_id": sid}


# ================================================================
# ENROLMENT — student fills admissions details + face from phone
# ================================================================

@app.post("/enroll")
async def enroll(
    student_id: str        = Form(...),
    full_name:  str        = Form(...),
    email:      str        = Form(""),
    program:    str        = Form(""),
    year:       int        = Form(1),
    phone:      str        = Form(""),
    face_image: UploadFile = File(...),   # JPEG/PNG from phone camera
):
    """
    Called from the student's phone during self-enrolment.
    Accepts admissions details + a frontal face photo.
    Extracts a 512-d InsightFace ArcFace embedding and stores
    everything in Supabase as a single upsert.
    """
    sid = student_id.strip()

    # ── verify student exists ────────────────────────────────
    result = (
        supabase.table("students")
        .select("student_id")
        .eq("student_id", sid)
        .maybe_single()
        .execute()
    )
    if not result.data:
        raise HTTPException(
            404,
            "Student ID not found. Please register your ID first, "
            "then complete full enrolment.",
        )

    # ── extract InsightFace embedding ────────────────────────
    img_bytes = await face_image.read()
    embedding = extract_embedding(img_bytes)
    if embedding is None:
        raise HTTPException(
            422,
            "No face detected in the uploaded image. "
            "Please use a clear, well-lit frontal photo with no obstructions.",
        )

    emb_b64 = embedding_to_b64(embedding)

    # ── upsert admissions details + embedding into Supabase ──
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        supabase.table("students").update({
            "full_name":   full_name.strip(),
            "email":       email.strip(),
            "program":     program.strip(),
            "year":        year,
            "phone":       phone.strip(),
            "embedding":   emb_b64,
            "enrolled_at": now_iso,
        }).eq("student_id", sid).execute()
    except Exception as e:
        raise HTTPException(500, f"Failed to save enrolment: {e}")

    logger.info("Enrolled %s (%s) — embedding dim=%d", sid, full_name, len(embedding))
    return {
        "status":        "enrolled",
        "student_id":    sid,
        "full_name":     full_name.strip(),
        "embedding_dim": int(len(embedding)),
    }


# ================================================================
# LIVENESS CHALLENGE
# ================================================================

@app.post("/challenge")
def generate_challenge(student_id: str = Form(...)):
    sid = student_id.strip()

    result = (
        supabase.table("students")
        .select("student_id")
        .eq("student_id", sid)
        .maybe_single()
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Student not registered")

    challenge    = random.choice(["LEFT", "RIGHT", "UP"])
    challenge_id = str(uuid.uuid4())[:12]
    ch_set(f"challenge:{sid}:{challenge_id}", challenge, ex=120)
    return {"challenge": challenge, "challenge_id": challenge_id, "student_id": sid}


# ================================================================
# MARK ATTENDANCE — liveness + face verification
# ================================================================

@app.post("/mark_attendance")
async def mark_attendance(
    student_id:   str        = Form(...),
    movement:     str        = Form(...),
    challenge_id: str        = Form(...),
    face_image:   UploadFile = File(None),   # required if student is enrolled
):
    sid = student_id.strip()
    mov = movement.strip().upper()

    # ── 1. Liveness check ───────────────────────────────────
    key      = f"challenge:{sid}:{challenge_id}"
    expected = ch_get(key)
    ch_del(key)   # consume — prevents replay attacks

    if not expected:
        raise HTTPException(400, "Challenge expired or invalid")
    if mov != expected.upper():
        raise HTTPException(403, "Liveness check failed — wrong head movement")

    # ── 2. Load student record from Supabase ─────────────────
    result = (
        supabase.table("students")
        .select("student_id,embedding")
        .eq("student_id", sid)
        .maybe_single()
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Student not registered")

    stored_b64: Optional[str] = result.data.get("embedding")
    face_score: Optional[float] = None

    # ── 3. Face verification (only if student has enrolled) ──
    if stored_b64:
        if face_image is None:
            raise HTTPException(
                422,
                "A face photo is required for enrolled students. "
                "Please allow camera access.",
            )
        img_bytes  = await face_image.read()
        live_emb   = extract_embedding(img_bytes)
        if live_emb is None:
            raise HTTPException(422, "No face detected in the live photo")

        stored_emb = b64_to_embedding(stored_b64)
        face_score = cosine_similarity(live_emb, stored_emb)

        if face_score < FACE_MATCH_THRESHOLD:
            raise HTTPException(
                403,
                f"Face verification failed "
                f"(similarity={face_score:.3f}, required≥{FACE_MATCH_THRESHOLD})",
            )
    # Students not yet enrolled → attendance recorded, face_verified=False

    # ── 4. Write attendance record to Supabase ───────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        supabase.table("attendance").insert({
            "student_id": sid,
            "timestamp":  now_iso,
            "face_score": round(face_score, 6) if face_score is not None else None,
        }).execute()
    except Exception as e:
        raise HTTPException(500, f"Failed to record attendance: {e}")

    # ── 5. Cache last-seen in Redis ───────────────────────────
    r = get_redis()
    if r:
        r.set(f"last_seen:{sid}", now_iso, ex=86400)

    return {
        "status":        "attendance_marked",
        "student_id":    sid,
        "face_verified": stored_b64 is not None,
        "face_score":    round(face_score, 4) if face_score is not None else None,
    }


# ================================================================
# FRONTEND — Single-Page Application
# ================================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

# ── HTML is in a raw string to avoid f-string / JS template conflicts ──
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>UniAttend v3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Cabinet+Grotesk:wght@400;500;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js" crossorigin="anonymous"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --ink:#0a0b10;
  --paper:#f5f4ef;
  --surface:#ffffff;
  --surface2:#f0efe9;
  --border:#e0dfd7;
  --accent:#1c3faa;
  --accent-lt:#e8ecf9;
  --green:#1a7a4a;
  --green-lt:#e6f4ed;
  --red:#c0392b;
  --red-lt:#fdf0ee;
  --amber:#b45309;
  --amber-lt:#fef3e2;
  --muted:#888070;
  --font-h:'Cabinet Grotesk',sans-serif;
  --font-m:'DM Mono',monospace;
  --r:8px;
  --r-lg:14px;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.06);
}

body{font-family:var(--font-h);background:var(--paper);color:var(--ink);min-height:100vh;overflow-x:hidden}

/* ── Noise texture overlay ── */
body::after{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");pointer-events:none;z-index:9999;opacity:.5}

.shell{max-width:860px;margin:0 auto;padding:24px 16px 80px}

/* ── Header ── */
.topbar{display:flex;align-items:center;gap:14px;margin-bottom:36px;padding-bottom:20px;border-bottom:2px solid var(--border)}
.logo{width:44px;height:44px;background:var(--accent);color:#fff;font-size:20px;display:grid;place-items:center;border-radius:10px;flex-shrink:0;font-family:var(--font-m)}
.brand-name{font-size:1.55rem;font-weight:900;letter-spacing:-.04em}
.brand-sub{font-family:var(--font-m);font-size:.65rem;color:var(--muted);letter-spacing:.08em}
.version-chip{margin-left:auto;background:var(--accent);color:#fff;font-family:var(--font-m);font-size:.62rem;padding:4px 10px;border-radius:20px;letter-spacing:.06em}

/* ── Tabs ── */
.tabs{display:flex;gap:2px;background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r-lg);padding:4px;margin-bottom:24px}
.tab-btn{flex:1;padding:10px 4px;border:none;background:transparent;font-family:var(--font-h);font-size:.83rem;font-weight:700;color:var(--muted);border-radius:10px;cursor:pointer;transition:.18s;letter-spacing:.01em}
.tab-btn:hover{color:var(--ink)}
.tab-btn.active{background:var(--accent);color:#fff;box-shadow:0 2px 10px rgba(28,63,170,.25)}

/* ── Panels ── */
.panel{display:none}
.panel.active{display:block;animation:rise .22s ease}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

/* ── Cards ── */
.card{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r-lg);padding:22px;margin-bottom:14px;box-shadow:var(--shadow)}
.card-label{font-family:var(--font-m);font-size:.62rem;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px}

/* ── Camera ── */
.cam-wrap{position:relative;border-radius:var(--r);overflow:hidden;background:#0a0b10;aspect-ratio:4/3}
.cam-wrap video{width:100%;height:100%;object-fit:cover;display:block;transform:scaleX(-1)}
.cam-ring{position:absolute;inset:0;border:2px solid transparent;border-radius:var(--r);pointer-events:none;transition:.3s}
.cam-wrap.detecting .cam-ring{border-color:var(--accent);animation:ring-pulse 1.2s infinite}
.cam-wrap.ok .cam-ring{border-color:var(--green)}
@keyframes ring-pulse{0%,100%{box-shadow:inset 0 0 0 2px rgba(28,63,170,.3)}50%{box-shadow:inset 0 0 0 4px rgba(28,63,170,.55)}}

/* Corner marks */
.cam-wrap::before,.cam-wrap::after{content:'';position:absolute;width:20px;height:20px;border-color:var(--accent);border-style:solid;z-index:2}
.cam-wrap::before{top:10px;left:10px;border-width:2px 0 0 2px}
.cam-wrap::after{bottom:10px;right:10px;border-width:0 2px 2px 0}

/* ── Forms ── */
.field{margin-bottom:11px}
.field label{display:block;font-family:var(--font-m);font-size:.62rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px}
input,select,textarea{width:100%;background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);color:var(--ink);font-family:var(--font-m);font-size:.88rem;padding:10px 13px;outline:none;transition:.18s;-webkit-appearance:none}
input:focus,select:focus,textarea:focus{border-color:var(--accent);background:#fff;box-shadow:0 0 0 3px rgba(28,63,170,.1)}
textarea{resize:vertical;min-height:100px}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24'%3E%3Cpath fill='%23888070' d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px;cursor:pointer}

/* ── Buttons ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:11px 22px;border:none;border-radius:var(--r);font-family:var(--font-h);font-size:.88rem;font-weight:700;cursor:pointer;transition:.18s;letter-spacing:.01em;text-decoration:none}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover:not(:disabled){background:#162e82;box-shadow:0 4px 16px rgba(28,63,170,.3);transform:translateY(-1px)}
.btn-green{background:var(--green);color:#fff}
.btn-green:hover:not(:disabled){box-shadow:0 4px 16px rgba(26,122,74,.3);transform:translateY(-1px)}
.btn-ghost{background:transparent;border:1.5px solid var(--border);color:var(--ink)}
.btn-ghost:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.btn-danger{background:transparent;border:1.5px solid #f5c6c2;color:var(--red)}
.btn-danger:hover{background:var(--red-lt)}
.btn-full{width:100%}

/* ── Status ── */
.status{display:none;align-items:center;gap:9px;padding:11px 14px;border-radius:var(--r);font-size:.84rem;margin:10px 0}
.status.show{display:flex;animation:rise .2s ease}
.status.ok{background:var(--green-lt);border:1.5px solid #b7dfc8;color:var(--green)}
.status.err{background:var(--red-lt);border:1.5px solid #f5c6c2;color:var(--red)}
.status.info{background:var(--accent-lt);border:1.5px solid #c3cfed;color:var(--accent)}
.status.warn{background:var(--amber-lt);border:1.5px solid #fcd48f;color:var(--amber)}

/* ── Challenge box ── */
.challenge{margin:14px 0;padding:18px;background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);text-align:center;min-height:84px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px}
.ch-lbl{font-family:var(--font-m);font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.ch-val{font-size:2rem;font-weight:900;color:var(--accent);letter-spacing:-.04em;line-height:1}
.ch-hint{font-size:.72rem;color:var(--muted);font-family:var(--font-m)}

/* ── Progress ── */
.prog{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin:10px 0}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--green));border-radius:2px;transition:width .4s ease;width:0%}

/* ── Split ── */
.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:600px){.split{grid-template-columns:1fr}}

/* ── Batch list ── */
.blist{border:1.5px solid var(--border);border-radius:var(--r);max-height:260px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--accent) transparent}
.bi{display:flex;justify-content:space-between;align-items:center;padding:9px 13px;border-bottom:1px solid var(--border);font-family:var(--font-m);font-size:.8rem}
.bi:last-child{border:none}
.bi.cur{background:var(--accent-lt)}
.pill{font-size:.65rem;padding:3px 9px;border-radius:20px}
.pill-pend{background:var(--surface2);color:var(--muted)}
.pill-cur{background:var(--accent-lt);color:var(--accent)}
.pill-ok{background:var(--green-lt);color:var(--green)}
.pill-fail{background:var(--red-lt);color:var(--red)}

/* ── Analytics ── */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}
@media(max-width:480px){.stats{grid-template-columns:1fr 1fr}}
.stat{background:var(--surface2);border:1.5px solid var(--border);border-radius:var(--r);padding:15px}
.stat-val{font-size:1.75rem;font-weight:900;letter-spacing:-.04em;color:var(--accent)}
.stat-lbl{font-family:var(--font-m);font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:3px}

/* ── Table ── */
.tbl-wrap{border:1.5px solid var(--border);border-radius:var(--r);overflow:auto;max-height:320px}
table{width:100%;border-collapse:collapse}
th{font-family:var(--font-m);font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;text-align:left;padding:8px 12px;border-bottom:1.5px solid var(--border);background:var(--surface2);position:sticky;top:0}
td{font-family:var(--font-m);font-size:.78rem;padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border:none}
tr:hover td{background:var(--surface2)}
.badge{display:inline-block;font-size:.62rem;padding:2px 7px;border-radius:10px}
.b-yes{background:var(--green-lt);color:var(--green)}
.b-no{background:var(--amber-lt);color:var(--amber)}

/* ── Enrolment steps ── */
.steps{display:flex;gap:0;margin-bottom:20px;position:relative}
.steps::before{content:'';position:absolute;top:16px;left:16px;right:16px;height:2px;background:var(--border);z-index:0}
.step{flex:1;display:flex;flex-direction:column;align-items:center;gap:5px;position:relative;z-index:1}
.step-dot{width:32px;height:32px;border-radius:50%;background:var(--surface2);border:2px solid var(--border);display:grid;place-items:center;font-family:var(--font-m);font-size:.72rem;font-weight:500;color:var(--muted);transition:.3s}
.step.done .step-dot{background:var(--green);border-color:var(--green);color:#fff}
.step.active .step-dot{background:var(--accent);border-color:var(--accent);color:#fff}
.step-lbl{font-family:var(--font-m);font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;text-align:center}
.step.active .step-lbl{color:var(--accent)}

/* ── Face preview ── */
.face-preview{width:100%;aspect-ratio:1;object-fit:cover;border-radius:var(--r);border:2px solid var(--green);display:none;margin-top:8px}
.face-preview.show{display:block}

/* ── Loading overlay ── */
#loader{position:fixed;inset:0;background:var(--paper);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:1000;transition:opacity .4s}
#loader.gone{opacity:0;pointer-events:none}
.ld-icon{font-size:2.2rem}
.ld-lbl{font-family:var(--font-m);font-size:.75rem;color:var(--muted)}
.ld-bar{width:180px;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.ld-fill{height:100%;background:var(--accent);border-radius:2px;animation:ldfill 2.5s ease forwards}
@keyframes ldfill{from{width:0}to{width:88%}}

/* ── Spinner ── */
.spin{width:15px;height:15px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spn .7s linear infinite;flex-shrink:0}
@keyframes spn{to{transform:rotate(360deg)}}
.spin-dark{border-color:rgba(0,0,0,.12);border-top-color:var(--ink)}
.divider{height:1.5px;background:var(--border);margin:18px 0}
</style>
</head>
<body>

<!-- Loading overlay -->
<div id="loader">
  <div class="ld-icon">👁</div>
  <div style="font-family:var(--font-h);font-weight:800;font-size:1.1rem">Initialising UniAttend</div>
  <div class="ld-lbl">Loading face detection model…</div>
  <div class="ld-bar"><div class="ld-fill"></div></div>
</div>

<div class="shell">

  <!-- Header -->
  <div class="topbar">
    <div class="logo">UA</div>
    <div>
      <div class="brand-name">UniAttend</div>
      <div class="brand-sub">FACIAL VERIFICATION + LIVENESS SYSTEM</div>
    </div>
    <div class="version-chip">v3.0</div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="tab(0)">Enrol</button>
    <button class="tab-btn" onclick="tab(1)">Attend</button>
    <button class="tab-btn" onclick="tab(2)">Batch</button>
    <button class="tab-btn" onclick="tab(3)">Admin</button>
  </div>

  <!-- ═══════════════════ ENROL ═══════════════════ -->
  <div id="p0" class="panel active">

    <!-- Step indicator -->
    <div class="steps" id="enrol-steps">
      <div class="step active" id="st0"><div class="step-dot">1</div><div class="step-lbl">Details</div></div>
      <div class="step"        id="st1"><div class="step-dot">2</div><div class="step-lbl">Face Scan</div></div>
      <div class="step"        id="st2"><div class="step-dot">3</div><div class="step-lbl">Done</div></div>
    </div>

    <!-- Step 0 — Admissions details -->
    <div id="enrol-step0" class="card">
      <div class="card-label">Admissions Details</div>
      <div class="split">
        <div>
          <div class="field"><label>Student ID *</label><input id="e-sid" placeholder="U2024001" autocomplete="off"></div>
          <div class="field"><label>Full Name *</label><input id="e-name" placeholder="Amara Okafor"></div>
          <div class="field"><label>Email</label><input id="e-email" type="email" placeholder="amara@university.edu"></div>
          <div class="field"><label>Phone</label><input id="e-phone" type="tel" placeholder="+234…"></div>
        </div>
        <div>
          <div class="field"><label>Programme</label>
            <select id="e-prog">
              <option value="">— Select —</option>
              <option>Computer Science</option>
              <option>Electrical Engineering</option>
              <option>Medicine</option>
              <option>Business Administration</option>
              <option>Law</option>
              <option>Architecture</option>
              <option>Economics</option>
              <option>Mathematics</option>
              <option>Physics</option>
              <option>Chemistry</option>
              <option>Other</option>
            </select>
          </div>
          <div class="field"><label>Year of Study</label>
            <select id="e-year">
              <option value="1">Year 1</option>
              <option value="2">Year 2</option>
              <option value="3">Year 3</option>
              <option value="4">Year 4</option>
              <option value="5">Year 5</option>
              <option value="6">Year 6</option>
            </select>
          </div>
        </div>
      </div>
      <div id="st-reg" class="status"></div>
      <button class="btn btn-primary btn-full" id="btn-next-enrol" onclick="enrolStep1()" style="margin-top:4px">
        Continue → Face Scan
      </button>
    </div>

    <!-- Step 1 — Face capture -->
    <div id="enrol-step1" style="display:none">
      <div class="card">
        <div class="card-label">Face Registration — Position your face in the frame</div>
        <div class="split">
          <div>
            <div class="cam-wrap" id="ecam"><video id="evid" autoplay playsinline muted></video><div class="cam-ring"></div></div>
            <button class="btn btn-ghost btn-full" style="margin-top:8px" onclick="captureEnrolFace()">📸 Capture Photo</button>
          </div>
          <div>
            <div id="e-face-status" class="status show info">Look straight at the camera. Good lighting is essential. Keep your face centred in the frame.</div>
            <img id="e-preview" class="face-preview" alt="Face preview">
            <div style="margin-top:12px;font-family:var(--font-m);font-size:.72rem;color:var(--muted);line-height:1.6">
              ✦ Remove glasses if possible<br>
              ✦ Neutral expression, mouth closed<br>
              ✦ Face forward, no tilt<br>
              ✦ Bright, even lighting
            </div>
          </div>
        </div>
        <div class="divider"></div>
        <div id="st-enrol" class="status"></div>
        <div style="display:flex;gap:8px;margin-top:4px">
          <button class="btn btn-ghost" onclick="goEnrolStep(0)">← Back</button>
          <button class="btn btn-green btn-full" id="btn-submit-enrol" onclick="submitEnrolment()" disabled>
            ✓ Complete Enrolment
          </button>
        </div>
      </div>
    </div>

    <!-- Step 2 — Done -->
    <div id="enrol-step2" style="display:none">
      <div class="card" style="text-align:center;padding:40px 20px">
        <div style="font-size:3rem;margin-bottom:12px">🎓</div>
        <div style="font-size:1.3rem;font-weight:900;letter-spacing:-.03em" id="enrol-done-name">Enrolled!</div>
        <div style="font-family:var(--font-m);font-size:.75rem;color:var(--muted);margin-top:6px" id="enrol-done-meta"></div>
        <div style="font-family:var(--font-m);font-size:.72rem;color:var(--green);margin-top:16px;padding:12px;background:var(--green-lt);border-radius:var(--r)">
          ✅ Facial embedding stored • You can now mark attendance
        </div>
        <button class="btn btn-ghost" style="margin-top:20px" onclick="resetEnrol()">Enrol another student</button>
      </div>
    </div>
  </div>

  <!-- ═══════════════════ INDIVIDUAL ATTEND ═══════════════════ -->
  <div id="p1" class="panel">
    <div class="split">
      <div>
        <div class="card-label">Camera</div>
        <div class="cam-wrap" id="acam"><video id="avid" autoplay playsinline muted></video><div class="cam-ring"></div></div>
      </div>
      <div>
        <div class="card-label">Mark Your Attendance</div>
        <div class="field"><label>Student ID</label><input id="a-sid" placeholder="U2024001" autocomplete="off"></div>
        <div class="challenge" id="ach">
          <div class="ch-lbl">Challenge</div>
          <div class="ch-val" id="ach-val">—</div>
          <div class="ch-hint" id="ach-hint">Press Start to begin</div>
        </div>
        <div id="st-attend" class="status"></div>
        <button class="btn btn-primary btn-full" id="btn-attend" onclick="startAttend()">▶ Mark Attendance</button>
      </div>
    </div>
  </div>

  <!-- ═══════════════════ BATCH ═══════════════════ -->
  <div id="p2" class="panel">
    <div class="card">
      <div class="card-label">Batch — Class Mode</div>
      <div class="field"><label>Student IDs (one per line)</label>
        <textarea id="b-ids" placeholder="U2024001&#10;U2024002&#10;U2024003"></textarea>
      </div>
      <button class="btn btn-green btn-full" id="btn-batch" onclick="startBatch()">🚀 Start Batch Attendance</button>
    </div>
    <div id="st-batch" class="status"></div>
    <div class="prog"><div class="prog-fill" id="bprog"></div></div>
    <div style="display:flex;gap:10px;margin-bottom:14px;align-items:center">
      <div class="cam-wrap" style="max-width:180px;flex-shrink:0">
        <video id="bvid" autoplay playsinline muted></video><div class="cam-ring" id="bcam-ring"></div>
      </div>
      <div style="flex:1">
        <div class="challenge" id="bch" style="margin:0">
          <div class="ch-lbl">Current</div>
          <div class="ch-val" id="bch-val">—</div>
          <div class="ch-hint" id="bch-hint">Waiting…</div>
        </div>
      </div>
    </div>
    <div class="blist" id="blist"></div>
  </div>

  <!-- ═══════════════════ ADMIN ═══════════════════ -->
  <div id="p3" class="panel">
    <div id="login-sec">
      <div class="card">
        <div class="card-label">Admin Login</div>
        <div class="field"><label>Username</label><input id="adm-u" autocomplete="username"></div>
        <div class="field"><label>Password</label><input id="adm-p" type="password" autocomplete="current-password"></div>
        <button class="btn btn-primary btn-full" onclick="admLogin()">Login</button>
        <div id="st-login" class="status" style="margin-top:10px"></div>
      </div>
    </div>

    <div id="adm-sec" style="display:none">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <div>
          <div style="font-weight:800;font-size:1.05rem;letter-spacing:-.02em">Dashboard</div>
          <div style="font-family:var(--font-m);font-size:.68rem;color:var(--muted)" id="adm-greet"></div>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn btn-ghost" onclick="admRefresh()">↻</button>
          <button class="btn btn-danger" onclick="admLogout()">Logout</button>
        </div>
      </div>

      <div class="stats">
        <div class="stat"><div class="stat-val" id="s-total">—</div><div class="stat-lbl">All Records</div></div>
        <div class="stat"><div class="stat-val" id="s-today">—</div><div class="stat-lbl">Today</div></div>
        <div class="stat"><div class="stat-val" id="s-enr">—</div><div class="stat-lbl">Face Enrolled</div></div>
      </div>

      <div class="card-label">Registered Students</div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Name</th><th>Programme</th><th>Yr</th><th>Email</th><th>Face</th><th>Enrolled</th>
            </tr>
          </thead>
          <tbody id="adm-tbody">
            <tr><td colspan="7" style="color:var(--muted);padding:18px;text-align:center">Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</div><!-- /shell -->

<script>
// ================================================================
// MEDIAPIPE
// ================================================================
let mp = null;
const WASM  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const THR   = 22;
const HIST  = 15;

async function initMP() {
  const {FaceLandmarker, FilesetResolver} = window.vision;
  const fs = await FilesetResolver.forVisionTasks(WASM);
  mp = await FaceLandmarker.createFromOptions(fs, {
    baseOptions: {modelAssetPath: MODEL},
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

// ================================================================
// CAMERA
// ================================================================
const streams = {};

async function startCam(vid) {
  if (streams[vid.id]) return;
  const s = await navigator.mediaDevices.getUserMedia({
    video: {facingMode: "user", width:{ideal:640}, height:{ideal:480}}
  });
  vid.srcObject = s;
  streams[vid.id] = s;
  await vid.play();
}

async function ensureCam(id) {
  const v = document.getElementById(id);
  if (!streams[v.id]) await startCam(v).catch(e => console.warn("cam:", e.message));
}

// ================================================================
// CAPTURE FRAME → Blob
// ================================================================
function captureBlob(vid) {
  const c  = document.createElement("canvas");
  c.width  = vid.videoWidth  || 640;
  c.height = vid.videoHeight || 480;
  c.getContext("2d").drawImage(vid, 0, 0);
  return new Promise(res => c.toBlob(res, "image/jpeg", 0.92));
}

// ================================================================
// HEAD MOVEMENT DETECTION
// ================================================================
let detecting = false;
let noseHist  = [];

function stopDetect() { detecting = false; }

function detect(vid, challenge, onOK, onTO) {
  detecting = true;
  noseHist  = [];
  const deadline = performance.now() + 30000;

  const loop = () => {
    if (!detecting) return;
    if (performance.now() > deadline) { detecting = false; onTO && onTO(); return; }
    try {
      const r = mp.detectForVideo(vid, performance.now());
      if (r.faceLandmarks && r.faceLandmarks.length) {
        const n = r.faceLandmarks[0][1];
        const W = vid.videoWidth  || 640;
        const H = vid.videoHeight || 480;
        noseHist.push({x: n.x * W, y: n.y * H});
        if (noseHist.length > HIST) noseHist.shift();
        if (noseHist.length >= 8) {
          const s = noseHist[0], e = noseHist[noseHist.length - 1];
          const dx = e.x - s.x, dy = e.y - s.y;
          let det = null;
          if (Math.abs(dx) > THR)          det = dx > 0 ? "LEFT" : "RIGHT";
          else if (Math.abs(dy) > THR*.8)  det = dy < 0 ? "UP" : "DOWN";
          if (det && det === challenge.toUpperCase()) {
            detecting = false; onOK(); return;
          }
        }
      }
    } catch(_) {}
    requestAnimationFrame(loop);
  };
  loop();
}

// ================================================================
// UTILS
// ================================================================
function st(id, msg, type) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = "status show " + type;
}
function stHide(id) { document.getElementById(id).className = "status"; }
async function api(path, init = {}) {
  const r = await fetch(path, init);
  let d; try { d = await r.json(); } catch { d = {}; }
  if (!r.ok) throw new Error(d.detail || "HTTP " + r.status);
  return d;
}
function fd(obj) {
  const f = new FormData();
  Object.entries(obj).forEach(([k, v]) => f.append(k, v));
  return f;
}
function tab(i) {
  document.querySelectorAll(".panel").forEach((p, j) => p.classList.toggle("active", i === j));
  document.querySelectorAll(".tab-btn").forEach((b, j) => b.classList.toggle("active", i === j));
  if (i === 0) ensureCam("evid");
  if (i === 1) ensureCam("avid");
  if (i === 2) ensureCam("bvid");
  if (i === 3 && admTok) admRefresh();
}

// ================================================================
// ENROLMENT
// ================================================================
let enrolFaceBlob = null;
let enrolStep = 0;

function goEnrolStep(n) {
  enrolStep = n;
  document.getElementById("enrol-step0").style.display = n === 0 ? "" : "none";
  document.getElementById("enrol-step1").style.display = n === 1 ? "" : "none";
  document.getElementById("enrol-step2").style.display = n === 2 ? "" : "none";
  ["st0","st1","st2"].forEach((id, i) => {
    const el = document.getElementById(id);
    el.className = "step" + (i < n ? " done" : i === n ? " active" : "");
  });
  if (n === 1) ensureCam("evid");
}

async function enrolStep1() {
  const sid  = document.getElementById("e-sid").value.trim();
  const name = document.getElementById("e-name").value.trim();
  if (!sid || !name) return st("st-reg", "Student ID and Full Name are required", "err");

  // Auto-register if not already registered
  try {
    await api("/register", {method:"POST", body: fd({student_id: sid})});
  } catch(e) {
    if (!e.message.includes("already")) {
      return st("st-reg", "❌ " + e.message, "err");
    }
  }
  stHide("st-reg");
  goEnrolStep(1);
}

async function captureEnrolFace() {
  const vid = document.getElementById("evid");
  if (!streams[vid.id]) {
    st("e-face-status", "Camera not ready. Please allow access.", "err");
    return;
  }
  enrolFaceBlob = await captureBlob(vid);

  // Show preview
  const url = URL.createObjectURL(enrolFaceBlob);
  const prev = document.getElementById("e-preview");
  prev.src = url;
  prev.classList.add("show");

  st("e-face-status", "✅ Photo captured — looking good! Click Complete Enrolment.", "ok");
  document.getElementById("btn-submit-enrol").disabled = false;
}

async function submitEnrolment() {
  if (!enrolFaceBlob) return st("st-enrol", "Capture your photo first", "err");

  const btn = document.getElementById("btn-submit-enrol");
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> Processing…';
  st("st-enrol", "Extracting facial embedding…", "info");

  const f = new FormData();
  f.append("student_id", document.getElementById("e-sid").value.trim());
  f.append("full_name",  document.getElementById("e-name").value.trim());
  f.append("email",      document.getElementById("e-email").value.trim());
  f.append("program",    document.getElementById("e-prog").value);
  f.append("year",       document.getElementById("e-year").value);
  f.append("phone",      document.getElementById("e-phone").value.trim());
  f.append("face_image", enrolFaceBlob, "face.jpg");

  try {
    const data = await api("/enroll", {method:"POST", body: f});
    document.getElementById("enrol-done-name").textContent = data.full_name + " — Enrolled!";
    document.getElementById("enrol-done-meta").textContent =
      data.student_id + " · Embedding dim: " + data.embedding_dim;
    goEnrolStep(2);
  } catch(e) {
    st("st-enrol", "❌ " + e.message, "err");
    btn.disabled = false;
    btn.innerHTML = "✓ Complete Enrolment";
  }
}

function resetEnrol() {
  ["e-sid","e-name","e-email","e-phone"].forEach(id => document.getElementById(id).value = "");
  document.getElementById("e-prog").value = "";
  document.getElementById("e-year").value = "1";
  document.getElementById("e-preview").classList.remove("show");
  enrolFaceBlob = null;
  goEnrolStep(0);
}

// ================================================================
// INDIVIDUAL ATTENDANCE
// ================================================================
let attending = false;

async function startAttend() {
  if (attending) return;
  const sid = document.getElementById("a-sid").value.trim();
  if (!sid) return st("st-attend", "Enter your student ID", "err");

  attending = true;
  const btn = document.getElementById("btn-attend");
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> Preparing…';
  stHide("st-attend");

  await ensureCam("avid");
  const vid = document.getElementById("avid");

  let chal;
  try {
    chal = await api("/challenge", {method:"POST", body: fd({student_id: sid})});
  } catch(e) {
    st("st-attend", "❌ " + e.message, "err");
    resetAttendBtn(); return;
  }

  const {challenge, challenge_id} = chal;
  document.getElementById("ach-val").textContent  = "↔ " + challenge;
  document.getElementById("ach-hint").textContent = "Move your head " + challenge.toLowerCase() + " now";
  document.getElementById("acam").classList.add("detecting");
  st("st-attend", "👁 Detection active — move your head " + challenge.toLowerCase(), "info");
  btn.innerHTML = "Detecting…";

  detect(vid, challenge, async () => {
    document.getElementById("acam").classList.replace("detecting","ok");
    document.getElementById("ach-val").textContent  = "✓";
    document.getElementById("ach-hint").textContent = "Capturing face…";
    st("st-attend", "Face captured — verifying identity…", "info");

    // Capture face for verification
    const blob = await captureBlob(vid);
    const f = new FormData();
    f.append("student_id",   sid);
    f.append("movement",     challenge);
    f.append("challenge_id", challenge_id);
    f.append("face_image",   blob, "live.jpg");

    try {
      const res = await api("/mark_attendance", {method:"POST", body: f});
      const score = res.face_score ? " (match " + (res.face_score * 100).toFixed(1) + "%)" : "";
      const verified = res.face_verified ? "✅ Identity verified" : "⚠️ Not enrolled — attendance recorded without face check";
      st("st-attend", verified + score, res.face_verified ? "ok" : "warn");
    } catch(e) {
      st("st-attend", "❌ " + e.message, "err");
    }

    setTimeout(() => {
      document.getElementById("acam").classList.remove("ok");
      document.getElementById("ach-val").textContent  = "—";
      document.getElementById("ach-hint").textContent = "Press Start to begin";
      resetAttendBtn();
    }, 3000);
  }, () => {
    st("st-attend", "⏱ Timeout — please try again", "err");
    document.getElementById("acam").classList.remove("detecting");
    document.getElementById("ach-val").textContent  = "—";
    document.getElementById("ach-hint").textContent = "Press Start to begin";
    resetAttendBtn();
  });
}

function resetAttendBtn() {
  attending = false;
  const btn = document.getElementById("btn-attend");
  btn.disabled = false;
  btn.innerHTML = "▶ Mark Attendance";
}

// ================================================================
// BATCH
// ================================================================
let bQ = [], bI = 0, bRunning = false;

async function startBatch() {
  if (bRunning) return;
  const txt = document.getElementById("b-ids").value.trim();
  if (!txt) return st("st-batch", "Enter at least one student ID", "err");
  bQ = txt.split("\n").map(s => s.trim()).filter(Boolean);
  bI = 0; bRunning = true;
  document.getElementById("btn-batch").disabled = true;
  renderBList();
  await ensureCam("bvid");
  bNext();
}

async function bNext() {
  if (bI >= bQ.length) {
    st("st-batch", "🎉 Complete — " + bQ.length + " students processed", "ok");
    document.getElementById("bprog").style.width = "100%";
    document.getElementById("btn-batch").disabled = false;
    document.getElementById("bch-val").textContent  = "✓";
    document.getElementById("bch-hint").textContent = "All done";
    bRunning = false; return;
  }

  const sid = bQ[bI];
  document.getElementById("bprog").style.width = Math.round(bI / bQ.length * 100) + "%";
  bUpdItem(bI, "pill-cur", "Active");
  st("st-batch", (bI+1) + "/" + bQ.length + ": " + sid, "info");

  let chal;
  try {
    chal = await api("/challenge", {method:"POST", body: fd({student_id: sid})});
  } catch(e) {
    bUpdItem(bI, "pill-fail", "❌ " + e.message);
    bI++; setTimeout(bNext, 300); return;
  }

  const {challenge, challenge_id} = chal;
  document.getElementById("bch-val").textContent  = "↔ " + challenge;
  document.getElementById("bch-hint").textContent = sid + " → " + challenge.toLowerCase();
  const vid = document.getElementById("bvid");
  stopDetect();

  detect(vid, challenge, async () => {
    bUpdItem(bI, "pill-ok", "Capturing…");
    const blob = await captureBlob(vid);
    const f = new FormData();
    f.append("student_id",   sid);
    f.append("movement",     challenge);
    f.append("challenge_id", challenge_id);
    f.append("face_image",   blob, "live.jpg");
    try {
      const res = await api("/mark_attendance", {method:"POST", body: f});
      const label = res.face_verified ? "✅ Verified" : "⚠️ No embed";
      bUpdItem(bI, res.face_verified ? "pill-ok" : "pill-cur", label);
    } catch(e) {
      bUpdItem(bI, "pill-fail", "❌ " + e.message);
    }
    bI++; setTimeout(bNext, 500);
  }, () => {
    bUpdItem(bI, "pill-fail", "⏱ Timeout");
    bI++; setTimeout(bNext, 300);
  });
}

function renderBList() {
  document.getElementById("blist").innerHTML = bQ.map((id, i) => `
    <div class="bi" id="bi-${i}">
      <span>${id}</span>
      <span class="pill pill-pend" id="bp-${i}">Pending</span>
    </div>`).join("");
}

function bUpdItem(i, cls, txt) {
  const pill = document.getElementById("bp-" + i);
  const row  = document.getElementById("bi-" + i);
  if (!pill) return;
  pill.className = "pill " + cls;
  pill.textContent = txt;
  if (row) {
    row.classList.toggle("cur", cls === "pill-cur");
    row.scrollIntoView({block:"nearest", behavior:"smooth"});
  }
}

// ================================================================
// ADMIN
// ================================================================
let admTok = null;

async function admLogin() {
  const u = document.getElementById("adm-u").value.trim();
  const p = document.getElementById("adm-p").value;
  if (!u || !p) return st("st-login", "Enter username and password", "err");
  try {
    const d = await api("/admin/login", {method:"POST", body: fd({username:u, password:p})});
    admTok = d.access_token;
    localStorage.setItem("admTok", admTok);
    showAdm(u); admRefresh();
  } catch(e) {
    st("st-login", "❌ " + e.message, "err");
  }
}

function showAdm(user) {
  document.getElementById("login-sec").style.display = "none";
  document.getElementById("adm-sec").style.display   = "block";
  document.getElementById("adm-greet").textContent   = "Signed in as " + user + " · session 2h";
}

function admLogout() {
  admTok = null; localStorage.removeItem("admTok");
  document.getElementById("login-sec").style.display = "block";
  document.getElementById("adm-sec").style.display   = "none";
}

async function admRefresh() {
  if (!admTok) return;
  const hdr = {Authorization: "Bearer " + admTok};
  try {
    const [a, s] = await Promise.all([
      api("/analytics",  {headers: hdr}),
      api("/students",   {headers: hdr}),
    ]);
    document.getElementById("s-total").textContent = a.total_attendance_records;
    document.getElementById("s-today").textContent = a.today_count;
    document.getElementById("s-enr").textContent   = a.enrolled_with_face + "/" + a.total_registered;

    const tbody = document.getElementById("adm-tbody");
    if (!s.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="color:var(--muted);padding:16px;text-align:center">No students yet</td></tr>';
    } else {
      tbody.innerHTML = s.map(r => `<tr>
        <td>${r.student_id}</td>
        <td>${r.full_name || '<span style="color:var(--muted)">—</span>'}</td>
        <td style="color:var(--muted)">${r.program || "—"}</td>
        <td style="color:var(--muted)">${r.year || "—"}</td>
        <td style="color:var(--muted);font-size:.72rem">${r.email || "—"}</td>
        <td><span class="badge ${r.has_face ? 'b-yes':'b-no'}">${r.has_face ? '✓ Yes':'Pending'}</span></td>
        <td style="color:var(--muted)">${r.enrolled_at ? new Date(r.enrolled_at).toLocaleDateString() : "—"}</td>
      </tr>`).join("");
    }
  } catch(e) {
    if (e.message.includes("401")) admLogout();
  }
}

// ================================================================
// BOOT
// ================================================================
window.addEventListener("load", async () => {
  try {
    await initMP();
  } catch(e) {
    alert("⚠️ Could not load face detection model. Check your internet connection.");
    return;
  } finally {
    document.getElementById("loader").classList.add("gone");
  }
  // Default tab camera
  ensureCam("evid");

  // Restore admin session
  const tok = localStorage.getItem("admTok");
  if (tok) {
    admTok = tok;
    try {
      const v = await api("/admin/verify-token", {method:"POST", headers:{Authorization:"Bearer "+tok}});
      showAdm(v.username);
    } catch(_) {
      localStorage.removeItem("admTok");
      admTok = null;
    }
  }
});
</script>
</body>
</html>"""
