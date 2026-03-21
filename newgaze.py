"""
interview_cheat_detector.py
════════════════════════════════════════════════════════════════════
Market-ready Interview Integrity Monitor
Iris tracking · Head pose · Blink analysis · Live + Session risk

Python 3.10+  |  opencv-python  mediapipe  numpy

Usage
─────
  python interview_cheat_detector.py [options]

  --source  INT|PATH   Camera index or video file path  (default: 0)
  --name    TEXT       Candidate name for the report    (default: "Candidate")
  --out     PATH       Report output directory          (default: ./reports)
  --config  PATH       JSON config file to override defaults
  --headless           Run without display (background service mode)
  --loglevel LEVEL     DEBUG | INFO | WARNING            (default: INFO)

Keys (during session)
─────────────────────
  Q   Quit & save report
  R   Reset session
  S   Save report now (without quitting)
  C   Recalibrate gaze baseline
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════════
def _setup_logging(level: str) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    return logging.getLogger("integrity")


log = logging.getLogger("integrity")


# ════════════════════════════════════════════════════════════════════
#  CONFIG  — defaults + JSON override support
# ════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Calibration
    calib_secs: float = 5.0

    # Gaze dead-zone (face-relative, after personal baseline).
    # Increase to be more forgiving; decrease to be stricter.
    gaze_h_thresh: float = 0.22  # horizontal dead-zone half-width
    gaze_v_thresh: float = 0.18  # vertical dead-zone half-height

    # Sustained gaze off-center before a FLAG fires (seconds)
    gaze_sustain: float = 4.0

    # Head pose (solvePnP yaw)
    head_yaw_thresh: float = 30.0  # degrees — deliberate large turn
    head_sustain: float = 3.0  # seconds

    # Face hidden before flagging
    face_lost_secs: float = 4.0

    # Blink stress detection
    ear_thresh: float = 0.18  # EAR below this = blink
    blink_count: int = 9  # blinks needed in window
    blink_window: float = 5.0  # seconds

    # Multi-face warning
    multi_face_warn: bool = True  # flag if >1 face appears

    # Per-event score weights
    score_gaze_lr: float = 2.0
    score_gaze_up: float = 1.5
    score_gaze_down: float = 0.8
    score_head: float = 3.0
    score_face_lost: float = 5.0
    score_blink: float = 1.5
    score_multi_face: float = 4.0

    # Minimum gap (sec) before the same flag type can fire again
    cooldown_secs: float = 15.0

    # Cumulative score thresholds → risk levels
    score_moderate: float = 4.0
    score_high: float = 10.0
    score_alert: float = 20.0

    # Score decay rate (pts/sec) — old events fade over time
    score_decay: float = 0.04

    # Live risk thresholds (fraction of threshold, not absolute)
    live_gaze_mod: float = 0.60  # gaze_dev fraction → MODERATE
    live_gaze_high: float = 1.00  # gaze_dev fraction → HIGH
    live_head_mod: float = 0.55
    live_head_high: float = 1.00

    # Temporal smoothing (frames)
    smooth_frames: int = 14

    # Alert overlay hold time (seconds)
    alert_hold_secs: float = 2.5

    # Display
    fps_cap: int = 30

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                log.warning("Unknown config key ignored: %s", k)
        return cfg

    def score_for(self, kind: str) -> float:
        return {
            "gaze_left": self.score_gaze_lr,
            "gaze_right": self.score_gaze_lr,
            "gaze_up": self.score_gaze_up,
            "gaze_down": self.score_gaze_down,
            "head_turn": self.score_head,
            "face_lost": self.score_face_lost,
            "rapid_blink": self.score_blink,
            "multi_face": self.score_multi_face,
        }.get(kind, 1.0)


# ════════════════════════════════════════════════════════════════════
#  ENUMS
# ════════════════════════════════════════════════════════════════════
class Zone(str, Enum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    NO_FACE = "NO FACE"
    MULTI = "MULTI FACE"


class Risk(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    ALERT = "ALERT"
    NO_FACE = "NO FACE"


# Severity ordering (used for comparisons)
RISK_RANK: dict[Risk, int] = {
    Risk.LOW: 0,
    Risk.MODERATE: 1,
    Risk.HIGH: 2,
    Risk.ALERT: 3,
    Risk.NO_FACE: 2,
}


def max_risk(a: Risk, b: Risk) -> Risk:
    return a if RISK_RANK[a] >= RISK_RANK[b] else b


# ════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ════════════════════════════════════════════════════════════════════
@dataclass
class IntegrityFlag:
    timestamp: float
    kind: str
    duration: float = 0.0
    detail: str = ""
    score: float = 0.0


@dataclass
class FrameResult:
    zone: Zone = Zone.NO_FACE
    iris_x: float = 0.0  # smoothed, baseline-relative
    iris_y: float = 0.0
    head_yaw: float = 0.0  # degrees, + = right
    head_pitch: float = 0.0
    l_ear: float = 0.0
    r_ear: float = 0.0
    face_count: int = 0
    blink: bool = False
    calibrating: bool = False
    calib_pct: float = 0.0
    # Risk signals
    live_risk: Risk = Risk.LOW
    sess_risk: Risk = Risk.LOW
    sess_score: float = 0.0
    # Normalised 0-1 intensities for UI bars
    gaze_intensity: float = 0.0
    head_intensity: float = 0.0
    # New flags fired this frame (cleared next frame)
    new_flags: list[IntegrityFlag] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES  (MediaPipe Face Mesh 468 + 10 iris)
# ════════════════════════════════════════════════════════════════════
class LM:
    # Iris centres (refine_landmarks=True)
    L_IRIS = 468
    R_IRIS = 473
    # Eye corners & lids
    L_TOP = 159
    L_BOT = 145
    L_LEFT = 33
    L_RIGHT = 133
    R_TOP = 386
    R_BOT = 374
    R_LEFT = 362
    R_RIGHT = 263
    # Face bounding references
    NOSE = 1
    CHIN = 152
    L_CHK = 234
    R_CHK = 454
    FORE = 10
    # Mouth corners (for head-pose solvePnP)
    M_LEFT = 61
    M_RIGHT = 291


# ════════════════════════════════════════════════════════════════════
#  CALIBRATOR  — learns per-person neutral gaze (median-robust)
# ════════════════════════════════════════════════════════════════════
class Calibrator:
    """
    Collects iris samples during the calibration window and computes
    a robust median baseline. Rejects samples with excessive jitter
    to avoid a bad calibration from head movements.
    """

    MAX_JITTER = 0.06  # face-relative units; sample discarded if > this

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._xs: list[float] = []
        self._ys: list[float] = []
        self.baseline_x: float = 0.0
        self.baseline_y: float = 0.0
        self.done: bool = False
        self.quality: float = 1.0  # 0-1; low = jittery calibration
        self._t0: float = time.time()

    def reset(self) -> None:
        self.__init__(self._cfg)

    @property
    def progress(self) -> float:
        return min(1.0, (time.time() - self._t0) / self._cfg.calib_secs)

    def feed(self, x: float, y: float) -> None:
        if self.done:
            return
        # Reject jittery samples mid-collection
        if self._xs:
            dx = abs(x - self._xs[-1])
            dy = abs(y - self._ys[-1])
            if dx > self.MAX_JITTER or dy > self.MAX_JITTER:
                log.debug("Calib sample rejected (jitter dx=%.3f dy=%.3f)", dx, dy)
                return
        self._xs.append(x)
        self._ys.append(y)

        if self.progress >= 1.0 and len(self._xs) >= 10:
            self.baseline_x = float(np.median(self._xs))
            self.baseline_y = float(np.median(self._ys))
            # Quality = 1 - normalised std (lower std = better)
            std = float(np.std(self._xs) + np.std(self._ys))
            self.quality = max(0.0, 1.0 - std / 0.05)
            self.done = True
            log.info(
                "Calibration complete — baseline (%.4f, %.4f)  quality=%.0f%%",
                self.baseline_x,
                self.baseline_y,
                self.quality * 100,
            )
            if self.quality < 0.5:
                log.warning(
                    "Calibration quality low (%.0f%%) — consider recalibrating (C)",
                    self.quality * 100,
                )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return x - self.baseline_x, y - self.baseline_y


# ════════════════════════════════════════════════════════════════════
#  SESSION  — cumulative score, zone time, flag history
# ════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, cfg: Config, candidate: str = "Candidate") -> None:
        self._cfg = cfg
        self.candidate = candidate
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = datetime.now()
        self.t0 = time.time()
        self.flags: list[IntegrityFlag] = []
        self._score: float = 0.0
        self._last_decay: float = time.time()
        self.blink_ts: deque[float] = deque(maxlen=60)
        self.zone_secs: dict[Zone, float] = {z: 0.0 for z in Zone}
        self._last_zone: Zone = Zone.CENTER
        self._last_tick: float = time.time()
        # Sustained-state onset timestamps
        self._gaze_off_t: Optional[float] = None
        self._head_off_t: Optional[float] = None
        self._face_lost_t: Optional[float] = None
        # Per-kind cooldown timestamps
        self._cooldowns: dict[str, float] = {}

    def reset(self, candidate: Optional[str] = None) -> None:
        name = candidate or self.candidate
        self.__init__(self._cfg, name)

    # ── Score with time-based decay ──────────────────────────────
    def _apply_decay(self) -> None:
        now = time.time()
        elapsed = now - self._last_decay
        self._score = max(0.0, self._score - self._cfg.score_decay * elapsed)
        self._last_decay = now

    @property
    def score(self) -> float:
        self._apply_decay()
        return self._score

    @property
    def risk(self) -> Risk:
        s = self.score
        if s < self._cfg.score_moderate:
            return Risk.LOW
        if s < self._cfg.score_high:
            return Risk.MODERATE
        if s < self._cfg.score_alert:
            return Risk.HIGH
        return Risk.ALERT

    # ── Zone time accounting ─────────────────────────────────────
    def tick_zone(self, zone: Zone) -> None:
        now = time.time()
        elapsed = now - self._last_tick
        self.zone_secs[self._last_zone] = (
            self.zone_secs.get(self._last_zone, 0.0) + elapsed
        )
        self._last_zone = zone
        self._last_tick = now

    @property
    def center_pct(self) -> float:
        total = sum(self.zone_secs.values())
        return self.zone_secs.get(Zone.CENTER, 0.0) / total * 100 if total else 100.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    # ── Flag firing with cooldown ─────────────────────────────────
    def _on_cooldown(self, kind: str) -> bool:
        return time.time() - self._cooldowns.get(kind, 0.0) < self._cfg.cooldown_secs

    def fire(self, flag: IntegrityFlag) -> bool:
        """
        Record a flag. Returns True if accepted, False if on cooldown.
        Scoring uses a single decay pass so reading `score` twice on the
        same call is idempotent.
        """
        if self._on_cooldown(flag.kind):
            return False
        self._apply_decay()
        self._score += flag.score
        self._cooldowns[flag.kind] = flag.timestamp
        self.flags.append(flag)
        log.warning(
            "[FLAG] %-18s  +%.1f pts  → total %.1f  |  %s",
            flag.kind,
            flag.score,
            self._score,
            flag.detail,
        )
        return True

    # ── Sustained-state checks ────────────────────────────────────
    def check_gaze(self, off: bool, zone: Zone) -> Optional[IntegrityFlag]:
        now = time.time()
        if off:
            if self._gaze_off_t is None:
                self._gaze_off_t = now
            elif now - self._gaze_off_t >= self._cfg.gaze_sustain:
                dur = now - self._gaze_off_t
                self._gaze_off_t = now  # reset so it fires again after next sustain
                kind = f"gaze_{zone.value.lower().replace(' ', '_')}"
                return IntegrityFlag(
                    timestamp=now,
                    kind=kind,
                    duration=round(dur, 1),
                    detail=f"Eyes {zone.value} for {dur:.1f}s",
                    score=self._cfg.score_for(kind),
                )
        else:
            self._gaze_off_t = None
        return None

    def check_head(self, yaw: float) -> Optional[IntegrityFlag]:
        now = time.time()
        if abs(yaw) > self._cfg.head_yaw_thresh:
            if self._head_off_t is None:
                self._head_off_t = now
            elif now - self._head_off_t >= self._cfg.head_sustain:
                dur = now - self._head_off_t
                self._head_off_t = now
                direction = "right" if yaw > 0 else "left"
                return IntegrityFlag(
                    timestamp=now,
                    kind="head_turn",
                    duration=round(dur, 1),
                    detail=f"Head turned {direction} {abs(yaw):.0f}° for {dur:.1f}s",
                    score=self._cfg.score_for("head_turn"),
                )
        else:
            self._head_off_t = None
        return None

    def check_face(self, present: bool) -> Optional[IntegrityFlag]:
        now = time.time()
        if not present:
            if self._face_lost_t is None:
                self._face_lost_t = now
            elif now - self._face_lost_t >= self._cfg.face_lost_secs:
                dur = now - self._face_lost_t
                self._face_lost_t = now
                return IntegrityFlag(
                    timestamp=now,
                    kind="face_lost",
                    duration=round(dur, 1),
                    detail=f"Face not visible for {dur:.1f}s",
                    score=self._cfg.score_for("face_lost"),
                )
        else:
            self._face_lost_t = None
        return None

    def check_blink(self) -> Optional[IntegrityFlag]:
        now = time.time()
        self.blink_ts.append(now)
        recent = [t for t in self.blink_ts if now - t <= self._cfg.blink_window]
        if len(recent) >= self._cfg.blink_count:
            self.blink_ts.clear()
            return IntegrityFlag(
                timestamp=now,
                kind="rapid_blink",
                detail=f"{len(recent)} blinks in {self._cfg.blink_window:.0f}s",
                score=self._cfg.score_for("rapid_blink"),
            )
        return None

    def check_multi_face(self, count: int) -> Optional[IntegrityFlag]:
        if count > 1 and self._cfg.multi_face_warn:
            return IntegrityFlag(
                timestamp=time.time(),
                kind="multi_face",
                detail=f"{count} faces detected simultaneously",
                score=self._cfg.score_for("multi_face"),
            )
        return None

    # ── Report ────────────────────────────────────────────────────
    def save_report(self, out_dir: str = "reports") -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in self.candidate)
        path = os.path.join(out_dir, f"{safe_name}_{self.session_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "candidate",
                    "session_id",
                    "started_at",
                    "duration_s",
                    "session_risk",
                    "score",
                    "center_gaze_pct",
                    "total_flags",
                ]
            )
            el = int(self.elapsed)
            w.writerow(
                [
                    self.candidate,
                    self.session_id,
                    self.started_at.isoformat(),
                    el,
                    self.risk.value,
                    round(self.score, 2),
                    round(self.center_pct, 1),
                    len(self.flags),
                ]
            )
            w.writerow([])
            w.writerow(["time", "kind", "duration_s", "score", "detail"])
            for fl in self.flags:
                w.writerow(
                    [
                        datetime.fromtimestamp(fl.timestamp).strftime("%H:%M:%S"),
                        fl.kind,
                        fl.duration,
                        fl.score,
                        fl.detail,
                    ]
                )
        log.info("Report saved → %s", path)
        return path


# ════════════════════════════════════════════════════════════════════
#  LIVE RISK  — computed from current-frame metrics ONLY
#  Returns (risk, gaze_intensity_0_1, head_intensity_0_1)
# ════════════════════════════════════════════════════════════════════
def compute_live_risk(
    ix: float,
    iy: float,
    yaw: float,
    face_count: int,
    cfg: Config,
) -> tuple[Risk, float, float]:

    if face_count == 0:
        return Risk.NO_FACE, 0.0, 0.0

    if face_count > 1:
        return Risk.ALERT, 1.0, 0.0

    # Normalised deviations (1.0 = at threshold, >1 = beyond)
    gaze_dev = max(abs(ix) / cfg.gaze_h_thresh, abs(iy) / cfg.gaze_v_thresh)
    head_dev = abs(yaw) / cfg.head_yaw_thresh

    g_int = min(1.0, gaze_dev)
    h_int = min(1.0, head_dev)

    # Risk classification — requires BOTH to be significantly off for HIGH/ALERT
    # to avoid false positives from natural glances
    both_off = gaze_dev >= cfg.live_gaze_high and head_dev >= cfg.live_head_mod

    if both_off:
        live = Risk.ALERT
    elif gaze_dev >= cfg.live_gaze_high or head_dev >= cfg.live_head_high:
        live = Risk.HIGH
    elif gaze_dev >= cfg.live_gaze_mod or head_dev >= cfg.live_head_mod:
        live = Risk.MODERATE
    else:
        live = Risk.LOW

    return live, g_int, h_int


# ════════════════════════════════════════════════════════════════════
#  DETECTOR  — MediaPipe wrapper + full per-frame analysis
# ════════════════════════════════════════════════════════════════════
class Detector:
    # Generic 3-D face model for solvePnP head-pose estimation
    _MODEL_3D = np.array(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -330.0, -65.0),  # chin
            (-225.0, 170.0, -135.0),  # left eye corner
            (225.0, 170.0, -135.0),  # right eye corner
            (-150.0, -150.0, -125.0),  # mouth left
            (150.0, -150.0, -125.0),  # mouth right
        ],
        dtype=np.float64,
    )

    def __init__(self, cfg: Config, candidate: str = "Candidate") -> None:
        self._cfg = cfg
        fm = mp.solutions.face_mesh
        self._mesh = fm.FaceMesh(
            refine_landmarks=True,
            max_num_faces=2,  # detect up to 2 so we can warn
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
            static_image_mode=False,
        )
        self._ix: deque[float] = deque(maxlen=cfg.smooth_frames)
        self._iy: deque[float] = deque(maxlen=cfg.smooth_frames)
        self._iyaw: deque[float] = deque(maxlen=cfg.smooth_frames)
        self._prev_blink: bool = False
        self._last_valid_pts: Optional[np.ndarray] = None  # for solvePnP caching
        self._last_yaw: float = 0.0
        self._last_pitch: float = 0.0

        self.session = Session(cfg, candidate)
        self.calibrator = Calibrator(cfg)

    def recalibrate(self) -> None:
        self.calibrator.reset()
        log.info("Recalibration started — look straight ahead.")

    def process(self, frame: np.ndarray) -> FrameResult:
        if frame is None or frame.size == 0:
            log.warning("Empty frame received — skipping.")
            return FrameResult()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        try:
            result = self._mesh.process(rgb)
        except Exception:
            log.error("MediaPipe error:\n%s", traceback.format_exc())
            return FrameResult()

        new_flags: list[IntegrityFlag] = []
        face_count = (
            len(result.multi_face_landmarks) if result.multi_face_landmarks else 0
        )

        # ── No face ─────────────────────────────────────────────
        if face_count == 0:
            flag = self.session.check_face(False)
            if flag and self.session.fire(flag):
                new_flags.append(flag)
            self.session.tick_zone(Zone.NO_FACE)
            lr, gi, hi = compute_live_risk(0, 0, 0, 0, self._cfg)
            return FrameResult(
                zone=Zone.NO_FACE,
                face_count=0,
                calibrating=not self.calibrator.done,
                calib_pct=self.calibrator.progress,
                live_risk=lr,
                sess_risk=self.session.risk,
                sess_score=round(self.session.score, 2),
                gaze_intensity=gi,
                head_intensity=hi,
                new_flags=new_flags,
            )

        self.session.check_face(True)

        # ── Multi-face detection ─────────────────────────────────
        if face_count > 1:
            flag = self.session.check_multi_face(face_count)
            if flag and self.session.fire(flag):
                new_flags.append(flag)
            self.session.tick_zone(Zone.MULTI)
            lr, gi, hi = compute_live_risk(0, 0, 0, face_count, self._cfg)
            return FrameResult(
                zone=Zone.MULTI,
                face_count=face_count,
                live_risk=lr,
                sess_risk=self.session.risk,
                sess_score=round(self.session.score, 2),
                gaze_intensity=gi,
                head_intensity=hi,
                new_flags=new_flags,
            )

        # ── Single face ──────────────────────────────────────────
        lm = result.multi_face_landmarks[0].landmark
        raw_x, raw_y = self._iris_position(lm)

        # Calibration phase
        if not self.calibrator.done:
            self.calibrator.feed(raw_x, raw_y)
            self.session.tick_zone(Zone.CENTER)
            return FrameResult(
                zone=Zone.CENTER,
                face_count=1,
                calibrating=True,
                calib_pct=self.calibrator.progress,
                live_risk=Risk.LOW,
                sess_risk=Risk.LOW,
                sess_score=0.0,
            )

        # Apply personal baseline + smooth
        dx, dy = self.calibrator.apply(raw_x, raw_y)
        self._ix.append(dx)
        self._iy.append(dy)
        sx = float(np.mean(self._ix))
        sy = float(np.mean(self._iy))

        # Gaze zone + flag check
        zone = self._classify_zone(sx, sy)
        is_off = zone != Zone.CENTER
        self.session.tick_zone(zone)

        flag = self.session.check_gaze(is_off, zone)
        if flag and self.session.fire(flag):
            new_flags.append(flag)

        # Blink / EAR
        l_ear = self._eye_aspect_ratio(lm, "left")
        r_ear = self._eye_aspect_ratio(lm, "right")
        blink = (l_ear + r_ear) / 2 < self._cfg.ear_thresh
        if blink and not self._prev_blink:
            flag2 = self.session.check_blink()
            if flag2 and self.session.fire(flag2):
                new_flags.append(flag2)
        self._prev_blink = blink

        # Head pose
        yaw, pitch = self._head_pose(lm, w, h)
        self._iyaw.append(yaw)
        smooth_yaw = float(np.mean(self._iyaw))

        flag3 = self.session.check_head(smooth_yaw)
        if flag3 and self.session.fire(flag3):
            new_flags.append(flag3)

        # Live risk (frame-level, no history)
        live_risk, g_int, h_int = compute_live_risk(sx, sy, smooth_yaw, 1, self._cfg)

        return FrameResult(
            zone=zone,
            iris_x=round(sx, 4),
            iris_y=round(sy, 4),
            head_yaw=round(smooth_yaw, 1),
            head_pitch=round(pitch, 1),
            l_ear=round(l_ear, 3),
            r_ear=round(r_ear, 3),
            face_count=1,
            blink=blink,
            calibrating=False,
            calib_pct=1.0,
            live_risk=live_risk,
            sess_risk=self.session.risk,
            sess_score=round(self.session.score, 2),
            gaze_intensity=g_int,
            head_intensity=h_int,
            new_flags=new_flags,
        )

    def release(self) -> None:
        self._mesh.close()
        log.info("Detector released.")

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Private helpers ──────────────────────────────────────────
    def _iris_position(self, lm) -> tuple[float, float]:
        """Return iris centre normalised to face bounding box."""
        li = lm[LM.L_IRIS]
        ri = lm[LM.R_IRIS]
        raw_x = (li.x + ri.x) / 2
        raw_y = (li.y + ri.y) / 2
        xs = [lm[LM.L_CHK].x, lm[LM.R_CHK].x]
        ys = [lm[LM.FORE].y, lm[LM.CHIN].y]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        fw = max(xs) - min(xs)
        fh = max(ys) - min(ys)
        if fw < 1e-6 or fh < 1e-6:
            return 0.0, 0.0
        return (raw_x - cx) / fw, (raw_y - cy) / fh

    def _classify_zone(self, x: float, y: float) -> Zone:
        if x < -self._cfg.gaze_h_thresh:
            return Zone.LEFT
        elif x > self._cfg.gaze_h_thresh:
            return Zone.RIGHT
        elif y < -self._cfg.gaze_v_thresh:
            return Zone.UP
        elif y > self._cfg.gaze_v_thresh:
            return Zone.DOWN
        return Zone.CENTER

    def _eye_aspect_ratio(self, lm, side: str) -> float:
        if side == "left":
            t, b, l, r = LM.L_TOP, LM.L_BOT, LM.L_LEFT, LM.L_RIGHT
        else:
            t, b, l, r = LM.R_TOP, LM.R_BOT, LM.R_LEFT, LM.R_RIGHT
        v = abs(lm[t].y - lm[b].y)
        h = abs(lm[l].x - lm[r].x)
        return v / (h + 1e-6)

    def _head_pose(self, lm, w: int, h: int) -> tuple[float, float]:
        img_pts = np.array(
            [
                (lm[LM.NOSE].x * w, lm[LM.NOSE].y * h),
                (lm[LM.CHIN].x * w, lm[LM.CHIN].y * h),
                (lm[LM.L_LEFT].x * w, lm[LM.L_LEFT].y * h),
                (lm[LM.R_RIGHT].x * w, lm[LM.R_RIGHT].y * h),
                (lm[LM.M_LEFT].x * w, lm[LM.M_LEFT].y * h),
                (lm[LM.M_RIGHT].x * w, lm[LM.M_RIGHT].y * h),
            ],
            dtype=np.float64,
        )

        # Skip solvePnP if points barely moved (saves ~2ms/frame)
        if self._last_valid_pts is not None:
            if np.max(np.abs(img_pts - self._last_valid_pts)) < 2.0:
                return self._last_yaw, self._last_pitch

        self._last_valid_pts = img_pts
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            self._MODEL_3D,
            img_pts,
            cam,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0
        rm, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rm[0, 0] ** 2 + rm[1, 0] ** 2)
        yaw = float(np.degrees(np.arctan2(rm[1, 0], rm[0, 0])))
        pitch = float(np.degrees(np.arctan2(-rm[2, 0], sy)))
        self._last_yaw = yaw
        self._last_pitch = pitch
        return yaw, pitch


# ════════════════════════════════════════════════════════════════════
#  UI — palette, helpers, calibration screen, main overlay
# ════════════════════════════════════════════════════════════════════

# BGR colour palette
P: dict[str, tuple[int, int, int]] = {
    "bg": (10, 10, 15),
    "panel": (15, 15, 21),
    "divider": (34, 34, 46),
    "text": (212, 212, 218),
    "dim": (90, 90, 104),
    "green": (48, 205, 88),
    "yellow": (22, 190, 230),
    "orange": (15, 125, 255),
    "red": (40, 40, 220),
    "cyan": (195, 205, 52),
    "blue": (220, 150, 45),
    "white": (230, 230, 235),
}

RISK_COL: dict[Risk, tuple[int, int, int]] = {
    Risk.LOW: P["green"],
    Risk.MODERATE: P["yellow"],
    Risk.HIGH: P["orange"],
    Risk.ALERT: P["red"],
    Risk.NO_FACE: P["dim"],
}
ZONE_COL: dict[Zone, tuple[int, int, int]] = {
    Zone.CENTER: P["green"],
    Zone.LEFT: P["yellow"],
    Zone.RIGHT: P["yellow"],
    Zone.UP: P["cyan"],
    Zone.DOWN: (145, 205, 255),
    Zone.NO_FACE: P["dim"],
    Zone.MULTI: P["red"],
}

FN = cv2.FONT_HERSHEY_SIMPLEX
FND = cv2.FONT_HERSHEY_DUPLEX
PW = 265  # panel width
TH = 62  # top-bar height
PAD = 11  # panel inner padding
LH = 22  # standard line height


def _text(
    img, s: str, x: int, y: int, col: tuple = None, sc: float = 0.48, th: int = 1
) -> None:
    cv2.putText(img, s, (x, y), FN, sc, col or P["text"], th, cv2.LINE_AA)


def _dim(img, s: str, x: int, y: int, col: tuple = None, sc: float = 0.40) -> None:
    cv2.putText(img, s, (x, y), FN, sc, col or P["dim"], 1, cv2.LINE_AA)


def _bar(
    img,
    x: int,
    y: int,
    w: int,
    h: int,
    pct: float,
    fill: tuple,
    bg: tuple = (26, 26, 36),
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fw = int(w * max(0.0, min(1.0, pct)))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (44, 44, 58), 1)


def _divider(img, x: int, y: int, w: int) -> None:
    cv2.line(img, (x, y), (x + w, y), P["divider"], 1)


def _section(img, title: str, x: int, y: int, w: int) -> int:
    """Draw section label + rule, return next y."""
    cv2.putText(img, title.upper(), (x, y), FN, 0.38, P["dim"], 1, cv2.LINE_AA)
    lx = x + len(title) * 5 + 8
    cv2.line(img, (lx, y - 3), (x + w - PAD, y - 3), P["divider"], 1)
    return y + LH - 2


def _badge(
    img, text: str, x: int, y: int, bw: int, bh: int, bg: tuple, fg: tuple = (8, 8, 12)
) -> None:
    cv2.rectangle(img, (x, y), (x + bw, y + bh), bg, -1)
    darker = tuple(max(0, c - 28) for c in bg)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), darker, 1)
    tw, th2 = cv2.getTextSize(text, FND, 0.56, 2)[0]
    cv2.putText(
        img,
        text,
        (x + (bw - tw) // 2, y + (bh + th2) // 2),
        FND,
        0.56,
        fg,
        2,
        cv2.LINE_AA,
    )


def _pill(img, text: str, x: int, y: int, col: tuple) -> int:
    """Outlined pill — returns x after the pill for chaining."""
    tw = cv2.getTextSize(text, FN, 0.44, 1)[0][0]
    cv2.rectangle(img, (x - 3, y - 12), (x + tw + 3, y + 4), col, 1)
    cv2.putText(img, text, (x, y), FN, 0.44, col, 1, cv2.LINE_AA)
    return x + tw + 12


def _row(
    img, label: str, value: str, x: int, y: int, bw: int, val_col: tuple = None
) -> int:
    """Key-value row. Returns next y."""
    _dim(img, label, x, y)
    vx = x + bw - len(value) * 8 - 2
    _text(img, value, vx, y, val_col or P["text"], 0.46)
    return y + LH - 2


# ── Calibration screen ───────────────────────────────────────────
def _draw_calibration(
    frame: np.ndarray, pct: float, quality: float = 1.0
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (fw, fh), (6, 6, 18), -1)
    cv2.addWeighted(ov, 0.76, out, 0.24, 0, out)

    cx, cy = fw // 2, fh // 2
    r = 56

    # Progress ring
    cv2.circle(out, (cx, cy), r, (24, 24, 40), 2)
    for a in range(0, int(360 * pct), 3):
        rad = np.radians(a - 90)
        px2 = int(cx + r * np.cos(rad))
        py2 = int(cy + r * np.sin(rad))
        cv2.circle(out, (px2, py2), 2, (65, 155, 255), -1)

    # Crosshair target
    cv2.circle(out, (cx, cy), 6, (65, 155, 255), -1)
    cv2.line(out, (cx - 14, cy), (cx + 14, cy), (65, 155, 255), 1)
    cv2.line(out, (cx, cy - 14), (cx, cy + 14), (65, 155, 255), 1)

    cv2.putText(
        out,
        "CALIBRATING",
        (cx - 112, cy - 78),
        FND,
        0.96,
        (155, 155, 248),
        2,
        cv2.LINE_AA,
    )
    _text(
        out,
        "Look straight at the camera and stay still",
        cx - 208,
        cy + 80,
        (108, 108, 178),
        0.52,
    )

    bw = 280
    _bar(out, cx - bw // 2, cy + 98, bw, 9, pct, (65, 155, 255))
    _text(out, f"{int(pct * 100)}%", cx - 14, cy + 126, (65, 155, 255), 0.58)

    if pct >= 1.0 and quality < 0.65:
        warn = "Low quality — press C to retry"
        _text(out, warn, cx - 148, cy + 152, P["orange"], 0.50)

    return out


# ── Main overlay ─────────────────────────────────────────────────
def render(
    frame: np.ndarray,
    r: FrameResult,
    sess: Session,
    fps: float,
    alert_ts: float,  # timestamp of last new flag (for hold timer)
    alert_msg: str,
    cfg: Config,
) -> np.ndarray:

    if r.calibrating:
        return _draw_calibration(frame, r.calib_pct)

    out = frame.copy()
    fh, fw = out.shape[:2]
    lrc = RISK_COL[r.live_risk]
    src = RISK_COL[r.sess_risk]
    zc = ZONE_COL[r.zone]
    px = fw - PW
    bw = PW - PAD * 2

    # ── TOP BAR ──────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (fw, TH), P["bg"], -1)
    _divider(out, 0, TH, fw)

    _badge(out, f"LIVE  {r.live_risk.value}", PAD, 9, 132, 44, lrc)
    _badge(
        out, f"SESSION  {r.sess_risk.value}", PAD + 142, 9, 162, 44, (20, 20, 30), src
    )
    _pill(out, r.zone.value, PAD + 316, 36, zc)

    # Candidate name + FPS (top-right of top bar, above panel)
    _dim(out, sess.candidate[:18], px + PAD, 22)
    _dim(out, f"{fps:.0f} fps", px + PAD, 50, P["dim"], 0.36)

    # ── VIDEO REGION ─────────────────────────────────────────────
    cv2.rectangle(out, (0, TH), (px, fh), (0, 0, 0), 1)

    if r.face_count == 1:
        vw = px
        vh = fh - TH
        rx2 = int(np.clip(vw // 2 + r.iris_x * vw * 0.50, 14, vw - 14))
        ry2 = int(np.clip(TH + vh // 2 + r.iris_y * vh * 0.50, TH + 14, fh - 14))
        cv2.circle(out, (rx2, ry2), 13, zc, 2)
        cv2.circle(out, (rx2, ry2), 3, zc, -1)
        cv2.line(out, (rx2 - 18, ry2), (rx2 + 18, ry2), zc, 1)
        cv2.line(out, (rx2, ry2 - 18), (rx2, ry2 + 18), zc, 1)

    elif r.face_count > 1:
        # Multi-face overlay warning on video
        cv2.putText(
            out,
            f"! {r.face_count} FACES DETECTED !",
            (14, TH + 36),
            FND,
            0.80,
            P["red"],
            2,
            cv2.LINE_AA,
        )

    # ── SIDE PANEL ───────────────────────────────────────────────
    cv2.rectangle(out, (px, TH), (fw, fh), P["panel"], -1)
    cv2.line(out, (px, TH), (px, fh), P["divider"], 1)

    y = TH + 16
    x = px + PAD

    # ── LIVE STATUS ──────────────────────────────────────────────
    y = _section(out, "Live Status", x, y, PW)

    _dim(out, "GAZE", x, y)
    _text(out, r.zone.value, x + bw - len(r.zone.value) * 7, y, zc, 0.44)
    y += 5
    gc = (
        P["green"]
        if r.zone == Zone.CENTER
        else (P["yellow"] if r.gaze_intensity < 0.75 else P["orange"])
    )
    _bar(out, x, y, bw, 8, r.gaze_intensity, gc)
    cv2.line(out, (x + bw - 1, y - 2), (x + bw - 1, y + 10), (95, 95, 115), 1)
    y += 16

    _dim(out, "HEAD YAW", x, y)
    yaw_c = (
        P["text"]
        if abs(r.head_yaw) < cfg.live_head_mod * cfg.head_yaw_thresh
        else (P["yellow"] if abs(r.head_yaw) < cfg.head_yaw_thresh else P["orange"])
    )
    _text(out, f"{r.head_yaw:+.1f}\xb0", x + bw - 46, y, yaw_c, 0.44)
    y += 5
    hc = (
        P["green"]
        if r.head_intensity < 0.5
        else (P["yellow"] if r.head_intensity < 0.85 else P["orange"])
    )
    _bar(out, x, y, bw, 8, r.head_intensity, hc)
    y += 16

    avg_ear = (r.l_ear + r.r_ear) / 2
    _dim(out, "EAR / BLINK", x, y)
    ec = (0, 210, 255) if r.blink else P["text"]
    _text(
        out,
        f"{'BLINK' if r.blink else 'open'}  {avg_ear:.2f}",
        x + bw - 82,
        y,
        ec,
        0.44,
    )
    y += 18
    _divider(out, x, y, bw)
    y += 10

    # ── LIVE RISK ────────────────────────────────────────────────
    y = _section(out, "Live Risk", x, y, PW)
    _badge(out, r.live_risk.value, x, y, bw, 26, lrc)
    y += 32

    drivers: list[tuple[str, tuple]] = []
    if r.gaze_intensity >= 0.60:
        drivers.append(("GAZE", P["yellow"] if r.gaze_intensity < 1.0 else P["orange"]))
    if r.head_intensity >= 0.50:
        drivers.append(("HEAD", P["yellow"] if r.head_intensity < 1.0 else P["orange"]))
    if r.face_count == 0:
        drivers.append(("NO FACE", P["orange"]))
    if r.face_count > 1:
        drivers.append(("MULTI FACE", P["red"]))
    if r.blink:
        drivers.append(("BLINK", P["cyan"]))

    if drivers:
        dx2 = x
        for label, col in drivers:
            if dx2 > x + bw - 30:
                break
            dx2 = _pill(out, label, dx2, y, col)
    else:
        _dim(out, "No active concerns", x + 4, y, P["green"], 0.40)
    y += 22
    _divider(out, x, y, bw)
    y += 10

    # ── SESSION SCORE ────────────────────────────────────────────
    y = _section(out, "Session Score", x, y, PW)

    _text(out, f"{r.sess_score:.1f}", x, y, src, 0.56)
    _dim(out, f"/ {cfg.score_alert:.0f}", x + 34, y, None, 0.40)
    _text(out, r.sess_risk.value, x + bw - len(r.sess_risk.value) * 7 - 2, y, src, 0.46)
    y += 9

    # Segmented background then fill
    seg_pairs = [
        (cfg.score_moderate / cfg.score_alert, P["green"]),
        ((cfg.score_high - cfg.score_moderate) / cfg.score_alert, P["yellow"]),
        ((cfg.score_alert - cfg.score_high) / cfg.score_alert, P["orange"]),
    ]
    sx2 = x
    for sp, _ in seg_pairs:
        sw = int(bw * sp)
        cv2.rectangle(out, (sx2, y), (sx2 + sw, y + 9), (20, 20, 30), -1)
        sx2 += sw
    filled = int(bw * min(1.0, r.sess_score / cfg.score_alert))
    if filled > 0:
        cv2.rectangle(out, (x, y), (x + filled, y + 9), src, -1)
    cv2.rectangle(out, (x, y), (x + bw, y + 9), (44, 44, 58), 1)
    for tv in [cfg.score_moderate, cfg.score_high]:
        tx = x + int(bw * tv / cfg.score_alert)
        cv2.line(out, (tx, y - 2), (tx, y + 11), (76, 76, 92), 1)
    y += 18
    _dim(out, "LOW      MODERATE      HIGH    ALERT", x, y, None, 0.28)
    y += 14
    _divider(out, x, y, bw)
    y += 10

    # ── ATTENTION ────────────────────────────────────────────────
    y = _section(out, "Attention", x, y, PW)

    cp = sess.center_pct
    cp_col = P["green"] if cp >= 65 else P["yellow"] if cp >= 45 else P["orange"]
    _dim(out, "CENTER GAZE", x, y)
    _text(out, f"{cp:.1f}%", x + bw - 40, y, cp_col, 0.48)
    y += 5
    _bar(out, x, y, bw, 7, cp / 100, cp_col)
    y += 14

    tot = sum(sess.zone_secs.values()) or 1.0
    pill_x = x
    for z in [Zone.LEFT, Zone.RIGHT, Zone.UP, Zone.DOWN]:
        zt = sess.zone_secs.get(z, 0.0)
        if zt < 1.0:
            continue
        label = f"{z.value[0]} {zt / tot * 100:.0f}%"
        if pill_x + 50 > x + bw:
            break
        pill_x = _pill(out, label, pill_x, y + 13, ZONE_COL[z])
    y += 24
    _divider(out, x, y, bw)
    y += 10

    # ── SESSION INFO ─────────────────────────────────────────────
    y = _section(out, "Session", x, y, PW)

    el = int(sess.elapsed)
    y = _row(out, "DURATION", f"{el // 60:02d}:{el % 60:02d}", x, y, bw)
    y = _row(out, "CANDIDATE", sess.candidate[:14], x, y, bw)
    y = _row(
        out, "FLAGS", str(len(sess.flags)), x, y, bw, src if sess.flags else P["dim"]
    )
    y += 2
    _divider(out, x, y, bw)
    y += 10

    # ── EVENT LOG ────────────────────────────────────────────────
    y = _section(out, "Event Log", x, y, PW)

    max_rows = max(1, (fh - y - 26) // (LH - 4))
    recent = sess.flags[-max_rows:][::-1]
    if not recent:
        _dim(out, "No flags yet", x + 4, y, P["green"], 0.40)
        y += LH
    else:
        for fl in recent:
            if y + LH > fh - 24:
                break
            ts = datetime.fromtimestamp(fl.timestamp).strftime("%H:%M:%S")
            ec = (
                P["red"]
                if "face" in fl.kind
                else P["orange"] if "head" in fl.kind else P["yellow"]
            )
            _dim(out, ts, x, y, None, 0.36)
            _text(out, fl.kind.replace("_", " "), x + 50, y, ec, 0.38)
            _dim(out, f"+{fl.score:.1f}", x + bw - 22, y, ec, 0.36)
            y += LH - 4

    # ── EYE COMPASS ──────────────────────────────────────────────
    cr = 34
    ccx = cr + 12
    ccy = fh - cr - 12
    cv2.circle(out, (ccx, ccy), cr, (18, 18, 28), -1)
    cv2.circle(out, (ccx, ccy), cr, (50, 50, 62), 1)
    sr = max(4, int(cr * 0.40))
    cv2.circle(out, (ccx, ccy), sr, (0, 75, 38), 1)
    cv2.line(out, (ccx - cr + 4, ccy), (ccx + cr - 4, ccy), (32, 32, 46), 1)
    cv2.line(out, (ccx, ccy - cr + 4), (ccx, ccy + cr - 4), (32, 32, 46), 1)
    dox = int(np.clip(ccx + r.iris_x * cr * 2.2, ccx - cr + 5, ccx + cr - 5))
    doy = int(np.clip(ccy + r.iris_y * cr * 2.2, ccy - cr + 5, ccy + cr - 5))
    cv2.circle(out, (dox, doy), 6, zc, -1)
    cv2.circle(out, (dox, doy), 6, (0, 0, 0), 1)
    _dim(out, "EYE", ccx - 9, ccy + cr + 12, None, 0.30)

    # ── ALERT OVERLAY (timed hold) ────────────────────────────────
    if alert_msg and (time.time() - alert_ts) < cfg.alert_hold_secs:
        alpha = max(0.0, 1.0 - (time.time() - alert_ts) / cfg.alert_hold_secs)
        fl2 = out.copy()
        cv2.rectangle(fl2, (0, 0), (px, fh), (0, 0, 80), -1)
        cv2.addWeighted(fl2, 0.18 * alpha, out, 1.0 - 0.18 * alpha, 0, out)
        tw = cv2.getTextSize(alert_msg, FN, 0.54, 1)[0][0]
        bx3 = max(12, min(px // 2 - tw // 2, px - tw - 16))
        mid = fh // 2
        cv2.rectangle(
            out, (bx3 - 8, mid - 20), (bx3 + tw + 8, mid + 8), (14, 14, 38), -1
        )
        cv2.rectangle(out, (bx3 - 8, mid - 20), (bx3 + tw + 8, mid + 8), P["red"], 1)
        col_fade = tuple(int(c * alpha) for c in P["red"])
        _text(out, alert_msg, bx3, mid + 2, col_fade, 0.54)

    # ── BORDER ───────────────────────────────────────────────────
    border = max_risk(r.live_risk, r.sess_risk)
    bc = RISK_COL[border] if border != Risk.LOW else (26, 26, 38)
    cv2.rectangle(out, (0, 0), (fw - 1, fh - 1), bc, 2)

    # ── HOTKEY BAR ───────────────────────────────────────────────
    cv2.rectangle(out, (0, fh - 20), (fw, fh), P["bg"], -1)
    _dim(out, "Q quit    R reset    S save    C recalibrate", PAD, fh - 6, None, 0.34)

    return out


# ════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════
def _print_summary(sess: Session) -> None:
    el = int(sess.elapsed)
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  INTERVIEW INTEGRITY REPORT")
    print(sep)
    print(f"  Candidate   : {sess.candidate}")
    print(f"  Session ID  : {sess.session_id}")
    print(f"  Started     : {sess.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration    : {el // 60:02d}:{el % 60:02d}")
    print(f"  Risk Level  : {sess.risk.value}")
    print(f"  Score       : {sess.score:.1f}")
    print(f"  Center Gaze : {sess.center_pct:.1f}%  (healthy ≥ 65%)")
    print(f"  Flags       : {len(sess.flags)}")
    if sess.flags:
        print()
        print(f"  {'TIME':8}  {'KIND':20}  {'DUR':5}  {'PTS':4}  DETAIL")
        print(f"  {'-'*8}  {'-'*20}  {'-'*5}  {'-'*4}  {'-'*30}")
        for fl in sess.flags:
            ts = datetime.fromtimestamp(fl.timestamp).strftime("%H:%M:%S")
            print(
                f"  {ts}  {fl.kind:20}  {fl.duration:4.1f}s  "
                f"{fl.score:4.1f}  {fl.detail}"
            )
    print(sep + "\n")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def run(
    source: int | str = 0,
    candidate: str = "Candidate",
    out_dir: str = "reports",
    cfg: Config = None,
    headless: bool = False,
) -> None:
    cfg = cfg or Config()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    delay = max(1, int(1000 / cfg.fps_cap))
    prev_t = time.time()
    alert_ts = 0.0
    alert_msg = ""
    fail_count = 0
    MAX_FAILS = 10

    log.info(
        "Session started | candidate=%s | source=%s | headless=%s",
        candidate,
        source,
        headless,
    )

    with Detector(cfg, candidate) as det:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count >= MAX_FAILS:
                        log.error(
                            "Camera/stream failed %d consecutive reads — exiting.",
                            MAX_FAILS,
                        )
                        break
                    log.warning(
                        "Frame read failed (%d/%d) — retrying.", fail_count, MAX_FAILS
                    )
                    time.sleep(0.05)
                    continue
                fail_count = 0

                frame = cv2.flip(frame, 1)
                r = det.process(frame)
                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now

                # Capture alert message for timed hold display
                if r.new_flags:
                    alert_ts = now
                    alert_msg = r.new_flags[-1].detail

                if not headless:
                    vis = render(frame, r, det.session, fps, alert_ts, alert_msg, cfg)
                    cv2.imshow("Interview Integrity Monitor", vis)
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        det.session.reset(candidate)
                        alert_msg = ""
                        log.info("Session reset.")
                    elif key == ord("s"):
                        path = det.session.save_report(out_dir)
                        log.info("Report saved: %s", path)
                    elif key == ord("c"):
                        det.recalibrate()

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        except Exception:
            log.critical("Unexpected error:\n%s", traceback.format_exc())
        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()

    _print_summary(det.session)
    saved = det.session.save_report(out_dir)
    log.info("Final report: %s", saved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interview Integrity Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", default=0, help="Camera index or video file path (default: 0)"
    )
    parser.add_argument(
        "--name", default="Candidate", help="Candidate name for the report"
    )
    parser.add_argument(
        "--out", default="reports", help="Report output directory (default: ./reports)"
    )
    parser.add_argument("--config", default=None, help="Path to JSON config file")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (background service mode)",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    _setup_logging(args.loglevel)

    cfg = Config.from_json(args.config) if args.config else Config()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run(
        source=source,
        candidate=args.name,
        out_dir=args.out,
        cfg=cfg,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
