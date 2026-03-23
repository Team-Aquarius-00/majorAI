"""
identity_verifier.py
════════════════════════════════════════════════════════════════════
Candidate Identity Verification Module
for AI Hiring Platform

Features:
  - Initial identity verification before interview starts
  - Periodic re-verification during interview (every N minutes)
  - Strict ArcFace confidence threshold (0.60)
  - Graceful error handling (no face, multiple faces, poor lighting)
  - Verification history log

Usage:
  verifier = IdentityVerifier(reference_image_path="candidate_photo.jpg")

  # At interview start
  granted = verifier.verify_initial()

  # During interview loop (call every frame or on a timer)
  verifier.check_periodic(frame)  # pass current webcam frame

  # Get integrity data for scoring
  report = verifier.get_report()
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("identity_verifier")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
)

# ────────────────────────────────────────────────────────────────────
# Try importing DeepFace — give a clear error if missing
# ────────────────────────────────────────────────────────────────────
try:
    from deepface import DeepFace

    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DEEPFACE_AVAILABLE = False
    log.error("DeepFace not installed. Run: pip install deepface")


# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
@dataclass
class VerifierConfig:
    # ArcFace distance threshold — lower = stricter
    # Default DeepFace threshold is 0.68; we use 0.60 for hiring
    distance_threshold: float = 0.60

    # How often to re-verify during interview (seconds)
    recheck_interval: float = 120.0  # every 2 minutes

    # How many consecutive failed re-verifications before flagging
    max_consecutive_failures: int = 2

    # Temp file path for live captures during re-verification
    temp_capture_path: str = "temp_reverify.jpg"

    # ArcFace model — most accurate for identity verification
    model_name: str = "ArcFace"


# ════════════════════════════════════════════════════════════════════
#  RESULT DATA CLASSES
# ════════════════════════════════════════════════════════════════════
@dataclass
class VerificationResult:
    success: bool
    distance: float = 0.0
    reason: str = ""  # human-readable explanation
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        ts = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return f"[{ts}] {status}  distance={self.distance:.3f}  reason={self.reason}"


@dataclass
class VerifierReport:
    initial_verified: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    consecutive_failures: int
    identity_score: float  # 0.0 – 1.0  (used for your 20% integrity marks)
    history: list[VerificationResult]


# ════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ════════════════════════════════════════════════════════════════════
class IdentityVerifier:
    """
    Handles both initial and periodic identity verification
    using DeepFace ArcFace model.

    Args:
        reference_image_path: Path to the candidate's registered photo.
        cfg: Optional VerifierConfig to override defaults.
    """

    def __init__(
        self,
        reference_image_path: str,
        cfg: Optional[VerifierConfig] = None,
    ) -> None:
        if not _DEEPFACE_AVAILABLE:
            raise RuntimeError("DeepFace is required. Run: pip install deepface")

        self.cfg = cfg or VerifierConfig()
        self.reference_path = reference_image_path

        if not Path(reference_image_path).exists():
            raise FileNotFoundError(
                f"Reference image not found: {reference_image_path}"
            )

        self.initial_verified: bool = False
        self._history: list[VerificationResult] = []
        self._consecutive_failures: int = 0
        self._last_check_time: float = 0.0
        self._total_checks: int = 0

        log.info(
            "IdentityVerifier ready | reference=%s | threshold=%.2f | interval=%.0fs",
            reference_image_path,
            self.cfg.distance_threshold,
            self.cfg.recheck_interval,
        )

    # ── Public API ───────────────────────────────────────────────

    def verify_initial(self) -> bool:
        """
        Capture a live image from webcam and compare with reference.
        Call this ONCE before the interview starts.

        Returns True if identity is confirmed, False otherwise.
        Prints clear messages for the candidate.
        """
        print("\n" + "═" * 55)
        print("  IDENTITY VERIFICATION")
        print("  Please look directly at the camera.")
        print("═" * 55)

        live_path = self._capture_from_webcam(prompt="Press SPACE to capture")
        if live_path is None:
            result = VerificationResult(
                success=False,
                reason="Camera capture cancelled or failed",
            )
            self._record(result)
            return False

        result = self._compare(live_path)
        self._record(result)

        if result.success:
            self.initial_verified = True
            self._last_check_time = time.time()
            print("\n  ✅ Identity verified. You may begin the interview.\n")
        else:
            print(f"\n  ❌ Verification failed: {result.reason}")
            print("  Please ensure good lighting and remove glasses if worn.\n")

        return result.success

    def check_periodic(self, frame: np.ndarray) -> Optional[VerificationResult]:
        """
        Call this on every frame (or in your main interview loop).
        It self-throttles — only runs a check every `recheck_interval` seconds.

        Args:
            frame: Current BGR webcam frame (numpy array from cv2).

        Returns:
            VerificationResult if a check was performed, None if skipped.
        """
        if not self.initial_verified:
            return None

        now = time.time()
        if now - self._last_check_time < self.cfg.recheck_interval:
            return None  # Not time yet

        self._last_check_time = now
        log.info("Periodic re-verification triggered.")

        # Save current frame to temp file for DeepFace
        temp_path = self.cfg.temp_capture_path
        saved = cv2.imwrite(temp_path, frame)
        if not saved:
            log.warning("Could not save temp frame for re-verification.")
            return None

        result = self._compare(temp_path)
        self._record(result)

        if result.success:
            log.info("Re-verification PASSED (distance=%.3f)", result.distance)
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            log.warning(
                "Re-verification FAILED (%d/%d consecutive) — %s",
                self._consecutive_failures,
                self.cfg.max_consecutive_failures,
                result.reason,
            )

            if self._consecutive_failures >= self.cfg.max_consecutive_failures:
                log.warning(
                    "⚠️  IDENTITY ALERT: %d consecutive failures — "
                    "possible candidate substitution!",
                    self._consecutive_failures,
                )

        return result

    def is_identity_alert(self) -> bool:
        """
        Returns True if consecutive failures hit the threshold.
        Use this in your main loop to flag a serious integrity issue.
        """
        return self._consecutive_failures >= self.cfg.max_consecutive_failures

    def seconds_until_next_check(self) -> float:
        """Useful for displaying a countdown in your UI."""
        elapsed = time.time() - self._last_check_time
        return max(0.0, self.cfg.recheck_interval - elapsed)

    def get_report(self) -> VerifierReport:
        """
        Returns a summary report for use in your integrity scoring system.
        identity_score is 1.0 if all checks passed, lower for failures.
        """
        passed = sum(1 for r in self._history if r.success)
        failed = len(self._history) - passed
        total = len(self._history)

        if total == 0:
            identity_score = 0.0
        else:
            identity_score = passed / total

        return VerifierReport(
            initial_verified=self.initial_verified,
            total_checks=total,
            passed_checks=passed,
            failed_checks=failed,
            consecutive_failures=self._consecutive_failures,
            identity_score=round(identity_score, 3),
            history=list(self._history),
        )

    def print_report(self) -> None:
        """Print a human-readable summary to console."""
        r = self.get_report()
        sep = "─" * 55
        print(f"\n{sep}")
        print("  IDENTITY VERIFICATION REPORT")
        print(sep)
        print(f"  Initial Verified : {'Yes' if r.initial_verified else 'No'}")
        print(f"  Total Checks     : {r.total_checks}")
        print(f"  Passed           : {r.passed_checks}")
        print(f"  Failed           : {r.failed_checks}")
        print(f"  Identity Score   : {r.identity_score:.1%}")
        print(f"\n  Check History:")
        for entry in r.history:
            print(f"    {entry}")
        print(sep + "\n")

    # ── Private helpers ──────────────────────────────────────────

    def _compare(self, live_image_path: str) -> VerificationResult:
        """
        Run DeepFace ArcFace comparison between reference and live image.
        Handles all error cases gracefully.
        """
        try:
            result = DeepFace.verify(
                img1_path=self.reference_path,
                img2_path=live_image_path,
                model_name=self.cfg.model_name,
                enforce_detection=True,  # raises if no face found
                distance_metric="cosine",
            )

            distance = float(result["distance"])

            # Apply our stricter threshold (0.60 instead of default 0.68)
            verified = distance < self.cfg.distance_threshold

            reason = (
                "Identity confirmed"
                if verified
                else f"Face mismatch (distance {distance:.3f} > threshold {self.cfg.distance_threshold})"
            )

            return VerificationResult(
                success=verified,
                distance=distance,
                reason=reason,
            )

        except ValueError as e:
            # DeepFace raises ValueError when no face is detected
            err = str(e).lower()
            if "face" in err or "detect" in err:
                reason = "No face detected — ensure good lighting and face the camera"
            else:
                reason = f"Detection error: {e}"
            log.warning("Face detection issue: %s", e)
            return VerificationResult(success=False, reason=reason)

        except Exception as e:
            # Catch-all for unexpected errors (model load, file issues, etc.)
            log.error("Unexpected verification error: %s", e)
            return VerificationResult(
                success=False,
                reason=f"System error during verification: {type(e).__name__}",
            )

    def _capture_from_webcam(
        self, prompt: str = "Press SPACE to capture"
    ) -> Optional[str]:
        """
        Opens webcam, shows live feed, captures on SPACE.
        Returns path to saved image or None if cancelled/failed.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Cannot access webcam.")
            return None

        save_path = self.cfg.temp_capture_path
        captured = None

        print(f"  {prompt} | ESC to cancel")

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read webcam frame.")
                break

            # Mirror for natural feel
            frame = cv2.flip(frame, 1)

            # Overlay instruction
            cv2.putText(
                frame,
                "SPACE = Capture | ESC = Cancel",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Identity Verification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                cv2.imwrite(save_path, frame)
                captured = save_path
                log.info("Live capture saved: %s", save_path)
                break
            elif key == 27:  # ESC
                log.info("Capture cancelled by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured

    def _record(self, result: VerificationResult) -> None:
        self._history.append(result)
        self._total_checks += 1


# ════════════════════════════════════════════════════════════════════
#  INTEGRATION EXAMPLE — how to use in your interview loop
# ════════════════════════════════════════════════════════════════════
def example_interview_loop(reference_image: str) -> None:
    """
    Example showing how to integrate IdentityVerifier
    into your main interview loop.
    """
    verifier = IdentityVerifier(
        reference_image_path=reference_image,
        cfg=VerifierConfig(
            distance_threshold=0.60,
            recheck_interval=120.0,  # re-verify every 2 minutes
            max_consecutive_failures=2,
        ),
    )

    # ── Step 1: Initial verification before interview ────────────
    if not verifier.verify_initial():
        print("Access denied. Cannot start interview.")
        return

    # ── Step 2: Interview loop ────────────────────────────────────
    cap = cv2.VideoCapture(0)
    print("Interview started. Press Q to end.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Periodic re-verification (self-throttled) ─────────────
        check_result = verifier.check_periodic(frame)

        if check_result is not None:
            status_text = "✅ ID OK" if check_result.success else "❌ ID FAIL"
            col = (0, 220, 80) if check_result.success else (40, 40, 220)
            cv2.putText(
                frame,
                status_text,
                (10, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                col,
                2,
                cv2.LINE_AA,
            )

        # ── Identity alert — flag for scoring system ──────────────
        if verifier.is_identity_alert():
            cv2.putText(
                frame,
                "⚠ IDENTITY ALERT",
                (10, 72),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (40, 40, 220),
                2,
                cv2.LINE_AA,
            )

        # Next check countdown
        secs = int(verifier.seconds_until_next_check())
        cv2.putText(
            frame,
            f"Next ID check: {secs}s",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 120),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Interview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Step 3: Final report ──────────────────────────────────────
    verifier.print_report()

    report = verifier.get_report()
    integrity_marks = round(report.identity_score * 20, 1)  # out of 20
    print(f"  Identity Integrity Marks: {integrity_marks} / 20")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    ref = sys.argv[1] if len(sys.argv) > 1 else "user_photo.jpg"
    example_interview_loop(ref)
