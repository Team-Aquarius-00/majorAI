"""
Audio Detection & Call Recording Module
========================================
Captures microphone + system audio (calls, browser, music, etc.)

Platform support
────────────────
  Windows  →  PyAudioWPatch WASAPI loopback  (the ONLY reliable fix for Errno -9999)
  macOS    →  BlackHole / Soundflower virtual device
  Linux    →  PulseAudio *.monitor source

Install
───────
  Windows (system audio capture):
      pip uninstall pyaudio
      pip install PyAudioWPatch numpy sounddevice

  macOS / Linux (mic only, or with virtual device):
      pip install pyaudio numpy sounddevice

  PyAudioWPatch is a drop-in pyaudio replacement that patches PortAudio
  with WASAPI loopback support. It is the only cross-driver solution that
  works without Stereo Mix and without Errno -9999.

  sounddevice's WasapiSettings does NOT support loopback — that feature was
  never added to the upstream sounddevice package (as of v0.5.x).

Usage (library)
───────────────
    from audio_detection import record_call_audio, record_on_voice, CONFIG

    CONFIG.vad_threshold = 800
    record_call_audio(duration=30, output_file="my_call.wav")
    record_on_voice(output_dir="segments/")

Usage (CLI)
───────────
    python audio_detection.py devices
    python audio_detection.py record -d 60
    python audio_detection.py vad

Version: 4.0.0
"""

from __future__ import annotations

import io
import logging
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sounddevice as sd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audio_detection")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AudioConfig:
    """Single source of truth for every tunable parameter."""

    sample_rate: int = 48_000  # Hz — VoIP standard
    channels: int = 2  # stereo captures both call sides
    dtype: str = "int16"  # 16-bit PCM
    blocksize: int = 1_024  # frames per callback
    vad_threshold: int = 1_200  # RMS amplitude — raise in noisy rooms
    silence_timeout: float = 3.0  # seconds of silence before segment ends
    progress_every: int = 4  # print progress every N seconds


CONFIG = AudioConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Platform helpers
# ──────────────────────────────────────────────────────────────────────────────
def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


# ──────────────────────────────────────────────────────────────────────────────
# Device discovery
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    hostapi_name: str = ""
    is_loopback: bool = False

    def __str__(self) -> str:
        tag = "  ← LOOPBACK / SYSTEM AUDIO" if self.is_loopback else ""
        return (
            f"  [{self.index:>3}]  {self.name:<50}"
            f"ch:{self.max_input_channels}  [{self.hostapi_name}]{tag}"
        )


_LOOPBACK_KEYWORDS = (
    "stereo mix",
    "loopback",
    "monitor",
    "what u hear",
    "what you hear",
    "blackhole",
    "soundflower",
    "virtual audio",
    "cable output",
    "vb-audio",
    "voicemeeter",
)


def list_input_devices() -> list[AudioDevice]:
    """Return every device that has at least one input channel."""
    apis = {i: sd.query_hostapis(i)["name"] for i in range(len(sd.query_hostapis()))}
    devices: list[AudioDevice] = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"]
        api = apis.get(dev["hostapi"], "")
        is_lb = any(k in name.lower() for k in _LOOPBACK_KEYWORDS)
        devices.append(
            AudioDevice(
                index=idx,
                name=name,
                max_input_channels=int(dev["max_input_channels"]),
                hostapi_name=api,
                is_loopback=is_lb,
            )
        )
    return devices


def print_devices() -> None:
    """Pretty-print every input device."""
    devices = list_input_devices()
    sep = "─" * 80
    print(f"\n{sep}\n  AUDIO INPUT DEVICES\n{sep}")
    for d in devices:
        print(d)
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# System-audio stream  (WASAPI loopback → named loopback fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _open_wasapi_loopback(
    frames: list[bytes],
    frames_lock: threading.Lock,
) -> Optional[Any]:
    """
    Open a WASAPI loopback stream via PyAudioWPatch.

    PyAudioWPatch is a patched pyaudio build that adds genuine WASAPI loopback
    support via get_loopback_device_info_generator() and p.open(as_loopback=True).
    This is the only solution that works reliably on Windows without Stereo Mix.

    sounddevice's WasapiSettings does NOT have a loopback parameter — that
    feature was never shipped in any version of sounddevice (confirmed: v0.5.x).

    Returns (pyaudio_instance, stream) tuple or None.
    """
    if not _is_windows():
        return None

    try:
        import pyaudiowpatch as pyaudio  # PyAudioWPatch drop-in
    except ImportError:
        log.warning(
            "PyAudioWPatch not installed — cannot capture system audio.\n"
            "  Fix:  pip uninstall pyaudio && pip install PyAudioWPatch"
        )
        return None

    p = pyaudio.PyAudio()

    # Find the default WASAPI loopback device (what's currently playing)
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        log.warning("WASAPI not available on this system.")
        p.terminate()
        return None

    default_speakers_idx = wasapi_info["defaultOutputDevice"]
    if default_speakers_idx < 0:
        log.warning("No default output device found.")
        p.terminate()
        return None

    default_speakers = p.get_device_info_by_index(default_speakers_idx)

    # Walk loopback devices to find the one matching the default output
    loopback_device = None
    for lb in p.get_loopback_device_info_generator():
        if default_speakers["name"] in lb["name"]:
            loopback_device = lb
            break

    if loopback_device is None:
        log.warning(
            "No WASAPI loopback device found for '%s'.", default_speakers["name"]
        )
        p.terminate()
        return None

    channels = min(CONFIG.channels, int(loopback_device["maxInputChannels"]))
    samplerate = int(loopback_device["defaultSampleRate"])

    def _loopback_cb(in_data, frame_count, time_info, status):
        with frames_lock:
            frames.append(in_data)
        return (None, pyaudio.paContinue)

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=samplerate,
            frames_per_buffer=CONFIG.blocksize,
            input=True,
            input_device_index=int(loopback_device["index"]),
            stream_callback=_loopback_cb,
        )
        log.info(
            "✓ WASAPI loopback stream ready: '%s'  (ch: %d @ %d Hz)",
            loopback_device["name"],
            channels,
            samplerate,
        )
        return (p, stream)
    except Exception as exc:
        log.warning("WASAPI loopback open failed: %s", exc)
        p.terminate()
        return None


def _open_named_loopback(
    audio_q: "queue.Queue[Optional[np.ndarray]]",
) -> Optional[sd.InputStream]:
    """
    Fallback: scan for Stereo Mix / BlackHole / PulseAudio monitor by name.
    Uses sounddevice (works fine for named devices — only WASAPI loopback
    requires PyAudioWPatch).
    """
    candidates = [d for d in list_input_devices() if d.is_loopback]

    for dev in candidates:
        ch = min(CONFIG.channels, dev.max_input_channels)
        for rate in (CONFIG.sample_rate, 44_100, 16_000):
            try:

                def _cb(indata: np.ndarray, frames, t, status, _n=dev.name) -> None:
                    if status:
                        log.debug("Loopback '%s': %s", _n, status)
                    audio_q.put(indata.copy())

                stream = sd.InputStream(
                    device=dev.index,
                    samplerate=rate,
                    channels=ch,
                    dtype=CONFIG.dtype,
                    blocksize=CONFIG.blocksize,
                    callback=_cb,
                )
                log.info("✓ Named loopback: '%s' @ %d Hz (ch: %d)", dev.name, rate, ch)
                return stream
            except Exception:
                continue

    return None


def open_system_audio_stream(
    audio_q: "queue.Queue[Optional[np.ndarray]]",
) -> tuple[Optional[Any], Optional[Any]]:
    """
    Best-effort system-audio capture.

    Returns (pyaudio_handle, stream) where pyaudio_handle is only set for
    PyAudioWPatch streams (needs separate cleanup), or (None, sd.InputStream)
    for named loopback via sounddevice, or (None, None) if unavailable.
    """
    # Windows: try PyAudioWPatch WASAPI loopback first
    if _is_windows():
        # PyAudioWPatch streams write directly into a shared list —
        # we bridge them into audio_q via a drain thread
        wpatch_frames: list[bytes] = []
        wpatch_lock = threading.Lock()
        result = _open_wasapi_loopback(wpatch_frames, wpatch_lock)
        if result is not None:
            p_handle, lb_stream = result

            def _drain():
                """Bridge PyAudioWPatch callback frames into audio_q."""
                while True:
                    time.sleep(0.02)
                    with wpatch_lock:
                        batch = wpatch_frames.copy()
                        wpatch_frames.clear()
                    for raw in batch:
                        arr = np.frombuffer(raw, dtype=np.int16)
                        # Reshape to (frames, channels) matching CONFIG
                        try:
                            arr = arr.reshape(-1, CONFIG.channels)
                        except ValueError:
                            arr = arr.reshape(-1, 1)
                        audio_q.put(arr)
                    # Sentinel check — stream stopped
                    if not lb_stream.is_active():
                        break

            threading.Thread(target=_drain, daemon=True).start()
            return (p_handle, lb_stream)

    # All platforms: try named loopback device
    sd_stream = _open_named_loopback(audio_q)
    if sd_stream is not None:
        return (None, sd_stream)

    _print_loopback_guide()
    return (None, None)


def _print_loopback_guide() -> None:
    print()
    log.warning("System audio capture unavailable — microphone only.")
    if _is_windows():
        print(
            "  ┌─ WINDOWS ─────────────────────────────────────────────────────────┐\n"
            "  │  Install PyAudioWPatch for WASAPI loopback (no Stereo Mix needed):│\n"
            "  │                                                                   │\n"
            "  │    pip uninstall pyaudio                                          │\n"
            "  │    pip install PyAudioWPatch                                      │\n"
            "  │                                                                   │\n"
            "  │  Also ensure a default Playback device is set:                    │\n"
            "  │    Settings → System → Sound → Output → choose a device           │\n"
            "  └───────────────────────────────────────────────────────────────────┘"
        )
    elif _is_macos():
        print(
            "  ┌─ macOS ────────────────────────────────────────────────────────────┐\n"
            "  │  Install BlackHole: https://existential.audio/blackhole/           │\n"
            "  │  Then set BlackHole as your Mac's audio output device.             │\n"
            "  └────────────────────────────────────────────────────────────────────┘"
        )
    elif _is_linux():
        print(
            "  ┌─ Linux ────────────────────────────────────────────────────────────┐\n"
            "  │  Run:  pactl list sources short                                    │\n"
            "  │  Look for a source ending in '.monitor' — it appears automatically.│\n"
            "  └────────────────────────────────────────────────────────────────────┘"
        )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# WAV helpers
# ──────────────────────────────────────────────────────────────────────────────
def _bytes_per_sample() -> int:
    return int(CONFIG.dtype.replace("int", "").replace("float", "")) // 8


def arrays_to_wav_bytes(chunks: list[np.ndarray]) -> bytes:
    raw = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.zeros((0, CONFIG.channels), dtype=CONFIG.dtype)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CONFIG.channels)
        wf.setsampwidth(_bytes_per_sample())
        wf.setframerate(CONFIG.sample_rate)
        wf.writeframes(raw.tobytes())
    return buf.getvalue()


def save_wav(chunks: list[np.ndarray], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(arrays_to_wav_bytes(chunks))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stop signal
# ──────────────────────────────────────────────────────────────────────────────
class StopSignal:
    """Thread-safe stop flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    def start_keyboard_listener(self) -> None:
        def _listen():
            try:
                import msvcrt

                while not self.is_set:
                    if msvcrt.kbhit() and msvcrt.getch().lower() == b"q":
                        log.info("Stop requested (q).")
                        self.request_stop()
                    time.sleep(0.05)
            except ImportError:
                pass

        threading.Thread(target=_listen, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# Voice-activity detection
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AudioSegment:
    detected: bool
    chunks: list[np.ndarray] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return sum(c.shape[0] for c in self.chunks) / CONFIG.sample_rate

    def to_wav_bytes(self) -> bytes:
        return arrays_to_wav_bytes(self.chunks)

    def save(self, path: str | Path) -> Path:
        return save_wav(self.chunks, path)


def audio_detection(
    audio_q: "queue.Queue[Optional[np.ndarray]]",
) -> AudioSegment:
    """
    Block until one voice segment is fully captured (sound → silence).

    Returns AudioSegment(detected=True) with audio, or
    AudioSegment(detected=False) on shutdown/interrupt.
    """
    active = False
    last_time = 0.0
    chunks: list[np.ndarray] = []

    while True:
        try:
            chunk = audio_q.get(timeout=0.5)
            if chunk is None:
                break  # shutdown sentinel

            amplitude = int(np.max(np.abs(chunk)))

            if amplitude > CONFIG.vad_threshold:
                if not active:
                    active = True
                last_time = time.monotonic()
                chunks.append(chunk)
            elif active:
                chunks.append(chunk)
                if (time.monotonic() - last_time) > CONFIG.silence_timeout:
                    return AudioSegment(detected=True, chunks=chunks)

        except queue.Empty:
            if active and (time.monotonic() - last_time) > CONFIG.silence_timeout:
                return AudioSegment(detected=True, chunks=chunks)
        except KeyboardInterrupt:
            break

    return AudioSegment(detected=False)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def record_call_audio(
    duration: Optional[float] = None,
    output_file: str | Path = "call_recording.wav",
    stop_signal: Optional[StopSignal] = None,
) -> bool:
    """
    Record microphone + system audio to a single WAV file.

    Args:
        duration:    Max seconds. None = unlimited.
        output_file: Destination .wav path.
        stop_signal: Optional external stop signal.

    Returns:
        True on success, False otherwise.
    """
    if stop_signal is None:
        stop_signal = StopSignal()
        stop_signal.start_keyboard_listener()

    mic_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=512)
    sys_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=512)
    mic_chunks: list[np.ndarray] = []
    sys_chunks: list[np.ndarray] = []

    def _mic_cb(indata: np.ndarray, frames, t, status) -> None:
        if status:
            log.debug("Mic: %s", status)
        mic_q.put(indata.copy())

    try:
        mic_stream = sd.InputStream(
            channels=CONFIG.channels,
            samplerate=CONFIG.sample_rate,
            dtype=CONFIG.dtype,
            blocksize=CONFIG.blocksize,
            callback=_mic_cb,
        )
    except Exception as exc:
        log.error("Cannot open microphone: %s", exc)
        return False

    sys_p_handle, sys_stream = open_system_audio_stream(sys_q)
    mode = "Microphone + System Audio" if sys_stream else "Microphone only"
    _print_recording_header(output_file, mode, duration)

    start = time.monotonic()
    last_print = start

    try:
        mic_stream.start()
        # sd.InputStream needs .start(); PyAudioWPatch stream is callback-driven
        if sys_stream and hasattr(sys_stream, "start_stream"):
            sys_stream.start_stream()
        elif sys_stream and hasattr(sys_stream, "start"):
            try:
                sys_stream.start()
            except Exception:
                pass

        while not stop_signal.is_set:
            if duration and (time.monotonic() - start) >= duration:
                break

            for q, store in ((mic_q, mic_chunks), (sys_q, sys_chunks)):
                while True:
                    try:
                        c = q.get_nowait()
                        if c is not None:
                            store.append(c)
                    except queue.Empty:
                        break

            now = time.monotonic()
            if now - last_print >= CONFIG.progress_every:
                print(f"  ⏺  {now - start:>7.1f}s recorded …", end="\r", flush=True)
                last_print = now

            time.sleep(0.02)

    except KeyboardInterrupt:
        log.info("Stopped by Ctrl+C.")
    finally:
        print()
        mic_stream.stop()
        mic_stream.close()
        if sys_stream:
            try:
                sys_stream.stop_stream()
                sys_stream.close()
            except AttributeError:
                try:
                    sys_stream.stop()
                    sys_stream.close()
                except Exception:
                    pass
        if sys_p_handle:
            try:
                sys_p_handle.terminate()
            except Exception:
                pass

    all_chunks = mic_chunks + sys_chunks
    if not all_chunks:
        log.warning("No audio captured.")
        return False

    out_path = save_wav(all_chunks, output_file)
    dur_s = sum(c.shape[0] for c in mic_chunks) / CONFIG.sample_rate
    size_kb = out_path.stat().st_size / 1_024

    print()
    log.info("✓ Saved: %s", out_path)
    log.info("  Mode: %s | Duration: %.1f s | Size: %.1f KB", mode, dur_s, size_kb)
    print()
    return True


def record_on_voice(
    output_dir: str | Path = "recordings",
    max_segments: int = 0,
    stop_signal: Optional[StopSignal] = None,
) -> list[Path]:
    """
    VAD-triggered recording: saves each utterance as its own WAV file.

    Args:
        output_dir:   Output folder.
        max_segments: Stop after N segments (0 = unlimited).
        stop_signal:  Optional external stop signal.
    """
    if stop_signal is None:
        stop_signal = StopSignal()
        stop_signal.start_keyboard_listener()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=512)

    def _cb(indata: np.ndarray, frames, t, status) -> None:
        if status:
            log.debug("VAD: %s", status)
        audio_q.put(indata.copy())

    try:
        stream = sd.InputStream(
            channels=CONFIG.channels,
            samplerate=CONFIG.sample_rate,
            dtype=CONFIG.dtype,
            blocksize=CONFIG.blocksize,
            callback=_cb,
        )
    except Exception as exc:
        log.error("Cannot open microphone: %s", exc)
        return []

    # Send sentinel when stop is requested
    def _sentinel():
        while not stop_signal.is_set:
            time.sleep(0.1)
        audio_q.put(None)

    threading.Thread(target=_sentinel, daemon=True).start()

    log.info("🎙  VAD active — listening …  (Ctrl+C or 'q' to stop)")
    saved: list[Path] = []
    count = 0

    try:
        stream.start()
        while not stop_signal.is_set:
            seg = audio_detection(audio_q)
            if not seg.detected:
                break
            count += 1
            fname = out_dir / f"segment_{count:04d}_{int(time.time())}.wav"
            path = seg.save(fname)
            saved.append(path)
            log.info("  Segment %d → %s  (%.2f s)", count, path, seg.duration_seconds)
            if max_segments and count >= max_segments:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

    log.info("Session ended. %d segment(s) saved.", len(saved))
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────
def _print_recording_header(path: str | Path, mode: str, dur: Optional[float]) -> None:
    sep = "─" * 72
    print(f"\n{sep}\n  ⏺  CALL RECORDING — ACTIVE\n{sep}")
    print(f"  Output  : {path}")
    print(f"  Mode    : {mode}")
    print(f"  Duration: {f'{dur:.0f} s' if dur else 'unlimited'}")
    print(f"  Stop    : Ctrl+C  or  'q'\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio Detection & Call Recording v3.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="List all audio input devices")

    p_rec = sub.add_parser("record", help="Record a call to a WAV file")
    p_rec.add_argument("-o", "--output", default="call_recording.wav")
    p_rec.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Max seconds (omit = unlimited)",
    )

    p_vad = sub.add_parser("vad", help="VAD-triggered segment recording")
    p_vad.add_argument("-o", "--output-dir", default="recordings")
    p_vad.add_argument("-n", "--max-segments", type=int, default=0)
    p_vad.add_argument(
        "--threshold",
        type=int,
        default=CONFIG.vad_threshold,
        help="RMS amplitude threshold",
    )

    args = parser.parse_args()

    if args.command == "devices":
        print_devices()
    elif args.command == "record":
        record_call_audio(duration=args.duration, output_file=args.output)
    elif args.command == "vad":
        CONFIG.vad_threshold = args.threshold
        record_on_voice(output_dir=args.output_dir, max_segments=args.max_segments)
