"""
Audio Detection & Call Recording Module
========================================
Monitors microphone and system audio for speaking/sound events.
Records all audio including calls happening on the system.

Platform Support:
    - Windows: WASAPI loopback (no Stereo Mix required)
    - macOS:   BlackHole / Soundflower virtual device
    - Linux:   PulseAudio monitor source

Requirements:
    pip install pyaudio numpy sounddevice

Author:  Senior ML/Audio Engineer
Version: 2.0.0
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if absent
# ---------------------------------------------------------------------------
try:
    import pyaudio

    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False

try:
    import sounddevice as sd

    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audio_detection")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """All tuneable parameters in one place."""

    # Recording quality
    sample_rate: int = 48_000  # Hz — 48 kHz is standard for VoIP/calls
    channels: int = 1  # 1 = mono (sufficient for calls); 2 = stereo
    chunk_size: int = 2_048  # Frames per read — balance latency vs CPU
    bit_depth: int = 16  # 16-bit PCM (paInt16)

    # Voice-activity detection
    vad_threshold: int = 1_500  # RMS amplitude — raise in noisy environments
    silence_timeout: float = 3.0  # Seconds of silence before segment ends

    # Logging / progress
    progress_interval: int = 100  # Print progress every N chunks


# Singleton config — override before calling any function
CONFIG = AudioConfig()


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    is_loopback: bool = False
    is_wasapi: bool = False

    def __str__(self) -> str:
        tags = []
        if self.is_loopback:
            tags.append("LOOPBACK")
        if self.is_wasapi:
            tags.append("WASAPI")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        return (
            f"  [{self.index:>3}] {self.name}  (ch: {self.max_input_channels}){tag_str}"
        )


def list_input_devices() -> list[AudioDevice]:
    """
    Return all available audio *input* devices.
    Prefers sounddevice (richer metadata) over pyaudio.
    """
    devices: list[AudioDevice] = []

    if _SOUNDDEVICE_AVAILABLE:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                name = dev["name"]
                is_lb = _looks_like_loopback(name)
                devices.append(
                    AudioDevice(
                        index=idx,
                        name=name,
                        max_input_channels=int(dev["max_input_channels"]),
                        is_loopback=is_lb,
                    )
                )
    elif _PYAUDIO_AVAILABLE:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                is_lb = _looks_like_loopback(name)
                devices.append(
                    AudioDevice(
                        index=i,
                        name=name,
                        max_input_channels=int(info["maxInputChannels"]),
                        is_loopback=is_lb,
                    )
                )
        p.terminate()

    return devices


def _looks_like_loopback(name: str) -> bool:
    keywords = (
        "stereo mix",
        "loopback",
        "monitor",
        "what u hear",
        "what you hear",
        "blackhole",
        "soundflower",
        "virtual",
    )
    return any(k in name.lower() for k in keywords)


def print_devices() -> None:
    """Pretty-print all input devices to stdout."""
    devices = list_input_devices()
    sep = "─" * 72
    print(f"\n{sep}")
    print("  AUDIO INPUT DEVICES")
    print(sep)
    for d in devices:
        print(d)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# System-audio stream factory
# ---------------------------------------------------------------------------
def _open_wasapi_loopback(p: "pyaudio.PyAudio") -> Optional["pyaudio.Stream"]:
    """
    Open a WASAPI loopback stream on Windows — the *correct* way to capture
    system audio without needing Stereo Mix.

    The key is ``as_loopback=True`` in the host-API-specific stream info.
    Falls back to any loopback-named device if WASAPI is unavailable.
    """
    if not _is_windows():
        return None

    try:
        # Find the WASAPI host API index
        wasapi_api_index: Optional[int] = None
        for i in range(p.get_host_api_count()):
            api_info = p.get_host_api_info_by_index(i)
            if api_info["type"] == pyaudio.paWASAPI:
                wasapi_api_index = i
                break

        if wasapi_api_index is None:
            log.warning("WASAPI host API not found — cannot open loopback stream.")
            return None

        # Find the default output device for that host API
        api_info = p.get_host_api_info_by_index(wasapi_api_index)
        default_out_idx: int = int(api_info["defaultOutputDevice"])

        log.info("Attempting WASAPI loopback on device index %d …", default_out_idx)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=CONFIG.channels,
            rate=CONFIG.sample_rate,
            input=True,
            input_device_index=default_out_idx,
            frames_per_buffer=CONFIG.chunk_size,
            input_host_api_specific_stream_info=(
                pyaudio.PaMacCoreStreamInfo()
                if False  # placeholder — overridden below via as_loopback
                else None
            ),
            # WASAPI loopback flag ─ this is what fixes Errno -9999
            as_loopback=True,  # type: ignore[call-arg]
        )
        log.info("✓ WASAPI loopback stream opened (device %d).", default_out_idx)
        return stream

    except AttributeError:
        # Some pyaudio builds don't expose as_loopback — try PyAudioWPatch
        log.warning(
            "as_loopback not available. Install PyAudioWPatch for WASAPI loopback:\n"
            "  pip install PyAudioWPatch"
        )
        return None
    except Exception as exc:
        log.warning("WASAPI loopback failed: %s", exc)
        return None


def _open_loopback_fallback(p: "pyaudio.PyAudio") -> Optional["pyaudio.Stream"]:
    """
    Fallback: scan for any device named 'Stereo Mix' / 'Loopback' / etc.
    Uses a lower sample-rate probe to avoid Errno -9999 on exotic hardware.
    """
    devices = list_input_devices()
    loopback_devs = [d for d in devices if d.is_loopback]

    for dev in loopback_devs:
        # Probe with several sample rates — some loopback devices are picky
        for rate in (CONFIG.sample_rate, 44_100, 16_000):
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=min(CONFIG.channels, dev.max_input_channels),
                    rate=rate,
                    input=True,
                    input_device_index=dev.index,
                    frames_per_buffer=CONFIG.chunk_size,
                )
                log.info("✓ Fallback loopback opened: '%s' @ %d Hz", dev.name, rate)
                return stream
            except Exception:
                continue

    return None


def open_system_audio_stream(
    p: "pyaudio.PyAudio",
) -> Optional["pyaudio.Stream"]:
    """
    Best-effort system-audio capture.

    Strategy (Windows):
        1. WASAPI loopback  ← fixes Errno -9999, no Stereo Mix needed
        2. Named loopback device (Stereo Mix, etc.)

    Strategy (macOS / Linux):
        1. Named loopback device (BlackHole, PulseAudio monitor, etc.)

    Returns the open stream or None if unavailable.
    """
    stream: Optional["pyaudio.Stream"] = None

    if _is_windows():
        stream = _open_wasapi_loopback(p)

    if stream is None:
        stream = _open_loopback_fallback(p)

    if stream is None:
        _print_loopback_setup_guide()

    return stream


def _print_loopback_setup_guide() -> None:
    """Print OS-specific instructions for enabling system audio capture."""
    print()
    log.warning("System audio capture not available.")
    if _is_windows():
        print(
            "  ┌─ Windows — Enable system audio capture ────────────────────────┐\n"
            "  │  Option A (recommended — no Stereo Mix needed):                │\n"
            "  │    pip install PyAudioWPatch                                   │\n"
            "  │    This replaces pyaudio and enables WASAPI loopback.          │\n"
            "  │                                                                │\n"
            "  │  Option B — Enable Stereo Mix manually:                        │\n"
            "  │    1. Right-click speaker icon → Sound settings                │\n"
            "  │    2. More sound settings → Recording tab                      │\n"
            "  │    3. Right-click empty area → Show Disabled Devices           │\n"
            "  │    4. Right-click 'Stereo Mix' → Enable                        │\n"
            "  └────────────────────────────────────────────────────────────────┘"
        )
    elif _is_macos():
        print(
            "  ┌─ macOS — Enable system audio capture ──────────────────────────┐\n"
            "  │    Install BlackHole (free virtual audio device):              │\n"
            "  │    https://existential.audio/blackhole/                        │\n"
            "  │    Then select 'BlackHole 2ch' as your audio output.           │\n"
            "  └────────────────────────────────────────────────────────────────┘"
        )
    elif _is_linux():
        print(
            "  ┌─ Linux — Enable system audio capture ──────────────────────────┐\n"
            "  │    PulseAudio monitor source should appear automatically.      │\n"
            "  │    Run:  pactl list sources short                              │\n"
            "  │    Look for a '.monitor' source (e.g. alsa_output.*.monitor)   │\n"
            "  └────────────────────────────────────────────────────────────────┘"
        )
    print()


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def frames_to_wav_bytes(
    frames: list[bytes],
    channels: int = CONFIG.channels,
    sampwidth: int = 2,
    framerate: int = CONFIG.sample_rate,
) -> bytes:
    """Pack raw PCM frames into a valid in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def save_wav(
    frames: list[bytes],
    path: str | Path,
    channels: int = CONFIG.channels,
    framerate: int = CONFIG.sample_rate,
) -> Path:
    """Write frames to a .wav file and return the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(frames_to_wav_bytes(frames, channels=channels, framerate=framerate))
    return out


# ---------------------------------------------------------------------------
# Voice-Activity Detection (VAD) segment recorder
# ---------------------------------------------------------------------------
@dataclass
class AudioSegment:
    """Result of a single voice-detected audio segment."""

    detected: bool
    frames: list[bytes] = field(default_factory=list)

    @property
    def raw_bytes(self) -> bytes:
        return b"".join(self.frames)

    @property
    def duration_seconds(self) -> float:
        return len(self.raw_bytes) / 2 / CONFIG.sample_rate

    def to_wav_bytes(self) -> bytes:
        return frames_to_wav_bytes(self.frames)

    def save(self, path: str | Path) -> Path:
        return save_wav(self.frames, path)


def audio_detection(stream: "pyaudio.Stream") -> AudioSegment:
    """
    Block until a voice segment is detected and ends.

    Uses simple RMS amplitude gating (sufficient for call recording).
    For production, replace with WebRTC VAD or Silero VAD.

    Args:
        stream: An open PyAudio input stream.

    Returns:
        AudioSegment with detected=True and captured frames, or
        AudioSegment with detected=False on interruption.
    """
    active = False
    last_time = 0.0
    frames: list[bytes] = []

    while True:
        try:
            data = stream.read(CONFIG.chunk_size, exception_on_overflow=False)
            amplitude = int(np.max(np.abs(np.frombuffer(data, dtype=np.int16))))

            if amplitude > CONFIG.vad_threshold:
                active = True
                last_time = time.monotonic()
                frames.append(data)
            elif active:
                frames.append(data)  # keep trailing silence for natural endings
                if (time.monotonic() - last_time) > CONFIG.silence_timeout:
                    return AudioSegment(detected=True, frames=frames)

        except KeyboardInterrupt:
            break
        except OSError as exc:
            log.error("Stream read error: %s", exc)
            break

    return AudioSegment(detected=False)


# ---------------------------------------------------------------------------
# Stop-signal helper (keyboard or external)
# ---------------------------------------------------------------------------
class StopSignal:
    """Thread-safe flag that can be set by 'q' key or external code."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def request_stop(self) -> None:
        self._flag.set()

    @property
    def is_set(self) -> bool:
        return self._flag.is_set()

    def start_keyboard_listener(self) -> None:
        """Background thread — sets flag when user presses 'q'."""

        def _listen():
            try:
                import msvcrt

                while not self.is_set:
                    if msvcrt.kbhit() and msvcrt.getch().lower() == b"q":
                        log.info("Stop requested via keyboard.")
                        self.request_stop()
                    time.sleep(0.05)
            except ImportError:
                pass  # Non-Windows: rely on KeyboardInterrupt (Ctrl+C)

        threading.Thread(target=_listen, daemon=True).start()


# ---------------------------------------------------------------------------
# High-level recording API
# ---------------------------------------------------------------------------
def record_call_audio(
    duration: Optional[float] = None,
    output_file: str | Path = "call_recording.wav",
    stop_signal: Optional[StopSignal] = None,
) -> bool:
    """
    Record microphone and (if available) system audio during a call.

    Args:
        duration:    Maximum recording length in seconds (None = unlimited).
        output_file: Destination .wav path.
        stop_signal: External StopSignal; creates one internally if omitted.

    Returns:
        True on success, False otherwise.
    """
    if not _PYAUDIO_AVAILABLE:
        log.error("pyaudio is not installed. Run: pip install pyaudio")
        return False

    if stop_signal is None:
        stop_signal = StopSignal()
        stop_signal.start_keyboard_listener()

    p = pyaudio.PyAudio()

    # --- Open streams ---------------------------------------------------------
    mic_stream: Optional["pyaudio.Stream"] = None
    sys_stream: Optional["pyaudio.Stream"] = None

    try:
        mic_stream = p.open(
            format=pyaudio.paInt16,
            channels=CONFIG.channels,
            rate=CONFIG.sample_rate,
            input=True,
            frames_per_buffer=CONFIG.chunk_size,
        )
        log.info("✓ Microphone stream opened.")
    except Exception as exc:
        log.error("Cannot open microphone: %s", exc)
        p.terminate()
        return False

    sys_stream = open_system_audio_stream(p)
    mode = "Microphone + System Audio" if sys_stream else "Microphone only"

    _print_recording_header(output_file, mode, duration)

    # --- Record loop ----------------------------------------------------------
    frames: list[bytes] = []
    start = time.monotonic()
    chunk_count = 0

    try:
        while not stop_signal.is_set:
            if duration and (time.monotonic() - start) >= duration:
                log.info("Duration limit reached (%.1fs).", duration)
                break

            try:
                data = mic_stream.read(CONFIG.chunk_size, exception_on_overflow=False)
                frames.append(data)
            except OSError as exc:
                log.warning("Mic read error: %s", exc)

            if sys_stream:
                try:
                    sys_data = sys_stream.read(
                        CONFIG.chunk_size, exception_on_overflow=False
                    )
                    frames.append(sys_data)
                except OSError:
                    pass  # Non-fatal — mic audio still captured

            chunk_count += 1
            if chunk_count % CONFIG.progress_interval == 0:
                elapsed = time.monotonic() - start
                print(f"  ⏺  {elapsed:>7.1f}s recorded …", end="\r", flush=True)

    except KeyboardInterrupt:
        log.info("Recording stopped by Ctrl+C.")
    finally:
        print()  # clear progress line
        _close_stream(mic_stream)
        _close_stream(sys_stream)
        p.terminate()

    # --- Save -----------------------------------------------------------------
    if not frames:
        log.warning("No audio captured.")
        return False

    out_path = save_wav(frames, output_file)
    total_bytes = sum(len(f) for f in frames)
    duration_s = total_bytes / 2 / CONFIG.sample_rate
    size_kb = out_path.stat().st_size / 1024

    print()
    log.info("✓ Recording saved: %s", out_path)
    log.info("  Mode:     %s", mode)
    log.info("  Duration: %.2f seconds", duration_s)
    log.info("  Size:     %.2f KB", size_kb)
    print()
    return True


def record_on_voice(
    output_dir: str | Path = "recordings",
    max_segments: int = 0,  # 0 = unlimited
    stop_signal: Optional[StopSignal] = None,
) -> list[Path]:
    """
    Continuously monitor microphone and save a new .wav file for each
    voice segment detected (hands-free, VAD-triggered recording).

    Args:
        output_dir:   Directory where segment files are written.
        max_segments: Stop after this many segments (0 = run forever).
        stop_signal:  Optional external stop signal.

    Returns:
        List of paths to saved segment files.
    """
    if not _PYAUDIO_AVAILABLE:
        log.error("pyaudio is not installed. Run: pip install pyaudio")
        return []

    if stop_signal is None:
        stop_signal = StopSignal()
        stop_signal.start_keyboard_listener()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CONFIG.channels,
            rate=CONFIG.sample_rate,
            input=True,
            frames_per_buffer=CONFIG.chunk_size,
        )
    except Exception as exc:
        log.error("Cannot open microphone: %s", exc)
        p.terminate()
        return []

    log.info("🎙  VAD mode active — listening for voice …  (Ctrl+C or 'q' to stop)")
    saved: list[Path] = []
    count = 0

    try:
        while not stop_signal.is_set:
            segment = audio_detection(stream)
            if not segment.detected:
                break

            count += 1
            fname = out_dir / f"segment_{count:04d}_{int(time.time())}.wav"
            path = segment.save(fname)
            saved.append(path)
            log.info(
                "  Segment %d saved → %s  (%.2fs)",
                count,
                path,
                segment.duration_seconds,
            )

            if max_segments and count >= max_segments:
                log.info("Max segments (%d) reached.", max_segments)
                break

    except KeyboardInterrupt:
        pass
    finally:
        _close_stream(stream)
        p.terminate()

    log.info("Session ended. %d segment(s) saved.", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _close_stream(stream: Optional["pyaudio.Stream"]) -> None:
    if stream is None:
        return
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass


def _print_recording_header(
    output_file: str | Path,
    mode: str,
    duration: Optional[float],
) -> None:
    sep = "─" * 72
    dur_str = f"{duration:.0f}s" if duration else "unlimited"
    print(f"\n{sep}")
    print("  ⏺  CALL RECORDING — ACTIVE")
    print(sep)
    print(f"  Output  : {output_file}")
    print(f"  Mode    : {mode}")
    print(f"  Duration: {dur_str}")
    print(f"  Stop    : press 'q'  or  Ctrl+C")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio Detection & Call Recording — v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- record ---------------------------------------------------------------
    p_rec = sub.add_parser("record", help="Record a call to a WAV file")
    p_rec.add_argument("-o", "--output", default="call_recording.wav")
    p_rec.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Max duration in seconds (omit for unlimited)",
    )

    # -- vad ------------------------------------------------------------------
    p_vad = sub.add_parser("vad", help="VAD-triggered segment recording")
    p_vad.add_argument("-o", "--output-dir", default="recordings")
    p_vad.add_argument("-n", "--max-segments", type=int, default=0)
    p_vad.add_argument(
        "--threshold",
        type=int,
        default=CONFIG.vad_threshold,
        help="RMS amplitude threshold (raise for noisy environments)",
    )

    # -- devices --------------------------------------------------------------
    sub.add_parser("devices", help="List all audio input devices")

    args = parser.parse_args()

    if args.command == "devices":
        print_devices()

    elif args.command == "record":
        record_call_audio(duration=args.duration, output_file=args.output)

    elif args.command == "vad":
        CONFIG.vad_threshold = args.threshold
        record_on_voice(output_dir=args.output_dir, max_segments=args.max_segments)
