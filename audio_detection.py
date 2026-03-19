"""
Audio Detection Module
======================
Monitors the microphone for speaking/sound events above a threshold.
Runs in a loop and returns when a speaking segment ends.

Requirements:
    pip install pyaudio numpy

⚠️  Important:
    - This module opens a PyAudio stream at import time.
      If no microphone is present, the import will fail.
    - audio_detection() is a BLOCKING call. Run it in a background thread.
    - The returned audio_data is raw PCM bytes (int16, mono, 48000 Hz).
      Use create_wav_bytes() to wrap it as a valid .wav file.

Usage:
    from threading import Thread
    from audio_detection import audio_detection, create_wav_bytes

    def monitor():
        while True:
            result = audio_detection()
            if result["audio_detected"]:
                wav = create_wav_bytes(result["audio_data"])
                with open("flagged_audio.wav", "wb") as f:
                    f.write(wav)

    Thread(target=monitor, daemon=True).start()
"""

import pyaudio
import wave
import io
import numpy as np
import time

# Audio capture settings
THRESHOLD = (
    2000  # Amplitude threshold — increase if too sensitive in noisy environments
)
CHUNK = 2048  # Frames per buffer (larger = smoother but more latency)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # 48 kHz sample rate
SOUND_END_DELAY = 4  # Seconds of silence before ending a recording segment

# Initialize PyAudio stream (opened once at import time)
_p = pyaudio.PyAudio()
_stream = _p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)


def audio_detection():
    """
    Monitor microphone and return when a speaking segment is detected and ends.

    This function blocks until:
      - Sound above THRESHOLD is detected, AND
      - Silence lasts for SOUND_END_DELAY seconds after the sound stops.

    Returns:
        dict: {
            "audio_detected": True,
            "audio_data": bytes   # raw PCM int16 bytes
        }
        or on KeyboardInterrupt / stream error:
        {
            "audio_detected": False,
            "audio_data": None
        }
    """
    sound_detected = False
    last_sound_time = 0
    frames = []

    while True:
        try:
            data = _stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if np.max(np.abs(audio_data)) > THRESHOLD:
                if not sound_detected:
                    sound_detected = True
                last_sound_time = time.time()
                frames.append(data)

            # Return segment once silence persists after sound
            if sound_detected and (time.time() - last_sound_time > SOUND_END_DELAY):
                audio_bytes = b"".join(frames)
                frames = []
                sound_detected = False
                return {"audio_detected": True, "audio_data": audio_bytes}

        except KeyboardInterrupt:
            break

    return {"audio_detected": False, "audio_data": None}


def create_wav_bytes(raw_audio, channels=1, sampwidth=2, framerate=48000):
    """
    Wrap raw PCM audio bytes with a WAV header so they can be saved/played.

    Args:
        raw_audio (bytes): Raw PCM bytes from audio_detection().
        channels (int): 1 = mono.
        sampwidth (int): 2 = 16-bit.
        framerate (int): Sample rate (must match RATE above).

    Returns:
        bytes: Valid WAV file bytes ready to write to disk or store in DB.
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(raw_audio)
    return wav_buffer.getvalue()


def cleanup():
    """Call this when your application shuts down to release audio resources."""
    _stream.stop_stream()
    _stream.close()
    _p.terminate()
