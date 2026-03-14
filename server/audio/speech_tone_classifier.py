from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from config.third_party import AudioClassificationPipelineProtocol, load_transformers_pipeline
from models.types import ClassifierResult
from numpy.typing import NDArray

class SpeechToneClassifierProtocol(Protocol):
    """Protocol for speech tone classification from audio."""

    def classify(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = ...,
    ) -> ClassifierResult:
        """
        Classify tone from audio features.

        Labels: "calm", "stressed", "monotone", "silent".
        """
        ...

# Maps HuggingFace emotion labels to protocol labels.
_EMOTION_TO_TONE: dict[str, str] = {
    "angry": "stressed",
    "disgusted": "stressed",
    "fearful": "stressed",
    "happy": "calm",
    "neutral": "calm",
    "sad": "monotone",
    "surprised": "stressed",
}

_MODEL_SAMPLE_RATE = 16_000
_SILENCE_RMS_THRESHOLD = 0.001


@runtime_checkable
class _SoundDeviceModuleProtocol(Protocol):
    def rec(
        self,
        frames: int,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
    ) -> NDArray[np.float32]: ...

    def wait(self) -> None: ...


def _load_sounddevice() -> _SoundDeviceModuleProtocol:
    module = importlib.import_module("sounddevice")
    if not isinstance(module, _SoundDeviceModuleProtocol):
        raise TypeError("sounddevice module does not match expected protocol")
    return module


class SpeechToneClassifier:
    """Classifies speech tone from audio using Hatman/audio-emotion-detection."""

    def __init__(self) -> None:
        self._pipe: AudioClassificationPipelineProtocol = load_transformers_pipeline()(
            "audio-classification",
            model="Hatman/audio-emotion-detection",
        )

    def classify(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = _MODEL_SAMPLE_RATE,
    ) -> ClassifierResult:
        """Classify tone from audio features.

        Labels: "calm", "stressed", "monotone", "silent".
        """

        # Collapse multi-channel input to mono before feature extraction.
        audio_chunk = _to_mono(audio_chunk)

        if _is_silent(audio_chunk):
            return ClassifierResult(label="silent", confidence=1.0)

        # Resample to 16 kHz if needed.
        if sample_rate != _MODEL_SAMPLE_RATE:
            audio_chunk = _resample(audio_chunk, sample_rate, _MODEL_SAMPLE_RATE)

        results: Sequence[dict[str, object]] = self._pipe(
            {"raw": audio_chunk, "sampling_rate": _MODEL_SAMPLE_RATE},
            top_k=len(_EMOTION_TO_TONE),
        )

        # Aggregate scores by protocol label.
        tone_scores: dict[str, float] = {"calm": 0.0, "stressed": 0.0, "monotone": 0.0}
        for r in results:
            label = r.get("label")
            score_value = r.get("score")
            if not isinstance(label, str):
                continue
            if not isinstance(score_value, int | float):
                continue
            emotion = label.lower()
            score = float(score_value)
            tone = _EMOTION_TO_TONE.get(emotion)
            if tone is not None:
                tone_scores[tone] += score

        best_label = max(tone_scores, key=lambda k: tone_scores[k])
        return ClassifierResult(label=best_label, confidence=tone_scores[best_label])


def _to_mono(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert multi-channel audio to 1-D mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    # (N, channels) → average across channels → (N,)
    return audio.mean(axis=1).astype(np.float32)


def _is_silent(audio: NDArray[np.float32]) -> bool:
    """Return True if RMS energy is below silence threshold."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms < _SILENCE_RMS_THRESHOLD


def _resample(
    audio: NDArray[np.float32], orig_sr: int, target_sr: int
) -> NDArray[np.float32]:
    """Simple linear interpolation resampling."""
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled: NDArray[np.float32] = np.interp(
        indices,
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
    return resampled


if __name__ == "__main__":
    SAMPLE_RATE = 16_000
    DURATION_SECONDS = 3

    sd = _load_sounddevice()

    print(f"Recording {DURATION_SECONDS}s of audio at {SAMPLE_RATE} Hz...")
    audio_data = np.asarray(
        sd.rec(
            int(DURATION_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        ),
        dtype=np.float32,
    )
    sd.wait()
    audio_data = _to_mono(audio_data)
    print(f"Recorded {len(audio_data)} samples.")

    print("Loading model (first run downloads ~1.2 GB)...")
    classifier = SpeechToneClassifier()

    result = classifier.classify(audio_data, SAMPLE_RATE)
    print(f"Label: {result.label}, Confidence: {result.confidence:.3f}")
