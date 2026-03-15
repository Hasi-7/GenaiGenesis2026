from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from config.third_party import (
    AudioClassificationPipelineProtocol,
    load_transformers_pipeline,
)
from models.types import ClassifierResult
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    _transformers_pipeline_factory = load_transformers_pipeline()
except Exception:  # pragma: no cover - optional dependency failure.
    _transformers_pipeline_factory = None


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
_DEFAULT_BACKEND = "heuristic"


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
    """Speech tone classifier with cheap heuristic default."""

    def __init__(self, backend: str = _DEFAULT_BACKEND) -> None:
        self._backend = backend.strip().lower()
        self._pipe: AudioClassificationPipelineProtocol | None = None
        self._pipe_failed = False

        if self._backend not in {"heuristic", "transformer"}:
            logger.warning(
                "Unknown SPEECH_TONE_BACKEND=%s; falling back to heuristic",
                self._backend,
            )
            self._backend = "heuristic"

        if self._backend == "heuristic":
            logger.debug("SpeechToneClassifier using heuristic backend")

    def classify(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = _MODEL_SAMPLE_RATE,
    ) -> ClassifierResult:
        audio_chunk = _to_mono(audio_chunk.squeeze())

        if _is_silent(audio_chunk):
            return ClassifierResult(label="silent", confidence=1.0)

        if self._backend == "transformer":
            transformer_result = self._classify_with_transformer(
                audio_chunk,
                sample_rate,
            )
            if transformer_result is not None:
                return transformer_result

        return _classify_with_heuristics(audio_chunk, sample_rate)

    def _classify_with_transformer(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int,
    ) -> ClassifierResult | None:
        pipe = self._ensure_transformer_pipeline()
        if pipe is None:
            return None

        model_audio = audio_chunk
        if sample_rate != _MODEL_SAMPLE_RATE:
            model_audio = _resample(audio_chunk, sample_rate, _MODEL_SAMPLE_RATE)

        try:
            results: Sequence[dict[str, object]] = pipe(
                {"raw": model_audio, "sampling_rate": _MODEL_SAMPLE_RATE},
                top_k=len(_EMOTION_TO_TONE),
            )
        except Exception:
            logger.exception(
                "SpeechToneClassifier transformer inference failed; "
                "falling back to heuristic backend"
            )
            self._pipe_failed = True
            return None

        tone_scores: dict[str, float] = {
            "calm": 0.0,
            "stressed": 0.0,
            "monotone": 0.0,
        }
        for result in results:
            label_value = result.get("label")
            score_value = result.get("score")
            emotion = str(label_value).lower()
            score = _coerce_float(score_value)
            tone = _EMOTION_TO_TONE.get(emotion)
            if tone is not None:
                tone_scores[tone] += score

        best_label = max(tone_scores, key=lambda key: tone_scores[key])
        return ClassifierResult(
            label=best_label,
            confidence=tone_scores[best_label],
        )

    def _ensure_transformer_pipeline(self):
        if self._pipe_failed:
            return None
        if self._pipe is not None:
            return self._pipe
        if _transformers_pipeline_factory is None:
            logger.warning(
                "transformers is unavailable; falling back to heuristic speech tone"
            )
            self._pipe_failed = True
            return None

        try:
            logger.debug(
                "Loading transformer speech tone backend; this may download "
                "model weights"
            )
            self._pipe = _transformers_pipeline_factory(
                "audio-classification",
                model="Hatman/audio-emotion-detection",
            )
        except Exception:
            logger.exception(
                "Failed to initialize transformer speech tone backend; "
                "falling back to heuristic"
            )
            self._pipe_failed = True
            self._pipe = None
            return None
        return self._pipe


def _to_mono(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _classify_with_heuristics(
    audio: NDArray[np.float32],
    sample_rate: int,
) -> ClassifierResult:
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    zero_crossings = float(
        np.mean(np.abs(np.diff(np.signbit(audio)).astype(np.float32)))
    )

    window = max(1, min(len(audio), sample_rate // 8))
    envelope = np.abs(audio[: len(audio) - (len(audio) % window)])
    if len(envelope) >= window:
        chunks = envelope.reshape(-1, window)
        chunk_energy = np.asarray(chunks.mean(axis=1), dtype=np.float32)
        mean_energy = sum(float(value) for value in chunk_energy) / len(chunk_energy)
        variance = sum(
            (float(value) - mean_energy) ** 2 for value in chunk_energy
        ) / len(chunk_energy)
        std_energy = variance**0.5
        variation = std_energy / (mean_energy + 1e-6)
    else:
        variation = 0.0

    if rms > 0.09 or peak > 0.72 or zero_crossings > 0.22:
        confidence = min(0.95, 0.55 + rms * 2.4 + zero_crossings)
        return ClassifierResult(label="stressed", confidence=confidence)

    if variation < 0.12 and zero_crossings < 0.08:
        confidence = min(0.9, 0.58 + (0.12 - variation) * 1.8)
        return ClassifierResult(label="monotone", confidence=confidence)

    confidence = min(0.88, 0.6 + variation * 0.6 + max(0.0, 0.12 - rms))
    return ClassifierResult(label="calm", confidence=confidence)


def _is_silent(audio: NDArray[np.float32]) -> bool:
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms < _SILENCE_RMS_THRESHOLD


def _resample(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    duration = len(audio) / orig_sr
    target_length = max(1, int(duration * target_sr))
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
    from config.settings import get_settings

    classifier = SpeechToneClassifier(backend=get_settings().speech_tone_backend)

    result = classifier.classify(audio_data, SAMPLE_RATE)
    print(f"Label: {result.label}, Confidence: {result.confidence:.3f}")
