from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from transformers import pipeline  # type: ignore[import-untyped]

from models.types import ClassifierResult

# Maps HuggingFace emotion labels → protocol labels ("calm", "stressed", "monotone", "silent")
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
_SILENCE_RMS_THRESHOLD = 0.01


class SpeechToneClassifier:
    """Classifies speech tone from audio using Hatman/audio-emotion-detection."""

    def __init__(self) -> None:
        self._pipe = pipeline(
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
        if _is_silent(audio_chunk):
            return ClassifierResult(label="silent", confidence=1.0)

        # Resample to 16 kHz if needed.
        if sample_rate != _MODEL_SAMPLE_RATE:
            audio_chunk = _resample(audio_chunk, sample_rate, _MODEL_SAMPLE_RATE)

        results: list[dict[str, object]] = self._pipe(
            {"raw": audio_chunk, "sampling_rate": _MODEL_SAMPLE_RATE},
            top_k=len(_EMOTION_TO_TONE),
        )

        # Aggregate scores by protocol label.
        tone_scores: dict[str, float] = {"calm": 0.0, "stressed": 0.0, "monotone": 0.0}
        for r in results:
            emotion = str(r["label"]).lower()
            score = float(r["score"])  # type: ignore[arg-type]
            tone = _EMOTION_TO_TONE.get(emotion)
            if tone is not None:
                tone_scores[tone] += score

        best_label = max(tone_scores, key=lambda k: tone_scores[k])
        return ClassifierResult(label=best_label, confidence=tone_scores[best_label])


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
    resampled: NDArray[np.float32] = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return resampled


if __name__ == "__main__":
    import sounddevice as sd

    SAMPLE_RATE = 16_000
    DURATION_SECONDS = 3

    print(f"Recording {DURATION_SECONDS}s of audio at {SAMPLE_RATE} Hz...")
    audio_data: NDArray[np.float32] = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio_data = audio_data.squeeze()
    print(f"Recorded {len(audio_data)} samples.")

    print("Loading model (first run downloads ~1.2 GB)...")
    classifier = SpeechToneClassifier()

    result = classifier.classify(audio_data, SAMPLE_RATE)
    print(f"Label: {result.label}, Confidence: {result.confidence:.3f}")
