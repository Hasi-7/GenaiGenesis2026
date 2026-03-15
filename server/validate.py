"""CognitiveSense classifier validation script.

Runs labelled test cases against every classifier and prints a rich
formatted report to the terminal AND saves it as validation_report.html.

Usage::

    uv run python validate.py

No camera, microphone, or API keys required.
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    _RICH_AVAILABLE = True
except ModuleNotFoundError:
    _RICH_AVAILABLE = False

    class _PlainBox:
        ROUNDED = None
        DOUBLE_EDGE = None

    box = _PlainBox()

    class Text(str):
        def __new__(cls, text: str, justify: str | None = None, style: str | None = None):
            del justify, style
            return str.__new__(cls, text)

    class Panel(str):
        def __new__(
            cls,
            renderable: object,
            subtitle: str | None = None,
            border_style: str | None = None,
        ):
            del border_style
            text = str(renderable)
            if subtitle:
                text = f"{text}\n{subtitle}"
            return str.__new__(cls, text)

    class Rule(str):
        def __new__(cls, title: str):
            return str.__new__(cls, f"\n=== {title} ===")

    class Table:
        def __init__(
            self,
            title: str = "",
            box: object | None = None,
            show_lines: bool = False,
            header_style: str | None = None,
            title_justify: str | None = None,
        ) -> None:
            del box, show_lines, header_style, title_justify
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[str, ...]] = []

        def add_column(self, name: str, **_: object) -> None:
            self.columns.append(name)

        def add_row(self, *values: object) -> None:
            self.rows.append(tuple(str(value) for value in values))

        def add_section(self) -> None:
            return

        def __str__(self) -> str:
            lines: list[str] = []
            if self.title:
                lines.append(self.title)
            if self.columns:
                lines.append(" | ".join(self.columns))
                lines.append("-" * len(lines[-1]))
            for row in self.rows:
                lines.append(" | ".join(row))
            return "\n".join(lines)

    class Console:
        def print(self, *objects: object, **_: object) -> None:
            text = " ".join(str(obj) for obj in objects)
            print(text)

# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    description: str
    inputs: str          # human-readable summary of inputs
    expected: str
    actual: str
    confidence: float
    passed: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Helper: build a rich Table from a list of CaseResults
# ---------------------------------------------------------------------------

def _build_table(title: str, results: list[CaseResult]) -> Table:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    title_color = "green" if passed == total else "yellow" if passed > 0 else "red"

    table = Table(
        title=f"[bold {title_color}]{title}[/bold {title_color}]  "
              f"[dim]({passed}/{total} passed)[/dim]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_justify="left",
    )

    table.add_column("Test Case", style="dim white", min_width=28)
    table.add_column("Inputs", style="white", min_width=34)
    table.add_column("Expected", style="cyan", min_width=12)
    table.add_column("Actual", min_width=12)
    table.add_column("Confidence", justify="right", min_width=10)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Note", style="dim", min_width=18)

    for r in results:
        actual_style = "green" if r.passed else "red"
        status = "[green]✓ PASS[/green]" if r.passed else "[red]✗ FAIL[/red]"
        conf_str = f"{r.confidence:.3f}" if r.confidence >= 0.0 else "—"
        table.add_row(
            r.description,
            r.inputs,
            r.expected,
            f"[{actual_style}]{r.actual}[/{actual_style}]",
            conf_str,
            status,
            r.note,
        )

    return table


# ---------------------------------------------------------------------------
# 1. (Removed – BlendshapeExpressionClassifier has been removed;
#     facial expression is now handled by the LLMStateTracker.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. EarBlinkDetector
# ---------------------------------------------------------------------------

def validate_blink() -> list[CaseResult]:
    from vision.blink_detector import EarBlinkDetector

    results: list[CaseResult] = []

    def make_landmarks(ear_value: float) -> Any:
        import numpy as np
        # Build a 478x3 landmark array that produces the given EAR.
        # EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        # We fix horizontal distance = 1.0 and set vertical accordingly.
        # Target: ear_value = vertical_sum / (2 * 1.0) -> vertical_sum = 2 * ear_value
        # Use P2 and P6 each at half: ||P2-P6|| = ear_value, ||P3-P5|| = ear_value
        lm = np.zeros((478, 3), dtype=np.float32)

        # Right eye indices: (33, 160, 158, 133, 153, 144)
        # EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        # P1=33 (outer), P4=133 (inner): horizontal = 1.0
        # P2/P6 and P3/P5 must share the same x so their distance is purely vertical
        lm[33]  = [0.0, 0.0, 0.0]
        lm[133] = [1.0, 0.0, 0.0]
        lm[160] = [0.25,  ear_value / 2, 0.0]   # P2 upper-outer
        lm[144] = [0.25, -ear_value / 2, 0.0]   # P6 lower-outer (same x as P2)
        lm[158] = [0.75,  ear_value / 2, 0.0]   # P3 upper-inner
        lm[153] = [0.75, -ear_value / 2, 0.0]   # P5 lower-inner (same x as P3)

        # Left eye indices: (362, 385, 387, 263, 380, 373) — same geometry
        lm[362] = [0.0, 0.0, 0.0]
        lm[263] = [1.0, 0.0, 0.0]
        lm[385] = [0.25,  ear_value / 2, 0.0]   # P2
        lm[373] = [0.25, -ear_value / 2, 0.0]   # P6 (same x)
        lm[387] = [0.75,  ear_value / 2, 0.0]   # P3
        lm[380] = [0.75, -ear_value / 2, 0.0]   # P5 (same x)

        return lm

    # --- EAR value round-trip test ---
    detector_ear = EarBlinkDetector()
    for ear_val, desc in [(0.30, "EAR=0.30 (open)"), (0.21, "EAR=0.21 (threshold)"), (0.10, "EAR=0.10 (closed)")]:
        lm = make_landmarks(ear_val)
        data = detector_ear.detect(lm)
        results.append(CaseResult(
            description=f"EAR round-trip: {desc}",
            inputs=f"synthetic landmarks -> target EAR={ear_val:.2f}",
            expected=f"avg~{ear_val:.2f}",
            actual=f"avg={data.ear_average:.3f}",
            confidence=-1.0,
            passed=abs(data.ear_average - ear_val) < 0.02,
            note="geometric accuracy check",
        ))

    # --- BPM -> label classification tests ---
    bpm_cases: list[tuple[float, str, str]] = [
        (0.0,  "unknown",  "No data yet"),
        (5.0,  "stressed", "bpm < 10 -> stressed"),
        (9.9,  "stressed", "bpm just below 10"),
        (12.0, "elevated", "10–15 bpm band"),
        (17.5, "normal",   "ideal centre of normal band"),
        (22.0, "elevated", "20–25 bpm band"),
        (30.0, "fatigued", "bpm > 25 -> fatigued"),
    ]

    from models.types import BlinkData
    for bpm, expected, note in bpm_cases:
        detector = EarBlinkDetector()
        blink_data = BlinkData(
            ear_left=0.28,
            ear_right=0.28,
            ear_average=0.28,
            blink_detected=False,
            blinks_per_minute=bpm,
        )
        result = detector.classify(blink_data)
        results.append(CaseResult(
            description=f"BPM={bpm:.1f} -> {expected}",
            inputs=f"blinks_per_minute={bpm:.1f}",
            expected=expected,
            actual=result.label,
            confidence=result.confidence,
            passed=result.label == expected,
            note=note,
        ))

    # --- Blink detection state machine test ---
    detector_sm = EarBlinkDetector(ear_threshold=0.21, consecutive_frames=3)
    open_lm  = make_landmarks(0.30)
    closed_lm = make_landmarks(0.10)

    # Feed 3 consecutive closed frames then open -> should register a blink
    blink_detected_frames = []
    for _ in range(3):
        d = detector_sm.detect(closed_lm)
        blink_detected_frames.append(d.blink_detected)
    d = detector_sm.detect(open_lm)
    blink_detected_frames.append(d.blink_detected)

    results.append(CaseResult(
        description="3 closed + 1 open -> blink registered",
        inputs="EAR=0.10 × 3 frames, then EAR=0.30",
        expected="True on 4th frame",
        actual=str(d.blink_detected),
        confidence=-1.0,
        passed=d.blink_detected is True,
        note="state machine: blink fires on eye-open",
    ))

    # 2 closed frames (below threshold) should NOT register a blink
    detector_sm2 = EarBlinkDetector(ear_threshold=0.21, consecutive_frames=3)
    for _ in range(2):
        detector_sm2.detect(closed_lm)
    d2 = detector_sm2.detect(open_lm)
    results.append(CaseResult(
        description="2 closed + 1 open -> no blink (below consec)",
        inputs="EAR=0.10 × 2 frames, then EAR=0.30",
        expected="False",
        actual=str(d2.blink_detected),
        confidence=-1.0,
        passed=d2.blink_detected is False,
        note="consecutive_frames=3 not reached",
    ))

    return results


# ---------------------------------------------------------------------------
# 3. IrisGazeDetector
# ---------------------------------------------------------------------------

def validate_gaze() -> list[CaseResult]:
    from vision.eye_movement_detector import IrisGazeDetector
    import numpy as np

    results: list[CaseResult] = []

    def make_gaze_landmarks(h_ratio: float, v_ratio: float) -> np.ndarray:
        """Build a 478x3 landmark array placing irises at known gaze ratios."""
        lm = np.zeros((478, 3), dtype=np.float32)

        # Left eye: outer=263, inner=362, top=386, bottom=374, iris=473
        # Right eye: outer=33, inner=133, top=159, bottom=145, iris=468
        # Horizontal: iris position along outer->inner axis maps to h_ratio in [-1,1]
        # We set eye width = 1.0, height = 0.4

        # Left eye corners (horizontal axis: outer->inner)
        lm[263] = [0.0, 0.0, 0.0]   # outer
        lm[362] = [1.0, 0.0, 0.0]   # inner
        lm[386] = [0.5, -0.2, 0.0]  # top
        lm[374] = [0.5,  0.2, 0.0]  # bottom

        # Right eye corners (axis used: inner->outer i.e. 133->33)
        lm[133] = [0.0, 0.0, 0.0]   # inner
        lm[33]  = [1.0, 0.0, 0.0]   # outer
        lm[159] = [0.5, -0.2, 0.0]  # top
        lm[145] = [0.5,  0.2, 0.0]  # bottom

        # h_ratio in [-1,1]: ratio = (proj/length)*2 - 1, proj = iris_x - outer_x
        # -> iris_x = (h_ratio + 1) / 2 * length  (outer_x=0, length=1.0)
        h_pos = (h_ratio + 1.0) / 2.0

        # v_ratio in [-1,1]: axis goes from top (-0.2) to bottom (+0.2)
        # ratio = ((iris_y - top_y) / eye_height) * 2 - 1
        # -> iris_y = top_y + (v_ratio + 1) / 2 * eye_height
        EYE_TOP_Y = -0.2
        EYE_HEIGHT = 0.4
        v_pos = EYE_TOP_Y + (v_ratio + 1.0) / 2.0 * EYE_HEIGHT

        lm[473] = [h_pos, v_pos, 0.0]   # left iris
        lm[468] = [h_pos, v_pos, 0.0]   # right iris

        return lm

    direction_cases: list[tuple[float, float, str, str]] = [
        (0.0,   0.0,  "center", "iris perfectly centred"),
        (0.8,   0.0,  "right",  "iris far right"),
        (-0.8,  0.0,  "left",   "iris far left"),
        (0.0,   0.7,  "down",   "iris looking down"),
        (0.0,  -0.7,  "up",     "iris looking up"),
        (0.15,  0.05, "center", "within ±0.2 threshold"),
    ]

    for h, v, expected_dir, note in direction_cases:
        lm = make_gaze_landmarks(h, v)
        detector = IrisGazeDetector(center_threshold=0.2)
        gaze = detector.detect(lm)
        results.append(CaseResult(
            description=f"Gaze direction: {note}",
            inputs=f"h={h:.2f}, v={v:.2f}",
            expected=expected_dir,
            actual=gaze.direction,
            confidence=-1.0,
            passed=gaze.direction == expected_dir,
            note=f"detected h={gaze.horizontal_ratio:.3f} v={gaze.vertical_ratio:.3f}",
        ))

    # Classify: fill window with centred gaze -> focused
    detector_focused = IrisGazeDetector(
        center_threshold=0.2, window=15,
        focused_dwell_seconds=0.0, distracted_dwell_seconds=0.0,
    )
    centre_lm = make_gaze_landmarks(0.0, 0.0)
    last_gaze = detector_focused.detect(centre_lm)
    for _ in range(14):
        last_gaze = detector_focused.detect(centre_lm)
    classify_result = detector_focused.classify(last_gaze)
    results.append(CaseResult(
        description="15 centred frames -> focused",
        inputs="h=0.0, v=0.0 × 15 frames (window filled)",
        expected="focused",
        actual=classify_result.label,
        confidence=classify_result.confidence,
        passed=classify_result.label == "focused",
        note="dwell satisfied, centre_ratio=1.0",
    ))

    # Classify: fill window with off-centre gaze -> distracted
    detector_distracted = IrisGazeDetector(
        center_threshold=0.2, window=15,
        focused_dwell_seconds=0.0, distracted_dwell_seconds=0.0,
    )
    right_lm = make_gaze_landmarks(0.8, 0.0)
    last_gaze2 = detector_distracted.detect(right_lm)
    for _ in range(14):
        last_gaze2 = detector_distracted.detect(right_lm)
    classify_result2 = detector_distracted.classify(last_gaze2)
    results.append(CaseResult(
        description="15 far-right frames -> distracted",
        inputs="h=0.8, v=0.0 × 15 frames (window filled)",
        expected="distracted",
        actual=classify_result2.label,
        confidence=classify_result2.confidence,
        passed=classify_result2.label == "distracted",
        note="centre_ratio=0.0, dwell satisfied",
    ))

    # Not enough history -> unknown
    detector_unknown = IrisGazeDetector(center_threshold=0.2, window=15)
    lm_u = make_gaze_landmarks(0.0, 0.0)
    g_u = detector_unknown.detect(lm_u)
    r_u = detector_unknown.classify(g_u)
    results.append(CaseResult(
        description="1 frame only -> unknown (insufficient history)",
        inputs="1 frame, window=15",
        expected="unknown",
        actual=r_u.label,
        confidence=r_u.confidence,
        passed=r_u.label == "unknown",
        note="window not filled yet",
    ))

    return results


# ---------------------------------------------------------------------------
# 4. MediaPipePostureDetector — test classify() directly (no camera needed)
# ---------------------------------------------------------------------------

def validate_posture() -> list[CaseResult]:
    from vision.posture_detector import MediaPipePostureDetector
    from models.types import PostureData

    detector = MediaPipePostureDetector()
    results: list[CaseResult] = []

    cases: list[tuple[PostureData, str, str]] = [
        (
            PostureData(shoulder_angle=2.0, head_tilt=5.0, is_slouching=False),
            "upright",
            "minimal angles, no slouch flag",
        ),
        (
            PostureData(shoulder_angle=5.0, head_tilt=10.0, is_slouching=False),
            "upright",
            "below all thresholds",
        ),
        (
            PostureData(shoulder_angle=20.0, head_tilt=5.0, is_slouching=True),
            "slouched",
            "shoulder angle > 8° with slouch flag",
        ),
        (
            PostureData(shoulder_angle=9.0, head_tilt=5.0, is_slouching=True),
            "slouched",
            "is_slouching=True drives classification",
        ),
        (
            PostureData(shoulder_angle=3.0, head_tilt=25.0, is_slouching=False),
            "leaning",
            "head tilt > 20° while not slouching",
        ),
        (
            PostureData(shoulder_angle=3.0, head_tilt=45.0, is_slouching=False),
            "leaning",
            "extreme head tilt",
        ),
        (
            PostureData(shoulder_angle=7.0, head_tilt=19.0, is_slouching=False),
            "upright",
            "both just below thresholds",
        ),
        (
            PostureData(shoulder_angle=8.1, head_tilt=5.0, is_slouching=True),
            "slouched",
            "just over slouch angle threshold",
        ),
    ]

    for posture_data, expected, note in cases:
        result = detector.classify(posture_data)
        results.append(CaseResult(
            description=f"shoulder={posture_data.shoulder_angle}°, "
                        f"head={posture_data.head_tilt}°, "
                        f"slouch={posture_data.is_slouching}",
            inputs=f"shoulder_angle={posture_data.shoulder_angle}, "
                   f"head_tilt={posture_data.head_tilt}, "
                   f"is_slouching={posture_data.is_slouching}",
            expected=expected,
            actual=result.label,
            confidence=result.confidence,
            passed=result.label == expected,
            note=note,
        ))

    detector.close()
    return results


# ---------------------------------------------------------------------------
# 5. SpeechToneClassifier (heuristic backend only — no model download needed)
# ---------------------------------------------------------------------------

def validate_speech_tone() -> list[CaseResult]:
    from audio.speech_tone_classifier import SpeechToneClassifier, _is_silent
    import numpy as np

    clf = SpeechToneClassifier(backend="heuristic")
    results: list[CaseResult] = []
    sr = 16_000

    def sine(freq: float, duration: float, amplitude: float) -> Any:
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def noise(duration: float, amplitude: float) -> Any:
        rng = np.random.default_rng(42)
        return (amplitude * rng.standard_normal(int(sr * duration))).astype(np.float32)

    cases: list[tuple[str, Any, str, str]] = [
        (
            "Near-silence (RMS≈0) -> silent",
            np.zeros(sr, dtype=np.float32),
            "silent",
            "all zeros, RMS < 0.001",
        ),
        (
            "Very quiet sine -> silent",
            sine(440, 1.0, 0.0005),
            "silent",
            "amplitude below silence threshold",
        ),
        (
            "Loud high-ZCR noise -> stressed",
            noise(1.0, 0.35),
            "stressed",
            "high RMS + high zero-crossing rate",
        ),
        (
            "High amplitude burst -> stressed",
            sine(200, 1.0, 0.15),
            "stressed",
            "RMS > 0.09 threshold",
        ),
        (
            "Loud clipped noise -> stressed",
            noise(1.0, 0.50),
            "stressed",
            "peak > 0.72 threshold",
        ),
        (
            "Low-energy flat tone -> monotone",
            sine(120, 2.0, 0.025),
            "monotone",
            "low variation, low ZCR",
        ),
        (
            "700 Hz sine -> calm",
            sine(700, 1.0, 0.045),
            "calm",
            "ZCR≈0.09 (>0.08 threshold), RMS<0.09 -> not stressed, not monotone",
        ),
        (
            "800 Hz mid-amplitude sine -> calm",
            sine(800, 1.0, 0.055),
            "calm",
            "ZCR≈0.10, RMS≈0.039 -> calm band",
        ),
    ]

    for desc, audio, expected, note in cases:
        result = clf.classify(audio, sample_rate=sr)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        results.append(CaseResult(
            description=desc,
            inputs=f"samples={len(audio)}, RMS={rms:.4f}, peak={peak:.4f}",
            expected=expected,
            actual=result.label,
            confidence=result.confidence,
            passed=result.label == expected,
            note=note,
        ))

    # Silence helper unit tests
    for arr, expected_silent, note in [
        (np.zeros(100, dtype=np.float32), True, "all zeros -> silent"),
        (np.full(100, 0.5, dtype=np.float32), False, "constant 0.5 -> not silent"),
    ]:
        actual = _is_silent(arr)
        results.append(CaseResult(
            description=f"_is_silent: {note}",
            inputs=f"array of {len(arr)} samples",
            expected=str(expected_silent),
            actual=str(actual),
            confidence=-1.0,
            passed=actual == expected_silent,
            note="internal silence helper",
        ))

    return results


# ---------------------------------------------------------------------------
# 6. LLMStateTracker — parse logic (no API call)
# ---------------------------------------------------------------------------

def validate_llm_state_tracker() -> list[CaseResult]:
    import json
    import time as _time
    from state.state_tracker import LLMStateTracker, _STATE_FROM_STRING
    from models.types import CognitiveState, CognitiveStateLabel

    results: list[CaseResult] = []

    # Build a minimal tracker with a mock client (parse_response is the target)
    class _MockClient:
        pass

    class _MockScreenshot:
        def encode_jpeg(self): return b""

    tracker = LLMStateTracker(
        client=_MockClient(),           # type: ignore[arg-type]
        screenshot_manager=_MockScreenshot(),  # type: ignore[arg-type]
    )

    current = CognitiveState(
        label=CognitiveStateLabel.FOCUSED,
        confidence=0.8,
        contributing_signals=[],
        timestamp=_time.time(),
    )

    def run_parse(raw: str, desc: str, expected_transition: bool, expected_state: str | None, note: str) -> CaseResult:
        result = tracker._parse_response(raw, current)
        transitioned = result is not None
        actual_state = result.new_state.label.value if result is not None else "none"
        passed = (transitioned == expected_transition) and (
            expected_state is None or actual_state == expected_state
        )
        return CaseResult(
            description=desc,
            inputs=f"raw JSON: {raw[:70]}{'…' if len(raw) > 70 else ''}",
            expected=f"transition={expected_transition}" + (f", state={expected_state}" if expected_state else ""),
            actual=f"transition={transitioned}, state={actual_state}",
            confidence=-1.0,
            passed=passed,
            note=note,
        )

    results.append(run_parse(
        json.dumps({"emotional_summary": "user looks calm", "transition": False,
                    "new_state": "focused", "confidence": 0.8, "reasoning": "stable"}),
        "No transition -> returns None",
        False, None, "transition=false path",
    ))

    results.append(run_parse(
        json.dumps({"emotional_summary": "user looks tired", "transition": True,
                    "new_state": "fatigued", "confidence": 0.75, "reasoning": "drooping"}),
        "Transition to fatigued",
        True, "fatigued", "valid transition",
    ))

    results.append(run_parse(
        json.dumps({"emotional_summary": "stressed out", "transition": True,
                    "new_state": "stressed", "confidence": 0.9, "reasoning": "tense face"}),
        "Transition to stressed",
        True, "stressed", "valid transition",
    ))

    results.append(run_parse(
        json.dumps({"emotional_summary": "looking away", "transition": True,
                    "new_state": "distracted", "confidence": 0.65, "reasoning": "gaze off"}),
        "Transition to distracted",
        True, "distracted", "valid transition",
    ))

    results.append(run_parse(
        "not valid json {{{",
        "Malformed JSON -> returns None",
        False, None, "JSON decode error handled gracefully",
    ))

    results.append(run_parse(
        json.dumps({"emotional_summary": "odd", "transition": True,
                    "new_state": "confused", "confidence": 0.5, "reasoning": "unknown"}),
        "Unknown state label -> returns None",
        False, None, "'confused' not in STATE_FROM_STRING",
    ))

    results.append(run_parse(
        json.dumps({"transition": True, "new_state": "focused",
                    "confidence": 0.7, "reasoning": "ok"}),
        "Missing emotional_summary -> still works",
        True, "focused", "summary is optional",
    ))

    results.append(run_parse(
        json.dumps({}),
        "Empty JSON -> no transition",
        False, None, "all fields missing, transition defaults to false",
    ))

    # Validate _STATE_FROM_STRING completeness
    for label in ["focused", "fatigued", "stressed", "distracted"]:
        present = label in _STATE_FROM_STRING
        results.append(CaseResult(
            description=f"_STATE_FROM_STRING has '{label}'",
            inputs=f"key='{label}'",
            expected="present",
            actual="present" if present else "MISSING",
            confidence=-1.0,
            passed=present,
            note="all cognitive labels must be mappable",
        ))

    return results


# ---------------------------------------------------------------------------
# 7. StateTracker (rule-based) — weighted voting & transition detection
# ---------------------------------------------------------------------------

def validate_state_tracker() -> list[CaseResult]:
    from state.state_tracker import StateTracker, _weighted_vote, _LABEL_TO_STATE
    from models.types import ClassifierResult, FrameAnalysis, CognitiveStateLabel

    results: list[CaseResult] = []

    # _weighted_vote unit tests
    vote_cases: list[tuple[list[ClassifierResult], CognitiveStateLabel, str]] = [
        ([], CognitiveStateLabel.UNKNOWN, "empty signals -> unknown"),
        ([ClassifierResult("neutral", 0.9)], CognitiveStateLabel.FOCUSED, "neutral -> focused"),
        ([ClassifierResult("calm", 0.9)], CognitiveStateLabel.FOCUSED, "calm -> focused"),
        ([ClassifierResult("upright", 0.9)], CognitiveStateLabel.FOCUSED, "upright -> focused"),
        ([ClassifierResult("tense", 0.9)], CognitiveStateLabel.STRESSED, "tense -> stressed"),
        ([ClassifierResult("stressed", 0.9)], CognitiveStateLabel.STRESSED, "stressed -> stressed"),
        ([ClassifierResult("fatigued", 0.9)], CognitiveStateLabel.FATIGUED, "fatigued -> fatigued"),
        ([ClassifierResult("slouched", 0.9)], CognitiveStateLabel.FATIGUED, "slouched -> fatigued"),
        ([ClassifierResult("monotone", 0.9)], CognitiveStateLabel.FATIGUED, "monotone -> fatigued"),
        ([ClassifierResult("left", 0.9)], CognitiveStateLabel.DISTRACTED, "left gaze -> distracted"),
        (
            [ClassifierResult("stressed", 0.9), ClassifierResult("tense", 0.8),
             ClassifierResult("neutral", 0.2)],
            CognitiveStateLabel.STRESSED,
            "majority stressed wins",
        ),
        (
            [ClassifierResult("calm", 0.7), ClassifierResult("upright", 0.8),
             ClassifierResult("normal", 0.9)],
            CognitiveStateLabel.FOCUSED,
            "unanimous focused",
        ),
    ]

    for signals, expected_label, note in vote_cases:
        label, conf = _weighted_vote(signals)
        results.append(CaseResult(
            description=f"_weighted_vote: {note}",
            inputs=", ".join(f"{s.label}({s.confidence:.1f})" for s in signals) or "[]",
            expected=expected_label.value,
            actual=label.value,
            confidence=conf,
            passed=label == expected_label,
            note=note,
        ))

    # _LABEL_TO_STATE coverage — ensure all known labels are mapped
    known_labels = [
        "fatigued", "elevated", "slouching", "slouched", "leaning", "monotone",
        "stressed", "tense", "distracted", "left", "right",
        "focused", "upright", "normal", "center", "calm", "relaxed", "neutral",
    ]
    for lbl in known_labels:
        mapped = lbl in _LABEL_TO_STATE
        results.append(CaseResult(
            description=f"_LABEL_TO_STATE maps '{lbl}'",
            inputs=f"key='{lbl}'",
            expected="mapped",
            actual="mapped" if mapped else "MISSING",
            confidence=-1.0,
            passed=mapped,
            note="all signal labels must resolve to a cognitive state",
        ))

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _build_summary(sections: list[tuple[str, list[CaseResult]]]) -> Table:
    table = Table(
        title="[bold white]Overall Validation Summary[/bold white]",
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        title_justify="left",
    )
    table.add_column("Classifier", style="white", min_width=35)
    table.add_column("Passed", justify="center", min_width=8)
    table.add_column("Failed", justify="center", min_width=8)
    table.add_column("Total", justify="center", min_width=8)
    table.add_column("Pass Rate", justify="right", min_width=10)

    grand_pass = grand_total = 0
    for name, results in sections:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        failed = total - passed
        rate = passed / total if total else 0.0
        rate_str = f"{rate * 100:.0f}%"
        color = "green" if rate == 1.0 else "yellow" if rate >= 0.7 else "red"
        table.add_row(
            name,
            f"[green]{passed}[/green]",
            f"[red]{failed}[/red]" if failed else f"[dim]{failed}[/dim]",
            str(total),
            f"[{color}]{rate_str}[/{color}]",
        )
        grand_pass += passed
        grand_total += total

    grand_rate = grand_pass / grand_total if grand_total else 0.0
    grand_color = "green" if grand_rate == 1.0 else "yellow" if grand_rate >= 0.7 else "red"
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold green]{grand_pass}[/bold green]",
        f"[bold red]{grand_total - grand_pass}[/bold red]",
        f"[bold]{grand_total}[/bold]",
        f"[bold {grand_color}]{grand_rate * 100:.0f}%[/bold {grand_color}]",
    )
    return table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SECTION_DESCRIPTIONS = {
    "EarBlinkDetector": (
        "Measures eye-openness via the Eye Aspect Ratio (EAR) and counts blinks per minute. "
        "Tests use synthetic landmark arrays built to produce exact EAR values, plus a state-machine "
        "check that verifies a blink only fires after the required number of consecutive closed frames."
    ),
    "IrisGazeDetector": (
        "Tracks where the user is looking (center / left / right / up / down) from iris landmarks. "
        "Tests inject synthetic landmark arrays at known horizontal/vertical ratios and confirm the "
        "classifier agrees. Window-fill tests check that 'focused' and 'distracted' labels need enough history."
    ),
    "MediaPipePostureDetector": (
        "Classifies posture as upright / slouched / leaning from shoulder angle and head tilt. "
        "Tests pass PostureData objects with known angles directly to classify() — no pose estimation runs."
    ),
    "SpeechToneClassifier (heuristic)": (
        "Labels speech tone as silent / monotone / calm / stressed from raw audio samples. "
        "Tests generate numpy sine waves and noise bursts with known RMS and zero-crossing rates, "
        "then check the heuristic rules produce the expected label."
    ),
    "LLMStateTracker (parse logic)": (
        "Parses the JSON response from Claude and decides whether a cognitive state transition occurred. "
        "Tests pass raw JSON strings (valid, malformed, missing fields) to _parse_response() directly "
        "— no API call is made."
    ),
    "StateTracker (weighted vote)": (
        "Combines classifier signals via weighted voting to produce a single CognitiveStateLabel. "
        "Tests call _weighted_vote() with lists of ClassifierResult objects and verify the winning "
        "label. Also checks that every known signal label is present in _LABEL_TO_STATE."
    ),
}


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_html_report(summary_data: list[tuple[str, list[CaseResult]]], timestamp: str) -> str:
    grand_pass = sum(1 for _, rs in summary_data for r in rs if r.passed)
    grand_total = sum(len(rs) for _, rs in summary_data)
    grand_rate = grand_pass / grand_total if grand_total else 0.0

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f4f6f9; color: #1a1a2e; padding: 24px; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .subtitle { color: #555; font-size: 0.85rem; margin-top: 4px; }
    .header { background: #1a1a2e; color: #fff; border-radius: 10px;
              padding: 24px 28px; margin-bottom: 24px; }
    .header h1 { color: #fff; }
    .header .subtitle { color: #aac4e0; }
    .grand-badge { display:inline-block; margin-top:12px; padding: 6px 16px;
                   border-radius: 20px; font-weight: 700; font-size: 1rem; }
    .badge-green  { background:#16a34a; color:#fff; }
    .badge-yellow { background:#d97706; color:#fff; }
    .badge-red    { background:#dc2626; color:#fff; }

    /* how-it-works box */
    .how-box { background:#fff; border-left: 4px solid #3b82f6;
               border-radius: 8px; padding: 18px 22px; margin-bottom: 24px;
               box-shadow: 0 1px 4px rgba(0,0,0,.07); }
    .how-box h2 { font-size: 1rem; color: #1d4ed8; margin-bottom: 10px; }
    .how-box p  { font-size: 0.88rem; color: #374151; line-height: 1.6; margin-bottom: 10px; }
    .how-box ul { margin: 8px 0 0 18px; font-size: 0.88rem; color: #374151; line-height: 1.8; }
    .pipeline { display:flex; align-items:center; flex-wrap:wrap; gap:6px;
                margin: 10px 0 14px; font-size: 0.82rem; }
    .pipe-step { background:#e2e8f0; color:#1e293b; padding:4px 10px;
                 border-radius:6px; font-weight:600; white-space:nowrap; }
    .pipe-step.highlight { background:#1d4ed8; color:#fff; }
    .pipe-arrow { color:#94a3b8; font-size:1rem; }
    .how-table { width:100%; border-collapse:collapse; font-size:0.83rem;
                 margin: 10px 0 14px; }
    .how-table th { background:#1e293b; color:#cbd5e1; padding:8px 12px;
                    text-align:left; font-weight:600; }
    .how-table td { padding:8px 12px; border-bottom:1px solid #e2e8f0;
                    color:#374151; vertical-align:top; line-height:1.5; }
    .how-table tr:nth-child(even) td { background:#f8fafc; }
    .how-table td:first-child { font-weight:600; color:#1e293b; white-space:nowrap; }
    .how-box code { background:#f1f5f9; color:#1d4ed8; padding:1px 5px;
                    border-radius:4px; font-size:0.82rem; }

    /* summary strip */
    .summary-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
    .summary-card { background: #fff; border-radius: 8px; padding: 14px 18px;
                    flex: 1 1 180px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
                    border-top: 4px solid #ccc; }
    .summary-card.all-pass  { border-color: #16a34a; }
    .summary-card.some-fail { border-color: #d97706; }
    .summary-card.all-fail  { border-color: #dc2626; }
    .summary-card .sc-name  { font-size: 0.75rem; font-weight: 600; color: #555;
                               text-transform: uppercase; letter-spacing: .04em; }
    .summary-card .sc-rate  { font-size: 1.6rem; font-weight: 800; margin: 4px 0 2px; }
    .summary-card .sc-sub   { font-size: 0.78rem; color: #777; }
    .green { color: #16a34a; } .yellow { color: #d97706; } .red { color: #dc2626; }

    /* section */
    .section { background: #fff; border-radius: 10px; margin-bottom: 28px;
               box-shadow: 0 1px 4px rgba(0,0,0,.07); overflow: hidden; }
    .section-header { background: #1e293b; color: #e2e8f0; padding: 14px 20px; }
    .section-header h2 { font-size: 1rem; font-weight: 700; }
    .section-header .src { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
    .section-desc { padding: 14px 20px; background: #f8fafc;
                    border-bottom: 1px solid #e2e8f0;
                    font-size: 0.85rem; color: #374151; line-height: 1.6; }
    .error-box { padding: 16px 20px; color: #dc2626; font-size: 0.88rem; }

    /* table */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
    thead th { background: #1e293b; color: #cbd5e1; padding: 10px 12px;
               text-align: left; white-space: nowrap; font-weight: 600; }
    tbody tr:nth-child(even) { background: #f8fafc; }
    tbody tr:hover { background: #eff6ff; }
    tbody td { padding: 9px 12px; border-bottom: 1px solid #e9ecef;
               vertical-align: top; color: #1a1a2e; }
    td.pass { color: #15803d; font-weight: 700; }
    td.fail { color: #dc2626; font-weight: 700; }
    td.expected { color: #1d4ed8; }
    td.actual-pass { color: #15803d; font-weight: 600; }
    td.actual-fail { color: #dc2626; font-weight: 600; }
    td.conf { color: #555; font-variant-numeric: tabular-nums; }
    td.note { color: #6b7280; font-size: 0.78rem; }
    """

    def rate_class(rate: float) -> str:
        if rate == 1.0: return "all-pass"
        if rate >= 0.7: return "some-fail"
        return "all-fail"

    def rate_color(rate: float) -> str:
        if rate == 1.0: return "green"
        if rate >= 0.7: return "yellow"
        return "red"

    # --- summary cards ---
    cards_html = ""
    for name, results in summary_data:
        p = sum(1 for r in results if r.passed)
        t = len(results)
        rate = p / t if t else 0.0
        pct = f"{rate*100:.0f}%"
        rc = rate_class(rate)
        color = rate_color(rate)
        cards_html += (
            f'<div class="summary-card {rc}">'
            f'<div class="sc-name">{_html_escape(name)}</div>'
            f'<div class="sc-rate {color}">{pct}</div>'
            f'<div class="sc-sub">{p} / {t} passed</div>'
            f'</div>'
        )

    grand_color = rate_color(grand_rate)
    grand_badge = f'<span class="grand-badge badge-{grand_color}">{grand_pass}/{grand_total} tests passed ({grand_rate*100:.0f}%)</span>'

    # --- how it works ---
    how_html = """
    <div class="how-box">
      <h2>How these tests work — no real person required</h2>
      <p>The full production pipeline looks like this:</p>
      <div class="pipeline">
        <span class="pipe-step">Camera</span>
        <span class="pipe-arrow">&#8594;</span>
        <span class="pipe-step">MediaPipe</span>
        <span class="pipe-arrow">&#8594;</span>
        <span class="pipe-step highlight">landmarks / blendshapes</span>
        <span class="pipe-arrow">&#8594;</span>
        <span class="pipe-step highlight">Classifier</span>
        <span class="pipe-arrow">&#8594;</span>
        <span class="pipe-step">label</span>
      </div>
      <p>These tests <strong>skip the Camera and MediaPipe steps entirely</strong> and inject
         hand-crafted numbers directly at the classifier stage. MediaPipe is Google&rsquo;s
         library — we trust it works. What we&rsquo;re testing is: <em>given the numbers
         MediaPipe would produce, does our logic classify them correctly?</em></p>
      <table class="how-table">
        <thead><tr><th>Classifier</th><th>What the test injects instead of a real person</th></tr></thead>
        <tbody>
          <tr>
            <td>Expression</td>
            <td>A hand-crafted Python dict of blendshape weights, e.g.
                <code>{"browDownLeft": 0.9, "mouthSmileLeft": 0.3}</code>.
                MediaPipe normally produces this dict from a camera frame — we just write it directly.</td>
          </tr>
          <tr>
            <td>Blink / Gaze</td>
            <td>A synthetic 478&times;3 numpy array of landmark coordinates. Only the handful of
                indices that matter (eye corners, iris centre) are set to mathematically exact
                positions; everything else is zero. This gives us a known EAR value or gaze ratio
                without needing a face.</td>
          </tr>
          <tr>
            <td>Posture</td>
            <td>A <code>PostureData(shoulder_angle=20.0, head_tilt=5.0, is_slouching=True)</code>
                dataclass passed straight to <code>classify()</code>. No pose estimation runs.</td>
          </tr>
          <tr>
            <td>Speech tone</td>
            <td>Numpy sine waves and noise arrays (<code>np.sin</code>, <code>np.random</code>)
                at known amplitudes, so the RMS and zero-crossing rate are predictable.</td>
          </tr>
          <tr>
            <td>LLM state tracker</td>
            <td>Raw JSON strings passed directly to the response-parsing function.
                No API call is made.</td>
          </tr>
          <tr>
            <td>State tracker</td>
            <td>Lists of <code>ClassifierResult</code> objects passed to the weighted-vote
                function directly.</td>
          </tr>
        </tbody>
      </table>
      <p style="margin-top:14px;"><strong>Column guide:</strong></p>
      <ul>
        <li><strong>Inputs</strong> — what was passed to the classifier</li>
        <li><strong>Expected</strong> — the label it <em>should</em> return for those inputs</li>
        <li><strong>Actual</strong> — what it actually returned at run-time</li>
        <li><strong>Confidence</strong> — the classifier&rsquo;s own score (&#8212; means N/A)</li>
        <li><strong>Status</strong> — PASS if Actual == Expected, FAIL otherwise</li>
        <li><strong>Note</strong> — why those inputs should produce that label</li>
      </ul>
    </div>
    """

    # --- sections ---
    sections_html = ""
    for name, results in summary_data:
        p = sum(1 for r in results if r.passed)
        t = len(results)
        rate = p / t if t else 0.0
        color = rate_color(rate)
        desc = SECTION_DESCRIPTIONS.get(name, "")

        rows = ""
        for r in results:
            status_cls = "pass" if r.passed else "fail"
            status_txt = "✓ PASS" if r.passed else "✗ FAIL"
            actual_cls = "actual-pass" if r.passed else "actual-fail"
            conf_str = f"{r.confidence:.3f}" if r.confidence >= 0.0 else "—"
            rows += (
                f"<tr>"
                f"<td>{_html_escape(r.description)}</td>"
                f"<td>{_html_escape(r.inputs)}</td>"
                f'<td class="expected">{_html_escape(r.expected)}</td>'
                f'<td class="{actual_cls}">{_html_escape(r.actual)}</td>'
                f'<td class="conf">{conf_str}</td>'
                f'<td class="{status_cls}">{status_txt}</td>'
                f'<td class="note">{_html_escape(r.note)}</td>'
                f"</tr>"
            )

        no_results = '<div class="error-box">No results produced (module failed to import or raised an error).</div>' if not results else ""

        sections_html += f"""
        <div class="section">
          <div class="section-header">
            <h2>{_html_escape(name)}
              <span class="{color}" style="font-size:0.85rem; margin-left:12px;">{p}/{t} passed</span>
            </h2>
          </div>
          {'<div class="section-desc">' + _html_escape(desc) + '</div>' if desc else ''}
          {no_results}
          {'<div class="tbl-wrap"><table><thead><tr>'
           '<th>Test Case</th><th>Inputs</th><th>Expected</th>'
           '<th>Actual</th><th>Confidence</th><th>Status</th><th>Note</th>'
           f'</tr></thead><tbody>{rows}</tbody></table></div>' if results else ''}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CognitiveSense · Validation Report</title>
<style>{css}</style>
</head>
<body>
  <div class="header">
    <h1>CognitiveSense &middot; Classifier Validation Report</h1>
    <div class="subtitle">{timestamp}</div>
    {grand_badge}
  </div>

  {how_html}

  <div class="summary-grid">{cards_html}</div>

  {sections_html}
</body>
</html>
"""


def main() -> None:
    console = Console()
    report_path = os.path.join(os.path.dirname(__file__), "validation_report.html")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    console.print()
    console.print(Panel(
        Text("CognitiveSense  ·  Classifier Validation Report", justify="center", style="bold cyan"),
        subtitle=f"[dim]{timestamp}[/dim]",
        border_style="bright_blue",
    ))
    console.print()

    sections = [
        ("EarBlinkDetector",                  validate_blink,             "vision/blink_detector.py"),
        ("IrisGazeDetector",                  validate_gaze,              "vision/eye_movement_detector.py"),
        ("MediaPipePostureDetector",          validate_posture,           "vision/posture_detector.py"),
        ("SpeechToneClassifier (heuristic)",  validate_speech_tone,       "audio/speech_tone_classifier.py"),
        ("LLMStateTracker (parse logic)",     validate_llm_state_tracker, "state/state_tracker.py"),
        ("StateTracker (weighted vote)",      validate_state_tracker,     "state/state_tracker.py"),
    ]

    summary_data: list[tuple[str, list[CaseResult]]] = []

    # Collect all results first
    for name, validate_fn, source in sections:
        try:
            results = validate_fn()
        except Exception as exc:
            import traceback
            print(f"ERROR running {name}: {exc}")
            print(traceback.format_exc())
            results = []
        summary_data.append((name, results))

    # Write HTML before touching the terminal (avoids Windows encoding crashes)
    html = _build_html_report(summary_data, timestamp)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Print terminal summary (best-effort; skip on encoding errors)
    try:
        for (name, results), (_, _, source) in zip(summary_data, sections):
            console.print(Rule(f"[bold cyan]{name}[/bold cyan]  [dim]{source}[/dim]"))
            console.print()
            if results:
                console.print(_build_table(name, results))
            else:
                console.print("[yellow]No results produced.[/yellow]")
            console.print()

        console.print(Rule("[bold magenta]Summary[/bold magenta]"))
        console.print()
        console.print(_build_summary(summary_data))
        console.print()
    except Exception:
        pass  # terminal encoding issue on Windows — HTML is already saved

    print(f"Report saved -> {report_path}")

    total_failed = sum(
        1 for _, results in summary_data for r in results if not r.passed
    )
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
