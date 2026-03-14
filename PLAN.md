# CognitiveSense Implementation Plan

## Context

Build the CognitiveSense prototype — an AI system that detects cognitive fatigue and stress from webcam + microphone in real time. Two environments (Desktop with webcam+mic, Mirror with Raspberry Pi camera, no mic) share one codebase. The goal is a clean modular Python architecture with placeholder classifiers so the team can plug in real models later.

---

## Step 0: Project Setup

**Update `server/.python-version`** to `3.11.15`

**Update `server/pyproject.toml`:**
- `requires-python = ">=3.11,<3.12"`
- `target-version = "py311"` (ruff)
- `pythonVersion = "3.11"` (pyright)
- Add `reportMissingModuleSource = false` to `[tool.pyright]`
- Add dependencies: `opencv-python>=4.8.0,<5.0`, `mediapipe>=0.10.9,<0.11`, `numpy>=1.24.0,<2.0`, `sounddevice>=0.4.6,<0.5`

**Run:** `uv python install 3.11.15 && uv venv --python 3.11.15 && uv sync`

**Create directory structure** with `__init__.py` in each package:
```
server/
├── models/
├── core/
├── input/
├── vision/
├── audio/
├── state/
├── reasoning/
└── ui/
```

---

## Step 1: Data Models — `server/models/types.py`

All structured data as dataclasses with `slots=True`. Key types:

| Dataclass | Purpose |
|-----------|---------|
| `ClassifierResult` | `label: str`, `confidence: float` — universal classifier output |
| `BlinkData` | `ear_left`, `ear_right`, `ear_average`, `blink_detected`, `blinks_per_minute` |
| `GazeData` | `horizontal_ratio`, `vertical_ratio`, `direction` |
| `PostureData` | `shoulder_angle`, `head_tilt`, `is_slouching` |
| `FrameAnalysis` | `timestamp` + optional results from each classifier |
| `CognitiveState` | `label: CognitiveStateLabel`, `confidence`, `contributing_signals` |
| `StateTransition` | `previous_state`, `new_state`, `transition_time` |
| `LLMRequest` | `frame_jpeg_bytes`, `current_state`, `transition`, `recent_analyses` |
| `LLMResponse` | `feedback_text`, `suggestions`, `estimated_cost_usd` |
| `CostTracker` | `total_spent_usd`, `cap_usd`, `call_count` — with `is_budget_exceeded` property |
| `PipelineConfig` | All tunables + `desktop()` / `mirror()` factory methods |

Enums: `CognitiveStateLabel` (FOCUSED, FATIGUED, STRESSED, DISTRACTED, UNKNOWN), `Environment` (DESKTOP, MIRROR).

---

## Step 2: Input Layer

### `server/input/camera_adapter.py`
- `CameraAdapter` class wrapping `cv2.VideoCapture`
- `read_frame() -> NDArray[np.uint8] | None` returns BGR frames
- Desktop uses OpenCV, Mirror uses picamera2 (try-import with fallback)

### `server/input/mic_adapter.py`
- `MicAdapter` class using `sounddevice.InputStream` in a background thread
- Audio chunks written to `queue.Queue[NDArray[np.float32]]` (maxsize=2)
- `get_latest_chunk()` is non-blocking, returns latest or None
- `start()` / `stop()` control the background thread

---

## Step 3: Vision Pipeline

### `server/vision/face_landmarks.py`
- Wraps MediaPipe Face Mesh (`refine_landmarks=True` for iris, 478 landmarks)
- `detect(frame_rgb) -> list[NDArray[np.float32]] | None`

### `server/vision/blink_detector.py`
- **EAR formula:** `(||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)`
- Right eye landmarks: 33, 160, 158, 133, 153, 144
- Left eye landmarks: 362, 385, 387, 263, 380, 373
- Blink = EAR < 0.21 for 3+ consecutive frames
- Tracks blinks/minute via timestamped deque (60s window)
- `classify()` maps rate to labels: normal (15-20/min), fatigued (>25), stressed (<10)

### `server/vision/eye_movement_detector.py`
- Uses iris center landmarks (468 left, 473 right) relative to eye corners
- Computes horizontal/vertical gaze ratio
- `classify()`: sustained center = "focused", frequent off-center = "distracted"

### `server/vision/posture_detector.py`
- Wraps MediaPipe Pose
- Uses shoulder landmarks (11, 12), ear (7, 8), nose (0)
- `classify()`: slouching (shoulder angle deviation >15deg), leaning, upright

### `server/vision/facial_expression_classifier.py`
- Placeholder heuristics from landmark geometry (mouth openness, brow position)
- Returns labels: neutral, tense, relaxed

---

## Step 4: Audio — `server/audio/speech_tone_classifier.py`

- Placeholder using numpy-only features: RMS energy, zero-crossing rate, spectral centroid
- Returns labels: calm, stressed, monotone, silent
- Skipped entirely when mic is disabled

---

## Step 5: State Tracker — `server/state/state_tracker.py`

- Maintains `deque[FrameAnalysis]` pruned to 10-second window
- **Change detection:** splits window into first half (T-10s to T-5s) and second half (T-5s to T)
- Each half: aggregate classifier labels → weighted vote → dominant `CognitiveStateLabel`
- Label mapping: fatigued/slouching/monotone→FATIGUED, stressed/tense→STRESSED, distracted/left/right→DISTRACTED, focused/upright/normal/center/calm→FOCUSED
- **Transition emitted when:** dominant label differs between halves AND new confidence > 0.6 AND confidence delta > 0.3 AND both halves sufficiently populated
- 5-second debounce between transitions

---

## Step 6: LLM Engine — `server/reasoning/llm_engine.py`

- Mock implementation returning canned feedback based on `CognitiveStateLabel`
- **Rate limiter:** 30-second cooldown between calls + $5 cumulative cap
- Cost estimate: ~$0.02/call → ~250 calls before cap
- `request_feedback(LLMRequest) -> LLMResponse | None` (None if rate limited)

---

## Step 7: UI Layer

### `server/ui/desktop_ui.py`
- OpenCV overlay: status bar with state + confidence, per-signal debug labels, LLM feedback text, budget/FPS counters
- Color-coded: green=FOCUSED, orange=FATIGUED, red=STRESSED, yellow=DISTRACTED
- Quit on 'q' key

### `server/ui/mirror_ui.py`
- Terminal output with ANSI clear, throttled to 1 update/second
- Quit via KeyboardInterrupt

---

## Step 8: Pipeline Controller — `server/core/pipeline_controller.py`

Main loop at `config.target_fps` (15 desktop, 10 mirror):

1. `camera.read_frame()` → BGR frame → convert to RGB
2. `face_detector.detect(rgb)` → landmarks
3. If landmarks: blink detect+classify, gaze detect+classify, expression classify
4. `posture_detector.detect(rgb)` → posture classify
5. If mic enabled: `mic.get_latest_chunk()` → speech tone classify
6. Build `FrameAnalysis`, feed to `state_tracker`
7. `state_tracker.detect_transition()` — if transition, JPEG-encode frame, build `LLMRequest`, call `llm_engine`
8. Render UI with state + optional LLM response
9. Frame rate sleep

---

## Step 9: Entry Point — `server/main.py`

- CLI arg: `python main.py [desktop|mirror]` (default: desktop)
- Creates `PipelineConfig.desktop()` or `.mirror()`
- Instantiates and runs `PipelineController`
- Handles KeyboardInterrupt gracefully

---

## Threading Model

- **Main thread:** camera capture → all vision → state → LLM → UI render
- **Audio thread (desktop only):** `sounddevice` callback → `queue.Queue` → main thread reads non-blocking
- Only shared state: `queue.Queue` (thread-safe by design)

---

## Pyright Strict Mode

- Keep `typeCheckingMode = "strict"` for our code
- Add `reportMissingModuleSource = false` for cv2/mediapipe
- Use `# type: ignore[import-untyped]` on cv2/mediapipe imports (~7-10 comments total, confined to 4 wrapper files)
- All wrapper functions explicitly annotate return types

---

## Implementation Order

1. Step 0 (setup) + Step 1 (types)
2. Step 2 (input layer)
3. Step 3 (vision pipeline)
4. Step 4 (audio)
5. Step 5 (state tracker)
6. Step 6 (LLM engine)
7. Step 7 (UI)
8. Step 8 (pipeline controller) + Step 9 (main.py)

---

## Verification

1. `uv run pyright .` — zero errors on our code
2. `uv run ruff check .` — zero lint violations
3. `uv run python main.py` — webcam opens, face mesh overlays, debug panel shows labels+confidence, blink counter ticks, state transitions trigger mock LLM feedback
4. Press 'q' to quit cleanly
5. `uv run python main.py mirror` — terminal output mode (requires Pi or falls back to OpenCV)
