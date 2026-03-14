# CognitiveSense Implementation Plan

## Context

Build the CognitiveSense prototype — an AI system that detects cognitive fatigue and stress from webcam + microphone in real time. Two environments (Desktop with webcam+mic, Mirror with Raspberry Pi camera, no mic) share one codebase. The goal is a clean modular Python architecture with placeholder classifiers so the team can plug in real models later.

---

## Step 0: Project Setup (DONE)

- Python 3.11.15 via uv
- Dependencies installed: `opencv-python`, `mediapipe`, `numpy`, `sounddevice`
- Dev tools: `pyright` (strict mode), `ruff`
- Directory structure created with `__init__.py` in each package

---

## Step 1: Data Models and Interfaces (DONE)

Shared types and protocols are defined in `server/models/`. These are the contracts that all modules must implement. **Read these files before starting any module.**

- **`server/models/types.py`** — All dataclasses and enums
- **`server/models/protocols.py`** — All Protocol classes defining module interfaces

### Dataclasses

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
| `LLMResponse` | `feedback_text`, `suggestions`, `estimated_cost_usd`, `timestamp` |
| `CostTracker` | `total_spent_usd`, `cap_usd`, `call_count` — with `is_budget_exceeded` property |
| `PipelineConfig` | All tunables + `desktop()` / `mirror()` factory methods |

Enums: `CognitiveStateLabel` (FOCUSED, FATIGUED, STRESSED, DISTRACTED, UNKNOWN), `Environment` (DESKTOP, MIRROR).

### Protocols (Interfaces)

Each module must implement the corresponding protocol from `server/models/protocols.py`.

#### Input Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `CameraSource` | `input/camera_adapter.py` | `read_frame() -> NDArray[np.uint8] \| None`, `release()`, `is_opened() -> bool`, `frame_width` (property), `frame_height` (property) |
| `MicSource` | `input/mic_adapter.py` | `start()`, `stop()`, `get_latest_chunk() -> NDArray[np.float32] \| None`, `is_recording` (property) |

#### Vision Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `FaceLandmarkSource` | `vision/face_landmarks.py` | `detect(frame_rgb) -> list[NDArray[np.float32]] \| None`, `close()` |
| `BlinkDetectorProtocol` | `vision/blink_detector.py` | `detect(landmarks) -> BlinkData`, `classify(BlinkData) -> ClassifierResult`, `reset()` |
| `EyeMovementDetectorProtocol` | `vision/eye_movement_detector.py` | `detect(landmarks) -> GazeData`, `classify(GazeData) -> ClassifierResult` |
| `ExpressionClassifierProtocol` | `vision/facial_expression_classifier.py` | `classify(landmarks) -> ClassifierResult` |
| `PostureDetectorProtocol` | `vision/posture_detector.py` | `detect(frame_rgb) -> PostureData \| None`, `classify(PostureData) -> ClassifierResult`, `close()` |

#### Audio Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `SpeechToneClassifierProtocol` | `audio/speech_tone_classifier.py` | `classify(audio_chunk, sample_rate=16000) -> ClassifierResult` |

#### State Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `StateTrackerProtocol` | `state/state_tracker.py` | `add_frame(FrameAnalysis)`, `get_current_state() -> CognitiveState`, `detect_transition() -> StateTransition \| None`, `get_recent_analyses(seconds) -> list[FrameAnalysis]` |

#### Reasoning Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `ReasoningEngine` | `reasoning/llm_engine.py` | `request_feedback(LLMRequest) -> LLMResponse \| None` |

#### UI Layer

| Protocol | File to implement | Methods |
|----------|-------------------|---------|
| `DesktopRenderer` | `ui/desktop_ui.py` | `render(frame, state, llm_response=None)`, `should_quit() -> bool`, `destroy()` |
| `MirrorRenderer` | `ui/mirror_ui.py` | `render(state, llm_response=None)`, `should_quit() -> bool`, `destroy()` |

---

## Step 2: Input Layer

### `server/input/camera_adapter.py`
- Implement `CameraSource` protocol
- Desktop: wrap `cv2.VideoCapture(camera_index)`
- Mirror: try-import `picamera2`, fallback to OpenCV
- `read_frame()` returns BGR frames

### `server/input/mic_adapter.py`
- Implement `MicSource` protocol
- Use `sounddevice.InputStream` callback in background thread
- Audio chunks written to `queue.Queue[NDArray[np.float32]]` (maxsize=2)
- `get_latest_chunk()` is non-blocking, drains queue and returns latest

---

## Step 3: Vision Pipeline

### `server/vision/face_landmarks.py`
- Implement `FaceLandmarkSource` protocol
- Wraps MediaPipe Face Mesh (`refine_landmarks=True` for iris, 478 landmarks)
- Returns list of landmark arrays, each shape (478, 3) with normalized (x, y, z)

### `server/vision/blink_detector.py`
- Implement `BlinkDetectorProtocol`
- **EAR formula:** `(||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)`
- Right eye landmarks: 33, 160, 158, 133, 153, 144
- Left eye landmarks: 362, 385, 387, 263, 380, 373
- Blink = EAR < 0.21 for 3+ consecutive frames
- Tracks blinks/minute via timestamped deque (60s window)
- `classify()` maps rate to labels: normal (15-20/min), fatigued (>25), stressed (<10)

### `server/vision/eye_movement_detector.py`
- Implement `EyeMovementDetectorProtocol`
- Uses iris center landmarks (468 left, 473 right) relative to eye corners
- Computes horizontal/vertical gaze ratio mapped to -1.0..1.0
- `classify()`: sustained center = "focused", frequent off-center = "distracted"

### `server/vision/posture_detector.py`
- Implement `PostureDetectorProtocol`
- Wraps MediaPipe Pose
- Uses shoulder landmarks (11, 12), ear (7, 8), nose (0)
- `classify()`: slouching (shoulder angle deviation >15deg), leaning (head tilt >20deg), upright

### `server/vision/facial_expression_classifier.py`
- Implement `ExpressionClassifierProtocol`
- Placeholder heuristics from landmark geometry (mouth openness, brow position)
- Returns labels: neutral, tense, relaxed

---

## Step 4: Audio — `server/audio/speech_tone_classifier.py`

- Implement `SpeechToneClassifierProtocol`
- Placeholder using numpy-only features: RMS energy, zero-crossing rate, spectral centroid
- Returns labels: calm, stressed, monotone, silent
- Skipped entirely when mic is disabled

---

## Step 5: State Tracker — `server/state/state_tracker.py`

- Implement `StateTrackerProtocol`
- Maintains `deque[FrameAnalysis]` pruned to 10-second window
- **Change detection:** splits window into first half (T-10s to T-5s) and second half (T-5s to T)
- Each half: aggregate classifier labels -> weighted vote -> dominant `CognitiveStateLabel`
- Label mapping: fatigued/slouching/monotone->FATIGUED, stressed/tense->STRESSED, distracted/left/right->DISTRACTED, focused/upright/normal/center/calm->FOCUSED
- **Transition emitted when:** dominant label differs between halves AND new confidence > 0.6 AND confidence delta > 0.3 AND both halves sufficiently populated
- 5-second debounce between transitions

---

## Step 6: LLM Engine — `server/reasoning/llm_engine.py`

- Implement `ReasoningEngine` protocol
- Mock implementation returning canned feedback based on `CognitiveStateLabel`
- **Rate limiter:** 30-second cooldown between calls + $5 cumulative cap
- Cost estimate: ~$0.02/call -> ~250 calls before cap
- `request_feedback(LLMRequest) -> LLMResponse | None` (None if rate limited)

---

## Step 7: UI Layer

### `server/ui/desktop_ui.py`
- Implement `DesktopRenderer` protocol
- OpenCV overlay: status bar with state + confidence, per-signal debug labels, LLM feedback text, budget/FPS counters
- Color-coded: green=FOCUSED, orange=FATIGUED, red=STRESSED, yellow=DISTRACTED
- Quit on 'q' key

### `server/ui/mirror_ui.py`
- Implement `MirrorRenderer` protocol
- Terminal output with ANSI clear, throttled to 1 update/second
- Quit via KeyboardInterrupt

---

## Step 8: Pipeline Controller — `server/core/pipeline_controller.py`

Main loop at `config.target_fps` (15 desktop, 10 mirror):

1. `camera.read_frame()` -> BGR frame -> convert to RGB
2. `face_detector.detect(rgb)` -> landmarks
3. If landmarks: blink detect+classify, gaze detect+classify, expression classify
4. `posture_detector.detect(rgb)` -> posture classify
5. If mic enabled: `mic.get_latest_chunk()` -> speech tone classify
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

- **Main thread:** camera capture -> all vision -> state -> LLM -> UI render
- **Audio thread (desktop only):** `sounddevice` callback -> `queue.Queue` -> main thread reads non-blocking
- Only shared state: `queue.Queue` (thread-safe by design)

---

## Pyright Strict Mode

- Keep `typeCheckingMode = "strict"` for our code
- Add `reportMissingModuleSource = false` for cv2/mediapipe
- Use `# type: ignore[import-untyped]` on cv2/mediapipe imports (~7-10 comments total, confined to 4 wrapper files)
- All wrapper functions explicitly annotate return types

---

## Implementation Order

Modules can be worked on independently as long as you implement the protocol from `models/protocols.py`.

1. **Step 0 + Step 1** (setup + types/protocols) — DONE
2. **Step 2** (input layer) — no dependencies on other steps
3. **Step 3** (vision pipeline) — no dependencies on other steps
4. **Step 4** (audio) — no dependencies on other steps
5. **Step 5** (state tracker) — depends on types only
6. **Step 6** (LLM engine) — depends on types only
7. **Step 7** (UI) — depends on types only
8. **Step 8 + Step 9** (pipeline controller + main.py) — depends on all above

Steps 2-7 can be done **in parallel** by different team members.

---

## Verification

1. `uv run pyright .` — zero errors on our code
2. `uv run ruff check .` — zero lint violations
3. `uv run python main.py` — webcam opens, face mesh overlays, debug panel shows labels+confidence, blink counter ticks, state transitions trigger mock LLM feedback
4. Press 'q' to quit cleanly
5. `uv run python main.py mirror` — terminal output mode (requires Pi or falls back to OpenCV)