# Cognitive Sense Writeup

## Overview

Cognitive Sense is an ambient AI system for real-time cognitive state detection.
The goal is to recognize altered mental states that people often miss in the moment,
then surface timely and actionable feedback before performance or wellbeing degrades.

The project supports two product environments:

1. Desktop environment
   - User works at a computer.
   - Camera and optional microphone are captured locally by the desktop shell.
   - The desktop app displays live cognitive state, indicators, and LLM feedback.

2. Hardware mirror environment
   - User stands in front of a smart mirror or other Raspberry Pi camera/display setup.
   - The Pi streams camera frames to the same backend used by the desktop system.
   - The mirror can show the same cognitive state, indicator, recommendation, and feedback output.

## Signals Analyzed

The system combines multiple low-level signals into one cognitive state estimate:

- facial expression
- posture
- blink rate
- eye engagement / gaze stability
- speech tone when audio is available

These signals are classified independently and then fused into a smoothed state label:

- focused
- fatigued
- stressed
- distracted

## Architecture Flow

The runtime flow follows the whiteboard design:

1. Sensors
   - camera
   - optional microphone

2. Signal processing
   - frame extraction
   - audio feature extraction when available

3. Classifier stage
   - blink detector
   - gaze detector
   - facial expression classifier
   - posture detector
   - speech tone classifier

4. State tracking
   - classifier results are fused into a smoothed cognitive state
   - state transitions are detected over time rather than on single-frame spikes

5. Event trigger and reasoning
   - when a meaningful transition occurs, a frame plus state context is sent to the LLM
   - the LLM produces higher-level feedback for the user

6. User-facing output
   - current state
   - confidence score
   - primary indicators
   - actionable recommendations
   - LLM feedback text

## Current Implementation

The system now uses one unified remote ingress server on port `9000`.

- Electron and Raspberry Pi clients both connect to the same backend listener.
- Each remote connection gets its own isolated analysis pipeline.
- Responses are returned to the same device that provided the media.
- The local webcam desktop mode remains separate for direct machine use.

This simplifies the architecture significantly:

- one network ingress instead of separate desktop and Pi servers
- one wire protocol family for video, audio, and control
- one per-session pipeline model instead of global source arbitration

## Wire Protocol

The wire format preserves the existing media packets:

- `CSJ1`: JPEG video frame
- `CSA1`: float32 PCM audio chunk
- `CSM1`: control / telemetry packet

The media packet framing remains unchanged.

`CSM1` is used for:

- session hello / capability advertisement
- structured binary state telemetry
- LLM feedback text

This keeps the C client simple while still allowing richer feedback on both the desktop app and Raspberry Pi display.

## Example Output

Example output for a fatigued user might include:

- cognitive fatigue detected
- confidence: 81%
- indicators:
  - blink rate elevated
  - posture slouched
  - speech tone monotone
- recommendations:
  - take a 10 minute break
  - hydrate
  - stretch and reset posture
- LLM feedback:
  - short contextual advice based on the current transition and recent signals

## Why This Design

This design matches the product intent:

- the desktop app and mirror are separate surfaces, but they share the same analysis core
- audio is optional rather than architecture-defining
- telemetry and feedback can reach both environments
- the system remains compatible with lightweight Pi clients while still supporting richer desktop behavior

## Next Steps

High-value next steps after this milestone:

1. strengthen session-level logging and diagnostics
2. add richer Pi display rendering beyond terminal text
3. improve structured indicator ranking using more classifier-specific thresholds
4. personalize recommendations based on repeated state patterns
5. add cost controls and policy for multiple simultaneous remote sessions
