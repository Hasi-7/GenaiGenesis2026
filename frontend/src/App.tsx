import { useEffect, useRef, useState } from 'react';

type CognitiveLabel = 'FOCUSED' | 'FATIGUED' | 'STRESSED' | 'DISTRACTED' | 'UNKNOWN';

type CognitiveState = {
  label: CognitiveLabel;
  confidence: number;
  indicators: string[];
  recommendations: string[];
};

type Feedback = {
  text: string;
  label?: string;
  timestamp: number;
  triggerKind?: string;
  severity?: string;
  shouldNotify?: boolean;
};

type TransportState = {
  connection: 'connecting' | 'connected' | 'disconnected';
  host: string;
  port: number;
  detail?: string;
};

const STATE_COLORS: Record<CognitiveLabel, string> = {
  FOCUSED: '#2fb36d',
  FATIGUED: '#d98a2b',
  STRESSED: '#d65545',
  DISTRACTED: '#c1a126',
  UNKNOWN: '#7b8694',
};

const FRAME_INTERVAL_MS = 200;
const JPEG_QUALITY = 0.68;
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const MIC_SETTING_KEY = 'cognitivesense.mic-enabled';
const AUDIO_SAMPLE_RATE = 16_000;
const AUDIO_FLUSH_INTERVAL_MS = 250;
const NEGATIVE_NOTIFICATION_DELAY_MS = 5_000;
const NEGATIVE_NOTIFICATION_REPEAT_MS = 120_000;

const NEGATIVE_LABELS = new Set<CognitiveLabel>(['FATIGUED', 'STRESSED', 'DISTRACTED']);

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const canvasToBlob = (canvas: HTMLCanvasElement, type: string, quality: number) =>
  new Promise<Blob | null>((resolve) => {
    canvas.toBlob((blob) => resolve(blob), type, quality);
  });

const formatClock = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
};

const buildNegativeNotificationText = (label: CognitiveLabel, durationMs: number) => {
  const seconds = Math.max(5, Math.round(durationMs / 1000));
  if (label === 'STRESSED') {
    return `You have looked stressed for about ${seconds} seconds. Take one slow breath and reset your posture.`;
  }
  if (label === 'FATIGUED') {
    return `You have looked fatigued for about ${seconds} seconds. Take a short break, hydrate, or stretch.`;
  }
  return `You have seemed distracted for about ${seconds} seconds. Try refocusing on one task and clearing distractions.`;
};

export const App = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const liveModeRef = useRef(true);
  const transportConnectionRef = useRef<TransportState['connection']>('disconnected');
  const feedbackTimerRef = useRef<number | null>(null);
  const lastStateLabelRef = useRef<CognitiveLabel>('UNKNOWN');
  const lastRecommendationRef = useRef('');
  const negativeEpisodeRef = useRef<{
    startedAt: number;
    lastNotifiedAt: number;
    latestLabel: CognitiveLabel;
  } | null>(null);

  const [cogState, setCogState] = useState<CognitiveState>({
    label: 'UNKNOWN',
    confidence: 0,
    indicators: [],
    recommendations: [],
  });
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [transport, setTransport] = useState<TransportState>({
    connection: 'disconnected',
    host: '127.0.0.1',
    port: 9000,
  });
  const [streamReady, setStreamReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [liveMode, setLiveMode] = useState(true);
  const [micEnabled, setMicEnabled] = useState(() => {
    try {
      return window.localStorage.getItem(MIC_SETTING_KEY) !== 'off';
    } catch {
      return true;
    }
  });
  const [audioCaptureState, setAudioCaptureState] = useState<
    'disabled' | 'starting' | 'ready' | 'unavailable'
  >('disabled');
  const [videoSeen, setVideoSeen] = useState(false);
  const [audioSeen, setAudioSeen] = useState(false);
  const [recentEvents, setRecentEvents] = useState<string[]>([]);

  const pushEvent = (text: string) => {
    setRecentEvents((current) => [text, ...current].slice(0, 4));
  };

  const debugLog = (message: string, data?: unknown) => {
    window.cognitiveSense.debugLog(message, data);
  };

  const buildAudioWorkletModuleUrl = () => {
    const source = `
      class CognitiveSenseAudioCaptureProcessor extends AudioWorkletProcessor {
        constructor(options) {
          super();
          const chunkMs = options?.processorOptions?.chunkMs ?? 250;
          this.targetSamples = Math.max(1024, Math.round(sampleRate * chunkMs / 1000));
          this.pending = [];
          this.pendingLength = 0;
        }

        process(inputs) {
          const input = inputs[0];
          const channel = input && input[0];
          if (!channel || channel.length === 0) {
            return true;
          }

          const copy = new Float32Array(channel.length);
          copy.set(channel);
          this.pending.push(copy);
          this.pendingLength += copy.length;

          if (this.pendingLength < this.targetSamples) {
            return true;
          }

          const merged = new Float32Array(this.pendingLength);
          let offset = 0;
          for (const chunk of this.pending) {
            merged.set(chunk, offset);
            offset += chunk.length;
          }

          this.pending = [];
          this.pendingLength = 0;
          this.port.postMessage(merged, [merged.buffer]);
          return true;
        }
      }

      registerProcessor('cognitivesense-audio-capture', CognitiveSenseAudioCaptureProcessor);
    `;

    return URL.createObjectURL(new Blob([source], { type: 'application/javascript' }));
  };

  const stopAudioCapture = async () => {
    audioWorkletNodeRef.current?.port.close();
    audioWorkletNodeRef.current?.disconnect();
    audioSourceRef.current?.disconnect();
    audioWorkletNodeRef.current = null;
    audioSourceRef.current = null;

    audioStreamRef.current?.getTracks().forEach((track) => track.stop());
    audioStreamRef.current = null;

    if (audioContextRef.current) {
      await audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
    }

    setAudioSeen(false);
  };

  useEffect(() => {
    liveModeRef.current = liveMode;
  }, [liveMode]);

  useEffect(() => {
    transportConnectionRef.current = transport.connection;
  }, [transport.connection]);

  useEffect(() => {
    let cancelled = false;

    const startVideo = async () => {
      try {
        debugLog('camera:getUserMedia:start');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 12, max: 15 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        videoStreamRef.current = stream;
        const videoTrack = stream.getVideoTracks()[0] ?? null;
        debugLog('camera:getUserMedia:success', {
          label: videoTrack?.label ?? null,
          settings: videoTrack?.getSettings() ?? null,
        });

        const video = videoRef.current;
        if (!video) {
          setMediaError('Camera preview element is unavailable.');
          debugLog('camera:preview-element:missing');
          return;
        }

        video.addEventListener(
          'loadedmetadata',
          () => {
            debugLog('camera:loadedmetadata', {
              width: video.videoWidth,
              height: video.videoHeight,
              readyState: video.readyState,
            });
          },
          { once: true },
        );

        video.addEventListener(
          'playing',
          () => {
            debugLog('camera:playing', {
              width: video.videoWidth,
              height: video.videoHeight,
            });
          },
          { once: true },
        );

        video.srcObject = stream;
        video.playsInline = true;
        video.muted = true;
        videoStreamRef.current = stream;

        await video.play();

        if (!cancelled) {
          setStreamReady(true);
        }
      } catch (error) {
        if (!cancelled) {
          setMediaError('Could not access the camera.');
          debugLog('camera:getUserMedia:failed', {
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
    };

    void startVideo();

    return () => {
      cancelled = true;
      if (feedbackTimerRef.current !== null) {
        clearTimeout(feedbackTimerRef.current);
        feedbackTimerRef.current = null;
      }
      void stopAudioCapture();
      videoStreamRef.current?.getTracks().forEach((track) => track.stop());
      videoStreamRef.current = null;
    };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(MIC_SETTING_KEY, micEnabled ? 'on' : 'off');
    } catch {
      // Ignore storage failures.
    }
  }, [micEnabled]);

  useEffect(() => {
    let cancelled = false;

    const startAudio = async () => {
      try {
        setAudioCaptureState('starting');
        debugLog('audio:getUserMedia:start');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: false,
          audio: {
            channelCount: { ideal: 1, max: 1 },
            sampleRate: { ideal: AUDIO_SAMPLE_RATE },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
          setAudioCaptureState('unavailable');
          debugLog('audio:capture:no-track');
          pushEvent('No microphone track available.');
          return;
        }

        const audioContext = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
        const audioSource = audioContext.createMediaStreamSource(stream);
        const workletUrl = buildAudioWorkletModuleUrl();
        try {
          await audioContext.audioWorklet.addModule(workletUrl);
        } finally {
          URL.revokeObjectURL(workletUrl);
        }

        const audioWorkletNode = new AudioWorkletNode(
          audioContext,
          'cognitivesense-audio-capture',
          {
            numberOfInputs: 1,
            numberOfOutputs: 0,
            channelCount: 1,
            channelCountMode: 'explicit',
            channelInterpretation: 'speakers',
            processorOptions: { chunkMs: AUDIO_FLUSH_INTERVAL_MS },
          },
        );

        audioWorkletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
          if (!liveModeRef.current || transportConnectionRef.current !== 'connected') {
            return;
          }

          const chunk = event.data;
          if (!(chunk instanceof Float32Array) || chunk.length === 0) {
            return;
          }

          const packet = new Float32Array(chunk.length);
          packet.set(chunk);
          window.cognitiveSense.sendAudio(packet.buffer, audioContext.sampleRate, 1);
        };

        audioSource.connect(audioWorkletNode);
        await audioContext.resume();

        audioStreamRef.current = stream;
        audioContextRef.current = audioContext;
        audioSourceRef.current = audioSource;
        audioWorkletNodeRef.current = audioWorkletNode;
        setAudioCaptureState('ready');

        debugLog('audio:capture:enabled', {
          label: audioTracks[0]?.label ?? null,
          sampleRate: audioContext.sampleRate,
          flushIntervalMs: AUDIO_FLUSH_INTERVAL_MS,
          transport: 'audio-worklet',
        });
        pushEvent('Audio uplink enabled.');
      } catch (error) {
        setAudioCaptureState('unavailable');
        debugLog('audio:getUserMedia:failed', {
          message: error instanceof Error ? error.message : String(error),
        });
        pushEvent('Microphone unavailable; continuing with video only.');
      }
    };

    if (!micEnabled) {
      setAudioCaptureState('disabled');
      void stopAudioCapture();
      pushEvent('Microphone uplink disabled.');
      return () => {
        cancelled = true;
      };
    }

    void startAudio();

    return () => {
      cancelled = true;
      void stopAudioCapture();
    };
  }, [micEnabled]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const ensurePlaying = () => {
      if (video.paused && video.srcObject) {
        void video.play().catch(() => undefined);
      }
    };

    video.addEventListener('pause', ensurePlaying);
    return () => {
      video.removeEventListener('pause', ensurePlaying);
    };
  }, []);

  useEffect(() => {
    const unsubscribeTelemetry = window.cognitiveSense.onTelemetry((payload) => {
      if (payload.type === 'state') {
        const label = String(payload.label ?? 'UNKNOWN') as CognitiveLabel;
        const nextState: CognitiveState = {
          label,
          confidence: Number(payload.confidence ?? 0),
          indicators: Array.isArray(payload.indicators)
            ? payload.indicators.map((indicator) => String(indicator))
            : [],
          recommendations: Array.isArray(payload.recommendations)
            ? payload.recommendations.map((item) => String(item))
            : [],
        };

        setCogState(nextState);
        const transportFlags = payload.transport as
          | { video?: boolean; audio?: boolean }
          | undefined;
        setVideoSeen(Boolean(transportFlags?.video));
        setAudioSeen(Boolean(transportFlags?.audio));

        if (lastStateLabelRef.current !== label && label !== 'UNKNOWN') {
          pushEvent(`State: ${label.toLowerCase()}`);
          lastStateLabelRef.current = label;
        }
        if (
          nextState.recommendations.length > 0 &&
          nextState.recommendations[0] !== lastRecommendationRef.current
        ) {
          pushEvent(`Recommend: ${nextState.recommendations[0]}`);
          lastRecommendationRef.current = nextState.recommendations[0];
        }

        const nowMs = Date.now();
        if (NEGATIVE_LABELS.has(label)) {
          if (negativeEpisodeRef.current === null) {
            negativeEpisodeRef.current = {
              startedAt: nowMs,
              lastNotifiedAt: 0,
              latestLabel: label,
            };
          } else {
            negativeEpisodeRef.current.latestLabel = label;
          }

          const episode = negativeEpisodeRef.current;
          const durationMs = nowMs - episode.startedAt;
          if (
            durationMs >= NEGATIVE_NOTIFICATION_DELAY_MS &&
            nowMs - episode.lastNotifiedAt >= NEGATIVE_NOTIFICATION_REPEAT_MS
          ) {
            const notificationText = buildNegativeNotificationText(
              episode.latestLabel,
              durationMs,
            );
            episode.lastNotifiedAt = nowMs;
            pushEvent(`Alert: ${notificationText}`);
            void window.cognitiveSense.notify('CognitiveSense', notificationText);
          }
        } else {
          negativeEpisodeRef.current = null;
        }
      }

      if (payload.type === 'feedback') {
        const nextFeedback: Feedback = {
          text: String(payload.text ?? ''),
          label: payload.label ? String(payload.label) : undefined,
          triggerKind: payload.triggerKind ? String(payload.triggerKind) : undefined,
          severity: payload.severity ? String(payload.severity) : undefined,
          shouldNotify: Boolean(payload.shouldNotify),
          timestamp: Number(payload.timestamp ?? Date.now() / 1000) * 1000,
        };
        setFeedback(nextFeedback);
        pushEvent(`Return: ${nextFeedback.text}`);
        if (feedbackTimerRef.current !== null) {
          clearTimeout(feedbackTimerRef.current);
        }
        feedbackTimerRef.current = window.setTimeout(() => {
          setFeedback(null);
          feedbackTimerRef.current = null;
        }, 12000);
      }
    });

    const unsubscribeTransport = window.cognitiveSense.onTransport((payload) => {
      setTransport(payload);
    });

    void window.cognitiveSense.getTransportState().then((nextTransport) => {
      setTransport(nextTransport);
    });

    return () => {
      unsubscribeTelemetry();
      unsubscribeTransport();
    };
  }, []);

  useEffect(() => {
    if (!streamReady) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;

    let stopped = false;
    let firstWaitLogged = false;
    let firstDrawLogged = false;
    let firstSendLogged = false;

    const setCanvasSize = () => {
      if (canvas.width !== CAPTURE_WIDTH) {
        canvas.width = CAPTURE_WIDTH;
      }
      if (canvas.height !== CAPTURE_HEIGHT) {
        canvas.height = CAPTURE_HEIGHT;
      }
    };

    const captureLoop = async () => {
      debugLog('capture:loop:start');
      while (!stopped) {
        if (
          !video.videoWidth ||
          !video.videoHeight ||
          video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
          if (!firstWaitLogged) {
            firstWaitLogged = true;
            debugLog('capture:waiting-for-video', {
              paused: video.paused,
              readyState: video.readyState,
              width: video.videoWidth,
              height: video.videoHeight,
            });
          }
          await sleep(100);
          continue;
        }

        setCanvasSize();
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (!firstDrawLogged) {
          firstDrawLogged = true;
          debugLog('capture:first-draw', {
            width: canvas.width,
            height: canvas.height,
            currentTime: video.currentTime,
          });
        }

        if (
          liveModeRef.current &&
          transportConnectionRef.current === 'connected'
        ) {
          const blob = await canvasToBlob(canvas, 'image/jpeg', JPEG_QUALITY);
          if (blob && !stopped) {
            const buffer = await blob.arrayBuffer();
            const sent = window.cognitiveSense.sendFrame(
              buffer,
              canvas.width,
              canvas.height,
            );
            if (!firstSendLogged) {
              firstSendLogged = true;
              debugLog('capture:first-send', {
                sent,
                bytes: buffer.byteLength,
                width: canvas.width,
                height: canvas.height,
              });
            }
          }
        }

        await sleep(
          liveModeRef.current && transportConnectionRef.current === 'connected'
            ? FRAME_INTERVAL_MS
            : 100,
        );
      }
    };

    void captureLoop();

    return () => {
      stopped = true;
    };
  }, [streamReady]);

  const stateColor = STATE_COLORS[cogState.label];
  const transportLabel =
    transport.connection === 'connected'
      ? 'Connected'
      : transport.connection === 'connecting'
        ? 'Connecting'
        : 'Offline';
  const audioLabel = !micEnabled
    ? 'Disabled'
    : audioSeen
      ? 'Live'
      : audioCaptureState === 'starting'
        ? 'Starting'
        : audioCaptureState === 'ready'
          ? 'Ready'
          : 'Unavailable';

  if (mediaError) {
    return (
      <main className="app error-shell">
        <div className="error-card">{mediaError}</div>
      </main>
    );
  }

  return (
    <main className="app">
      <video ref={videoRef} className="webcam" autoPlay muted playsInline />
      <canvas ref={canvasRef} className="capture-canvas" />

      <header className="titlebar">
        <div className="titlebar-drag">
          <span className="titlebar-mark" />
          <span className="titlebar-label">CognitiveSense</span>
        </div>
        <div className="titlebar-actions">
          <button className="titlebar-button" onClick={() => void window.cognitiveSense.minimize()}>
            Minimize
          </button>
          <button className="titlebar-button danger" onClick={() => void window.cognitiveSense.quit()}>
            Quit
          </button>
        </div>
      </header>

      <section className="hud hud-left">
        <div className="panel status-panel">
          <div className="badge-row">
            <div className="state-chip" style={{ borderColor: stateColor }}>
              <span className="dot" style={{ backgroundColor: stateColor }} />
              <span>{cogState.label}</span>
              <strong>{Math.round(cogState.confidence * 100)}%</strong>
            </div>
            <div className={`transport-chip ${transport.connection}`}>
              <span className="dot" />
              <span>{transportLabel}</span>
            </div>
          </div>

          <h1>Remote analysis server</h1>
          <p className="support-copy">
            Keep the preview live here, send frames upstream, and wait for the server to decide
            when a return is worth surfacing.
          </p>

          <div className="button-row">
            <button className={`primary-button ${liveMode ? 'active' : ''}`} onClick={() => setLiveMode((value) => !value)}>
              {liveMode ? 'Pause uplink' : 'Resume uplink'}
            </button>
            <button
              className={`secondary-button ${micEnabled ? 'active' : ''}`}
              onClick={() => setMicEnabled((value) => !value)}
            >
              {micEnabled ? 'Mic on' : 'Mic off'}
            </button>
            <button className="secondary-button" onClick={() => void window.cognitiveSense.quit()}>
              Kill app
            </button>
          </div>
        </div>

          <div className="panel signal-panel">
          <div className="panel-title">Indicators</div>
          <div className="signal-list">
            {cogState.indicators.length > 0 ? (
              cogState.indicators.map((indicator) => (
                <div className="signal-row" key={indicator}>
                  <span>{indicator}</span>
                </div>
              ))
            ) : (
              <div className="empty-copy">No primary indicators yet.</div>
            )}
          </div>
        </div>

        <div className="panel signal-panel">
          <div className="panel-title">Recommendations</div>
          <div className="signal-list">
            {cogState.recommendations.length > 0 ? (
              cogState.recommendations.map((recommendation) => (
                <div className="signal-row" key={recommendation}>
                  <span>{recommendation}</span>
                </div>
              ))
            ) : (
              <div className="empty-copy">Recommendations appear after analysis settles.</div>
            )}
          </div>
        </div>
      </section>

      <aside className="hud hud-right">
        <div className="panel meta-panel">
          <div className="meta-row">
            <span>Server</span>
            <strong>{transport.host}:{transport.port}</strong>
          </div>
          <div className="meta-row">
            <span>Video</span>
            <strong>{videoSeen ? 'Live' : 'Waiting'}</strong>
          </div>
          <div className="meta-row">
            <span>Audio</span>
            <strong>{audioLabel}</strong>
          </div>
          <div className="meta-row">
            <span>Mode</span>
            <strong>{liveMode ? 'Remote uplink' : 'Preview only'}</strong>
          </div>
          <div className="meta-row detail-row">
            <span>Status</span>
            <strong>{transport.detail ?? 'Nominal'}</strong>
          </div>
        </div>

        <div className="panel events-panel">
          <div className="panel-title">Recent</div>
          <div className="event-list">
            {recentEvents.length > 0 ? (
              recentEvents.map((entry) => (
                <div className="event-row" key={entry}>
                  {entry}
                </div>
              ))
            ) : (
              <div className="empty-copy">Nothing back from the server yet.</div>
            )}
          </div>
        </div>
      </aside>

      {feedback && (
        <div className="feedback-toast" onClick={() => setFeedback(null)}>
          <div className="toast-kicker">Server return</div>
          <div className="toast-text">{feedback.text}</div>
          <div className="toast-time">{formatClock(feedback.timestamp)}</div>
        </div>
      )}
    </main>
  );
};
