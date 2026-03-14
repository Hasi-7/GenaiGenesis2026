import { useEffect, useRef, useState } from 'react';

type CognitiveLabel = 'FOCUSED' | 'FATIGUED' | 'STRESSED' | 'DISTRACTED' | 'UNKNOWN';

type Signal = {
  label: string;
  confidence: number;
};

type CognitiveState = {
  label: CognitiveLabel;
  confidence: number;
  signals: Signal[];
};

type Feedback = {
  text: string;
  label?: string;
  timestamp: number;
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

const prettifySignal = (value: string) => {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
};

export const App = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const liveModeRef = useRef(true);
  const transportConnectionRef = useRef<TransportState['connection']>('disconnected');
  const feedbackTimerRef = useRef<number | null>(null);
  const lastStateLabelRef = useRef<CognitiveLabel>('UNKNOWN');

  const [cogState, setCogState] = useState<CognitiveState>({
    label: 'UNKNOWN',
    confidence: 0,
    signals: [],
  });
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [transport, setTransport] = useState<TransportState>({
    connection: 'disconnected',
    host: '127.0.0.1',
    port: 9100,
  });
  const [streamReady, setStreamReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [liveMode, setLiveMode] = useState(true);
  const [videoSeen, setVideoSeen] = useState(false);
  const [audioSeen, setAudioSeen] = useState(false);
  const [recentEvents, setRecentEvents] = useState<string[]>([]);

  const pushEvent = (text: string) => {
    setRecentEvents((current) => [text, ...current].slice(0, 4));
  };

  const debugLog = (message: string, data?: unknown) => {
    window.cognitiveSense.debugLog(message, data);
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

        await video.play();
        if (!cancelled) {
          setStreamReady(true);
        }
      } catch {
        if (!cancelled) {
          setMediaError('Could not access the camera.');
          debugLog('camera:getUserMedia:failed');
        }
      }
    };

    void window.cognitiveSense.getTransportState().then((nextTransport) => {
      if (!cancelled) {
        setTransport(nextTransport);
      }
    });

    pushEvent('Audio uplink temporarily disabled while stabilizing video.');
    debugLog('audio:capture:disabled-for-stability');
    void startVideo();

    return () => {
      cancelled = true;
      if (feedbackTimerRef.current !== null) {
        clearTimeout(feedbackTimerRef.current);
        feedbackTimerRef.current = null;
      }
      videoStreamRef.current?.getTracks().forEach((track) => track.stop());
      videoStreamRef.current = null;
    };
  }, []);

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
          signals: Array.isArray(payload.signals)
            ? payload.signals.map((signal) => ({
                label: String((signal as { label?: string }).label ?? 'unknown'),
                confidence: Number((signal as { confidence?: number }).confidence ?? 0),
              }))
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
      }

      if (payload.type === 'feedback') {
        const nextFeedback: Feedback = {
          text: String(payload.text ?? ''),
          label: payload.label ? String(payload.label) : undefined,
          timestamp: Number(payload.timestamp ?? Date.now() / 1000) * 1000,
        };
        setFeedback(nextFeedback);
        pushEvent(`Return: ${nextFeedback.text}`);
        void window.cognitiveSense.notify('CognitiveSense', nextFeedback.text);
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
            <button className="secondary-button" onClick={() => void window.cognitiveSense.quit()}>
              Kill app
            </button>
          </div>
        </div>

        <div className="panel signal-panel">
          <div className="panel-title">Signals</div>
          <div className="signal-list">
            {cogState.signals.length > 0 ? (
              cogState.signals.slice(0, 5).map((signal) => (
                <div className="signal-row" key={`${signal.label}-${signal.confidence}`}>
                  <span>{prettifySignal(signal.label)}</span>
                  <strong>{Math.round(signal.confidence * 100)}%</strong>
                </div>
              ))
            ) : (
              <div className="empty-copy">No server-side signal bundle yet.</div>
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
            <strong>{audioSeen ? 'Live' : 'Off'}</strong>
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
