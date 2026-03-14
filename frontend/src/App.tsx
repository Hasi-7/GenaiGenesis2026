import { useEffect, useRef, useState } from 'react';

type CognitiveLabel = 'FOCUSED' | 'FATIGUED' | 'STRESSED' | 'DISTRACTED' | 'UNKNOWN';

type CognitiveState = {
  label: CognitiveLabel;
  confidence: number;
  signals: string[];
};

type Feedback = {
  text: string;
  suggestions: string[];
};

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

const STATE_COLORS: Record<CognitiveLabel, string> = {
  FOCUSED: '#22c55e',
  FATIGUED: '#f97316',
  STRESSED: '#ef4444',
  DISTRACTED: '#eab308',
  UNKNOWN: '#71717a',
};

const WS_URL = 'ws://localhost:8765';
const FRAME_INTERVAL_MS = 100; // ~10 fps
const MAX_WS_BUFFERED_BYTES = 512 * 1024;

type ImageCaptureLike = {
  grabFrame: () => Promise<ImageBitmap>;
};

export const App = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoStreamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const feedbackTimerRef = useRef<number | null>(null);

  const [cogState, setCogState] = useState<CognitiveState>({
    label: 'UNKNOWN',
    confidence: 0,
    signals: [],
  });
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('disconnected');
  const [streamReady, setStreamReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  // Start webcam preview first so audio setup cannot stall video.
  useEffect(() => {
    let cancelled = false;

    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 15, max: 24 },
            facingMode: 'user',
          },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        videoStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current
            .play()
            .then(() => {
              if (!cancelled) {
                setStreamReady(true);
              }
            })
            .catch(() => {
              if (!cancelled) {
                setMediaError('Unable to start camera preview');
              }
            });
          return;
        }
        setStreamReady(true);
      } catch {
        if (!cancelled) setMediaError('Camera access denied');
      }
    };

    void startVideo();

    return () => {
      cancelled = true;
      if (feedbackTimerRef.current !== null) {
        clearTimeout(feedbackTimerRef.current);
        feedbackTimerRef.current = null;
      }
      videoStreamRef.current?.getTracks().forEach((t) => t.stop());
      videoStreamRef.current = null;
    };
  }, []);

  // WebSocket connection with auto-reconnect
  useEffect(() => {
    let alive = true;
    let ws: WebSocket;
    let timer: number;

    const connect = () => {
      if (!alive) return;
      setConnection('connecting');
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnection('connected');

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data as string);
          if (msg.type === 'state') {
            setCogState({
              label: msg.label,
              confidence: msg.confidence,
              signals: msg.signals ?? [],
            });
          } else if (msg.type === 'feedback') {
            const fb: Feedback = { text: msg.text, suggestions: msg.suggestions ?? [] };
            setFeedback(fb);
            window.cognitiveSense?.notify('CognitiveSense', msg.text);
            if (feedbackTimerRef.current !== null) {
              clearTimeout(feedbackTimerRef.current);
            }
            feedbackTimerRef.current = window.setTimeout(() => {
              setFeedback(null);
              feedbackTimerRef.current = null;
            }, 12_000);
          }
        } catch {
          /* ignore malformed messages */
        }
      };

      ws.onclose = () => {
        setConnection('disconnected');
        wsRef.current = null;
        if (alive) timer = window.setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      alive = false;
      clearTimeout(timer);
      ws?.close();
    };
  }, []);

  // Send video frames to server as binary JPEG
  useEffect(() => {
    if (!streamReady) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let cancelled = false;
    const videoTrack = videoStreamRef.current?.getVideoTracks()[0] ?? null;
    const imageCapture =
      videoTrack && 'ImageCapture' in window
        ? (new ImageCapture(videoTrack) as unknown as ImageCaptureLike)
        : null;

    const setCanvasSize = () => {
      if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    };

    const ensurePlaying = () => {
      if (cancelled || video.srcObject === null || !video.paused) return;
      void video.play().catch(() => {
        /* ignore transient autoplay interruptions */
      });
    };

    video.addEventListener('loadedmetadata', setCanvasSize);
    video.addEventListener('pause', ensurePlaying);
    setCanvasSize();
    ensurePlaying();

    let sending = false;
    const interval = window.setInterval(() => {
      if (sending || wsRef.current?.readyState !== WebSocket.OPEN) return;
      if (wsRef.current.bufferedAmount > MAX_WS_BUFFERED_BYTES) return;
      if (!video.videoWidth) return;

      sending = true;

      const sendCanvasBlob = () => {
        canvas.toBlob(
          (blob) => {
            sending = false;
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
              void blob.arrayBuffer().then((buf) => wsRef.current?.send(buf));
            }
          },
          'image/jpeg',
          0.7,
        );
      };

      if (imageCapture) {
        void imageCapture
          .grabFrame()
          .then((bitmap: ImageBitmap) => {
            if (cancelled) {
              sending = false;
              return;
            }

            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            bitmap.close();
            sendCanvasBlob();
          })
          .catch(() => {
            if (cancelled) {
              sending = false;
              return;
            }

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            sendCanvasBlob();
          });
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      sendCanvasBlob();
    }, FRAME_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
      video.removeEventListener('loadedmetadata', setCanvasSize);
      video.removeEventListener('pause', ensurePlaying);
    };
  }, [streamReady]);

  const color = STATE_COLORS[cogState.label] ?? STATE_COLORS.UNKNOWN;

  if (mediaError) {
    return (
      <main className="app">
        <div className="center-msg">{mediaError}</div>
      </main>
    );
  }

  return (
    <main className="app">
      <video ref={videoRef} className="webcam" autoPlay muted playsInline />
      <canvas ref={canvasRef} className="capture-canvas" />

      <div className="top-bar">
        <div className="state-badge" style={{ borderColor: color }}>
          <span className="state-dot" style={{ backgroundColor: color }} />
          <span className="state-label">{cogState.label}</span>
          <span className="state-confidence">
            {Math.round(cogState.confidence * 100)}%
          </span>
        </div>

        <div className={`conn-badge ${connection}`}>
          <span className="conn-dot" />
          {connection === 'connected'
            ? 'Connected'
            : connection === 'connecting'
              ? 'Connecting...'
              : 'Offline'}
        </div>
      </div>

      {feedback && (
        <div className="feedback-toast" onClick={() => setFeedback(null)}>
          <p className="feedback-text">{feedback.text}</p>
          {feedback.suggestions.length > 0 && (
            <div className="feedback-suggestions">
              {feedback.suggestions.map((s, i) => (
                <span key={i} className="suggestion-chip">
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </main>
  );
};
