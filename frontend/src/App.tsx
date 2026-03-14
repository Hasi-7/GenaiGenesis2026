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

export const App = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [cogState, setCogState] = useState<CognitiveState>({
    label: 'UNKNOWN',
    confidence: 0,
    signals: [],
  });
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [connection, setConnection] = useState<ConnectionStatus>('disconnected');
  const [streamReady, setStreamReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  // Start webcam + mic
  useEffect(() => {
    let cancelled = false;

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: { sampleRate: 16000, channelCount: 1 },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStreamReady(true);
      } catch {
        if (!cancelled) setMediaError('Camera or microphone access denied');
      }
    };

    void start();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
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
            setTimeout(() => setFeedback(null), 12_000);
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

    const setCanvasSize = () => {
      if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
    };

    video.addEventListener('loadedmetadata', setCanvasSize);
    setCanvasSize();

    let sending = false;
    const interval = window.setInterval(() => {
      if (sending || wsRef.current?.readyState !== WebSocket.OPEN) return;
      if (!video.videoWidth) return;

      ctx.drawImage(video, 0, 0);
      sending = true;

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
    }, FRAME_INTERVAL_MS);

    return () => {
      clearInterval(interval);
      video.removeEventListener('loadedmetadata', setCanvasSize);
    };
  }, [streamReady]);

  // Capture and send audio chunks
  useEffect(() => {
    if (!streamReady || !streamRef.current) return;

    const stream = streamRef.current;
    if (stream.getAudioTracks().length === 0) return;

    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    const gain = audioCtx.createGain();
    gain.gain.value = 0; // mute output to prevent mic feedback

    processor.onaudioprocess = (e) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) return;

      const pcm = new Float32Array(e.inputBuffer.getChannelData(0));
      wsRef.current.send(
        JSON.stringify({
          type: 'audio',
          data: float32ToBase64(pcm),
          sampleRate: audioCtx.sampleRate,
        }),
      );
    };

    source.connect(processor);
    processor.connect(gain);
    gain.connect(audioCtx.destination);

    return () => {
      processor.disconnect();
      source.disconnect();
      void audioCtx.close();
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

function float32ToBase64(f32: Float32Array): string {
  const bytes = new Uint8Array(f32.buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
