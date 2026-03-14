/// <reference types="vite/client" />

interface Window {
  cognitiveSense: {
    notify: (title: string, body: string) => Promise<void>;
    show: () => Promise<void>;
    hide: () => Promise<void>;
    minimize: () => Promise<void>;
    quit: () => Promise<void>;
    debugLog: (message: string, data?: unknown) => void;
    sendFrame: (data: ArrayBuffer, width: number, height: number) => boolean;
    sendAudio: (data: ArrayBuffer, sampleRate: number, channels: number) => boolean;
    getBackendTarget: () => Promise<{ host: string; port: number }>;
    getTransportState: () => Promise<{ connection: 'connecting' | 'connected' | 'disconnected'; host: string; port: number; detail?: string }>;
    onTelemetry: (callback: (payload: Record<string, unknown>) => void) => () => void;
    onTransport: (callback: (payload: { connection: 'connecting' | 'connected' | 'disconnected'; host: string; port: number; detail?: string }) => void) => () => void;
  };
}
