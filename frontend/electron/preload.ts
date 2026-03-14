import { contextBridge, ipcRenderer } from 'electron';

type TelemetryEvent = Record<string, unknown>;

type TransportEvent = {
  connection: 'connecting' | 'connected' | 'disconnected';
  host: string;
  port: number;
  detail?: string;
};

const subscribe = <T>(channel: string, callback: (payload: T) => void) => {
  const listener = (_event: unknown, payload: T) => callback(payload);
  ipcRenderer.on(channel, listener);
  return () => {
    ipcRenderer.removeListener(channel, listener);
  };
};

contextBridge.exposeInMainWorld('cognitiveSense', {
  notify: (title: string, body: string) =>
    ipcRenderer.invoke('shell:notify', title, body) as Promise<void>,
  show: () => ipcRenderer.invoke('shell:show') as Promise<void>,
  hide: () => ipcRenderer.invoke('shell:hide') as Promise<void>,
  minimize: () => ipcRenderer.invoke('shell:minimize') as Promise<void>,
  quit: () => ipcRenderer.invoke('shell:quit') as Promise<void>,
  debugLog: (message: string, data?: unknown) => {
    ipcRenderer.send('diagnostic:log', { message, data });
  },
  sendFrame: (data: ArrayBuffer, width: number, height: number) => {
    ipcRenderer.send('stream:frame', { data, width, height });
    return true;
  },
  sendAudio: (data: ArrayBuffer, sampleRate: number, channels: number) => {
    ipcRenderer.send('stream:audio', { data, sampleRate, channels });
    return true;
  },
  getBackendTarget: () =>
    ipcRenderer.invoke('stream:backend-target') as Promise<{ host: string; port: number }>,
  getTransportState: () =>
    ipcRenderer.invoke('telemetry:transport-state') as Promise<TransportEvent>,
  onTelemetry: (callback: (payload: TelemetryEvent) => void) =>
    subscribe<TelemetryEvent>('telemetry:event', callback),
  onTransport: (callback: (payload: TransportEvent) => void) =>
    subscribe<TransportEvent>('telemetry:transport', callback),
});
