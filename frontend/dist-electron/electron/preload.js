"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const subscribe = (channel, callback) => {
    const listener = (_event, payload) => callback(payload);
    electron_1.ipcRenderer.on(channel, listener);
    return () => {
        electron_1.ipcRenderer.removeListener(channel, listener);
    };
};
electron_1.contextBridge.exposeInMainWorld('cognitiveSense', {
    notify: (title, body) => electron_1.ipcRenderer.invoke('shell:notify', title, body),
    show: () => electron_1.ipcRenderer.invoke('shell:show'),
    hide: () => electron_1.ipcRenderer.invoke('shell:hide'),
    minimize: () => electron_1.ipcRenderer.invoke('shell:minimize'),
    quit: () => electron_1.ipcRenderer.invoke('shell:quit'),
    debugLog: (message, data) => {
        electron_1.ipcRenderer.send('diagnostic:log', { message, data });
    },
    sendFrame: (data, width, height) => {
        electron_1.ipcRenderer.send('stream:frame', { data, width, height });
        return true;
    },
    sendAudio: (data, sampleRate, channels) => {
        electron_1.ipcRenderer.send('stream:audio', { data, sampleRate, channels });
        return true;
    },
    getBackendTarget: () => electron_1.ipcRenderer.invoke('stream:backend-target'),
    getTransportState: () => electron_1.ipcRenderer.invoke('telemetry:transport-state'),
    onTelemetry: (callback) => subscribe('telemetry:event', callback),
    onTransport: (callback) => subscribe('telemetry:transport', callback),
});
//# sourceMappingURL=preload.js.map