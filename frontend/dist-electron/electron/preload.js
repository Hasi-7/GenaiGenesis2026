"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('cognitiveSense', {
    notify: (title, body) => electron_1.ipcRenderer.invoke('shell:notify', title, body),
    show: () => electron_1.ipcRenderer.invoke('shell:show'),
    hide: () => electron_1.ipcRenderer.invoke('shell:hide'),
});
//# sourceMappingURL=preload.js.map