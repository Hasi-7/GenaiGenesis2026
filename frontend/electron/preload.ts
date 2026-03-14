import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('cognitiveSense', {
  notify: (title: string, body: string) =>
    ipcRenderer.invoke('shell:notify', title, body) as Promise<void>,
  show: () => ipcRenderer.invoke('shell:show') as Promise<void>,
  hide: () => ipcRenderer.invoke('shell:hide') as Promise<void>,
});
