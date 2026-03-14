"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const node_fs_1 = __importDefault(require("node:fs"));
const node_net_1 = __importDefault(require("node:net"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const enableTray = process.platform !== 'linux';
const desktopFileName = 'cognitivesense-shell.desktop';
const mediaHost = process.env.COGNITIVESENSE_SERVER_HOST ?? '127.0.0.1';
const mediaPort = Number(process.env.COGNITIVESENSE_SERVER_PORT ?? '9100');
const useFakeMedia = process.env.COGNITIVESENSE_FAKE_MEDIA === '1';
const disableGpu = process.env.COGNITIVESENSE_DISABLE_GPU === '1';
const runtimeLogPath = node_path_1.default.join(node_os_1.default.tmpdir(), 'cognitivesense-electron-runtime.log');
const headerSize = 24;
const frameMagic = 'CSJ1';
const audioMagic = 'CSA1';
const eventMagic = 'CSM1';
const maxSocketBufferedBytes = 512 * 1024;
electron_1.app.setName('CognitiveSense');
const writeRuntimeLog = (source, message, data) => {
    const suffix = data === undefined ? '' : ` ${JSON.stringify(data)}`;
    const line = `${new Date().toISOString()} [${source}] ${message}${suffix}\n`;
    try {
        node_fs_1.default.appendFileSync(runtimeLogPath, line, 'utf8');
    }
    catch {
        // Ignore diagnostic logging failures.
    }
};
if (disableGpu) {
    electron_1.app.disableHardwareAcceleration();
}
if (process.platform === 'linux') {
    process.env.CHROME_DESKTOP = desktopFileName;
}
electron_1.app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
if (disableGpu) {
    electron_1.app.commandLine.appendSwitch('disable-gpu');
    electron_1.app.commandLine.appendSwitch('disable-gpu-compositing');
}
if (useFakeMedia) {
    electron_1.app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
    electron_1.app.commandLine.appendSwitch('use-fake-device-for-media-stream');
}
let mainWindow = null;
let tray = null;
let isQuitting = false;
let backendSocket = null;
let backendReconnectTimer = null;
let backendBuffer = Buffer.alloc(0);
let backendConnection = 'disconnected';
let forwardedFrameCount = 0;
let forwardedAudioCount = 0;
let droppedFrameCount = 0;
let droppedAudioCount = 0;
let lastForwardedAt = 0;
const createTrayIcon = () => {
    return electron_1.nativeImage
        .createFromPath(node_path_1.default.join(electron_1.app.getAppPath(), 'assets', 'tray-icon.svg'))
        .resize({ width: 18, height: 18 });
};
const escapeDesktopEntryValue = (value) => {
    return value.replace(/\\/g, '\\\\').replace(/\n/g, ' ');
};
const quoteDesktopExecArg = (value) => {
    return `"${value.replace(/(["\\])/g, '\\$1')}"`;
};
const ensureLinuxDesktopEntry = () => {
    if (process.platform !== 'linux')
        return;
    const applicationsDir = node_path_1.default.join(node_os_1.default.homedir(), '.local', 'share', 'applications');
    const desktopFilePath = node_path_1.default.join(applicationsDir, desktopFileName);
    const execTarget = process.defaultApp
        ? `${quoteDesktopExecArg(process.execPath)} ${quoteDesktopExecArg(electron_1.app.getAppPath())}`
        : quoteDesktopExecArg(process.execPath);
    const iconPath = node_path_1.default.join(electron_1.app.getAppPath(), 'assets', 'tray-icon.svg');
    const desktopEntry = [
        '[Desktop Entry]',
        'Version=1.0',
        'Type=Application',
        'Name=CognitiveSense',
        'Comment=Ambient cognitive sense monitor',
        `Exec=${execTarget}`,
        `Icon=${escapeDesktopEntryValue(iconPath)}`,
        'Terminal=false',
        'StartupNotify=true',
        'Categories=Utility;',
        'StartupWMClass=CognitiveSense',
    ].join('\n');
    node_fs_1.default.mkdirSync(applicationsDir, { recursive: true });
    node_fs_1.default.writeFileSync(desktopFilePath, `${desktopEntry}\n`, 'utf8');
};
const getRendererUrl = () => {
    if (isDev)
        return process.env.VITE_DEV_SERVER_URL;
    return `file://${node_path_1.default.join(__dirname, '../../dist/index.html')}`;
};
const emitToRenderer = (channel, payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(channel, payload);
    }
};
const getTransportPayload = (payload = {}) => {
    return {
        connection: payload.connection ?? backendConnection,
        host: mediaHost,
        port: mediaPort,
        detail: payload.detail,
    };
};
const emitTransport = (payload = {}) => {
    emitToRenderer('telemetry:transport', getTransportPayload(payload));
};
const transportDetail = () => {
    if (backendConnection !== 'connected') {
        return undefined;
    }
    if (lastForwardedAt === 0) {
        return 'TCP connected; waiting for media';
    }
    return `frames ${forwardedFrameCount} dropped ${droppedFrameCount} audio ${forwardedAudioCount}`;
};
const emitTransportSnapshot = () => {
    emitTransport({ detail: transportDetail() });
};
const scheduleBackendReconnect = () => {
    if (backendReconnectTimer)
        return;
    backendReconnectTimer = setTimeout(() => {
        backendReconnectTimer = null;
        connectBackend();
    }, 3000);
};
const closeBackendSocket = () => {
    if (!backendSocket)
        return;
    backendSocket.removeAllListeners();
    backendSocket.destroy();
    backendSocket = null;
};
const parseBackendPackets = (chunk) => {
    backendBuffer = Buffer.concat([backendBuffer, chunk]);
    while (backendBuffer.length >= headerSize) {
        const magic = backendBuffer.toString('ascii', 0, 4);
        const payloadSize = backendBuffer.readUInt32BE(12);
        const packetSize = headerSize + payloadSize;
        if (backendBuffer.length < packetSize)
            return;
        const payload = backendBuffer.subarray(headerSize, packetSize);
        backendBuffer = backendBuffer.subarray(packetSize);
        if (magic !== eventMagic) {
            continue;
        }
        try {
            emitToRenderer('telemetry:event', JSON.parse(payload.toString('utf8')));
        }
        catch {
            emitTransport({ detail: 'Malformed telemetry payload from server' });
        }
    }
};
const connectBackend = () => {
    if (backendSocket)
        return;
    backendConnection = 'connecting';
    writeRuntimeLog('main', 'backend:connecting', { host: mediaHost, port: mediaPort });
    emitTransport();
    const socket = node_net_1.default.createConnection({ host: mediaHost, port: mediaPort });
    backendSocket = socket;
    socket.on('connect', () => {
        backendConnection = 'connected';
        backendBuffer = Buffer.alloc(0);
        writeRuntimeLog('main', 'backend:connected', { host: mediaHost, port: mediaPort });
        emitTransportSnapshot();
    });
    socket.on('data', parseBackendPackets);
    socket.on('error', (error) => {
        writeRuntimeLog('main', 'backend:error', { message: error.message });
        emitTransport({ detail: error.message });
    });
    socket.on('close', () => {
        backendSocket = null;
        backendConnection = 'disconnected';
        writeRuntimeLog('main', 'backend:closed');
        emitTransport({ detail: undefined });
        if (!isQuitting) {
            scheduleBackendReconnect();
        }
    });
};
const packetTimestamp = () => BigInt(Date.now()) * 1000000n;
const writePacket = (magic, meta1, meta2, payload) => {
    const socket = backendSocket;
    if (!socket || backendConnection !== 'connected' || socket.destroyed)
        return false;
    if (socket.writableLength > maxSocketBufferedBytes || socket.writableNeedDrain) {
        return false;
    }
    const header = Buffer.alloc(headerSize);
    header.write(magic, 0, 4, 'ascii');
    header.writeUInt32BE(meta1 >>> 0, 4);
    header.writeUInt32BE(meta2 >>> 0, 8);
    header.writeUInt32BE(payload.length >>> 0, 12);
    header.writeBigUInt64BE(packetTimestamp(), 16);
    return socket.write(Buffer.concat([header, payload]));
};
const reportMediaCounters = () => {
    if ((forwardedFrameCount + droppedFrameCount + forwardedAudioCount + droppedAudioCount) % 25 === 0) {
        emitTransportSnapshot();
    }
};
const toBuffer = (data) => {
    if (Buffer.isBuffer(data))
        return data;
    if (data instanceof Uint8Array) {
        return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    }
    return Buffer.from(data);
};
const showWindow = async () => {
    if (!mainWindow)
        return;
    mainWindow.setSkipTaskbar(false);
    mainWindow.show();
    if (process.platform === 'darwin' && electron_1.app.dock) {
        electron_1.app.dock.show();
    }
    mainWindow.focus();
};
const hideWindow = () => {
    if (!mainWindow)
        return;
    mainWindow.hide();
    mainWindow.setSkipTaskbar(true);
    if (process.platform === 'darwin' && !isQuitting && electron_1.app.dock) {
        electron_1.app.dock.hide();
    }
};
const createWindow = async () => {
    mainWindow = new electron_1.BrowserWindow({
        width: 1180,
        height: 780,
        minWidth: 860,
        minHeight: 620,
        show: true,
        frame: false,
        titleBarStyle: 'hidden',
        backgroundColor: '#07111a',
        title: 'CognitiveSense',
        autoHideMenuBar: true,
        resizable: true,
        webPreferences: {
            backgroundThrottling: false,
            preload: node_path_1.default.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    mainWindow.webContents.on('did-finish-load', () => {
        emitTransportSnapshot();
    });
    await mainWindow.loadURL(getRendererUrl()).catch((error) => {
        console.error('Failed to load renderer URL', error);
    });
    await showWindow();
};
const syncTrayMenu = () => {
    if (!tray)
        return;
    const contextMenu = electron_1.Menu.buildFromTemplate([
        { label: 'Show', click: () => void showWindow() },
        { label: 'Hide', click: () => hideWindow() },
        { type: 'separator' },
        {
            label: 'Quit',
            click: () => {
                isQuitting = true;
                electron_1.app.quit();
            },
        },
    ]);
    tray.setContextMenu(contextMenu);
    tray.setToolTip('CognitiveSense');
};
const createTray = () => {
    if (!enableTray)
        return;
    tray = new electron_1.Tray(createTrayIcon());
    tray.on('click', () => {
        if (mainWindow?.isVisible()) {
            hideWindow();
            return;
        }
        void showWindow();
    });
    syncTrayMenu();
};
electron_1.app.whenReady().then(async () => {
    electron_1.app.setAppUserModelId('com.genesis.cognitivesense');
    ensureLinuxDesktopEntry();
    connectBackend();
    electron_1.session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
        callback(['media', 'mediaKeySystem'].includes(permission));
    });
    electron_1.ipcMain.handle('shell:show', async () => {
        await showWindow();
    });
    electron_1.ipcMain.handle('shell:hide', () => {
        hideWindow();
    });
    electron_1.ipcMain.handle('shell:minimize', () => {
        mainWindow?.minimize();
    });
    electron_1.ipcMain.handle('shell:quit', () => {
        isQuitting = true;
        electron_1.app.quit();
    });
    electron_1.ipcMain.handle('shell:notify', (_event, title, body) => {
        if (electron_1.Notification.isSupported()) {
            new electron_1.Notification({
                title,
                body,
                urgency: 'normal',
                ...(process.platform === 'linux' ? {} : { icon: createTrayIcon() }),
            }).show();
        }
    });
    electron_1.ipcMain.on('stream:frame', (_event, payload) => {
        const sent = writePacket(frameMagic, payload.width, payload.height, toBuffer(payload.data));
        if (sent) {
            forwardedFrameCount += 1;
            lastForwardedAt = Date.now();
            if (forwardedFrameCount === 1) {
                writeRuntimeLog('main', 'media:first-frame-forwarded', {
                    width: payload.width,
                    height: payload.height,
                    bytes: toBuffer(payload.data).byteLength,
                });
            }
        }
        else {
            droppedFrameCount += 1;
            if (droppedFrameCount === 1) {
                writeRuntimeLog('main', 'media:first-frame-dropped');
            }
        }
        reportMediaCounters();
    });
    electron_1.ipcMain.on('stream:audio', (_event, payload) => {
        const sent = writePacket(audioMagic, payload.sampleRate, payload.channels, toBuffer(payload.data));
        if (sent) {
            forwardedAudioCount += 1;
            lastForwardedAt = Date.now();
        }
        else {
            droppedAudioCount += 1;
        }
        reportMediaCounters();
    });
    electron_1.ipcMain.handle('stream:backend-target', () => ({ host: mediaHost, port: mediaPort }));
    electron_1.ipcMain.handle('telemetry:transport-state', () => getTransportPayload({ detail: transportDetail() }));
    electron_1.ipcMain.on('diagnostic:log', (_event, payload) => {
        writeRuntimeLog('renderer', payload.message, payload.data);
    });
    writeRuntimeLog('main', 'app:ready', {
        runtimeLogPath,
        disableGpu,
        useFakeMedia,
    });
    await createWindow();
    createTray();
    electron_1.app.on('activate', () => {
        if (mainWindow === null) {
            void createWindow();
            return;
        }
        void showWindow();
    });
});
electron_1.app.on('before-quit', () => {
    isQuitting = true;
    if (backendReconnectTimer) {
        clearTimeout(backendReconnectTimer);
        backendReconnectTimer = null;
    }
    closeBackendSocket();
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        return;
});
//# sourceMappingURL=main.js.map