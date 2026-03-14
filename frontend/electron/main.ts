import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  Notification,
  Tray,
  nativeImage,
  session,
} from 'electron';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const enableTray = process.platform !== 'linux';
const desktopFileName = 'cognitivesense-shell.desktop';
const mediaHost = process.env.COGNITIVESENSE_SERVER_HOST ?? '127.0.0.1';
const mediaPort = Number(process.env.COGNITIVESENSE_SERVER_PORT ?? '9000');
const useFakeMedia = process.env.COGNITIVESENSE_FAKE_MEDIA === '1';
const disableGpu = process.env.COGNITIVESENSE_DISABLE_GPU === '1';
const runtimeLogPath = path.join(os.tmpdir(), 'cognitivesense-electron-runtime.log');

const headerSize = 24;
const frameMagic = 'CSJ1';
const audioMagic = 'CSA1';
const eventMagic = 'CSM1';
const maxSocketBufferedBytes = 512 * 1024;
const controlHello = 1;
const controlState = 2;
const controlFeedback = 3;
const controlVersion = 1;
const sourceDesktop = 1;
const capabilitySendVideo = 1 << 0;
const capabilitySendAudio = 1 << 1;
const capabilityReceiveState = 1 << 2;
const capabilityBinaryControl = 1 << 4;
const streamVideoRecent = 1 << 0;
const streamAudioRecent = 1 << 1;

const stateLabels: Record<number, string> = {
  0: 'UNKNOWN',
  1: 'FOCUSED',
  2: 'FATIGUED',
  3: 'STRESSED',
  4: 'DISTRACTED',
};

const indicatorLabels: Record<number, string> = {
  1: 'Blink rate elevated',
  2: 'Blink rate suppressed',
  3: 'Posture slouched',
  4: 'Posture leaning',
  5: 'Eye movement distracted',
  6: 'Facial tension detected',
  7: 'Speech tone stressed',
  8: 'Speech tone monotone',
  9: 'Posture upright',
  10: 'Eye engagement focused',
  11: 'Facial expression relaxed',
  12: 'Speech tone calm',
};

const recommendationLabels: Record<number, string> = {
  1: 'Take a 10 minute break',
  2: 'Hydrate',
  3: 'Stretch and reset posture',
  4: 'Take a breathing pause',
  5: 'Refocus on one task',
  6: 'Silence distractions',
  7: 'Keep your current pace',
  8: 'Reset your posture',
};

type RendererTransportState = 'connecting' | 'connected' | 'disconnected';

type TransportPayload = {
  connection: RendererTransportState;
  host: string;
  port: number;
  detail?: string;
};

app.setName('CognitiveSense');

const writeRuntimeLog = (source: string, message: string, data?: unknown) => {
  const suffix = data === undefined ? '' : ` ${JSON.stringify(data)}`;
  const line = `${new Date().toISOString()} [${source}] ${message}${suffix}\n`;
  try {
    fs.appendFileSync(runtimeLogPath, line, 'utf8');
  } catch {
    // Ignore diagnostic logging failures.
  }
};

if (disableGpu) {
  app.disableHardwareAcceleration();
}

if (process.platform === 'linux') {
  process.env.CHROME_DESKTOP = desktopFileName;
}

app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
if (disableGpu) {
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-gpu-compositing');
}
if (useFakeMedia) {
  app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
  app.commandLine.appendSwitch('use-fake-device-for-media-stream');
}

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let isQuitting = false;
let backendSocket: net.Socket | null = null;
let backendReconnectTimer: NodeJS.Timeout | null = null;
let backendBuffer = Buffer.alloc(0);
let backendConnection: RendererTransportState = 'disconnected';
let forwardedFrameCount = 0;
let forwardedAudioCount = 0;
let droppedFrameCount = 0;
let droppedAudioCount = 0;
let lastForwardedAt = 0;

const createTrayIcon = () => {
  return nativeImage
    .createFromPath(path.join(app.getAppPath(), 'assets', 'tray-icon.svg'))
    .resize({ width: 18, height: 18 });
};

const escapeDesktopEntryValue = (value: string) => {
  return value.replace(/\\/g, '\\\\').replace(/\n/g, ' ');
};

const quoteDesktopExecArg = (value: string) => {
  return `"${value.replace(/(["\\])/g, '\\$1')}"`;
};

const ensureLinuxDesktopEntry = () => {
  if (process.platform !== 'linux') return;

  const applicationsDir = path.join(os.homedir(), '.local', 'share', 'applications');
  const desktopFilePath = path.join(applicationsDir, desktopFileName);
  const execTarget = process.defaultApp
    ? `${quoteDesktopExecArg(process.execPath)} ${quoteDesktopExecArg(app.getAppPath())}`
    : quoteDesktopExecArg(process.execPath);
  const iconPath = path.join(app.getAppPath(), 'assets', 'tray-icon.svg');
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

  fs.mkdirSync(applicationsDir, { recursive: true });
  fs.writeFileSync(desktopFilePath, `${desktopEntry}\n`, 'utf8');
};

const getRendererUrl = () => {
  if (isDev) return process.env.VITE_DEV_SERVER_URL as string;
  return `file://${path.join(__dirname, '../../dist/index.html')}`;
};

const emitToRenderer = (channel: string, payload: unknown) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
};

const getTransportPayload = (payload: Partial<TransportPayload> = {}): TransportPayload => {
  return {
    connection: payload.connection ?? backendConnection,
    host: mediaHost,
    port: mediaPort,
    detail: payload.detail,
  };
};

const emitTransport = (payload: Partial<TransportPayload> = {}) => {
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
  if (backendReconnectTimer) return;
  backendReconnectTimer = setTimeout(() => {
    backendReconnectTimer = null;
    connectBackend();
  }, 3000);
};

const closeBackendSocket = () => {
  if (!backendSocket) return;
  backendSocket.removeAllListeners();
  backendSocket.destroy();
  backendSocket = null;
};

const sendControlPacket = (messageType: number, flags: number, payload: Buffer) => {
  return writePacket(eventMagic, messageType, flags, payload);
};

const sendHelloPacket = () => {
  const payload = Buffer.alloc(4);
  payload.writeUInt8(controlVersion, 0);
  payload.writeUInt8(sourceDesktop, 1);
  payload.writeUInt16BE(0, 2);

  const capabilityFlags =
    capabilitySendVideo |
    capabilitySendAudio |
    capabilityReceiveState |
    capabilityBinaryControl;

  return sendControlPacket(controlHello, capabilityFlags, payload);
};

const emitStateTelemetry = (payload: Buffer) => {
  if (payload.length < 8) {
    emitTransport({ detail: 'Malformed state telemetry payload from server' });
    return;
  }

  const stateId = payload.readUInt8(1);
  const confidence = payload.readUInt8(2) / 255;
  const streamFlags = payload.readUInt8(3);
  const indicators = payload.length >= 11
    ? [payload.readUInt8(8), payload.readUInt8(9), payload.readUInt8(10)]
        .map((code) => indicatorLabels[code])
        .filter((value): value is string => Boolean(value))
    : [];
  const recommendations = payload.length >= 14
    ? [payload.readUInt8(11), payload.readUInt8(12), payload.readUInt8(13)]
        .map((code) => recommendationLabels[code])
        .filter((value): value is string => Boolean(value))
    : [];

  emitToRenderer('telemetry:event', {
    type: 'state',
    label: stateLabels[stateId] ?? 'UNKNOWN',
    confidence,
    indicators,
    recommendations,
    transport: {
      video: Boolean(streamFlags & streamVideoRecent),
      audio: Boolean(streamFlags & streamAudioRecent),
    },
  });
};

const emitFeedbackTelemetry = (payload: Buffer) => {
  const text = payload.toString('utf8').trim();
  if (!text) {
    return;
  }
  emitToRenderer('telemetry:event', {
    type: 'feedback',
    text,
    timestamp: Date.now() / 1000,
  });
};

const parseBackendPackets = (chunk: Buffer) => {
  backendBuffer = Buffer.concat([backendBuffer, chunk]);

  while (backendBuffer.length >= headerSize) {
    const magic = backendBuffer.toString('ascii', 0, 4);
    const meta1 = backendBuffer.readUInt32BE(4);
    const payloadSize = backendBuffer.readUInt32BE(12);
    const packetSize = headerSize + payloadSize;
    if (backendBuffer.length < packetSize) return;

    const payload = backendBuffer.subarray(headerSize, packetSize);
    backendBuffer = backendBuffer.subarray(packetSize);

    if (magic !== eventMagic) {
      continue;
    }

    if (meta1 === controlState) {
      emitStateTelemetry(payload);
    } else if (meta1 === controlFeedback) {
      emitFeedbackTelemetry(payload);
    }
  }
};

const connectBackend = () => {
  if (backendSocket) return;

  backendConnection = 'connecting';
  writeRuntimeLog('main', 'backend:connecting', { host: mediaHost, port: mediaPort });
  emitTransport();

  const socket = net.createConnection({ host: mediaHost, port: mediaPort });
  backendSocket = socket;

  socket.on('connect', () => {
    backendConnection = 'connected';
    backendBuffer = Buffer.alloc(0);
    writeRuntimeLog('main', 'backend:connected', { host: mediaHost, port: mediaPort });
    sendHelloPacket();
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

const writePacket = (magic: string, meta1: number, meta2: number, payload: Buffer) => {
  const socket = backendSocket;
  if (!socket || backendConnection !== 'connected' || socket.destroyed) return false;
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

const toBuffer = (data: ArrayBuffer | Uint8Array | Buffer) => {
  if (Buffer.isBuffer(data)) return data;
  if (data instanceof Uint8Array) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  }
  return Buffer.from(data);
};

const showWindow = async () => {
  if (!mainWindow) return;
  mainWindow.setSkipTaskbar(false);
  mainWindow.show();
  if (process.platform === 'darwin' && app.dock) {
    app.dock.show();
  }
  mainWindow.focus();
};

const hideWindow = () => {
  if (!mainWindow) return;
  mainWindow.hide();
  mainWindow.setSkipTaskbar(true);
  if (process.platform === 'darwin' && !isQuitting && app.dock) {
    app.dock.hide();
  }
};

const createWindow = async () => {
  mainWindow = new BrowserWindow({
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
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.webContents.on('did-finish-load', () => {
    emitTransportSnapshot();
  });

  await mainWindow.loadURL(getRendererUrl()).catch((error: unknown) => {
    console.error('Failed to load renderer URL', error);
  });

  await showWindow();
};

const syncTrayMenu = () => {
  if (!tray) return;

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show', click: () => void showWindow() },
    { label: 'Hide', click: () => hideWindow() },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);
  tray.setToolTip('CognitiveSense');
};

const createTray = () => {
  if (!enableTray) return;

  tray = new Tray(createTrayIcon());
  tray.on('click', () => {
    if (mainWindow?.isVisible()) {
      hideWindow();
      return;
    }
    void showWindow();
  });
  syncTrayMenu();
};

app.whenReady().then(async () => {
  app.setAppUserModelId('com.genesis.cognitivesense');
  ensureLinuxDesktopEntry();
  connectBackend();

  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(['media', 'mediaKeySystem'].includes(permission));
  });

  ipcMain.handle('shell:show', async () => {
    await showWindow();
  });
  ipcMain.handle('shell:hide', () => {
    hideWindow();
  });
  ipcMain.handle('shell:minimize', () => {
    mainWindow?.minimize();
  });
  ipcMain.handle('shell:quit', () => {
    isQuitting = true;
    app.quit();
  });
  ipcMain.handle('shell:notify', (_event, title: string, body: string) => {
    if (Notification.isSupported()) {
      new Notification({
        title,
        body,
        urgency: 'normal',
        ...(process.platform === 'linux' ? {} : { icon: createTrayIcon() }),
      }).show();
    }
  });
  ipcMain.on(
    'stream:frame',
    (_event, payload: { data: ArrayBuffer | Uint8Array; width: number; height: number }) => {
      const sent = writePacket(
        frameMagic,
        payload.width,
        payload.height,
        toBuffer(payload.data),
      );
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
    } else {
      droppedFrameCount += 1;
      if (droppedFrameCount === 1) {
        writeRuntimeLog('main', 'media:first-frame-dropped');
      }
    }
    reportMediaCounters();
  },
  );
  ipcMain.on(
    'stream:audio',
    (_event, payload: { data: ArrayBuffer | Uint8Array; sampleRate: number; channels: number }) => {
      const sent = writePacket(
        audioMagic,
        payload.sampleRate,
        payload.channels,
        toBuffer(payload.data),
      );
      if (sent) {
        forwardedAudioCount += 1;
        lastForwardedAt = Date.now();
      } else {
        droppedAudioCount += 1;
      }
      reportMediaCounters();
    },
  );
  ipcMain.handle('stream:backend-target', () => ({ host: mediaHost, port: mediaPort }));
  ipcMain.handle('telemetry:transport-state', () => getTransportPayload({ detail: transportDetail() }));
  ipcMain.on('diagnostic:log', (_event, payload: { message: string; data?: unknown }) => {
    writeRuntimeLog('renderer', payload.message, payload.data);
  });

  writeRuntimeLog('main', 'app:ready', {
    runtimeLogPath,
    disableGpu,
    useFakeMedia,
  });
  await createWindow();
  createTray();

  app.on('activate', () => {
    if (mainWindow === null) {
      void createWindow();
      return;
    }
    void showWindow();
  });
});

app.on('before-quit', () => {
  isQuitting = true;
  if (backendReconnectTimer) {
    clearTimeout(backendReconnectTimer);
    backendReconnectTimer = null;
  }
  closeBackendSocket();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') return;
});
