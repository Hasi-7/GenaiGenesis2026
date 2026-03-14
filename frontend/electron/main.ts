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
import os from 'node:os';
import path from 'node:path';

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const desktopEnv = (process.env.XDG_CURRENT_DESKTOP ?? process.env.DESKTOP_SESSION ?? '').toLowerCase();
const useWindowFallback = process.platform === 'linux' && desktopEnv.includes('gnome');
const enableTray = process.platform !== 'linux';
const desktopFileName = 'cognitivesense-shell.desktop';

app.setName('CognitiveSense');

if (process.platform === 'linux') {
  process.env.CHROME_DESKTOP = desktopFileName;
}

app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let isQuitting = false;

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
  if (useWindowFallback) {
    mainWindow.minimize();
    return;
  }
  mainWindow.hide();
  mainWindow.setSkipTaskbar(true);
  if (process.platform === 'darwin' && !isQuitting && app.dock) {
    app.dock.hide();
  }
};

const createWindow = async () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    minWidth: 480,
    minHeight: 360,
    show: true,
    backgroundColor: '#0a0a0a',
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

  mainWindow.on('close', (event) => {
    if (isQuitting) return;
    event.preventDefault();
    hideWindow();
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

  // Grant camera and microphone permissions automatically
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(['media', 'mediaKeySystem'].includes(permission));
  });

  ipcMain.handle('shell:show', async () => {
    await showWindow();
  });
  ipcMain.handle('shell:hide', () => {
    hideWindow();
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
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') return;
});
