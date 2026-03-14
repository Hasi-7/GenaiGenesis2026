"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const node_fs_1 = __importDefault(require("node:fs"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);
const desktopEnv = (process.env.XDG_CURRENT_DESKTOP ?? process.env.DESKTOP_SESSION ?? '').toLowerCase();
const useWindowFallback = process.platform === 'linux' && desktopEnv.includes('gnome');
const enableTray = process.platform !== 'linux';
const desktopFileName = 'cognitivesense-shell.desktop';
electron_1.app.setName('CognitiveSense');
if (process.platform === 'linux') {
    process.env.CHROME_DESKTOP = desktopFileName;
}
electron_1.app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
let mainWindow = null;
let tray = null;
let isQuitting = false;
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
    if (useWindowFallback) {
        mainWindow.minimize();
        return;
    }
    mainWindow.hide();
    mainWindow.setSkipTaskbar(true);
    if (process.platform === 'darwin' && !isQuitting && electron_1.app.dock) {
        electron_1.app.dock.hide();
    }
};
const createWindow = async () => {
    mainWindow = new electron_1.BrowserWindow({
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
            preload: node_path_1.default.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    mainWindow.on('close', (event) => {
        if (isQuitting)
            return;
        event.preventDefault();
        hideWindow();
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
    // Grant camera and microphone permissions automatically
    electron_1.session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
        callback(['media', 'mediaKeySystem'].includes(permission));
    });
    electron_1.ipcMain.handle('shell:show', async () => {
        await showWindow();
    });
    electron_1.ipcMain.handle('shell:hide', () => {
        hideWindow();
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
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        return;
});
//# sourceMappingURL=main.js.map