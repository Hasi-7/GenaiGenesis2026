import { execSync } from 'node:child_process';

const killPort = (port) => {
  if (process.platform === 'win32') {
    try {
      const output = execSync('netstat -ano -p tcp', { encoding: 'utf8' });
      const pids = new Set();
      for (const line of output.split(/\r?\n/)) {
        if (!line.includes(`:${port}`)) continue;
        const parts = line.trim().split(/\s+/);
        if (parts.length < 5) continue;
        const localAddress = parts[1];
        const pid = parts[parts.length - 1];
        if (!localAddress.endsWith(`:${port}`)) continue;
        if (!/^\d+$/.test(pid)) continue;
        pids.add(pid);
      }
      for (const pid of pids) {
        execSync(`taskkill /F /PID ${pid}`, { stdio: 'ignore' });
      }
    } catch {
      // Port already free or Windows tooling unavailable.
    }
    return;
  }

  try {
    const output = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf8' }).trim();
    if (!output) return;
    for (const pid of output.split(/\s+/)) {
      execSync(`kill -9 ${pid}`, { stdio: 'ignore' });
    }
    return;
  } catch {
    // Fall through to other Unix tools.
  }

  try {
    execSync(`fuser -k ${port}/tcp`, { stdio: 'ignore' });
  } catch {
    // Port already free or cleanup tool unavailable.
  }
};

killPort(5173);
