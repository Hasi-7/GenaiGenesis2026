import { execSync } from 'node:child_process';

const killPort = (port) => {
  try {
    const output = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf8' }).trim();
    if (!output) return;
    for (const pid of output.split(/\s+/)) {
      execSync(`kill -9 ${pid}`);
    }
  } catch {
    // Port already free or lsof unavailable.
  }
};

killPort(5173);
