import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import net from 'node:net';
import path from 'node:path';

import { _electron as electron } from 'playwright-core';

const require = createRequire(import.meta.url);
const electronBinary = require('electron');

const HEADER_SIZE = 24;
const FRAME_MAGIC = 'CSJ1';
const AUDIO_MAGIC = 'CSA1';
const EVENT_MAGIC = 'CSM1';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const waitFor = async (predicate, timeoutMs, message) => {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await predicate()) {
      return;
    }
    await sleep(100);
  }
  throw new Error(message);
};

const encodeEventPacket = (payload) => {
  const body = Buffer.from(JSON.stringify(payload), 'utf8');
  const header = Buffer.alloc(HEADER_SIZE);
  header.write(EVENT_MAGIC, 0, 4, 'ascii');
  header.writeUInt32BE(0, 4);
  header.writeUInt32BE(0, 8);
  header.writeUInt32BE(body.length, 12);
  header.writeBigUInt64BE(BigInt(Date.now()) * 1000000n, 16);
  return Buffer.concat([header, body]);
};

const startMockServer = async () => {
  let buffer = Buffer.alloc(0);
  let connections = 0;
  let framePackets = 0;
  let audioPackets = 0;
  let currentSocket = null;

  const server = net.createServer((socket) => {
    connections += 1;
    currentSocket = socket;
    socket.on('data', (chunk) => {
      buffer = Buffer.concat([buffer, chunk]);
      while (buffer.length >= HEADER_SIZE) {
        const magic = buffer.toString('ascii', 0, 4);
        const payloadSize = buffer.readUInt32BE(12);
        const packetSize = HEADER_SIZE + payloadSize;
        if (buffer.length < packetSize) {
          return;
        }

        buffer = buffer.subarray(packetSize);
        if (magic === FRAME_MAGIC) {
          framePackets += 1;
        } else if (magic === AUDIO_MAGIC) {
          audioPackets += 1;
        }
      }
    });

    setTimeout(() => {
      if (socket.destroyed) {
        return;
      }
      socket.write(
        encodeEventPacket({
          type: 'state',
          label: 'FOCUSED',
          confidence: 0.91,
          signals: [{ label: 'steady_gaze', confidence: 0.88 }],
          transport: { video: true, audio: true },
          updatedAt: Date.now() / 1000,
        }),
      );
    }, 500);
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Mock server failed to bind to a TCP port');
  }

  return {
    port: address.port,
    getStats: () => ({ connections, framePackets, audioPackets }),
    close: async () => {
      currentSocket?.destroy();
      await new Promise((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve();
        });
      });
    },
  };
};

const main = async () => {
  const artifactDir = path.resolve('e2e-artifacts');
  await fs.mkdir(artifactDir, { recursive: true });

  const mockServer = await startMockServer();
  console.log(`mock-backend:${mockServer.port}`);
  const app = await electron.launch({
    executablePath: electronBinary,
    args: ['dist-electron/electron/main.js'],
    env: {
      ...process.env,
      COGNITIVESENSE_SERVER_HOST: '127.0.0.1',
      COGNITIVESENSE_SERVER_PORT: String(mockServer.port),
    },
  });

  let page;
  try {
    page = await app.firstWindow();
    console.log('window-opened');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('.webcam', { timeout: 15000 });
    console.log('webcam-selector-ready');

    await waitFor(
      async () => mockServer.getStats().connections > 0,
      10000,
      'Frontend never connected to the mock backend',
    );
    console.log('backend-connected');

    await page.waitForFunction(
      () => {
        const video = document.querySelector('video.webcam');
        return Boolean(video && !video.paused && video.readyState >= 2);
      },
      undefined,
      { timeout: 15000 },
    );
    console.log('video-playing');

    const pageState = await page.evaluate(() => {
      const video = document.querySelector('video.webcam');
      return {
        paused: video?.paused ?? null,
        readyState: video?.readyState ?? null,
        currentTime: video?.currentTime ?? null,
        videoWidth: video?.videoWidth ?? null,
        videoHeight: video?.videoHeight ?? null,
        bodyText: document.body.innerText,
      };
    });
    console.log(`page-state:${JSON.stringify(pageState)}`);

    await waitFor(
      async () => mockServer.getStats().framePackets >= 5,
      15000,
      'Frontend never streamed enough video frames to the backend',
    );
    console.log(`frame-streaming:${mockServer.getStats().framePackets}`);

    const rafCount = await page.evaluate(
      () =>
        new Promise((resolve) => {
          let count = 0;
          const started = performance.now();
          const step = () => {
            count += 1;
            if (performance.now() - started >= 1000) {
              resolve(count);
              return;
            }
            requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }),
    );
    assert(rafCount >= 20, `UI thread appears stalled: ${rafCount} animation frames in 1s`);

    const videoMetrics = await page.evaluate(
      () =>
        new Promise((resolve) => {
          const video = document.querySelector('video.webcam');
          const startTime = video.currentTime;
          const started = performance.now();
          let frameCallbacks = 0;

          const finish = () => {
            resolve({
              paused: video.paused,
              readyState: video.readyState,
              startTime,
              endTime: video.currentTime,
              frameCallbacks,
              supportsVideoFrameCallback:
                typeof video.requestVideoFrameCallback === 'function',
            });
          };

          if (typeof video.requestVideoFrameCallback === 'function') {
            const step = () => {
              frameCallbacks += 1;
              if (performance.now() - started >= 1500) {
                finish();
                return;
              }
              video.requestVideoFrameCallback(step);
            };
            video.requestVideoFrameCallback(step);
            return;
          }

          setTimeout(finish, 1500);
        }),
    );

    assert.equal(videoMetrics.paused, false, 'Video element is paused');
    assert(videoMetrics.readyState >= 2, `Video readyState too low: ${videoMetrics.readyState}`);
    assert(
      videoMetrics.endTime - videoMetrics.startTime > 0.5,
      `Video currentTime barely moved: ${videoMetrics.startTime} -> ${videoMetrics.endTime}`,
    );
    if (videoMetrics.supportsVideoFrameCallback) {
      assert(
        videoMetrics.frameCallbacks >= 10,
        `Camera frames appear frozen: only ${videoMetrics.frameCallbacks} video frame callbacks`,
      );
    }

    await page.getByRole('button', { name: 'Pause uplink' }).click();
    await page.getByRole('button', { name: 'Resume uplink' }).waitFor({ timeout: 5000 });

    const pausedAt = mockServer.getStats().framePackets;
    await sleep(1200);
    const pausedAfter = mockServer.getStats().framePackets;
    assert(
      pausedAfter - pausedAt <= 1,
      `Frame uplink did not pause cleanly: ${pausedAt} -> ${pausedAfter}`,
    );

    await page.getByRole('button', { name: 'Resume uplink' }).click();
    await waitFor(
      async () => mockServer.getStats().framePackets >= pausedAfter + 3,
      8000,
      'Frame uplink did not resume after clicking Resume uplink',
    );

    await page.screenshot({ path: path.join(artifactDir, 'e2e-smoke.png') });

    console.log(
      JSON.stringify(
        {
          ok: true,
          rafCount,
          videoMetrics,
          backendStats: mockServer.getStats(),
          screenshot: path.join('e2e-artifacts', 'e2e-smoke.png'),
        },
        null,
        2,
      ),
    );
  } catch (error) {
    console.error('stats-on-failure', mockServer.getStats());
    if (page) {
      try {
        await page.screenshot({ path: path.join(artifactDir, 'e2e-failure.png') });
      } catch {
        // Ignore screenshot failure while handling test failure.
      }
    }
    throw error;
  } finally {
    await app.close();
    await mockServer.close();
  }
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
