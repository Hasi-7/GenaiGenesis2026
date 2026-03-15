# Remote Deployment

## Services

The project now has two deployable backend services:

1. `analysis`
   - raw TCP media ingress on `9000`
   - runs the cognitive-state inference pipeline

2. `control`
   - FastAPI control-plane API on `8080`
   - serves the basic website
   - handles account registration, login, device claiming, and period summaries

Both services share the same SQLite file by default.

## Local Docker Compose

```bash
docker compose up --build
```

Exposed ports:

- `9000` -> media ingress
- `8080` -> control API and website

## Remote Machine Setup

Recommended minimum:

- Docker + Docker Compose
- persistent volume for `/app/data`
- firewall rules for `9000` and `8080`
- reverse proxy/TLS in front of `8080` for production

## Environment Variables

- `OPENAI_API_KEY`
- `COGNITIVESENSE_SERVER_PORT` (defaults to `9000`)
- `COGNITIVESENSE_DB_PATH` (defaults to `/app/data/control.db` in Docker)

## Website / Control Plane

Open:

```text
http://<server-host>:8080/
```

Available API routes:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/me`
- `GET /api/devices/unclaimed`
- `POST /api/devices/claim`
- `GET /api/devices`
- `GET /api/overview`
- `GET /api/timeline`

## Device Association Model

- Each device reports a stable `deviceId` in the media hello packet.
- The control plane records it as an unclaimed device.
- A signed-in user can claim that device from the website.
- Period scores aggregate across all devices claimed by that user.

## Current Persistence Model

Stored today:

- users
- auth tokens
- devices
- monitoring sessions
- state samples
- feedback events

Not stored by default:

- raw frames
- raw audio

## Production Next Steps

- replace SQLite with Postgres if multi-process write volume grows
- put `control` behind HTTPS
- add backup for the data volume
- add signed installers + first-run onboarding to point desktop clients at the remote host
