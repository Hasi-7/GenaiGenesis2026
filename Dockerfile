FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libportaudio2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

COPY . .

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/server"
ENV COGNITIVESENSE_DB_PATH="/app/data/control.db"

EXPOSE 9000 8080

CMD ["/app/.venv/bin/python", "main.py", "server"]
