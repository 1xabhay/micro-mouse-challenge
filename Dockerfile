# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uv from the official image — no curl/install script needed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer: cached until pyproject.toml changes.
# CPU-only torch keeps the image ~2GB smaller than the default CUDA wheels.
COPY pyproject.toml README.md ./
RUN mkdir -p src/micromouse && touch src/micromouse/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      -e ".[rl,dash,dev]"

COPY . .
RUN uv pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "micromouse.cli", "dash", "--host", "0.0.0.0", "--port", "8000"]
