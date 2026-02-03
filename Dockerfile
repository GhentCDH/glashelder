FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cifs-utils \
    curl \
    git \
    vim \
    tmux \
    just \
    libgl1 \
    libglib2.0-0 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user with UID matching host user (default 1000, override with --build-arg)
ARG UID=1000
RUN useradd -m -u $UID -s /bin/bash dev

# Prepare mount point (needs to be accessible by dev user)
RUN mkdir -p /mnt/share && chown dev:dev /mnt/share

# Create cache directory for uv
RUN mkdir -p /home/dev/.cache && chown -R dev:dev /home/dev/.cache

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
