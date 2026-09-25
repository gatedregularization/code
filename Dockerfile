FROM ubuntu:24.04 AS dev

# Python / build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    neovim \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    zsh && \
    pip3 install --no-cache-dir uv --break-system-packages && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user matching the host user's UID/GID so that
# files written by the container are not owned by root on the host.
ARG USER_ID=1000
ARG GROUP_ID=1000
# 1. Rename the "ubuntu" group to "devuser" and set the GID
# 2. Rename the "ubuntu" user to "devuser", set the UID, and change the shell
RUN groupmod -n devuser -g ${GROUP_ID} ubuntu && \
    usermod -l devuser -u ${USER_ID} -m -d /home/devuser -s /bin/zsh ubuntu && \
    touch /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser /app
ENV HOME=/home/devuser \
    PATH="/home/devuser/.local/bin:${PATH}"

USER devuser

# Activate the venv for all subsequent layers and at runtime.
# Placed in home dir so the .:/app bind mount in Compose doesn't shadow it.
ENV VIRTUAL_ENV=/home/devuser/.venv \
    PATH="/home/devuser/.venv/bin:/home/devuser/.local/bin:${PATH}"
RUN --mount=type=bind,src=requirements.txt,target=requirements.txt \
    --mount=type=cache,gid=${GROUP_ID},uid=${USER_ID},target=/home/devuser/.cache/uv \
    uv venv --python 3.12 /home/devuser/.venv && \
    uv pip install -r requirements.txt

# Keeps the container alive by default (Dev Container)
CMD ["sleep", "infinity"]


FROM ubuntu:24.04 AS run

# Python / build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user matching the host user's UID/GID so that
# files written by the container are not owned by root on the host.
# Named "devuser" (same as dev stage) to keep venv paths consistent and simple.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupmod -n devuser -g ${GROUP_ID} ubuntu && \
    usermod -l devuser -u ${USER_ID} -m -d /home/devuser ubuntu && \
    chown -R devuser:devuser /home/devuser /app
ENV HOME=/home/devuser \
    PATH="/home/devuser/.local/bin:${PATH}"

USER devuser

COPY --from=dev /home/devuser/.venv /home/devuser/.venv
ENV VIRTUAL_ENV=/home/devuser/.venv \
    PATH="/home/devuser/.venv/bin:/home/devuser/.local/bin:${PATH}"

ENTRYPOINT [ "python" ]
CMD ["--help"]
