#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Usage:
  setup_docker_host.sh [--help]

Installs Docker Engine, Docker Buildx/Compose, and NVIDIA Container Toolkit on
an Ubuntu host. This script requires sudo privileges.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
  TARGET_USER="${SUDO_USER:-root}"
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Docker host dependencies." >&2
    exit 1
  fi
  SUDO="sudo"
  TARGET_USER="${USER}"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported host: /etc/os-release not found." >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "Unsupported host OS: ${ID:-unknown}. Ubuntu is required." >&2
  exit 1
fi

ARCH="$(dpkg --print-architecture)"
CODENAME="${VERSION_CODENAME:-}"
if [[ -z "${CODENAME}" ]]; then
  echo "Could not determine Ubuntu codename from /etc/os-release." >&2
  exit 1
fi

echo "[1/7] Installing Docker apt prerequisites..."
${SUDO} apt-get update
${SUDO} apt-get install -y ca-certificates curl gnupg lsb-release

echo "[2/7] Configuring Docker apt repository..."
${SUDO} install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | ${SUDO} gpg --yes --dearmor -o /etc/apt/keyrings/docker.gpg
${SUDO} chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "[3/7] Installing Docker Engine..."
${SUDO} apt-get update
${SUDO} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[4/7] Configuring NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${SUDO} gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

echo "[5/7] Installing NVIDIA Container Toolkit..."
${SUDO} apt-get update
${SUDO} apt-get install -y nvidia-container-toolkit

echo "[6/7] Enabling NVIDIA runtime for Docker..."
${SUDO} nvidia-ctk runtime configure --runtime=docker
${SUDO} systemctl restart docker

echo "[7/7] Adding ${TARGET_USER} to docker group..."
if getent group docker >/dev/null 2>&1; then
  ${SUDO} usermod -aG docker "${TARGET_USER}" || true
fi

echo
echo "Docker host setup completed."
echo "Open a new login shell or run: newgrp docker"
echo "Quick checks:"
echo "  docker --version"
echo "  docker compose version"
echo "  docker run --rm hello-world"
echo "  docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi"
