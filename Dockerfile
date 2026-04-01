# 1. Base Image 선언 (PyTorch 및 CUDA 환경)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]

ARG PYTHON_VERSION=3.11
ARG ISAACSIM_PIP_VERSION=5.1.0
ARG ISAACLAB_VERSION=v2.3.2

# 2. 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TERM=xterm-256color \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/isaaclab-env \
    PATH=/opt/isaaclab-env/bin:$PATH \
    PYTHON_BIN=/opt/isaaclab-env/bin/python \
    ISAACLAB_PATH=/workspace/IsaacLab \
    SIMGEN_IN_DOCKER=1 \
    SIMGEN_ARTIFACT_ROOT=/workspace/artifacts \
    ADC_PATH=/workspace/Simulation-Generation-Agent/external/AutoDataCollector \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

# 3. 시스템 패키지 설치 (필요시)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      gpg \
      build-essential \
      cmake \
      pkg-config \
      ncurses-bin \
      ncurses-term \
      wget \
      libasound2 \
      libdbus-1-3 \
      libegl1 \
      libfontconfig1 \
      libfreetype6 \
      libgl1 \
      libglib2.0-0 \
      libglu1-mesa \
      libgtk-3-0 \
      libnss3 \
      libsm6 \
      libvulkan1 \
      libx11-6 \
      libxcursor1 \
      libxext6 \
      libxinerama1 \
      libxi6 \
      libxkbcommon0 \
      libxrandr2 \
      libxrender1 \
      libxt6 \
      libxtst6 \
      libxxf86vm1 \
      mesa-vulkan-drivers \
      vulkan-tools && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv && \
    mkdir -p /usr/share/vulkan/icd.d && \
    ln -sf /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json && \
    rm -rf /var/lib/apt/lists/*

# 4. Python 가상환경 + Isaac Sim + PyTorch
RUN python${PYTHON_VERSION} -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
      "isaacsim[all,extscache]==${ISAACSIM_PIP_VERSION}" \
      --extra-index-url https://pypi.nvidia.com && \
    python -m pip install --no-cache-dir \
      -U torch==2.7.0 torchvision==0.22.0 \
      --index-url https://download.pytorch.org/whl/cu128

# 5. IsaacLab 소스 설치
RUN git clone --branch "${ISAACLAB_VERSION}" --depth 1 \
      https://github.com/isaac-sim/IsaacLab.git \
      "${ISAACLAB_PATH}" && \
    cd "${ISAACLAB_PATH}" && \
    python -m pip install --no-cache-dir toml && \
    ./isaaclab.sh -i none

# 6. 작업 디렉토리 설정
WORKDIR /workspace/Simulation-Generation-Agent

# 7. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir \
      numpy==1.26.0 \
      typing_extensions==4.12.2 \
      packaging==23.0 \
      pyparsing==3.0.9 && \
    python -m pip install --no-cache-dir --no-build-isolation flatdict==4.0.1 && \
    python -m pip install --no-cache-dir --no-build-isolation -e "${ISAACLAB_PATH}/source/isaaclab"

# 8. 소스 코드 복사
COPY . .

# 9. 실행 권한 부여
RUN chmod +x run_agent.sh && \
    mkdir -p /workspace/artifacts

# 10. (중요) API Key 등은 빌드 시 넣지 말고 환경변수로 받도록 설정
ENV OPENAI_API_KEY="" \
    AZURE_OPENAI_API_KEY="" \
    HF_TOKEN=""

# 11. 컨테이너 시작 시 실행 명령어
ENTRYPOINT ["./run_agent.sh"]
CMD ["--help"]
