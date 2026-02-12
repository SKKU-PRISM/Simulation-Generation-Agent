# 설치 가이드

이 문서는 Simulation-Generation-Agent의 전체 설치 과정을 안내합니다.

---

## 전체 의존성 구조

```
Simulation-Generation-Agent
├── Isaac Sim 4.2.0+ ─────── Isaac Sim Pipeline (씬 빌드 + VLM 평가)
│   └── isaac-sim-mcp ────── MCP 서버 (TCP 소켓 통신)
├── IsaacLab v2.3.2+ ─────── IsaacLab Pipeline (코드 생성 + 실행)
│   └── conda env: env_isaaclab
└── Azure OpenAI API ──────── 두 파이프라인 공용 (LLM + VLM)
```

**어떤 파이프라인을 사용하느냐에 따라 설치 범위가 다릅니다:**

| 사용 목적 | 필수 설치 |
|-----------|----------|
| IsaacLab Pipeline만 (코드 생성) | 이 프로젝트 + IsaacLab + Azure OpenAI 키 |
| Isaac Sim Pipeline만 (씬 빌드) | 이 프로젝트 + Isaac Sim + isaac-sim-mcp + VLM 키 |
| 두 파이프라인 모두 | 전부 |

---

## 1. 시스템 요구사항

- **OS**: Ubuntu 22.04+ (또는 동등한 Linux)
- **GPU**: NVIDIA GPU (RTX 2070 이상 권장)
- **CUDA**: 12.x
- **Python**: 3.10+
- **Conda**: Miniconda 또는 Anaconda (IsaacLab Pipeline 사용 시)
- **디스크**: Isaac Sim ~15GB, IsaacLab ~5GB

---

## 2. 이 프로젝트 설치

```bash
# 레포지토리 클론
git clone <repo-url>
cd Simulation-Generation-Agent

# Python 의존성 설치
pip install -e .
```

### 환경변수 설정

```bash
# .env 파일 생성 (API 키는 절대 git에 커밋하지 않음)
cp .env.example .env
```

`.env` 파일을 편집하여 다음을 설정합니다:

```bash
# === 필수 (두 파이프라인 공용) ===
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/

# === 선택: VLM 백엔드 (Isaac Sim Pipeline에서 VLM 평가 시) ===
# Azure가 최우선으로 사용되므로, 위 키만 있으면 VLM도 Azure로 동작합니다.
# 아래는 추가 백엔드가 필요한 경우에만 설정:
# ANTHROPIC_API_KEY=your-anthropic-key    # Claude VLM
# GOOGLE_API_KEY=your-google-key          # Gemini VLM (무료)

# === 선택: 모델명 오버라이드 ===
# AZURE_OPENAI_MODEL=gpt-5-mini           # 기본값
# CLAUDE_MODEL=claude-sonnet-4-20250514
# GEMINI_MODEL=gemini-2.5-flash
# OLLAMA_MODEL=llava:7b
# OLLAMA_BASE_URL=http://localhost:11434

# === 선택: 경로 오버라이드 ===
# ISAACLAB_PATH=/path/to/IsaacLab
```

---

## 3. Isaac Sim Pipeline 전용: Isaac Sim + MCP 설치

> IsaacLab Pipeline만 사용한다면 이 섹션을 건너뛰세요.

### 3.1 Isaac Sim 설치

1. [NVIDIA Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/) 설치
2. Launcher에서 **Isaac Sim 4.2.0+** 설치
3. 설치 확인:
   ```bash
   # Isaac Sim 실행 확인 (GUI 모드)
   ~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh
   ```

### 3.2 isaac-sim-mcp 설치 (MCP 서버)

isaac-sim-mcp는 Isaac Sim과 이 프로젝트를 TCP 소켓으로 연결하는 MCP 서버입니다.

```bash
# 이 프로젝트와 같은 워크스페이스에 클론
cd ~/workspace  # 또는 원하는 디렉토리
git clone https://github.com/isaac-sim/isaac-sim-mcp.git
```

#### MCP 서버 시작

```bash
# Isaac Sim을 스트리밍 모드로 시작 + MCP 서버 자동 로드
cd ~/workspace/isaac-sim-mcp
./run_isaac_mcp_streaming.sh 0   # 0 = GPU ID
```

서버가 시작되면 `localhost:8766`에서 TCP 소켓을 수신합니다.

#### 연결 설정 (`.mcp.json`)

이 프로젝트 루트의 `.mcp.json`에서 isaac-sim-mcp 경로를 **본인의 경로로 수정**해야 합니다:

```json
{
  "mcpServers": {
    "isaac-sim": {
      "command": "python3",
      "args": ["/your/path/to/isaac-sim-mcp/isaac_mcp/server.py"]
    }
  }
}
```

#### 연결 확인

```bash
# MCP 서버가 실행 중인 상태에서:
python3 tests/test_components.py connection

# 기대 출력:
# ✓ MCP connection successful
```

---

## 4. IsaacLab Pipeline 전용: IsaacLab 설치

> Isaac Sim Pipeline만 사용한다면 이 섹션을 건너뛰세요.

### 4.1 IsaacLab 클론 및 설치

```bash
cd ~/workspace  # 또는 원하는 디렉토리
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

IsaacLab은 자체 conda 환경이 필요합니다. 공식 설치 가이드를 따릅니다:

> 참고: [IsaacLab 공식 설치 문서](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

요약:

```bash
# conda 환경 생성 (IsaacLab 공식 스크립트 사용)
./isaaclab.sh --install

# 또는 수동으로 conda 환경 생성 후 설치
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
pip install -e .
```

설치 확인:

```bash
# conda 환경에서 import 테스트
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab OK:', isaaclab.__version__)"
```

### 4.2 이 프로젝트와 연결

IsaacLab 경로를 설정합니다. 두 가지 방법 중 택일:

**방법 A: 환경변수 (권장)**

```bash
# .env 파일에 추가
echo 'ISAACLAB_PATH=/path/to/IsaacLab' >> .env
```

**방법 B: config 파일 직접 수정**

`configs/isaaclab_agent_config.yaml`에서:

```yaml
isaaclab:
  path: "/path/to/IsaacLab"  # null → 실제 경로로 변경
  conda_env: "env_isaaclab"   # conda 환경 이름 (변경 시 수정)
```

### 4.3 연결 확인

```bash
# isaaclab.sh가 실행 가능한지 확인
conda run -n env_isaaclab --no-capture-output \
  bash -c "echo 'Conda env OK'"

# IsaacLab 참조 코드 디렉토리 존재 확인
ls $ISAACLAB_PATH/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/
# 기대 출력: stack/ lift/ reach/ 등의 디렉토리
```

---

## 5. 설치 검증 체크리스트

### 공통

- [ ] `pip install -e .` 성공
- [ ] `.env` 파일에 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` 설정

### Isaac Sim Pipeline

- [ ] Isaac Sim 실행 가능
- [ ] isaac-sim-mcp 클론 완료
- [ ] `.mcp.json` 경로 수정 완료
- [ ] `./run_isaac_mcp_streaming.sh 0` 으로 MCP 서버 시작
- [ ] `python3 tests/test_components.py connection` 성공

### IsaacLab Pipeline

- [ ] IsaacLab 클론 및 설치 완료
- [ ] conda 환경 `env_isaaclab` 생성 완료
- [ ] `ISAACLAB_PATH` 환경변수 또는 config 설정 완료
- [ ] `conda run -n env_isaaclab -- python -c "import isaaclab"` 성공

---

## 다음 단계

설치가 완료되었으면 [사용법](usage.md)을 참고하세요.
