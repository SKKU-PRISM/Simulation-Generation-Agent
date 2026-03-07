# 설치 가이드

이 레포는 세 경로를 제공합니다. **기본 설치 대상은 IsaacLab Pipeline**이며, Isaac Sim과 Data Collection은 선택 확장입니다.

## 권장 설치 순서

1. 이 프로젝트 + Python 의존성 설치
2. Azure OpenAI 환경변수 설정
3. IsaacLab 설치 및 연결
4. 필요 시 Isaac Sim + MCP 연결
5. 필요 시 Data Collection extra + ADC 서브모듈 초기화

## 1. 공통 설치

```bash
git clone <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

`.env` 최소 설정:

```bash
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
```

선택 설정:

```bash
# VLM backends
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...
# OLLAMA_BASE_URL=http://localhost:11434

# Paths
# ISAACLAB_PATH=/home/you/workspace/IsaacLab
# ADC_PATH=/home/you/workspace/AutoDataCollector
```

## 2. IsaacLab 설치 (권장, 주력 파이프라인)

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

프로젝트에서 IsaacLab 위치를 인식시키는 방법:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

또는 `configs/isaaclab_agent_config.yaml`의 `isaaclab.path`를 직접 설정합니다.

### IsaacLab 연결 확인

```bash
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

성공 기준:

- `isaaclab.sh`가 존재한다
- `env_isaaclab` conda 환경이 동작한다
- `--dry-run`이 `outputs/isaaclab/<task>_<timestamp>/`에 코드를 생성한다

## 3. Isaac Sim + MCP 설치 (선택)

Isaac Sim은 주로 YAML task의 **시각적 검증**에 사용합니다.

### Isaac Sim

- Isaac Sim 5.x 계열 기준으로 사용 중
- GPU 및 NVIDIA 드라이버가 필요합니다

### isaac-sim-mcp

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/isaac-sim-mcp.git
```

Isaac Sim을 MCP extension과 함께 실행:

```bash
cd ~/workspace/isaac-sim
./isaac-sim.streaming.sh \
  --ext-folder /home/$USER/workspace/isaac-sim-mcp \
  --enable isaac.sim.mcp_extension
```

정상 기동 시 `localhost:8766`에서 TCP 소켓을 수신합니다.

### `.mcp.json`의 역할

루트의 `.mcp.json`은 **Claude Code 등의 MCP 클라이언트가 `isaac_mcp/server.py`를 띄울 수 있게 하는 설정**입니다.  
실제 파이프라인 런타임은 여전히 Isaac Sim extension의 TCP 서버(`localhost:8766`)와 통신합니다.

### Isaac Sim 연결 확인

```bash
python3 tests/test_components.py connection
```

## 4. Data Collection 설치 (선택)

Data Collection은 IsaacLab 환경 위에서 동작합니다. 즉, **IsaacLab 설치가 먼저**입니다.

```bash
pip install -e ".[data-collection]"
pip install pin                    # Pinocchio — 6-DOF IK 필수
git submodule update --init external/AutoDataCollector
```

**필수 의존성**:
- `pinocchio` (`pip install pin`): 6-DOF IK solver (FK 기반 `robot_xyzrpy` 계산에도 사용)
- ADC 서브모듈: FK/IK 엔진 + judge 프롬프트 재사용 (없으면 fallback)

### Data Collection 연결 확인

```bash
# ADC 서브모듈 확인
python3 -c "from src.data_collection.adc_imports import is_adc_available; print(is_adc_available())"

# Pinocchio 확인
python3 -c "import pinocchio; print('Pinocchio OK')"

# CLI 확인
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. 첫 실행 권장 순서

### IsaacLab 우선 검증

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

확인할 산출물:

- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/run_env.py`
- `outputs/isaaclab/<run_dir>/eval_report.json`

### Isaac Sim 시각 검증

```bash
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto
```

확인할 산출물:

- `outputs/isaac_sim/<run_dir>/run_report.json`
- `outputs/isaac_sim/<run_dir>/iter_*.png`

### Data Collection

```bash
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml
```

확인할 산출물:

- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

## 6. 체크리스트

### 공통

- [ ] `pip install -e .` 성공
- [ ] `.env`에 `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` 설정

### IsaacLab

- [ ] `ISAACLAB_PATH` 설정 또는 config 반영
- [ ] `env_isaaclab` 사용 가능
- [ ] `python3 scripts/run_isaac_lab.py ... --dry-run` 성공

### Isaac Sim

- [ ] Isaac Sim 실행 가능
- [ ] `isaac-sim-mcp` 설치 완료
- [ ] Isaac Sim extension이 `localhost:8766`에서 수신
- [ ] `python3 tests/test_components.py connection` 성공

### Data Collection

- [ ] `pip install -e ".[data-collection]"` 성공
- [ ] `git submodule update --init external/AutoDataCollector` 완료
- [ ] ADC 없이도 fallback으로 실행 가능함을 이해

## 관련 문서

- `docs/usage.md`
- `docs/evaluation.md`
- `docs/task_yaml_spec.md`
- `docs/troubleshooting.md`
