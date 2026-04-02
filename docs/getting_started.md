# 설치 가이드

대상: 처음 이 레포를 실행하는 사용자
세부 CLI 옵션과 결과 해석: `docs/usage.md`

## 1. 공통 설치

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

# submodule이 빠졌을 경우 수동 초기화
git submodule update --init --recursive

pip install -e .
cp .env.example .env
```

필수 `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

선택 설정:

```bash
# VLM backends (Isaac Sim 시각 검증용)
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# Paths
# ISAACLAB_PATH=/home/you/workspace/IsaacLab

# 토큰 사용량 추적
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLab 설치 (주력 경로)

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

또는 `configs/isaaclab_agent_config.yaml`의 `isaaclab.path`를 직접 설정한다.

연결 확인:

```bash
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCP 설치 (선택)

Isaac Sim은 시각 검증 경로에서만 필요하다.

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/isaac-sim-mcp.git
```

```bash
cd ~/workspace/isaac-sim
./isaac-sim.streaming.sh \
  --ext-folder /home/$USER/workspace/isaac-sim-mcp \
  --enable isaac.sim.mcp_extension
```

정상 기동 시 `localhost:8766`에서 TCP 소켓을 수신한다.

연결 확인:

```bash
python3 tests/test_components.py connection
```

## 4. Data Collection 설치 (선택)

Data Collection은 IsaacLab 위에서 동작한다.

```bash
pip install -e ".[data-collection]"
pip install pin
```

확인:

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. Task Spec Agent 설치 (NL → YAML)

Task Spec Agent는 자연어 입력을 YAML 태스크 명세로 변환하는 Stage 1 파이프라인입니다.

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

벡터 스토어는 첫 실행 시 `data/vector_store/`에 자동 생성됩니다 (tasks/ 디렉토리의 기존 YAML을 인덱싱).

확인:

```bash
cd scripts/task_spec_agent
python3 -c "from rag_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

단독 실행:

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. 첫 성공 실행 권장 순서

### Full Pipeline (NL → YAML → IsaacLab → CaP → Video)

```bash
bash scripts/run_full_test.sh
```

13개 Franka 태스크에 대해 3단계를 순차 실행합니다:
1. **NL → YAML**: `task_spec_agent`가 자연어를 YAML로 변환
2. **YAML → IsaacLab**: LLM이 환경 Python 코드를 생성하고 실행 검증
3. **CaP → Data**: LLM이 스킬 코드를 생성하고, IK로 실행하고, 성공 판정 후 데이터 수집

### IsaacLab 단독 실행

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

성공 기준:
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/eval_report.json`

### Data Collection 단독 실행

```bash
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>
```

성공 기준:
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### 학습 준비 경로

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. 다음에 볼 문서

- 파이프라인 아키텍처 상세: `docs/architecture.md`
- 실제 옵션/출력 구조/결과 해석: `docs/usage.md`
