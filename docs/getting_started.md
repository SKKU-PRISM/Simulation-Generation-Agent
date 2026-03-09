# 설치 가이드

대상: 처음 이 레포를 실행하는 사용자  
이 문서가 다루는 것: 설치, 외부 의존성 연결, 첫 성공 실행  
세부 CLI 옵션과 결과 해석: `docs/usage.md`

이 레포는 세 경로를 제공한다. 기본 설치 대상은 **IsaacLab Pipeline**이며, Isaac Sim과 Data Collection은 선택 확장이다.

## 1. 공통 설치

```bash
git clone <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

필수 `.env`:

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
git submodule update --init external/AutoDataCollector
```

확인:

```bash
python3 -c "from src.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. 첫 성공 실행 권장 순서

### IsaacLab

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

성공 기준:

- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/eval_report.json`

### Data Collection

```bash
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>
```

성공 기준:

- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`
- `outputs/data_collection/<run_dir>/COLLECTION_COMPLETE_MARKER`

### 기본 학습 준비 경로

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### 선택: local LeRobot dataset 생성

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/<dataset_name>
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/<dataset_name> \
  --repo-id local/<dataset_name>
```

필요 시 아래처럼 Hub 업로드를 수행한다.

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/<dataset_name> \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/<dataset_name> \
  --private
```

## 6. 다음에 볼 문서

- 실제 옵션/출력 구조/결과 해석: `docs/usage.md`
- 평가 점수 해석: `docs/evaluation.md`
- dataset/export/preprocess 기준: `docs/dataset_alignment_and_export.md`
- 오류 대응: `docs/troubleshooting.md`
