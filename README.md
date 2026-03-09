# Simulation-Generation-Agent

YAML task 문서에서 시작해 IsaacLab 환경 생성, 선택적 Isaac Sim 시각 검증, IsaacLab 기반 데이터 수집까지 연결하는 프레임워크입니다. 현재 기준 주력 경로는 **IsaacLab Pipeline**입니다.

## 문서 시작점

- 빠른 문서 맵: `docs/README.md`
- 설치와 첫 실행: `docs/getting_started.md`
- 실제 CLI 사용법과 출력 해석: `docs/usage.md`
- dataset/export/preprocess 기준: `docs/dataset_alignment_and_export.md`
- task 상태 보드: `docs/tasks_overview.md`
- task 한글 설명: `docs/task_descriptions_ko.md`
- 문제 해결: `docs/troubleshooting.md`

## 주력 파이프라인

1. **IsaacLab Pipeline**  
   task YAML -> LLM 코드 생성 -> `isaaclab.sh` 실행 -> 선택적 평가
2. **Isaac Sim Pipeline**  
   동일 task YAML -> MCP 씬 빌드 -> 스크린샷 -> VLM 시각 검증
3. **Data Collection**  
   IsaacLab env 위에서 detect -> plan -> execute -> judge -> record -> `raw_dataset/`
4. **학습 준비**  
   `raw_dataset/` -> `export_dataset.py` (기본 `adc_compatible`) -> `preprocess_dataset.py`
5. **선택적 LeRobot 변환/배포**  
   `convert_lerobot_dataset.py` -> `check_lerobot_dataset.py` -> `publish_lerobot_dataset.py`

## Quick Start

### 1. 공통 설치

```bash
git clone <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

필수 `.env`:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
```

### 2. IsaacLab 연결

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

### 3. Data Collection + 학습 준비

```bash
pip install -e ".[data-collection]"
pip install pin
git submodule update --init external/AutoDataCollector

python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>

python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets

# optional: local LeRobot dataset + Hub upload
python3 scripts/convert_lerobot_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/franka_stack_sim
python3 scripts/check_lerobot_dataset.py outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
python3 scripts/publish_lerobot_dataset.py outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id <org>/franka_stack_sim --local-repo-id local/franka_stack_sim --private
```

세부 설치, 옵션, 결과 해석은 `docs/getting_started.md`, `docs/usage.md`, `docs/dataset_alignment_and_export.md`를 본다.

## 프로젝트 구조

```text
Simulation-Generation-Agent/
├── src/common/              # Azure/OpenAI, MCP 공용 클라이언트
├── src/isaac_lab/           # 주력 파이프라인: 코드 생성 + 평가
├── src/isaac_sim/           # 시각 검증 파이프라인
├── src/data_collection/     # IsaacLab 기반 데이터 수집 / export / preprocess
├── src/task_search/         # task YAML 카탈로그/검색 메타데이터
├── scripts/                 # CLI 진입점
├── configs/                 # agent / pipeline / eval / robot profile 설정
├── tasks/                   # task YAML corpus
├── docs/                    # 사용자 문서
├── assets/                  # 로컬 USD/URDF 자산
└── external/AutoDataCollector/  # ADC 서브모듈, read-only
```

## 현재 task corpus

- 총 task YAML: `78`
- 실사용 robot corpus: `franka`, `openarm`, `so101`, `ur10e`
- `tasks/templates/`는 빈 placeholder 디렉터리다

세부 집계와 최신 검증 상태는 아래 문서를 본다.

- `docs/tasks_overview.md`
- `docs/task_taxonomy.md`
- `docs/task_descriptions_ko.md`

## 참고

- `external/AutoDataCollector/`는 read-only 서브모듈이다.
- 생성 산출물은 `outputs/` 아래에 저장되며 git에 커밋하지 않는다.
- MCP 기반 Isaac Sim 검증은 `localhost:8766`에 의존한다.
