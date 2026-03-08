# 트러블슈팅

대상: 실행 중 실패를 직접 디버깅하는 사용자  
이 문서가 다루는 것: 자주 발생하는 증상, 우선 확인할 명령, 해결 방향  
정상 사용법과 출력 구조: `docs/usage.md`

최근 실제 실행에서 자주 마주친 문제를 정리한다. 우선순위는 **IsaacLab -> Isaac Sim/MCP -> Azure/VLM -> Data Collection -> export/preprocess** 순서로 본다.

## 1. IsaacLab Pipeline

### `env_isaaclab`를 찾지 못함

```text
conda run: error: argument -n: env 'env_isaaclab' not found
```

확인:

```bash
conda env list
```

해결:

- 실제 환경 이름을 `configs/isaaclab_agent_config.yaml`의 `isaaclab.conda_env`에 반영
- `ISAACLAB_PATH`와 conda 환경을 함께 점검

### `isaaclab.sh`를 찾지 못함

```bash
echo $ISAACLAB_PATH
ls $ISAACLAB_PATH/isaaclab.sh
```

### `simulation_app.close()`가 hang됨

현재 레포는 marker file 기반으로 우회한다.

- `run_env.py`는 성공 시 `.success_marker`를 먼저 남김
- evaluator도 result 파일과 marker를 먼저 기록

### `AppLauncher` import 순서 문제

원칙:

```python
from isaaclab.app import AppLauncher
# AppLauncher 초기화 후 physics/torch/env import
```

### 평가 점수가 낮은데 실행은 됨

확인:

```bash
cat outputs/isaaclab/<run_dir>/eval_report.json
```

대표 케이스:

- `rewards=None`
- reward가 전부 0
- goal 함수/threshold가 코드에 반영되지 않음

## 2. Isaac Sim / MCP

### `localhost:8766` 연결 거부

```bash
python3 tests/test_components.py connection
ss -ltnp | rg 8766
```

### MCP 응답 파싱 실패

현재 `MCPClient`는 newline JSON, prefix noise, broken pipe/reset 후 1회 재연결을 자동 처리한다. 그래도 실패하면 Isaac Sim 로그부터 확인한다.

### `execute_script` 결과가 항상 `null`

데이터를 회수하려면 스크립트 내부에서 파일로 써야 한다.

```python
import json
with open('/tmp/isaac_results.json', 'w') as f:
    json.dump({'ok': True}, f)
```

### 로컬 `assets/...` USD가 안 보임

`SceneBuilder`는 `assets/...`를 **레포 루트 기준 절대경로**로 해석한다. 경로가 실제로 존재하는지 먼저 확인한다.

## 3. Azure / VLM

### 인증 오류

```bash
cat .env | rg AZURE_OPENAI
```

필수:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`

### `max_tokens` unsupported, `max_completion_tokens` 사용 요구

최신 Azure GPT-5 계열 deployment에서 발생할 수 있다. 현재 코드는 `max_completion_tokens` 우선, 구버전 deployment에는 `max_tokens` fallback으로 동작한다.

### `MockVLMEvaluator`만 선택됨

```bash
cat .env | rg 'AZURE_OPENAI|ANTHROPIC|GOOGLE'
```

## 4. Data Collection

### ADC 서브모듈 없이 실행 가능 여부

가능하다. 다만 submodule이 없으면 judge prompt/일부 helper는 fallback을 쓴다.

### Pinocchio IK가 안 붙음

```bash
git submodule status
python3 -c "from src.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
```

### Data Collection이 env 생성 단계에서 실패

먼저 IsaacLab 단독 실행이 성공하는지 본다.

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

그다음 `--env-dir`를 명시해 수집 경로만 좁혀서 본다.

### `pipeline_completed=true`인데 task는 실패

이건 정상적으로 가능한 상태다.

- `pipeline_completed=true`: 수집과 정리는 끝남
- `target_met=false`: 목표 성공 episode 수는 못 채움

즉 task 성공 여부는 `target_met`, `successful_episodes`로 판단한다.

### `COLLECTION_SUCCESS_MARKER`를 찾았는데 동작이 이상함

현재 정상 완료 marker는 `COLLECTION_COMPLETE_MARKER`다. `COLLECTION_SUCCESS_MARKER`는 legacy 호환 흔적일 수 있다.

### raw dataset은 있는데 학습에 바로 못 씀

정상이다. raw는 수집 포맷이고, 학습 전에는 아래 경로를 거친다.

```bash
python3 scripts/export_dataset.py <raw_dataset_dir> --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py <export_dir> --output-dir outputs/preprocessed_datasets
```

## 5. Export / Preprocess

### export는 됐는데 schema가 헷갈림

기본값은 `adc_compatible`이다. current TCP가 필요한 분석용 확장 schema만 `canonical_training`을 명시적으로 선택한다.

### `adc_raw` export가 실패

이 경로는 host Python에 `pyarrow`, `pandas`, `datasets`, `lerobot`가 필요하다. 실제 ADC dataset root와 robot config도 함께 필요하다.

### raw를 바로 학습에 넣고 싶음

권장하지 않는다. 기본 경로는 `raw -> adc_compatible export -> preprocess`다.

## 6. 일반

### `ModuleNotFoundError: No module named 'src'`

프로젝트 루트에서 실행해야 한다.

```bash
cd /path/to/Simulation-Generation-Agent
python3 scripts/run_isaac_lab.py ...
```

### `outputs/`가 너무 커짐

`outputs/`는 git 대상이 아니므로 불필요한 run 디렉터리만 선택적으로 정리한다.

## 관련 문서

- `docs/getting_started.md`
- `docs/usage.md`
- `docs/dataset_alignment_and_export.md`
- `docs/evaluation.md`
