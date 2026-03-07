# 트러블슈팅

최근 실제 실행에서 자주 마주치는 문제를 정리합니다. 우선순위는 **IsaacLab -> Isaac Sim/MCP -> Azure/VLM -> Data Collection** 순서로 봅니다.

## 1. IsaacLab Pipeline

### `env_isaaclab`를 찾지 못함

증상:

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

원인:

- `ISAACLAB_PATH`가 잘못됐거나 설치가 끝나지 않음

해결:

```bash
echo $ISAACLAB_PATH
ls $ISAACLAB_PATH/isaaclab.sh
```

### `simulation_app.close()`가 hang됨

현재 레포는 marker file 기반으로 우회합니다.

- `run_env.py`는 성공 시 `.success_marker`를 먼저 남김
- evaluator의 `eval_runner.py`도 `eval_results.json`과 marker를 먼저 기록
- subprocess가 timeout되어도 marker/result 파일이 있으면 성공으로 판단 가능

### `AppLauncher` import 순서 문제

증상:

- IsaacLab import 에러
- physics 관련 segfault
- 실행 직후 프로세스 비정상 종료

원인:

- `AppLauncher`보다 먼저 physics 모듈을 import함

원칙:

```python
from isaaclab.app import AppLauncher
# AppLauncher 초기화
# 그 다음에 torch / isaaclab.envs import
```

### 평가 점수가 낮은데 실행은 됨

대표 케이스:

- `rewards=None`: 현재 기본 설정에서는 `FAIL`이 아니라 `3/5 WARN`
- reward가 전부 0: `reward_computation`이 `3/5 WARN`
- custom goal 함수가 없거나 threshold가 코드에 반영되지 않음

확인:

```bash
cat outputs/isaaclab/<run_dir>/eval_report.json
```

### 상대 출력 디렉토리로 evaluator 실행 시 실패

예전에는 상대 `output_dir`에서 `eval_runner.py` 경로가 꼬일 수 있었습니다. 현재는 절대경로로 정규화합니다.

권장:

```bash
python3 scripts/evaluate.py outputs/isaaclab/<run_dir> tasks/franka/stack/franka_stack.yaml
```

## 2. Isaac Sim / MCP

### `localhost:8766` 연결 거부

증상:

```text
ConnectionRefusedError: [Errno 111] Connection refused
```

해결:

```bash
python3 tests/test_components.py connection
ss -ltnp | rg 8766
```

Isaac Sim은 MCP extension과 함께 실행되어야 합니다.

### MCP 응답 파싱 실패

현재 `MCPClient`는 다음을 자동 처리합니다.

- newline-delimited JSON 응답
- JSON 앞뒤에 붙은 불필요한 prefix
- broken pipe / reset / timeout 발생 시 1회 재연결 후 재시도

그래도 실패하면:

- Isaac Sim 쪽 로그 확인
- `tests/test_components.py connection`부터 다시 점검

### `execute_script` 결과가 항상 `null`

이 제약은 여전히 남아 있습니다. 데이터를 회수하려면 스크립트 내부에서 파일로 써야 합니다.

```python
import json
with open("/tmp/isaac_results.json", "w") as f:
    json.dump({"ok": True}, f)
```

### `reset_scene`가 동작하지 않음

MCP extension의 제약입니다. 새 stage가 필요하면 `execute_script`에서 직접 초기화합니다.

```python
omni.usd.get_context().new_stage()
```

### 로컬 `assets/...` USD가 안 보임

현재 `SceneBuilder`는 `assets/...`를 **레포 루트 기준 절대경로**로 해석합니다. 특히 SO-101 로컬 자산은 이 규칙을 기대합니다.

확인:

- 경로가 `src/assets/...`가 아니라 `assets/...`로 시작하는지
- 파일이 실제로 존재하는지

## 3. Azure / VLM

### 인증 오류

증상:

```text
openai.AuthenticationError: Error code: 401
```

해결:

```bash
cat .env | rg AZURE_OPENAI
```

필수:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`

`AZURE_OPENAI_BASE_URL`은 `/openai/v1/`로 끝나는 값을 사용합니다.

### `temperature is not supported`

`gpt-5-mini` 같은 모델은 `temperature`를 지원하지 않을 수 있습니다. 현재 `src/common/llm_client.py`가 자동 fallback합니다.

### `max_tokens` unsupported, `max_completion_tokens` 사용 요구

증상:

```text
Unsupported parameter: 'max_tokens' is not supported with this model.
Use 'max_completion_tokens' instead.
```

설명:

- 최신 Azure GPT-5 계열 chat deployment에서 발생 가능
- 현재 `src/isaac_sim/vlm_evaluator.py`는 `max_completion_tokens` 우선, 구버전 deployment에는 `max_tokens` fallback으로 동작

### `MockVLMEvaluator`만 선택됨

원인:

- 실제 VLM backend가 모두 비활성화됨

해결:

```bash
cat .env | rg 'AZURE_OPENAI|ANTHROPIC|GOOGLE'
```

가장 간단한 경로는 Azure 설정입니다.

## 4. Isaac Sim screenshot 관련

### 첫 캡처는 실패하고 다음 반복에서 성공함

Isaac Sim 러너는 실제로 이런 경우를 허용합니다.

- iteration 1: build 성공, screenshot 실패
- iteration 2: screenshot 성공, VLM 평가 성공

즉 1회 실패만으로 전체 파이프라인 실패라고 보지 않습니다. 최종 결과는 `run_report.json`의 `success`와 `iterations`를 기준으로 판단합니다.

### 캡처 산출물이 `iter_*.png` 외에 `rgb_*.png`로 많이 생김

Replicator fallback 경로에서 생기는 정상 산출물입니다. 디버깅에는 유용하지만, 최종 기준 이미지는 `iter_*.png`를 우선 봅니다.

## 5. Data Collection

### ADC 서브모듈 없이 실행 가능 여부

가능합니다. 현재는 fallback 경로가 있습니다.

- IK: IsaacLab DifferentialIK fallback
- Judge prompt: 내장 prompt fallback
- interpolation: 내부 fallback

다만 정밀한 IK와 ADC prompt 재사용이 필요하면 submodule 초기화를 권장합니다.

### Pinocchio IK가 안 붙음

확인:

```bash
git submodule status
python3 -c "from src.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
```

필요 시:

```bash
git submodule update --init external/AutoDataCollector
```

또는 `ADC_PATH`를 외부 경로로 설정합니다.

### Data Collection이 env 생성 단계에서 실패

이 파이프라인은 내부적으로 IsaacLab env를 먼저 확보해야 합니다. 우선 아래 명령이 단독으로 성공하는지 확인합니다.

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

그 다음 `--env-dir`로 명시해 수집 파이프라인을 좁혀서 디버깅합니다.

## 6. 일반

### `ModuleNotFoundError: No module named 'src'`

프로젝트 루트에서 실행해야 합니다.

```bash
cd /path/to/Simulation-Generation-Agent
python3 scripts/run_isaac_lab.py ...
```

### `outputs/`가 너무 커짐

출력은 git 대상이 아니므로 주기적으로 정리합니다.

```bash
ls outputs
```

불필요한 run 디렉토리만 선택적으로 삭제합니다.

## 관련 문서

- `docs/getting_started.md`
- `docs/usage.md`
- `docs/evaluation.md`
