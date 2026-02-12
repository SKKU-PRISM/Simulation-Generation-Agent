# 트러블슈팅

자주 발생하는 문제와 해결책을 정리합니다.

---

## IsaacLab Pipeline (Pipeline 2)

### `simulation_app.close()`가 무한 대기 (hang)

**증상**: IsaacLab 코드 실행 후 프로세스가 종료되지 않고 멈춤.

**원인**: IsaacLab의 `SimulationApp.close()` 메서드가 특정 조건에서 무한 블로킹됨.

**해결**: 이 프로젝트는 marker file 방식으로 우회합니다.
- 생성된 `run_env.py`는 step loop 완료 후 `.success_marker` 파일을 먼저 생성
- 그 다음 `simulation_app.close()` 호출
- Agent는 subprocess 종료와 관계없이 marker file 존재 여부로 성공 판단
- subprocess가 타임아웃(기본 300초)으로 강제 종료되어도 문제 없음

### `AppLauncher` import 순서 에러

**증상**: `run_env.py` 실행 시 segfault 또는 import error.

```
ImportError: cannot import name 'ManagerBasedRLEnv' from 'isaaclab.envs'
```

**원인**: `AppLauncher`가 초기화되기 전에 physics 관련 모듈을 import함.

**해결**: `run_env.py`에서 반드시 다음 순서를 지켜야 합니다:

```python
# 1. AppLauncher 초기화 (최상단)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 이후에 physics 모듈 import
import torch
from isaaclab.envs import ManagerBasedRLEnv
```

LLM이 이 순서를 틀리게 생성하면 Agent가 자동으로 감지하여 에러 피드백에 포함시킵니다.

### conda 환경을 찾을 수 없음

**증상**:
```
conda run: error: argument -n: env 'env_isaaclab' not found
```

**해결**:
```bash
# conda 환경 목록 확인
conda env list

# 환경 이름이 다른 경우, config에서 수정
# configs/isaaclab_agent_config.yaml:
isaaclab:
  conda_env: "your_env_name"   # 실제 환경 이름으로 변경
```

### LLM이 존재하지 않는 MDP 함수를 생성

**증상**: 생성된 코드에 `object_obs`, `cubes_stacked`, `ee_frame_pos` 같은 함수가 사용됨.

**원인**: LLM이 존재하지 않는 IsaacLab MDP 함수를 환각(hallucinate)함.

**해결**: 프롬프트에 이미 Available/Unavailable 양면 제약이 포함되어 있습니다. 계속 발생하면:
- `prompts/isaaclab_generation.md`의 Common Mistakes 섹션에 해당 함수 추가
- 또는 `configs/isaaclab_eval_config.yaml`의 `valid_mdp_functions` 목록 확인

### `RewTerm`에 `weight` 누락

**증상**:
```
TypeError: RewTerm.__init__() missing required keyword argument: 'weight'
```

**해결**: IsaacLab의 `RewTerm`은 `weight` 파라미터가 필수입니다:

```python
# 잘못된 예
my_reward = RewTerm(func=mdp.action_rate_l2, params={})

# 올바른 예
my_reward = RewTerm(func=mdp.action_rate_l2, weight=-0.01, params={})
```

---

## Isaac Sim MCP (Pipeline 1)

### 연결 거부

**증상**:
```
ConnectionRefusedError: [Errno 111] Connection refused (localhost:8766)
```

**원인**: MCP 서버가 실행되지 않았거나 다른 포트에서 실행 중.

**해결**:
```bash
# 1. MCP 서버 시작
cd ~/workspace/isaac-sim-mcp
./run_isaac_mcp_streaming.sh 0

# 2. 연결 테스트
python3 tests/test_components.py connection

# 3. 포트가 다른 경우
python3 tests/test_components.py connection --port 8767
```

### `execute_script` 결과가 항상 `null`

**원인**: Isaac Sim MCP 확장의 `execute_script` 핸들러가 결과를 하드코딩으로 `null` 반환.

**우회**: 스크립트 내에서 결과를 temp 파일에 기록하고 별도로 읽어옴.

```python
# 스크립트 내에서
import json
results = {"bbox": [1.0, 2.0, 3.0]}
with open("/tmp/isaac_results.json", "w") as f:
    json.dump(results, f)
```

### `reset_scene` 핸들러 없음

**원인**: MCP 확장에 씬 초기화 기능이 구현되지 않음.

**우회**: `execute_script`로 직접 초기화:
```python
# MCP execute_script로 실행
omni.usd.get_context().new_stage()
```

---

## LLM / Azure OpenAI

### API 키 에러

**증상**:
```
openai.AuthenticationError: Error code: 401
```

**해결**:
```bash
# .env 파일 확인
cat .env | grep AZURE_OPENAI

# 두 값 모두 설정되어 있어야 함:
# AZURE_OPENAI_API_KEY=your-key
# AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
```

주의: `AZURE_OPENAI_BASE_URL`은 `/openai/v1/`로 끝나야 합니다.

### gpt-5-mini temperature 에러

**증상**:
```
BadRequestError: temperature is not supported for this model
```

**원인**: `gpt-5-mini`는 temperature 파라미터를 지원하지 않음.

**해결**: 자동으로 처리됩니다. `src/common/llm_client.py`에서 첫 호출 실패 시 temperature를 제거하고 재시도하는 fallback 로직이 구현되어 있습니다.

---

## VLM 평가

### VLM 백엔드 자동 탐지 우선순위

```
Azure OpenAI (AZURE_OPENAI_API_KEY + AZURE_OPENAI_BASE_URL)
  → Gemini (GOOGLE_API_KEY, 무료)
    → Claude (ANTHROPIC_API_KEY)
      → Ollama (localhost:11434)
        → Mock (테스트용, 실제 평가 불가)
```

어떤 백엔드가 선택되었는지 확인:
```bash
# VLM 평가 실행 시 로그에 "Using backend: AzureVLMEvaluator" 등이 출력됨
python3 tests/test_components.py vlm <screenshot> <yaml>
```

### Gemini 무료 API 제한

- 분당 15 요청
- 일일 1,500 요청
- 이미지당 최대 4MB

대량 평가 시 rate limit에 걸릴 수 있습니다. Azure 백엔드를 사용하거나 요청 간 대기 시간을 설정하세요.

### MockVLMEvaluator만 선택됨

**증상**: 평가 실행 시 "Using MockVLMEvaluator" 경고.

**원인**: 설정된 API 키가 없어 모든 실제 백엔드가 비활성화됨.

**해결**: `.env` 파일에 최소 하나의 API 키를 설정:
```bash
# 가장 간단 (Azure가 이미 있다면)
# AZURE_OPENAI_API_KEY와 AZURE_OPENAI_BASE_URL이 .env에 있으면 자동 사용

# 무료로 시작하려면
GOOGLE_API_KEY=your-google-api-key   # Gemini 무료 tier
```

---

## 일반

### `outputs/` 디렉토리가 커짐

생성된 코드와 에러 로그가 계속 축적됩니다.

```bash
# 오래된 출력물 정리
ls -la outputs/isaaclab/
rm -rf outputs/isaaclab/FrankaStack_20260101_*/  # 필요 없는 것만 삭제
```

`outputs/`는 `.gitignore`에 포함되어 있어 git에는 커밋되지 않습니다.

### Python import 에러

```
ModuleNotFoundError: No module named 'src'
```

**해결**: 프로젝트 루트 디렉토리에서 실행해야 합니다:
```bash
cd /path/to/Simulation-Generation-Agent
python3 scripts/run_isaac_lab.py ...
```

---

## 관련 문서

- [설치 가이드](getting_started.md) — 환경 구성
- [사용법](usage.md) — CLI 옵션
- [평가 시스템](evaluation.md) — 평가 결과 해석
