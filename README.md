# Simulation-Generation-Agent

YAML task document 기반 로보틱스 시뮬레이션 환경 자동 구성 프레임워크. 두 가지 파이프라인을 지원합니다:

- **Isaac Sim Pipeline**: MCP 소켓으로 Isaac Sim 씬을 빌드하고 VLM으로 평가
- **IsaacLab Pipeline**: LLM이 IsaacLab ManagerBasedRLEnv Python 코드를 생성하고 자동 실행

## Architecture

```
                        62 Task YAML Documents
                    tasks/{robot}/{category}/*.yaml
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   Isaac Sim Pipeline               IsaacLab Pipeline
              |                                 |
   +----------v-----------+          +----------v-----------+
   |    SceneBuilder       |          |    IsaacLabAgent      |
   |    (MCP → localhost   |          |    (YAML 파싱 →       |
   |     :8766 TCP)        |          |     LLM 프롬프트)     |
   +----------+------------+          +----------+-----------+
              |                                  |
   +----------v-----------+          +-----------v----------+
   |  ScreenshotCapture    |          |  Azure OpenAI        |
   |  (Replicator API)     |          |  gpt-5-mini          |
   +----------+------------+          |  (코드 생성)          |
              |                       +-----------+----------+
   +----------v-----------+                       |
   |    VLM Evaluator      |          +-----------v----------+
   |  Azure/Claude/Gemini/ |          |  Generated Code       |
   |  Ollama               |          |  env_cfg.py + run_env  |
   |  (score 0-100)        |          +-----------+----------+
   +----------+------------+                      |
              |                         +---------v---------+
        score >= 80?                    |  isaaclab.sh       |
        YES → 완료                      |  (headless 실행)    |
        NO  → 재시도 (max 5)           +---------+---------+
                                                  |
                                           SUCCESS? → 실패시
                                           에러 피드백 → LLM
                                           재생성 (max 5회)
```

## 문서

| 문서 | 설명 |
|------|------|
| **[설치 가이드](docs/getting_started.md)** | 환경 구성, 외부 의존성(Isaac Sim, IsaacLab, MCP) 설치 및 연결 방법 |
| **[사용법](docs/usage.md)** | 두 파이프라인 CLI 사용법, 옵션, 실행 예시, 출력 구조 |
| **[Task YAML 명세](docs/task_yaml_spec.md)** | 태스크 문서 포맷, 필드 정의, 에셋 규약, 로봇별 참고사항 |
| **[평가 시스템](docs/evaluation.md)** | 4카테고리 100점 자동 평가, 결과 해석법 |
| **[트러블슈팅](docs/troubleshooting.md)** | 자주 발생하는 문제와 해결책 |

## Quick Start

```bash
# 설치
git clone <repo-url>
cd Simulation-Generation-Agent
pip install -e .
cp .env.example .env   # AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL 설정
```

자세한 설치 과정 (Isaac Sim, IsaacLab, MCP 서버 연결)은 **[설치 가이드](docs/getting_started.md)**를 참고하세요.

```bash
# IsaacLab Pipeline — 코드 생성 + 실행
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 코드 생성 + 실행 + 평가 (100점 만점)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 배치 처리
python3 scripts/run_isaac_lab.py --batch tasks/franka/

# Isaac Sim Pipeline — 씬 빌드 (MCP 서버 필요)
python3 scripts/build_scene.py tasks/franka/stack/franka_stack.yaml
```

더 많은 옵션은 **[사용법](docs/usage.md)**을 참고하세요.

## Supported Robots & Tasks

| Robot | DOF | Gripper | Tasks (62 total) |
|-------|-----|---------|-------------------|
| **Franka Panda** | 7+2 | Parallel jaw (8cm) | stack(2), lift(2), pick_place(9), cabinet(3), sort(3), peg_insert(1) |
| **OpenArm** | 7+2 | Parallel jaw (8.8cm) | stack(2), lift(2), pick_place(9), reach(1), cabinet(3), sort(3) |
| **UR10** | 6 | Suction | stack(2), cabinet(3), pick_place(7), reach(1) |
| **SO-101** | 5+1 | Claw (5cm) | stack(1), lift(1), pick_place(3), reach(1), sort(3) |

## Project Structure

```
Simulation-Generation-Agent/
├── src/                               # 핵심 소스 코드
│   ├── common/                        # 공유 유틸리티 (MCPClient, LLMClient)
│   ├── isaac_sim/                     # Isaac Sim Pipeline (SceneBuilder, Screenshot, VLM)
│   └── isaac_lab/                     # IsaacLab Pipeline (Agent, Evaluator)
├── scripts/                           # CLI 진입점
├── tests/                             # 컴포넌트 테스트
├── tasks/                             # 62 Task YAML documents
├── configs/                           # 설정 파일 (pipeline, VLM, agent, eval, robot profiles)
├── prompts/                           # LLM/VLM 프롬프트
├── docs/                              # 문서
└── research_notes/                    # 연구 노트
```
