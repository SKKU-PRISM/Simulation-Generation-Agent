# Simulation-Generation-Agent

YAML task document 기반 로보틱스 시뮬레이션 환경 자동 구성 프레임워크. 두 가지 파이프라인을 지원합니다:

1. **Isaac Sim Pipeline**: MCP 소켓으로 Isaac Sim 씬을 빌드하고 VLM으로 평가
2. **IsaacLab Pipeline**: LLM이 IsaacLab ManagerBasedRLEnv Python 코드를 생성하고 자동 실행

## Architecture

```
                        62 Task YAML Documents
                    tasks/{robot}/{category}/*.yaml
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   Pipeline 1: Isaac Sim + VLM      Pipeline 2: IsaacLab + LLM
              |                                 |
   +----------v-----------+          +----------v-----------+
   |    SceneBuilder       |          |    IsaacLabAgent      |
   |    (MCP → localhost   |          |    (YAML 파싱 →       |
   |     :8766 TCP)        |          |     LLM 프롬프트)     |
   +----------+------------+          +----------+-----------+
              |                                  |
   +----------v-----------+          +-----------v----------+
   |  ScreenshotCapture    |          |  Azure OpenAI        |
   |  (Replicator API)     |          |  gpt-5-mini        |
   +----------+------------+          |  (코드 생성)          |
              |                       +-----------+----------+
   +----------v-----------+                       |
   |    VLM Evaluator      |          +-----------v----------+
   |  Azure/Claude/Gemini/Ollama |          |  Generated Code       |
   |  (score 0-100)        |          |  env_cfg.py + run_env  |
   +----------+------------+          +-----------+----------+
              |                                   |
        score >= 80?                    +---------v---------+
        YES → 완료                      |  isaaclab.sh       |
        NO  → 재시도 (max 5)           |  (headless 실행)    |
                                        +---------+---------+
                                                  |
                                           SUCCESS? → 실패시
                                           에러 피드백 → LLM
                                           재생성 (max 5회)
```

## Prerequisites

- **Python 3.10+**
- **Isaac Sim** (2023.1.1+) + MCP 확장 — Pipeline 1
- **IsaacLab** (v2.3.2+) + conda env `env_isaaclab` — Pipeline 2
- Isaac Sim MCP 서버: [isaac-sim-mcp](https://github.com/isaac-sim/isaac-sim-mcp)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Pipeline 1: Isaac Sim + VLM

```bash
# Isaac Sim MCP 서버 시작
/path/to/isaac-sim-mcp/run_isaac_mcp_streaming.sh 0

# 씬 빌드
python3 scripts/build_scene.py tasks/franka/stack/franka_stack.yaml
```

### Pipeline 2: IsaacLab Code Generation

```bash
# .env에 API 키 설정
echo 'AZURE_OPENAI_API_KEY=your-key' >> .env
echo 'AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/' >> .env

# 단일 task → IsaacLab 코드 생성 + 실행
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 코드 생성 + 실행 + 품질 평가 (100점 만점)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 코드 생성만 (실행 안 함)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# 기존 output 평가만 (standalone)
python3 scripts/evaluate.py outputs/isaaclab/<dir>/ tasks/franka/stack/franka_stack.yaml

# 기존 output 평가 (agent 경유)
python3 scripts/run_isaac_lab.py <yaml_path> --eval-only <output_dir>

# 폴더 전체 batch 변환
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### 컴포넌트 테스트

```bash
python3 tests/test_components.py connection     # MCP 연결
python3 tests/test_components.py scene tasks/franka/stack/franka_stack.yaml
python3 tests/test_components.py full --skip-vlm # 전체 (VLM 제외)
```

## Project Structure

```
Simulation-Generation-Agent/
├── src/                               # 핵심 소스 코드
│   ├── common/                        # 파이프라인 공유 유틸리티
│   │   ├── mcp_client.py             # MCPClient (TCP socket)
│   │   └── llm_client.py             # AzureOpenAIClient
│   ├── isaac_sim/                     # Isaac Sim + VLM
│   │   ├── scene_builder.py          # SceneBuilder (MCP 씬 빌드)
│   │   ├── screenshot.py             # ScreenshotCapture (Replicator)
│   │   └── vlm_evaluator.py          # VLM 평가 (multi-backend)
│   └── isaac_lab/                     # IsaacLab + LLM
│       ├── agent.py                   # IsaacLabAgent (코드 생성 + 실행)
│       └── evaluator/                 # 평가 시스템
│           ├── __init__.py            # IsaacLabEvaluator (오케스트레이터)
│           ├── parser.py              # EnvCfgParser, GoalNormalizer
│           ├── scene_fidelity.py      # SceneFidelityChecker (30점)
│           ├── mdp_correctness.py     # MDPCorrectnessChecker (25점)
│           ├── task_alignment.py      # TaskAlignmentChecker (25점)
│           └── runtime_validity.py    # RuntimeValidityChecker (20점)
├── scripts/                           # CLI 진입점 (thin wrappers)
│   ├── run_isaac_lab.py              # IsaacLab 코드 생성 실행
│   ├── evaluate.py                   # Standalone 평가
│   └── build_scene.py                # Pipeline 1 씬 빌드
├── tests/
│   └── test_components.py            # Pipeline 1 컴포넌트 테스트
├── tasks/                             # 62 Task YAML documents
│   ├── franka/                        # Franka Panda (20 tasks)
│   ├── openarm/                       # OpenArm (20 tasks)
│   ├── ur10/                          # UR10 (13 tasks)
│   ├── so101/                         # SO-101 (9 tasks)
│   └── templates/                     # v2.0.0 템플릿
├── configs/
│   ├── pipeline_config.yaml           # MCP 연결, 씬 기본값
│   ├── vlm_config.yaml                # VLM 모델, 평가 가중치
│   ├── isaaclab_agent_config.yaml     # IsaacLab Agent 설정
│   ├── isaaclab_eval_config.yaml      # IsaacLab 평가 설정
│   └── robot_profiles/                # 로봇별 kinematics (4종)
├── prompts/
│   ├── vlm_evaluation.md              # VLM 평가 프롬프트
│   ├── isaaclab_generation.md         # IsaacLab 코드 생성 프롬프트
│   └── isaaclab_error_fix.md          # 에러 수정 프롬프트
├── .claude/skills/                    # Claude Code 스킬 (4종)
├── .mcp.json                          # Isaac Sim MCP 서버 설정
└── requirements.txt
```

## Supported Robots & Tasks

| Robot | DOF | Gripper | Tasks (62 total) |
|-------|-----|---------|-------------------|
| **Franka Panda** | 7+2 | Parallel jaw (8cm) | stack(2), lift(2), pick_place(9), cabinet(3), sort(3), peg_insert(1) |
| **OpenArm** | 7+2 | Parallel jaw (8.8cm) | stack(2), lift(2), pick_place(9), reach(1), cabinet(3), sort(3) |
| **UR10** | 6 | Suction | stack(2), cabinet(3), pick_place(7), reach(1) |
| **SO-101** | 5+1 | Claw (5cm) | stack(1), lift(1), pick_place(3), reach(1), sort(3) |

Task document 경로: `tasks/{robot}/{category}/{robot}_{task}.yaml`

## Task Document Format

```yaml
task:
  name: FrankaStack
  description: "Stack three colored cubes"

simulation:
  gravity: [0, 0, -9.81]
  timestep: 0.01
  decimation: 5
  episode_length: 30.0

scene:
  lighting: { type: dome, intensity: 3000 }
  ground: { enabled: true, position: [0, 0, -1.05] }

assets:
  - name: robot
    type: articulation
    robot_type: franka
    position: [0, 0, 0]
    rotation: [1, 0, 0, 0]           # wxyz quaternion
    initial_joints: { ... }

  - name: table
    type: static
    asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/..."

  - name: cube_1
    type: rigid
    asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
    physics: { rigid_body: true, collision: true }
    randomize:
      position: { type: absolute, x: [0.4, 0.6], y: [-0.1, 0.1] }

goal:
  description: "Stack cubes: Blue -> Red -> Green"
  success_criteria:
    xy_threshold: 0.04
    height_diff: 0.0468

camera:
  position: [1.5, 1.2, 1.0]
  target: [0.25, 0, 0.3]
  fov: 60
```

## Configuration

| File | Pipeline | Description |
|------|----------|-------------|
| `configs/pipeline_config.yaml` | Isaac Sim | MCP 연결, 평가 임계치, 씬 기본값 |
| `configs/vlm_config.yaml` | Isaac Sim | VLM 모델, 평가 가중치, 재시도 설정 |
| `configs/isaaclab_agent_config.yaml` | IsaacLab | IsaacLab 경로, LLM 설정, 실행 타임아웃 |
| `configs/isaaclab_eval_config.yaml` | IsaacLab | 평가 가중치, 유효 MDP 함수, 런타임 설정 |
| `configs/robot_profiles/*.yaml` | Both | 로봇별 kinematics, workspace, gripper 정보 |

## Environment Variables

```bash
# Isaac Sim Pipeline (VLM 평가)
ANTHROPIC_API_KEY        # Claude VLM 백엔드
GOOGLE_API_KEY           # Gemini VLM 백엔드 (무료 tier)
CLAUDE_MODEL             # Claude 모델명 (default: claude-sonnet-4-20250514)
GEMINI_MODEL             # Gemini 모델명 (default: gemini-2.5-flash)
OLLAMA_MODEL             # Ollama 모델명 (default: llava:7b)
OLLAMA_BASE_URL          # Ollama 서버 URL (default: http://localhost:11434)

# Azure OpenAI (Isaac Sim VLM + IsaacLab LLM 공용)
AZURE_OPENAI_API_KEY     # Azure OpenAI API 키 (필수)
AZURE_OPENAI_BASE_URL    # Azure 엔드포인트 (필수)
AZURE_OPENAI_MODEL       # 모델명 (default: gpt-5-mini)
```
