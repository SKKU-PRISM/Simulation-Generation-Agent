# AutoEnvConstruction

MCP 기반 Isaac Sim 로보틱스 시뮬레이션 환경 자동 구성 파이프라인. YAML task document를 읽어 Isaac Sim 씬을 빌드하고, 스크린샷을 캡처하여 VLM으로 검증합니다.

## Architecture

```
YAML Task Document
       |
       v
 +-------------+    +---------------+    +--------------+
 | SceneBuilder|--->|ScreenshotCapt.|--->| VLM Evaluator|
 |   (MCP)     |    | (Replicator)  |    |(Claude/Gemini)|
 +-------------+    +---------------+    +--------------+
```

- **SceneBuilder**: YAML을 파싱하여 MCP 소켓(localhost:8766)을 통해 Isaac Sim에 USD prim 생성
- **ScreenshotCapture**: Omni Replicator API로 뷰포트 캡처
- **VLM Evaluator**: Claude/Gemini/Ollama 등으로 씬 품질 평가 (0-100점)

## Prerequisites

- **Isaac Sim** (2023.1.1+) + MCP 확장 활성화
- **Python 3.10+**
- Isaac Sim MCP 서버: [isaac-sim-mcp](https://github.com/isaac-sim/isaac-sim-mcp)

## Installation

```bash
pip install -r requirements.txt
```

### Isaac Sim MCP 시작

씬 빌드 전 Isaac Sim MCP 서버가 실행 중이어야 합니다:

```bash
# GPU 0번으로 Isaac Sim + MCP 시작
/path/to/isaac-sim-mcp/run_isaac_mcp_streaming.sh 0
```

## Quick Start

```bash
cd scripts

# 1. 씬 빌드
python3 scene_builder.py ../tasks/franka/stack/franka_stack.yaml

# 2. 스크린샷 캡처
python3 screenshot_capture.py ../outputs/test.png

# 3. VLM 평가 (API 키 필요)
python3 vlm_evaluator.py \
  --screenshot ../outputs/test.png \
  --document ../tasks/franka/stack/franka_stack.yaml \
  --output ../outputs/eval.json
```

### 컴포넌트 테스트

```bash
cd scripts

# MCP 연결 테스트
python3 test_components.py connection

# 씬 빌드 테스트
python3 test_components.py scene ../tasks/franka/stack/franka_stack.yaml

# 전체 테스트 (연결 -> 빌드 -> 스크린샷)
python3 test_components.py full --skip-vlm
```

## Project Structure

```
AutoEnvConstruction/
├── scripts/
│   ├── scene_builder.py          # YAML -> Isaac Sim 씬 (MCP)
│   ├── screenshot_capture.py     # 뷰포트 스크린샷 캡처
│   ├── vlm_evaluator.py          # VLM 기반 씬 평가
│   └── test_components.py        # 컴포넌트 개별 테스트
├── tasks/                        # Task document (YAML)
│   ├── franka/                   # Franka Panda (7-DOF + 2-finger)
│   ├── openarm/                  # OpenArm (7-DOF + 2-finger)
│   └── ur10/                     # UR10 (6-DOF)
├── configs/
│   ├── pipeline_config.yaml      # MCP 연결, 씬 기본값
│   ├── vlm_config.yaml           # VLM 모델, 평가 가중치
│   └── robot_profiles/           # 로봇별 kinematics 프로파일
├── prompts/
│   └── vlm_evaluation.md         # VLM 평가 프롬프트 템플릿
└── requirements.txt
```

## Supported Robots & Tasks

| Robot | Tasks |
|-------|-------|
| **Franka Panda** | stack, lift, cabinet, pick_place |
| **OpenArm** | stack, lift, cabinet, reach |
| **UR10** | stack, cabinet, pick_place, reach |

Task document 경로: `tasks/{robot}/{task}/{robot}_{task}.yaml`

## Task Document Format

```yaml
task:
  name: FrankaStack
  description: "Stack three colored cubes"

simulation:
  gravity: [0, 0, -9.81]
  timestep: 0.01

scene:
  lighting: { type: dome, intensity: 3000 }
  ground: { position: [0, 0, -1.05] }

assets:
  - name: robot
    type: articulation
    robot_type: franka
    prim_path: /World/Robot
    position: [0, 0, 0]
    rotation: [1, 0, 0, 0]     # wxyz quaternion
    initial_joints: { ... }

  - name: table
    type: static
    asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/..."
    position: [0.5, 0, 0]

  - name: cube_1
    type: rigid
    primitive: cube
    position: [0.4, 0, 0.0203]
    physics: { rigid_body: true, collision: true }
    randomize:
      position: { x: [0.4, 0.6], y: [-0.1, 0.1] }

camera:
  position: [1.5, 1.2, 1.0]
  target: [0.25, 0, 0.3]
  fov: 60

goal:
  description: "Stack cubes: Blue -> Red -> Green"
```

## Configuration

| File | Description |
|------|-------------|
| `configs/pipeline_config.yaml` | MCP 연결(host/port), 씬 기본값(physics, lighting) |
| `configs/vlm_config.yaml` | VLM 모델 설정, 평가 가중치, 재시도 설정 |
| `configs/robot_profiles/*.yaml` | 로봇별 kinematics, workspace, gripper 정보 |

## Environment Variables

```bash
ANTHROPIC_API_KEY    # Claude VLM 백엔드
GOOGLE_API_KEY       # Gemini VLM 백엔드 (무료 tier 사용 가능)
```
