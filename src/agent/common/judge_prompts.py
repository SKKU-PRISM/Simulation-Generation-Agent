"""
Judge Prompt Template

태스크 완료 여부를 판단하는 VLM 프롬프트 템플릿
Chain-of-Thought 방식으로 단계별 추론을 유도합니다.
"""

from typing import Dict, List, Optional, Tuple, Union


JUDGE_SYSTEM_PROMPT = """You are a Task Completion Judge for robotic manipulation tasks.

You will evaluate whether a robot successfully completed a pick-and-place task using a structured, step-by-step analysis.

## Evaluation Process

You MUST complete all 6 steps and mark each step as PASS or FAIL.

**Step 1: Identify Objects in Initial Image**
- Use the provided pixel coordinates to locate each object
- Verify the object matches the expected description
- Mark PASS if all objects are correctly identified

**Step 2: Identify Objects in Final Image**
- Find the same objects in the final image
- Note their new approximate pixel positions
- Mark PASS if all objects are visible and identifiable

**Step 3: Analyze Movement**
- Compare initial vs final positions
- The "pick object" (object being manipulated) should have moved
- Mark PASS if the pick object moved from its original position

**Step 4: Verify Goal Achievement**
- Check if the pick object reached the intended destination
- Mark PASS if the object is at or near the target location

**Step 5: Check Spatial Relationships**
- Verify the final spatial relationship matches the instruction
- "place on X" → object should be on/above X
- "stack on X" → object should be stacked on X
- Mark PASS if the relationship is correct

**Step 6: Final Judgment**
- Count PASS/FAIL results from steps 1-5
- TRUE: Steps 3, 4, and 5 all PASS (core task achieved)
- FALSE: Any of steps 3, 4, or 5 FAIL
- UNCERTAIN: Cannot determine (e.g., object not visible)

Be fair in evaluation. Focus on whether the core task objective was achieved.
"""


JUDGE_USER_PROMPT_TEMPLATE = """## Task Evaluation Request

### 1. Goal Instruction
{instruction}

### 2. Image Information
- **Image Resolution**: {image_width} x {image_height} pixels
- **Image 1 (Initial State)**: Before task execution
- **Image 2 (Final State)**: After task execution

### 3. Object Positions in Initial Image
{initial_positions_text}

### 4. Executed Python Code
```python
{executed_code}
```

---

## Your Evaluation

Analyze each step and provide your findings in the format below.

### Output Format (You MUST follow this exact format)

**STEP 1 - Identify Objects in Initial Image**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 1 Result: [PASS/FAIL] - [brief reason]

**STEP 2 - Identify Objects in Final Image**
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Object: [object name]
  - Found at pixel: ([x], [y])
  - Description: [what you see at that location]
- Step 2 Result: [PASS/FAIL] - [brief reason]

**STEP 3 - Analyze Movement**
- Object: [object name]
  - Initial: ([x1], [y1]) → Final: ([x2], [y2])
  - Movement: [moved/not moved], Direction: [left/right/up/down/none]
- Step 3 Result: [PASS/FAIL] - [Did the pick object move as expected?]

**STEP 4 - Verify Goal Achievement**
- Goal: "{instruction}"
- Pick object picked up: [YES/NO]
- Reached destination: [YES/NO]
- Step 4 Result: [PASS/FAIL] - [brief reason]

**STEP 5 - Check Spatial Relationships**
- Expected relationship: [e.g., "red cup should be on carrier"]
- Observed relationship: [what you actually see in final image]
- Step 5 Result: [PASS/FAIL] - [brief reason]

**STEP 6 - Final Judgment**
- Steps passed: [list which steps passed]
- Steps failed: [list which steps failed, if any]

PREDICTION: [TRUE/FALSE/UNCERTAIN]

REASONING: [One sentence summary of why you made this prediction based on the step results]
"""


def build_judge_prompt(
    instruction: str,
    object_positions: Dict[str, Union[List[float], Tuple[float, float, float], Dict, None]],
    executed_code: str,
    image_resolution: Tuple[int, int] = None,
) -> str:
    """
    Judge 프롬프트 생성 (Chain-of-Thought 구조)

    Args:
        instruction: 목표 명령어
        object_positions: 객체 위치 딕셔너리
            Extended format: {
                name: {
                    "position": [x, y, z],
                    "pixel_coords": (cx, cy),
                    "bbox_pixels": (x1, y1, x2, y2),
                    ...
                }
            }
        executed_code: 실행된 Python 코드
        image_resolution: 이미지 해상도 (width, height)

    Returns:
        포맷된 프롬프트 문자열
    """
    # 이미지 해상도 (기본값)
    if image_resolution is None:
        image_resolution = (640, 480)
    image_width, image_height = image_resolution

    # 객체 위치 텍스트 생성 (픽셀 좌표 포함)
    positions_lines = []
    for name, info in object_positions.items():
        # 메타데이터 키는 건너뛰기
        if name.startswith("_"):
            continue

        if info is None:
            positions_lines.append(f"- **{name}**: Not detected")
        elif isinstance(info, dict) and "position" in info:
            # Extended format with pixel coordinates
            pos = info["position"]
            pixel_coords = info.get("pixel_coords")
            bbox_pixels = info.get("bbox_pixels")

            line = f"- **{name}**:\n"
            line += f"  - World position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] meters\n"

            if pixel_coords:
                cx, cy = pixel_coords
                line += f"  - Pixel center: ({cx}, {cy})\n"

            if bbox_pixels:
                x1, y1, x2, y2 = bbox_pixels
                line += f"  - Bounding box: ({x1}, {y1}) to ({x2}, {y2})"

            positions_lines.append(line)
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            # Legacy format: [x, y, z]
            positions_lines.append(
                f"- **{name}**: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}] meters"
            )
        else:
            positions_lines.append(f"- **{name}**: Not detected")

    initial_positions_text = "\n".join(positions_lines) if positions_lines else "No objects detected"

    # 프롬프트 생성
    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        image_width=image_width,
        image_height=image_height,
        initial_positions_text=initial_positions_text,
        executed_code=executed_code,
    )

    return prompt


def get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    return JUDGE_SYSTEM_PROMPT


# 테스트
if __name__ == "__main__":
    # 테스트 데이터 (extended format with pixel coords)
    test_instruction = "pick up the green block and place it on the blue dish"
    test_positions = {
        "_image_resolution": (640, 480),
        "green block": {
            "position": [0.1500, 0.0500, 0.0200],
            "pixel_coords": (280, 320),
            "bbox_pixels": (250, 290, 310, 350),
            "confidence": 0.92,
        },
        "blue dish": {
            "position": [0.2000, -0.0300, 0.0100],
            "pixel_coords": (400, 280),
            "bbox_pixels": (350, 230, 450, 330),
            "confidence": 0.88,
        },
    }
    test_code = """
from skills.skills_lerobot import LeRobotSkills

skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
skills.connect()

# Pick up green block
skills.execute_pick_object([0.15, 0.05, 0.02])

# Place on blue dish
skills.execute_place_object([0.20, -0.03, 0.03])

skills.disconnect()
"""

    prompt = build_judge_prompt(
        test_instruction,
        test_positions,
        test_code,
        image_resolution=(640, 480),
    )

    print("=" * 70)
    print("System Prompt:")
    print("=" * 70)
    print(get_system_prompt())
    print("\n" + "=" * 70)
    print("User Prompt:")
    print("=" * 70)
    print(prompt)
