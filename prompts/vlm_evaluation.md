# VLM Scene Evaluation Prompt

You are evaluating a robot simulation scene based ONLY on the task document requirements. There is no ground truth image - you must assess the scene solely against the document specifications.

## Generated Scene Screenshot
[Image provided separately]

## Task Document Requirements
{task_document}

## Evaluation Criteria

Evaluate the scene against these criteria. Be STRICT and precise. A perfect score (90+) on the first attempt is extremely rare - most initial scenes have noticeable issues.

### 1. Asset Presence (30 points)
Check if ALL assets specified in the document are visible in the scene:
- Robot (type, position)
- Table/surfaces
- Objects (cubes, tools, etc.)
- Ground plane
- Lighting setup

Scoring guide:
- **25-30**: ALL assets present with correct type/appearance
- **15-24**: 1 asset missing OR 1 asset has wrong type/appearance
- **5-14**: 2 assets missing or incorrect
- **0-4**: 3+ assets missing or major asset failures

For each asset, explicitly state: present/missing/wrong type.

### 2. Position/Layout (30 points)
Assess if asset positions match the document specifications:
- Relative positions between objects
- Spacing and distances
- Orientations/rotations
- Objects properly placed (on surfaces, not floating)

Scoring guide:
- **25-30**: All positions visually match document specs, objects properly grounded
- **15-24**: Minor deviations - objects are on correct surfaces but positions slightly off
- **5-14**: Noticeable errors - objects floating, overlapping, or significantly misplaced
- **0-4**: Major layout failures - objects in completely wrong locations

Deduct points for: floating objects (-5), interpenetrating objects (-5), objects fallen through surfaces (-8).

### 3. Visual Correctness (20 points)
Check if visual properties match the document:
- Colors (exact RGB if specified)
- Sizes/scales
- Materials/textures (if specified)
- Lighting quality

Scoring guide:
- **16-20**: All colors and sizes match document specs
- **8-15**: 1-2 color/size mismatches
- **3-7**: Multiple visual property mismatches
- **0-2**: Most visual properties incorrect

### 4. Scene Completeness (20 points)
Assess if the scene is ready for the described task:
- Physics setup appears correct
- Objects properly grounded (not floating)
- No unexpected collisions or interpenetrations
- Camera angle allows task verification

Scoring guide:
- **16-20**: Scene fully ready for task execution, all physics/lighting correct
- **8-15**: Minor issues (e.g., slightly off camera angle, minor lighting issue)
- **3-7**: Notable issues affecting task readiness
- **0-2**: Scene not suitable for task execution

## Strict Scoring Rules

1. **Itemize every deduction**: For each point deducted, explain exactly why
2. **No benefit of the doubt**: If something looks wrong, deduct points
3. **Compare precisely**: Check each asset against its document specification one by one
4. **Camera angle limitation**: If the camera angle obscures assets, note this as an issue (deduct from Scene Completeness)
5. **First attempts typically score 40-75**: Be calibrated - a score above 85 means the scene is nearly perfect

## Output Format

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):

```json
{
  "score": <0-100>,
  "breakdown": {
    "asset_presence": {
      "score": <0-30>,
      "details": "<list each asset: present/missing/wrong, with deduction reason>"
    },
    "position_layout": {
      "score": <0-30>,
      "details": "<list each position check with specific deduction reasons>"
    },
    "visual_correctness": {
      "score": <0-20>,
      "details": "<list each visual property check with deduction reasons>"
    },
    "scene_completeness": {
      "score": <0-20>,
      "details": "<list completeness checks with deduction reasons>"
    }
  },
  "feedback": {
    "missing_assets": ["<asset_name>", ...],
    "position_errors": ["<description of position error>", ...],
    "visual_issues": ["<description of visual issue>", ...],
    "general_issues": ["<other issues>", ...]
  },
  "document_improvements": [
    "<specific, actionable suggestion for improving the document>",
    ...
  ]
}
```

## Important Notes

- Be objective and consistent in scoring
- Only assess what's visible in the screenshot
- If the camera angle obscures some assets, note this in feedback AND deduct from scene_completeness
- Document improvements should be specific and actionable (e.g., "Change cube_blue position from [0.4, 0, 0.02] to [0.4, 0, 0.82] to place it on the table surface")
- Do not make assumptions about physics behavior from a static image
