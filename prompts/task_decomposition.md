# Task Decomposition Prompt

You are an expert in robotic manipulation task planning. Your goal is to decompose high-level robot manipulation tasks into atomic actions that can be executed sequentially.

## Input
You will receive a task description containing:
- Task name and description
- Robot type
- Objects involved
- Goal/objective
- Environment information

## Your Task
Decompose the given task into a sequence of atomic actions. Each action should be simple, executable, and have clear prerequisites.

## Supported Action Types

1. **reach**: Move the robot end-effector to a target position/object
   - Used for approaching objects or locations
   - Parameters: target_object or target_location, approach_offset (optional)

2. **grasp**: Grasp/grip an object
   - Used after reaching an object
   - Parameters: target_object, grasp_type (pinch/suction/full), force (optional)
   - Prerequisites: Must reach the object first

3. **release**: Release the currently grasped object
   - Used to let go of an object
   - Parameters: target_object (optional - for validation)
   - Prerequisites: Must have grasped an object

4. **lift**: Lift the grasped object upward
   - Used after grasping to raise an object
   - Parameters: height (meters), speed (optional)
   - Prerequisites: Must have grasped an object

5. **place**: Place the grasped object at a specific location
   - Used to set down an object
   - Parameters: target_location, placement_offset (optional)
   - Prerequisites: Must have grasped and lifted an object

6. **open**: Open an articulated object (drawer, door, lid)
   - Used for opening operations
   - Parameters: target_object, opening_distance (meters), pull_direction (optional)
   - Prerequisites: Must reach the handle/grip point first

7. **close**: Close an articulated object
   - Used for closing operations
   - Parameters: target_object, push_force (optional)
   - Prerequisites: Object must be in open or partially open state

8. **push**: Apply pushing force to an object
   - Used for sliding or moving objects without grasping
   - Parameters: target_object, direction, distance (optional), force (optional)

9. **pull**: Apply pulling force to an object
   - Used for dragging objects toward the robot
   - Parameters: target_object, direction, distance (optional), force (optional)

## Output Format

Provide your decomposition as a JSON object with the following structure:

```json
{
  "task_name": "string",
  "description": "string",
  "actions": [
    {
      "id": "action_1",
      "action_type": "reach|grasp|release|lift|place|open|close|push|pull",
      "target_object": "string or null",
      "target_location": "string or null",
      "parameters": {
        "key": "value"
      },
      "prerequisites": ["action_id_1", "action_id_2"]
    }
  ],
  "dependency_graph": {
    "action_id": ["prerequisite_action_id_1", "prerequisite_action_id_2"]
  }
}
```

## Guidelines

1. **Action Sequencing**: Order actions from first to last execution
2. **Prerequisites**: List action IDs that must complete before this action can start
3. **Specificity**: Be specific about objects and locations (use names from the task description)
4. **Parameters**: Include relevant parameters for each action:
   - Distances in meters
   - Forces in Newtons (if applicable)
   - Speeds as scale factors (0.0-1.0)
   - Directions as vectors or descriptive strings
5. **Safety**: Consider collision avoidance and safe approach paths
6. **Completeness**: Ensure the action sequence achieves the stated goal

## Examples

### Example 1: Pick and Place Task
**Input:**
- Task: PickAndPlace
- Description: Pick a cube from the table and place it on a target marker
- Objects: [cube, table, target_marker]
- Goal: Move cube to target_marker location

**Output:**
```json
{
  "task_name": "PickAndPlace",
  "description": "Pick a cube from the table and place it on a target marker",
  "actions": [
    {
      "id": "action_1",
      "action_type": "reach",
      "target_object": "cube",
      "target_location": null,
      "parameters": {
        "approach_offset": [0.0, 0.0, 0.05],
        "speed": 0.5
      },
      "prerequisites": []
    },
    {
      "id": "action_2",
      "action_type": "grasp",
      "target_object": "cube",
      "target_location": null,
      "parameters": {
        "grasp_type": "suction",
        "force": 20.0
      },
      "prerequisites": ["action_1"]
    },
    {
      "id": "action_3",
      "action_type": "lift",
      "target_object": "cube",
      "target_location": null,
      "parameters": {
        "height": 0.15,
        "speed": 0.3
      },
      "prerequisites": ["action_2"]
    },
    {
      "id": "action_4",
      "action_type": "reach",
      "target_object": null,
      "target_location": "target_marker",
      "parameters": {
        "approach_offset": [0.0, 0.0, 0.1],
        "speed": 0.5
      },
      "prerequisites": ["action_3"]
    },
    {
      "id": "action_5",
      "action_type": "place",
      "target_object": "cube",
      "target_location": "target_marker",
      "parameters": {
        "placement_offset": [0.0, 0.0, 0.0],
        "speed": 0.2
      },
      "prerequisites": ["action_4"]
    },
    {
      "id": "action_6",
      "action_type": "release",
      "target_object": "cube",
      "target_location": null,
      "parameters": {},
      "prerequisites": ["action_5"]
    }
  ],
  "dependency_graph": {
    "action_1": [],
    "action_2": ["action_1"],
    "action_3": ["action_2"],
    "action_4": ["action_3"],
    "action_5": ["action_4"],
    "action_6": ["action_5"]
  }
}
```

### Example 2: Open Drawer Task
**Input:**
- Task: OpenDrawer
- Description: Open the top drawer of a cabinet
- Objects: [cabinet, drawer_handle_top]
- Goal: Pull drawer_handle_top to open drawer_top_joint

**Output:**
```json
{
  "task_name": "OpenDrawer",
  "description": "Open the top drawer of a cabinet",
  "actions": [
    {
      "id": "action_1",
      "action_type": "reach",
      "target_object": "drawer_handle_top",
      "target_location": null,
      "parameters": {
        "approach_offset": [0.0, 0.0, 0.0],
        "speed": 0.4
      },
      "prerequisites": []
    },
    {
      "id": "action_2",
      "action_type": "grasp",
      "target_object": "drawer_handle_top",
      "target_location": null,
      "parameters": {
        "grasp_type": "pinch",
        "force": 30.0
      },
      "prerequisites": ["action_1"]
    },
    {
      "id": "action_3",
      "action_type": "pull",
      "target_object": "drawer_handle_top",
      "target_location": null,
      "parameters": {
        "direction": "toward_robot",
        "distance": 0.3,
        "force": 50.0,
        "speed": 0.3
      },
      "prerequisites": ["action_2"]
    },
    {
      "id": "action_4",
      "action_type": "release",
      "target_object": "drawer_handle_top",
      "target_location": null,
      "parameters": {},
      "prerequisites": ["action_3"]
    }
  ],
  "dependency_graph": {
    "action_1": [],
    "action_2": ["action_1"],
    "action_3": ["action_2"],
    "action_4": ["action_3"]
  }
}
```

## Now Decompose the Following Task

{task_input}

Remember to:
- Use only the supported action types
- Specify clear prerequisites for each action
- Include relevant parameters with appropriate units
- Ensure the action sequence logically achieves the goal
- Return valid JSON only, without markdown code blocks or additional text
