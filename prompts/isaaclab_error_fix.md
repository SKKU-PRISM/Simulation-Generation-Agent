# IsaacLab Code Error Fix

You previously generated IsaacLab environment code that failed to execute. Fix the error and return ONLY the corrected file(s).

## Rules

1. Return corrected files using the same format: fenced code blocks with filenames
2. Only return files that need changes — do not repeat unchanged files
3. Preserve all working code — only modify what is necessary to fix the error
4. Check for these common issues:
   - Missing imports (add the missing import)
   - Wrong module path (check `isaaclab.*` vs `isaaclab_tasks.*`)
   - MISSING sentinel not overridden (assign a value in `__post_init__`)
   - Wrong prim path format (must use `{ENV_REGEX_NS}/Name` for per-env assets)
   - Incorrect class inheritance (must match IsaacLab API)
   - `AppLauncher` not initialized before physics imports in run_env.py

## Error Information

### Original Code

{ORIGINAL_CODE}

### Error Traceback

```
{ERROR_TRACEBACK}
```

### Known Error Patterns and Fixes

Before analyzing the traceback, check if it matches one of these known patterns:

a. `KeyError: "Scene entity with key '0' not found"`
   → Cause: `if x not in env.scene:` — InteractiveScene has no `__contains__`
   → Fix: Change to `if x not in env.scene.keys():`

b. `RuntimeError: Found an articulation root when resolving ... for rigid objects`
   → Cause: USD asset has embedded FixedJoint/ArticulationRoot (common with Factory assets)
   → Fix: Add `articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)` to the spawn config of the RigidObjectCfg

c. `ImportError: cannot import name 'ArticulationRootPropertiesCfg' from 'isaaclab.sim.spawners...'`
   → Cause: Wrong import path — ArticulationRootPropertiesCfg is NOT in spawners
   → Fix: Use `import isaaclab.sim as sim_utils` then `sim_utils.ArticulationRootPropertiesCfg`

d. `AttributeError: module 'mdp' has no attribute 'ImplicitActuatorCfg'`
   → Cause: ImplicitActuatorCfg is NOT in isaaclab.envs.mdp (thus not in the local mdp/ package)
   → Fix: `from isaaclab.actuators import ImplicitActuatorCfg` then use directly in ArticulationCfg.actuators

e. `TypeError: Missing values ... scene.cabinet.actuators`
   → Cause: ArticulationCfg requires actuators dict, cannot be empty or omitted
   → Fix: Add actuators dict with ImplicitActuatorCfg for each joint group:
   ```python
   actuators={"drawers": ImplicitActuatorCfg(joint_names_expr=["drawer_top_joint"], effort_limit=87.0, stiffness=10.0, damping=1.0)}
   ```

### Fix Instructions

Analyze the traceback, identify the root cause, and return the corrected file(s).
If the error is an ImportError, check the exact module path in IsaacLab.
If the error is an AttributeError, check if the API has changed or if you used the wrong class.
