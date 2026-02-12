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

### Fix Instructions

Analyze the traceback, identify the root cause, and return the corrected file(s).
If the error is an ImportError, check the exact module path in IsaacLab.
If the error is an AttributeError, check if the API has changed or if you used the wrong class.
