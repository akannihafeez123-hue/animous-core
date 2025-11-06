# executor.py
import time
import traceback
from typing import Dict, Any

# imports from your bot and memory modules
from bot import approved_plugins, blocked_plugins
from memory import update_plugin_score

# provider helper for generating submission content or solving tasks
from llm_utils import provider_generate

# local task history
task_log = []

def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a task dict and return a result dict:
    {
      "status": "completed" | "failed" | "requires_human" | "error",
      "pay": float,
      "detail": str
    }
    Behavior:
    - Skip blocked plugins
    - Require approval for unapproved plugins
    - For ai_solvable tasks, attempt to generate a solution via provider_generate
    - Update plugin score via memory.update_plugin_score
    """
    platform = task.get("platform")
    title = task.get("title", "<untitled>")
    pay = float(task.get("pay", 0) or 0)
    ai_solvable = task.get("ai_solvable", True)

    # Quick safety checks
    if platform in blocked_plugins:
        detail = f"Plugin {platform} is blocked"
        print(f"🚫 {detail}")
        result = {"status": "failed", "pay": 0, "detail": detail}
        task_log.append({**task, **result})
        return result

    if platform not in approved_plugins:
        detail = f"Plugin {platform} not approved"
        print(f"⏸ {detail}")
        result = {"status": "requires_human", "pay": 0, "detail": detail}
        task_log.append({**task, **result})
        return result

    # If not AI-solvable, require human
    if not ai_solvable:
        detail = "Marked not AI-solvable"
        print(f"🛑 {detail} for {title}")
        result = {"status": "requires_human", "pay": 0, "detail": detail}
        task_log.append({**task, **result})
        return result

    # Attempt automated execution
    try:
        print(f"🤖 Executing task '{title}' on {platform} (pay ${pay})")
        prompt = (
            f"You are an autonomous agent tasked with completing a platform task.\n"
            f"Platform: {platform}\nTask title: {title}\n"
            f"Task details: {task.get('description','')}\n"
            "Produce a concise, step-by-step result that can be used to submit or verify the task.\n"
            "If you cannot complete exactly, explain what needs human action.\n"
            "Return in plain text."
        )
        solution = provider_generate(prompt, max_tokens=400, temperature=0.15)
        # Basic success heuristics: solution length and presence of actionable text
        success = bool(solution and len(solution.strip()) > 20)
        status = "completed" if success else "requires_human"
        detail = solution.strip() if solution else "No solution generated"

        # Append to task log and update scores
        result = {"status": status, "pay": pay if success else 0, "detail": detail}
        task_log.append({**task, **result})
        update_plugin_score(platform, success, pay if success else 0)

        print(f"✅ Task '{title}' result: {status}; earned ${result['pay']}")
        return result

    except Exception:
        traceback.print_exc()
        detail = "Execution error"
        result = {"status": "error", "pay": 0, "detail": detail}
        task_log.append({**task, **result})
        update_plugin_score(platform, False, 0)
        return result
