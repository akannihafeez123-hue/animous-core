from bot import approved_plugins, blocked_plugins
from memory import update_plugin_score

task_log = []

def execute_task(task):
    platform = task.get("platform")
    if platform in blocked_plugins:
        print(f"🚫 Skipping blocked plugin: {platform}")
        return False
    if platform not in approved_plugins:
        print(f"⏸ Plugin not approved: {platform}")
        return False
    print(f"🤖 Executing: {task['title']} for ${task['pay']}")
    success = True
    task_log.append({
        "platform": platform,
        "title": task["title"],
        "pay": task["pay"],
        "status": "completed" if success else "failed"
    })
    update_plugin_score(platform, success, task["pay"])
    return success
