import time, os
from reader import parse_platform
from synthesizer import generate_plugin
from memory import store_platform
from executor import execute_task
from bot import auto_learn_enabled, auto_retrain_enabled

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

def heartbeat():
    print(f"💓 Animous Core is alive. Wallet: {WALLET_ADDRESS}")

def main_loop():
    while True:
        heartbeat()
        platforms = [
            {"name": "LabelNet", "url": "https://labelnet.ai/docs"},
            {"name": "TaskHive", "url": "https://taskhive.io/help"},
        ]
        for p in platforms:
            doc = parse_platform(p["url"])
            if auto_learn_enabled:
                plugin = generate_plugin(doc)
                store_platform(p["name"], plugin)
                if auto_retrain_enabled:
                    for task in plugin.get("tasks", []):
                        execute_task({**task, "platform": p["name"]})
        time.sleep(300)

if __name__ == "__main__":
    main_loop()
