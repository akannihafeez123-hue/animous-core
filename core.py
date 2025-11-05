import time, os
from reader import parse_platform
from synthesizer import generate_plugin
from executor import execute_task

# 🔐 Environment
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# 🧠 Updated memory system
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

embedding_model = OpenAIEmbeddings()
docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
memory_db = FAISS.from_documents(docs, embedding_model)

platforms = {}
plugin_scores = {}

def store_platform(name, plugin):
    platforms[name] = plugin
    doc = Document(page_content=str(plugin), metadata={"source": name})
    memory_db.add_documents([doc])
    print(f"🧠 Stored plugin for {name}")

def search_memory(query):
    return memory_db.similarity_search(query, k=3)

def update_plugin_score(name, success, payout):
    score = plugin_scores.get(name, {"success": 0, "fail": 0, "earned": 0})
    if success:
        score["success"] += 1
        score["earned"] += payout
    else:
        score["fail"] += 1
    plugin_scores[name] = score

# 🔁 Core Loop
auto_learn_enabled = True
auto_retrain_enabled = True

def heartbeat():
    print(f"💓 Animous Core is alive. Wallet: {WALLET_ADDRESS}")

def main_loop():
    while True:
        heartbeat()
        platforms_to_learn = [
            {"name": "LabelNet", "url": "https://labelnet.ai/docs"},
            {"name": "TaskHive", "url": "https://taskhive.io/help"},
        ]
        for p in platforms_to_learn:
            doc = parse_platform(p["url"])
            if auto_learn_enabled:
                plugin = generate_plugin(doc)
                if plugin:
                    store_platform(p["name"], plugin)
                    if auto_retrain_enabled:
                        for task in plugin.get("tasks", []):
                            execute_task({**task, "platform": p["name"]})
        time.sleep(300)

if __name__ == "__main__":
    main_loop()
