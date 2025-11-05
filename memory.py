from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

embedding_model = OpenAIEmbeddings()
memory_db = FAISS.new(embedding_model)

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
