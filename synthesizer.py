from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 🧠 Embedded memory logic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

embedding_model = OpenAIEmbeddings()
memory_db = FAISS(embedding_model.embed_query, embedding_model)

def search_memory(query):
    return memory_db.similarity_search(query, k=3)

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

def generate_plugin(doc):
    similar = search_memory("task format: " + doc["content"][:100])
    context = "\n\n".join([s.page_content for s in similar])
    prompt = PromptTemplate.from_template("""
You are an AI agent that reads platform documentation and generates Python plugins to interact with it.

New Platform Documentation:
{doc}

Similar Platforms:
{context}

Generate a plugin that:
- Extracts tasks
- Estimates payout
- Submits results if AI-solvable
- Returns a dictionary with 'platform', 'tasks', and 'executor'

Respond with Python code only.
""")
    response = llm.predict(prompt.format(doc=doc["content"], context=context))
    try:
        plugin = eval(response, {"__builtins__": {}})
        return plugin
    except Exception as e:
        print(f"Plugin generation failed: {e}")
        return None
