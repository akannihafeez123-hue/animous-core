from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# 🧠 Memory system
embedding_model = OpenAIEmbeddings()
docs = [Document(page_content="initial memory", metadata={"source": "bootstrap"})]
memory_db = FAISS.from_documents(docs, embedding_model)

def search_memory(query):
    return memory_db.similarity_search(query, k=3)

# 🚀 GPT-5 model
llm = ChatOpenAI(model="gpt-5", temperature=0.3)

def generate_plugin(doc):
    similar = search_memory("task format: " + doc["content"][:100])
    context = "\n\n".join([s.page_content for s in similar])
    prompt = PromptTemplate.from_template("""
You are Animous Core — a living AI that reads platform documentation and generates Python plugins to interact with it.

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
