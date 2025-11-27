from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# ---- Load .env (same trick as before) ----
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
# ------------------------------------------

root = Path(__file__).resolve().parents[1]
persist_dir = root / "data" / "manrag" / "chroma_store"

# Re-open the existing Chroma store
embeddings = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=str(persist_dir),
    embedding_function=embeddings,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ask_rag(question: str):
    # 1) Retrieve relevant chunks (new API)
    docs = retriever.invoke(question)

    print("\n--- Retrieved context ---")
    for i, d in enumerate(docs, start=1):
        print(f"\n[Doc {i}]")
        print(d.page_content[:600], "...\n")

    # 2) Send context + question to the LLM
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""
You are a manufacturing expert assistant.

Answer the question STRICTLY using the retrieved context below.
If the context contains procedures, steps, recommendations, or instructions, transform them into a clean answer.

If the answer isn't present, respond: "The available documents do not contain that information."

---

📄 CONTEXT:
{context}

---

❓ QUESTION:
{question}

---

📌 FORMAT YOUR RESPONSE LIKE THIS:

- Summary (1–2 sentences)
- Key Points or Steps (bullet points)
- If relevant: preventive actions or follow-up

Keep it concise but useful.

"""

    answer = llm.invoke(prompt)
    print("\n--- Answer ---")
    print(answer.content)

if __name__ == "__main__":
    # Try a few example questions:
    ask_rag("What is the procedure for handling material quality issues?")
    ask_rag("What do operator notes say about machine M11?")
    ask_rag("What kind of alignment checks are recommended for M-series machines?")
