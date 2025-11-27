from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
import shutil



# --- load .env from project root ---
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)
# -----------------------------------

root = Path(__file__).resolve().parents[1]
docs_dir = root / "data" / "manrag" / "unstructured"
persist_dir = root / "data" / "manrag" / "chroma_store"

if persist_dir.exists():
    shutil.rmtree(persist_dir)

docs = []
for file in docs_dir.rglob("*.txt"):
    docs.extend(TextLoader(str(file)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

vectordb = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory=str(persist_dir),
)


print(f"✅ Vector store created at: {persist_dir}")
