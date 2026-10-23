"""
Optional local check of the document search (needs OPENAI_API_KEY in .env).

The app builds this index in memory on the first chat question, so nothing
needs to be saved to the repo. Run this to see what retrieval returns:
    python code/vector_store.py "alignment procedure for M11"
"""
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

import assistant as ai
import plant_metrics as pm

load_dotenv(pm.ROOT / ".env")
query = " ".join(sys.argv[1:]) or "What is the procedure for handling material quality issues?"
search, n_chunks = ai.build_doc_search(pm.DOCS_DIR, OpenAIEmbeddings(model="text-embedding-3-small"))
print(f"Indexed {n_chunks} chunks. Top results for: {query}\n")
for hit in search(query, 4):
    print(f"[{hit['folder']}/{hit['source']}]\n{hit['text'][:300]}\n")
