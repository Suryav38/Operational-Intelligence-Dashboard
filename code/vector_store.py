"""
Optional local check of document search (needs OPENAI_API_KEY in .env).
The app builds this index in memory on the first copilot question.
    python code/vector_store.py "EUV scanner decision"
"""
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

import assistant as ai
import fab_data as fd

load_dotenv(fd.ROOT / ".env")
query = " ".join(sys.argv[1:]) or "What did the planning review decide about Fab B etch?"
search, n_chunks = ai.build_doc_search(fd.DOCS, OpenAIEmbeddings(model="text-embedding-3-small"))
print(f"Indexed {n_chunks} chunks. Top results for: {query}\n")
for hit in search(query, 4):
    print(f"[{hit['folder']}/{hit['source']}]\n{hit['text'][:300]}\n")
