"""Quick script to verify the ChromaDB index is queryable.

Loads the persisted index, embeds a sample query, retrieves the top-3
chunks, and prints their IDs, source types, and distances.
"""

from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore

store = VectorStore()
print(f"Collection count: {store.count()}")
print()

embedder = Embedder()
query = "deployment issue"
print(f"Query: {query!r}")
print()

result = store.query(embedder.embed_query(query), top_k=3)

print("Top-3 results:")
for i in range(len(result["ids"][0])):
    chunk_id = result["ids"][0][i]
    meta = result["metadatas"][0][i]
    dist = result["distances"][0][i]
    doc_preview = result["documents"][0][i][:100].replace("\n", " ")
    print(f"  {i + 1}. {chunk_id}")
    print(f"     doc_id:     {meta['doc_id']}")
    print(f"     source:     {meta['source_type']}")
    print(f"     distance:   {dist:.4f}")
    print(f"     preview:    {doc_preview}...")
    print()