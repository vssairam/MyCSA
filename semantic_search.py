import json
import os
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_store"

load_dotenv(BASE_DIR / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Minimum cosine similarity for a relevant match.
# This is an initial value; we will calibrate it
# using relevant and unrelated test questions.

MIN_SIMILARITY = float(
    os.getenv("RAG_MIN_SIMILARITY", "0.30")
)


# --------------------------------------------------
# LOAD FAISS INDEX AND METADATA
# --------------------------------------------------

def load_vector_store():

    index_path = VECTOR_DIR / "policies.index"
    metadata_path = VECTOR_DIR / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "FAISS vector store not found. "
            "Run python build_vector_store.py first."
        )

    index = faiss.read_index(str(index_path))

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    chunks = metadata["chunks"]
    embedding_model = metadata["embedding_model"]

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and metadata contain different "
            "numbers of records."
        )

    return index, chunks, embedding_model


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def search_semantic_kb(query: str, top_k: int = 3):

    if not query.strip():
        return []

    index, chunks, embedding_model = load_vector_store()

    # Generate an embedding for the customer question.
    response = client.embeddings.create(
        model=embedding_model,
        input=query
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    # Normalize for cosine similarity.
    faiss.normalize_L2(query_embedding)

    # Search the most similar policy chunks.
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        similarity = float(score)

        if similarity < MIN_SIMILARITY:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "source": chunk["source"],
            "content": chunk["content"],
            "score": round(similarity, 4)
        })

    return results


# --------------------------------------------------
# TEST SEARCH
# --------------------------------------------------

if __name__ == "__main__":

    test_query = (
        "My parcel has not arrived and the tracking "
        "information has not changed."
    )

    print("\nCustomer Complaint:")
    print(test_query)

    print("\nSearching FAISS Knowledge Base...")

    matches = search_semantic_kb(test_query)

    if matches:

        for number, match in enumerate(matches, start=1):

            print(f"\nResult {number}")
            print(f"Source: {match['source']}")
            print(f"Similarity: {match['score']}")
            print(f"Content:\n{match['content']}")

    else:

        print(
            "No sufficiently relevant company policy "
            "was found."
        )