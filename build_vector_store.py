import os
import json
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DIR = BASE_DIR / "vector_store"

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

load_dotenv(BASE_DIR / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------
# LOAD KNOWLEDGE BASE DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = []

    for file_path in sorted(KB_DIR.glob("*.md")):

        content = file_path.read_text(encoding="utf-8")

        if content.strip():

            documents.append({
                "source": file_path.name,
                "content": content
            })

    return documents


# --------------------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def chunk_documents(documents, chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):

    import re

    chunks = []

    for document in documents:

        text = document["content"]

        # Split at Markdown headings while preserving headings.
        sections = re.split(
            r"(?=^#{1,6}\s+)",
            text,
            flags=re.MULTILINE
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            # Preserve short policy sections as complete chunks.
            if len(section) <= chunk_size:

                chunks.append({
                    "source": document["source"],
                    "content": section
                })

                continue

            # Split longer sections at paragraph boundaries.
            paragraphs = re.split(r"\n\s*\n", section)

            current_chunk = ""

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                # Keep paragraphs together where possible.
                if (
                    current_chunk
                    and len(current_chunk) + len(paragraph) + 2 > chunk_size
                ):

                    chunks.append({
                        "source": document["source"],
                        "content": current_chunk.strip()
                    })

                    current_chunk = ""

                # Split exceptionally long paragraphs.
                if len(paragraph) > chunk_size:

                    if current_chunk:
                        chunks.append({
                            "source": document["source"],
                            "content": current_chunk.strip()
                        })
                        current_chunk = ""

                    words = paragraph.split()
                    piece = ""

                    for word in words:

                        if piece and len(piece) + len(word) + 1 > chunk_size:

                            chunks.append({
                                "source": document["source"],
                                "content": piece
                            })

                            piece = ""

                        piece = f"{piece} {word}".strip()

                    if piece:
                        current_chunk = piece

                else:

                    current_chunk = (
                        f"{current_chunk}\n\n{paragraph}"
                        if current_chunk
                        else paragraph
                    )

            if current_chunk.strip():

                chunks.append({
                    "source": document["source"],
                    "content": current_chunk.strip()
                })

    return chunks


# --------------------------------------------------
# GENERATE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(chunks):

    texts = [chunk["content"] for chunk in chunks]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )

    return embeddings


# --------------------------------------------------
# BUILD AND SAVE FAISS INDEX
# --------------------------------------------------

def build_vector_store():

    print("Loading knowledge base documents...")

    documents = load_documents()

    if not documents:
        raise ValueError("No Markdown policy documents found.")

    print(f"Documents loaded: {len(documents)}")

    print("Splitting documents into chunks...")

    chunks = chunk_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    print("Generating OpenAI embeddings...")

    embeddings = generate_embeddings(chunks)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimensions = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimensions)

    index.add(embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        str(VECTOR_DIR / "policies.index")
    )

    # Save chunk metadata
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": chunks
    }

    (VECTOR_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nFAISS Vector Store Created Successfully!")
    print(f"Vector count: {index.ntotal}")
    print(f"Vector dimensions: {dimensions}")
    print(f"Saved to: {VECTOR_DIR}")


if __name__ == "__main__":
    build_vector_store()