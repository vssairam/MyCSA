from pathlib import Path
import re

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
KB_FOLDER = BASE_DIR / "knowledge_base"

STOP_WORDS = {
    "the", "is", "am", "are", "was", "were",
    "a", "an", "and", "or", "to", "of",
    "in", "on", "at", "for", "with", "my",
    "i", "me", "you", "your", "it", "this",
    "that", "do", "does", "did", "what",
    "how", "can", "could", "should", "would",
    "have", "has", "had", "been", "be",
    "please", "want", "need", "through",
    "our", "their", "they", "them", "from",
    "after", "before", "within", "more",
    "than", "not", "cannot", "about",

    # Generic e-commerce words
    "store", "demo", "ai", "customer",
    "customers", "order", "orders",
    "days", "day", "business",
    "support", "team", "information"
}

SYNONYMS = {
    "late": "delayed",
    "delay": "delayed",
    "delays": "delayed",
    "delivery": "shipping",
    "delivered": "shipping",
    "parcel": "package",
    "money": "payment",
    "charged": "charge",
    "charging": "charge",
    "twice": "duplicate",
    "signin": "login",
    "logon": "login",
    "broken": "damaged",
    "faulty": "defective",
    "cancel": "cancellation",
    "cancelled": "cancellation",
    "canceled": "cancellation"
}


# =====================================================
# STEP 1: EXTRACT KEYWORDS
# =====================================================

def extract_keywords(text):

    words = re.findall(r"\b\w+\b", text.lower())

    keywords = set()

    for word in words:

        if word in STOP_WORDS:
            continue

        normalized = SYNONYMS.get(word, word)

        if len(normalized) > 2:
            keywords.add(normalized)

    return keywords


# =====================================================
# STEP 2: LOAD KNOWLEDGE BASE DOCUMENTS
# =====================================================

def load_kb_documents():

    documents = []

    if not KB_FOLDER.exists():
        print("ERROR: Knowledge Base folder not found.")
        return documents

    for file_path in sorted(KB_FOLDER.glob("*.md")):

        content = file_path.read_text(encoding="utf-8")

        # Split into sections using Markdown headings
        sections = re.split(
            r"(?=^## )",
            content,
            flags=re.MULTILINE
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            lines = section.splitlines()

            heading = lines[0].strip()

            # Ignore standalone document headings
            if heading.startswith("# ") and not heading.startswith("## "):
                continue

            # Ignore empty sections
            if len(lines) < 2:
                continue

            documents.append({
                "source": file_path.name,
                "heading": heading,
                "content": section
            })

    return documents


# =====================================================
# STEP 3: SEARCH KNOWLEDGE BASE
# =====================================================

def search_kb(query, top_k=3):

    documents = load_kb_documents()

    query_keywords = extract_keywords(query)

    if not query_keywords:
        return []

    results = []

    for document in documents:

        heading_keywords = extract_keywords(
            document["heading"]
        )

        content_keywords = extract_keywords(
            document["content"]
        )

        # Keywords appearing in both query and content
        matched_keywords = query_keywords.intersection(
            content_keywords
        )

        if not matched_keywords:
            continue

        # Give higher importance to section headings
        heading_matches = query_keywords.intersection(
            heading_keywords
        )

        # Weighted relevance score
        score = (
            len(matched_keywords)
            + (2 * len(heading_matches))
        )

        # Require meaningful relevance
        if score < 2:
            continue

        results.append({
            "source": document["source"],
            "heading": document["heading"],
            "content": document["content"],
            "score": score,
            "matched_keywords": sorted(matched_keywords)
        })

    # Sort by relevance
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# =====================================================
# STEP 4: INTERACTIVE TEST
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print(" THE AI DEMO STORE - KNOWLEDGE BASE SEARCH")
    print("=" * 60)

    documents = load_kb_documents()

    print(f"\nKnowledge Base Sections Loaded: {len(documents)}")

    while True:

        print("\n" + "-" * 60)

        query = input(
            "Enter customer complaint (or type exit): "
        ).strip()

        if query.lower() == "exit":
            print("\nExiting Knowledge Base Search.")
            break

        if not query:
            print("Please enter a customer complaint.")
            continue

        results = search_kb(query)

        if not results:

            print("\nNo relevant knowledge found.")
            print(
                "This complaint may require human escalation."
            )

        else:

            print(f"\nFound {len(results)} relevant sections:\n")

            for index, result in enumerate(
                results, start=1
            ):

                print("=" * 60)
                print(f"RESULT {index}")
                print("=" * 60)

                print(f"Source: {result['source']}")
                print(f"Section: {result['heading']}")
                print(f"Relevance Score: {result['score']}")

                print(
                    "Matched Keywords:",
                    ", ".join(result["matched_keywords"])
                )

                print("\nPolicy Information:")
                print(result["content"])