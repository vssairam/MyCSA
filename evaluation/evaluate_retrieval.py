import json
import sys
from pathlib import Path

# Allow imports from the MyCSA project root.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents.knowledge_agent import search_company_knowledge


def load_test_cases():
    """Load the evaluation dataset."""

    test_file = BASE_DIR / "evaluation" / "test_cases.json"

    with open(test_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["test_cases"]


def evaluate_retrieval():
    """Evaluate Knowledge Agent retrieval accuracy."""

    test_cases = load_test_cases()

    total_hit_rate = 0
    total_precision = 0
    total_recall = 0

    print("\nMyCSA RAG Retrieval Evaluation")
    print("=" * 65)

    for test in test_cases:

        query = test["query"]

        expected = {
            (item["source"], item["section"])
            for item in test["expected_sections"]
        }

        response = json.loads(search_company_knowledge(query))

        retrieved = {
            (item["source"], item["section"])
            for item in response.get("matches", [])
        }

        correct = expected.intersection(retrieved)

        hit_rate = 1.0 if correct else 0.0

        precision = (
            len(correct) / len(retrieved)
            if retrieved else 0.0
        )

        recall = (
            len(correct) / len(expected)
            if expected else 0.0
        )

        total_hit_rate += hit_rate
        total_precision += precision
        total_recall += recall

        print(f"\nTest Case: {test['id']}")
        print(f"Category: {test['category']}")
        print(f"Query: {query}")

        print("Expected:", sorted(expected))
        print("Retrieved:", sorted(retrieved))

        print(f"Hit Rate@3: {hit_rate:.2f}")
        print(f"Precision@3: {precision:.2f}")
        print(f"Recall@3: {recall:.2f}")

    count = len(test_cases)

    print("\n" + "=" * 65)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 65)

    print(f"Total Test Cases: {count}")
    print(f"Average Hit Rate@3: {total_hit_rate / count:.2%}")
    print(f"Average Precision@3: {total_precision / count:.2%}")
    print(f"Average Recall@3: {total_recall / count:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()