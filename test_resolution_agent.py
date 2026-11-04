import asyncio
import json
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.resolution_agent import create_resolution_agent
from kb_search import search_kb


async def main():

    load_dotenv()

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        # Step 1: Customer complaint
        complaint = (
            "My order delivery is delayed by three days. "
            "I am frustrated because I have not received any updates."
        )

        # Step 2: Retrieve relevant company policies
        kb_results = [
            result
            for result in search_kb(complaint, top_k=20)
            if result["source"] == "shipping_policy.md"
        ]

        if not kb_results:
            print("ERROR: No relevant shipping policy found.")
            return

        # Select the most relevant policy sections
        kb_results = kb_results[:3]

        policy_context = "\n\n".join(
            f"Source: {item['source']}\n"
            f"Section: {item['heading']}\n"
            f"Policy:\n{item['content']}"
            for item in kb_results
        )

        # Step 3: Provide classification and policy information
        task = f"""
CUSTOMER COMPLAINT:
{complaint}

COMPLAINT CLASSIFICATION:
Category: Shipping
Urgency: Medium
Sentiment: Frustrated

RETRIEVED COMPANY POLICIES:
{policy_context}

Generate a professional customer response
using only the retrieved policies.

Return valid JSON.
"""

        print("=" * 60)
        print("RESOLUTION AGENT TEST")
        print("=" * 60)

        print("\nCustomer Complaint:")
        print(complaint)

        # Step 4: Run Resolution Agent
        agent = create_resolution_agent(model_client)

        result = await agent.run(task=task)

        response = result.messages[-1].content

        # Step 5: Validate JSON
        resolution = json.loads(response)

        required_fields = {
            "response",
            "confidence",
            "sources",
            "requires_human_review",
            "review_reason"
        }

        assert set(resolution.keys()) == required_fields
        assert isinstance(resolution["response"], str)
        assert resolution["response"].strip()
        assert isinstance(resolution["confidence"], (int, float))
        assert 0 <= resolution["confidence"] <= 1
        assert isinstance(resolution["sources"], list)
        assert isinstance(resolution["requires_human_review"], bool)
        assert isinstance(resolution["review_reason"], str)

        allowed_sources = {
            item["source"] for item in kb_results
        }

        assert set(resolution["sources"]).issubset(
            allowed_sources
        ), "Agent cited a source that was not retrieved."

        print("\nResolution Agent Output:")
        print("-" * 60)
        print(json.dumps(resolution, indent=4, ensure_ascii=False))

        print("\nSUCCESS: Resolution Agent returned valid JSON!")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())