import asyncio
import json
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.query_agent import create_query_agent


async def main():

    load_dotenv()

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        agent = create_query_agent(model_client)

        complaint = (
            "My order is delayed by five days. "
            "I am very frustrated because nobody has updated me."
        )

        print("=" * 55)
        print("CUSTOMER QUERY AGENT TEST")
        print("=" * 55)

        print("\nCustomer Complaint:")
        print(complaint)

        result = await agent.run(task=complaint)

        response = result.messages[-1].content

        print("\nAgent Classification:")
        print("-" * 55)

        classification = json.loads(response)

        print(json.dumps(classification, indent=4))

        required_fields = {
            "category", "urgency", "sentiment", "summary"
        }

        assert set(classification.keys()) == required_fields

        assert classification["category"] in {
            "Shipping", "Refund", "Account",
            "Product", "Billing", "Other"
        }

        assert classification["urgency"] in {
            "Low", "Medium", "High"
        }

        assert classification["sentiment"] in {
            "Positive", "Neutral", "Frustrated", "Angry"
        }

        assert isinstance(classification["summary"], str)
        assert classification["summary"].strip()

        print("\nSUCCESS: Customer Query Agent returned valid classification!")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())