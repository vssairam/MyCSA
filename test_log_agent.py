import asyncio
import csv
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.log_agent import (
    create_log_agent,
    log_customer_ticket,
    LOG_FILE
)


def test_escalation_rules():
    """
    Test deterministic CSV logging and escalation.
    """

    scenarios = [
        {
            "name": "Normal Shipping Delay",
            "complaint": "My order delivery is delayed by three days.",
            "category": "Shipping",
            "urgency": "Medium",
            "sentiment": "Frustrated",
            "summary": "Customer reports delayed delivery.",
            "resolution": "Please contact support with your Order ID.",
            "confidence": 0.95,
            "sources": ["shipping_policy.md"],
            "kb_found": True,
            "requires_human_review": False,
            "expected_escalation": False
        },
        {
            "name": "Angry Customer - High Urgency",
            "complaint": "I am extremely angry about an unauthorized payment!",
            "category": "Billing",
            "urgency": "High",
            "sentiment": "Angry",
            "summary": "Customer reports unauthorized payment.",
            "resolution": "This requires human billing review.",
            "confidence": 0.80,
            "sources": ["billing_policy.md"],
            "kb_found": True,
            "requires_human_review": True,
            "expected_escalation": True
        },
        {
            "name": "No Relevant Knowledge Base Policy",
            "complaint": "Can you arrange international flight tickets?",
            "category": "Other",
            "urgency": "Low",
            "sentiment": "Neutral",
            "summary": "Customer requests flight booking.",
            "resolution": "This request requires human review.",
            "confidence": 0.30,
            "sources": [],
            "kb_found": False,
            "requires_human_review": True,
            "expected_escalation": True
        }
    ]

    for index, scenario in enumerate(scenarios, start=1):

        print(f"\nTEST {index}: {scenario['name']}")
        print("-" * 60)

        result = log_customer_ticket(
            complaint=scenario["complaint"],
            category=scenario["category"],
            urgency=scenario["urgency"],
            sentiment=scenario["sentiment"],
            summary=scenario["summary"],
            resolution=scenario["resolution"],
            confidence=scenario["confidence"],
            sources=scenario["sources"],
            kb_found=scenario["kb_found"],
            requires_human_review=scenario["requires_human_review"]
        )

        print("Ticket ID:", result["ticket_id"])
        print("Logged:", result["logged"])
        print("Escalated:", result["escalated"])
        print("Reasons:", result["escalation_reasons"])

        assert result["logged"] is True
        assert result["escalated"] == scenario["expected_escalation"]

        print("RESULT: PASSED")

    assert LOG_FILE.exists(), "CSV log file was not created."

    with LOG_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= len(scenarios)

    print("\nSUCCESS: All escalation and CSV logging tests passed!")


async def test_autogen_log_agent():
    """
    Verify that the AutoGen agent invokes the logging tool.
    """

    load_dotenv()

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        agent = create_log_agent(model_client)

        task = """
Log this customer support ticket using the provided details.

Original complaint:
My order delivery is delayed by three days.

Category: Shipping
Urgency: Medium
Sentiment: Frustrated
Summary: Customer reports delayed delivery.
Resolution: Please contact support with your Order ID.
Confidence: 0.95
Sources: ["shipping_policy.md"]
KB found: true
Requires human review: false

Call the logging tool and report its result.
"""

        result = await agent.run(task=task)

        print("\nAUTOGEN LOG AGENT TEST")
        print("-" * 60)

        for message in result.messages:
            print(
                type(message).__name__,
                ":",
                str(message.content)[:1000]
            )

        assert any(
            type(message).__name__ == "ToolCallExecutionEvent"
            for message in result.messages
        ), "AutoGen agent did not execute the logging tool."

        print("\nSUCCESS: AutoGen Log Agent invoked the logging tool!")

    finally:
        await model_client.close()


async def main():

    print("=" * 60)
    print("LOG AGENT TEST")
    print("=" * 60)

    test_escalation_rules()

    await test_autogen_log_agent()

    print("\n" + "=" * 60)
    print("ALL LOG AGENT TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())