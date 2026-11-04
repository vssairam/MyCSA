import asyncio
import csv

from multi_agent_workflow import run_customer_support
from agents.log_agent import LOG_FILE


# Customer complaint test scenarios
TEST_SCENARIOS = [
    {
        "name": "Shipping Delay",
        "complaint": (
            "My order delivery is delayed by three days. "
            "I have not received any updates."
        ),
        "expected_category": "Shipping",
        "expected_source": "shipping_policy.md",
        "expected_escalation": False
    },
    {
        "name": "Refund Request",
        "complaint": (
            "I returned my product last week, but I have "
            "not received my refund yet."
        ),
        "expected_category": "Refund",
        "expected_source": "refund_policy.md",
        "expected_escalation": None
    },
    {
        "name": "Duplicate Payment",
        "complaint": (
            "I was charged twice for the same order. "
            "This is a payment dispute and I need help."
        ),
        "expected_category": "Billing",
        "expected_source": "billing_policy.md",
        "expected_escalation": True
    },
    {
        "name": "Account Login Issue",
        "complaint": (
            "I cannot log in to my account because "
            "I forgot my password."
        ),
        "expected_category": "Account",
        "expected_source": "account_and_login.md",
        "expected_escalation": None
    },
    {
        "name": "Unrelated Request",
        "complaint": (
            "Can you book an international flight ticket "
            "for my vacation?"
        ),
        "expected_category": "Other",
        "expected_source": None,
        "expected_escalation": True
    }
]


def read_ticket(ticket_id):
    """
    Retrieve a ticket from the CSV log.
    """

    if not LOG_FILE.exists():
        return None

    with LOG_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["ticket_id"] == ticket_id:
                return row

    return None


async def test_scenarios():

    print("=" * 65)
    print("MYCSA - MULTI-SCENARIO INTEGRATION TEST")
    print("=" * 65)

    passed = 0
    failed = 0

    for index, scenario in enumerate(TEST_SCENARIOS, start=1):

        print(f"\nTEST {index}: {scenario['name']}")
        print("-" * 65)

        try:
            # Run all four AutoGen agents
            result = await run_customer_support(
                scenario["complaint"]
            )

            # Find successful logging tool executions
            ticket_ids = []

            for message in result.messages:

                if (
                    message.source == "Log_Agent"
                    and type(message).__name__
                    == "ToolCallExecutionEvent"
                ):

                    for tool_result in message.content:

                        if (
                            tool_result.name == "log_customer_ticket"
                            and not tool_result.is_error
                        ):

                            import ast

                            data = ast.literal_eval(
                                tool_result.content
                            )

                            if data.get("logged") is True:
                                ticket_ids.append(
                                    data["ticket_id"]
                                )

            # Each complaint should create exactly one ticket
            assert len(ticket_ids) == 1, (
                f"Expected 1 ticket, found {len(ticket_ids)}"
            )

            ticket = read_ticket(ticket_ids[0])

            assert ticket is not None, (
                "Ticket not found in CSV"
            )

            print("\nVALIDATION RESULTS")
            print("Ticket ID:", ticket["ticket_id"])
            print("Category:", ticket["category"])
            print("Escalated:", ticket["escalated"])
            print("Sources:", ticket["sources"])

            # Validate complaint category
            assert (
                ticket["category"].lower()
                == scenario["expected_category"].lower()
            ), "Incorrect complaint category"

            # Validate knowledge base source
            if scenario["expected_source"] is not None:

                assert (
                    scenario["expected_source"]
                    in ticket["sources"]
                ), "Incorrect knowledge base source"

            else:
                assert ticket["sources"] in (
                    "[]",
                    ""
                ), "Unexpected knowledge base source"

            # Validate escalation when explicitly expected
            if scenario["expected_escalation"] is not None:

                actual_escalation = (
                    ticket["escalated"].lower() == "true"
                )

                assert (
                    actual_escalation
                    == scenario["expected_escalation"]
                ), "Incorrect escalation decision"

            print("RESULT: PASSED")
            passed += 1

        except Exception as error:

            print(f"RESULT: FAILED - {error}")
            failed += 1

    print("\n" + "=" * 65)
    print("FINAL TEST SUMMARY")
    print("=" * 65)

    print("Passed:", passed)
    print("Failed:", failed)
    print("Total:", len(TEST_SCENARIOS))


if __name__ == "__main__":
    asyncio.run(test_scenarios())