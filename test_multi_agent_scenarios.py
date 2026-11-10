import asyncio
import csv

from selector_workflow import run_selector_customer_support
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
            result = await run_selector_customer_support(
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

            # Validate the order of the four agents.
            # Ignore tool-call events because they are intermediate messages.
            expected_agent_order = [
                "Customer_Query_Agent",
                "Knowledge_Base_Agent",
                "Resolution_Agent",
                "Log_Agent"
            ]

            actual_agent_order = []

            for message in result.messages:

                if message.source not in expected_agent_order:
                    continue

                if type(message).__name__ in (
                    "ToolCallRequestEvent",
                    "ToolCallExecutionEvent",
                    "ModelClientStreamingChunkEvent"
                ):
                    continue

                actual_agent_order.append(message.source)

            assert actual_agent_order == expected_agent_order, (
                "Incorrect agent execution order.\n"
                f"Expected: {expected_agent_order}\n"
                f"Actual: {actual_agent_order}"
            )

            print("Agent execution order: PASSED")
            print("Exactly one ticket logging event: PASSED")

            # Extract actual Knowledge Base retrieval sources
            import json

            kb_sources = set()
            kb_search_count = 0

            for message in result.messages:

                if (
                    message.source == "Knowledge_Base_Agent"
                    and type(message).__name__
                    == "ToolCallExecutionEvent"
                ):

                    for tool_result in message.content:

                        if tool_result.name == "search_company_knowledge":

                            assert not tool_result.is_error, (
                                "Knowledge Base search tool failed"
                            )

                            kb_search_count += 1

                            kb_data = json.loads(
                                tool_result.content
                            )

                            assert isinstance(kb_data, dict), (
                                "Knowledge Base result must be a JSON object"
                            )

                            if kb_data.get("found") is True:

                                kb_sources.update(
                                    kb_data.get("sources", [])
                                )

            assert kb_search_count >= 1, (
                "Knowledge Base search tool was not executed"
            )

            print(
                "Knowledge Base retrieval sources:",
                sorted(kb_sources)
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

            # Validate response grounding when no policy is found
            if scenario["expected_source"] is None:

                resolution_messages = [
                    message
                    for message in result.messages
                    if (
                        message.source == "Resolution_Agent"
                        and type(message).__name__ == "TextMessage"
                    )
                ]

                assert len(resolution_messages) == 1, (
                    "Expected exactly one Resolution Agent response"
                )

                import json

                resolution_data = json.loads(
                    resolution_messages[0].content
                )

                response_text = resolution_data["response"].lower()

                # The agent must acknowledge missing policy information.
                assert (
                    "could not find information" in response_text
                ), "Resolution did not acknowledge missing knowledge"

                # The agent must not claim that a service is unavailable.
                unsupported_phrases = [
                    "do not provide",
                    "does not provide",
                    "do not offer",
                    "does not offer",
                    "no services related to",
                    "do not have relevant company policies or services"
                ]

                assert not any(
                    phrase in response_text
                    for phrase in unsupported_phrases
                ), "Resolution contains an unsupported service claim"

                # Validate required escalation fields.
                assert resolution_data["requires_human_review"] is True, (
                    "Missing-knowledge request must require human review"
                )

                assert resolution_data["sources"] == [], (
                    "Missing-knowledge response must have no sources"
                )

                assert resolution_data["confidence"] < 0.7, (
                    "Missing-knowledge confidence must be below 0.7"
                )

                print("Knowledge grounding validation: PASSED")

            # Validate Resolution Agent sources against logged sources
            import json
            import ast

            resolution_messages = [
                message
                for message in result.messages
                if (
                    message.source == "Resolution_Agent"
                    and type(message).__name__ == "TextMessage"
                )
            ]

            assert len(resolution_messages) == 1, (
                "Expected exactly one Resolution Agent response"
            )

            resolution_data = json.loads(
                resolution_messages[0].content
            )

            resolution_sources = resolution_data["sources"]

            logged_sources = ast.literal_eval(
                ticket["sources"]
            ) if ticket["sources"] else []

            assert isinstance(resolution_sources, list), (
                "Resolution sources must be a list"
            )

            assert sorted(resolution_sources) == sorted(logged_sources), (
                "Resolution Agent sources do not match logged sources.\n"
                f"Resolution sources: {resolution_sources}\n"
                f"Logged sources: {logged_sources}"
            )

            if scenario["expected_source"] is not None:
                assert scenario["expected_source"] in resolution_sources, (
                    "Expected policy source missing from Resolution Agent"
                )

            print("Resolution-to-log source consistency: PASSED")

            # Validate Knowledge Base-to-Resolution source traceability
            resolution_source_set = set(resolution_sources)

            unsupported_sources = resolution_source_set - kb_sources

            assert not unsupported_sources, (
                "Resolution Agent cited policies not retrieved "
                "by the Knowledge Base.\n"
                f"Knowledge Base sources: {sorted(kb_sources)}\n"
                f"Resolution sources: {sorted(resolution_source_set)}\n"
                f"Unsupported sources: {sorted(unsupported_sources)}"
            )

            if not kb_sources:
                assert not resolution_source_set, (
                    "Resolution Agent must not cite policies "
                    "when the Knowledge Base returned no sources"
                )

            print("Knowledge Base-to-Resolution traceability: PASSED")

            # Validate refund policy factual accuracy
            if scenario["expected_category"] == "Refund":

                response_text = resolution_data["response"].lower()

                unsupported_refund_claims = [
                    "refunds typically process within 7 business days",
                    "refunds are processed within 7 business days",
                    "refunds typically take 7 business days",
                    "refunds will be processed within 7 business days",
                ]

                assert not any(
                    claim in response_text
                    for claim in unsupported_refund_claims
                ), (
                    "Refund response contains an unsupported "
                    "processing timeline"
                )

                assert "refund_policy.md" in resolution_data["sources"], (
                    "Refund response must reference refund_policy.md"
                )

                print("Refund policy factual accuracy: PASSED")

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
