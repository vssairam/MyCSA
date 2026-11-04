from types import SimpleNamespace

from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_core.models import FunctionExecutionResult

from selector_workflow import ticket_logged


def make_logging_event(content, is_error=False):
    """
    Simulate a Log Agent tool execution event.
    """

    result = FunctionExecutionResult(
        content=content,
        name="log_customer_ticket",
        call_id="test-call-001",
        is_error=is_error
    )

    return ToolCallExecutionEvent(
        source="Log_Agent",
        content=[result]
    )


def run_test(name, messages, expected):

    actual = ticket_logged(messages)

    status = "PASS" if actual == expected else "FAIL"

    print(f"{name}: {status}")

    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")

    return actual == expected


tests = [
    (
        "Successful Logging",
        [
            make_logging_event(
                "{'ticket_id': 'CSA-TEST123', 'logged': True}"
            )
        ],
        True
    ),
    (
        "Failed Logging",
        [
            make_logging_event(
                "{'logged': False}"
            )
        ],
        False
    ),
    (
        "Logging Tool Error",
        [
            make_logging_event(
                "Database connection failed",
                is_error=True
            )
        ],
        False
    ),
    (
        "Missing Ticket ID",
        [
            make_logging_event(
                "{'logged': True}"
            )
        ],
        False
    ),
    (
        "No Logging Event",
        [],
        False
    )
]


print("\n" + "=" * 60)
print("MYCSA - SELECTOR TERMINATION TEST")
print("=" * 60)

passed = 0

for name, messages, expected in tests:

    if run_test(name, messages, expected):
        passed += 1


print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(f"Total Tests: {len(tests)}")
print(f"Passed: {passed}")
print(f"Failed: {len(tests) - passed}")

print("=" * 60)