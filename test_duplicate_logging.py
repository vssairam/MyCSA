from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_core.models import FunctionExecutionResult

from selector_workflow import count_successful_ticket_logs


def make_logging_event(ticket_id=None, logged=True, is_error=False):
    """
    Create a simulated ticket logging event.
    """

    result = {
        "ticket_id": ticket_id,
        "logged": logged
    }

    return ToolCallExecutionEvent(
        source="Log_Agent",
        content=[
            FunctionExecutionResult(
                content=str(result),
                name="log_customer_ticket",
                call_id="test-call",
                is_error=is_error
            )
        ]
    )


def run_test(name, messages, expected):

    actual = count_successful_ticket_logs(messages)

    status = "PASS" if actual == expected else "FAIL"

    print(f"{name}: {status}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")

    return actual == expected


tests = [
    (
        "No Logging",
        [],
        0
    ),
    (
        "Single Successful Logging",
        [
            make_logging_event("CSA-001")
        ],
        1
    ),
    (
        "Duplicate Successful Logging",
        [
            make_logging_event("CSA-001"),
            make_logging_event("CSA-002")
        ],
        2
    ),
    (
        "Three Successful Logs",
        [
            make_logging_event("CSA-001"),
            make_logging_event("CSA-002"),
            make_logging_event("CSA-003")
        ],
        3
    ),
    (
        "Failed Logging",
        [
            make_logging_event(
                ticket_id=None,
                logged=False
            )
        ],
        0
    ),
    (
        "Logging Tool Error",
        [
            make_logging_event(
                ticket_id="CSA-004",
                logged=True,
                is_error=True
            )
        ],
        0
    )
]


print("\n" + "=" * 60)
print("MYCSA - DUPLICATE LOGGING TEST")
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