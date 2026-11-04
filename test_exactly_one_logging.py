from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_core.models import FunctionExecutionResult

from selector_workflow import count_successful_ticket_logs


def make_logging_event(ticket_id):
    """Create a simulated successful ticket logging event."""

    result = {
        "ticket_id": ticket_id,
        "logged": True
    }

    return ToolCallExecutionEvent(
        source="Log_Agent",
        content=[
            FunctionExecutionResult(
                content=str(result),
                name="log_customer_ticket",
                call_id=f"test-{ticket_id}",
                is_error=False
            )
        ]
    )


def validate_exactly_one(messages):
    """Apply the same validation rules as selector_workflow.py."""

    successful_logs = count_successful_ticket_logs(messages)

    if successful_logs == 0:
        raise RuntimeError(
            "MyCSA workflow ended without successfully "
            "logging a customer support ticket."
        )

    if successful_logs > 1:
        raise RuntimeError(
            "MyCSA detected duplicate ticket logging! "
            f"Successful logging events: {successful_logs}"
        )

    return True


tests = [
    ("No Successful Logging", [], False),

    (
        "Exactly One Successful Log",
        [make_logging_event("CSA-001")],
        True
    ),

    (
        "Two Successful Logs",
        [
            make_logging_event("CSA-001"),
            make_logging_event("CSA-002")
        ],
        False
    ),

    (
        "Three Successful Logs",
        [
            make_logging_event("CSA-001"),
            make_logging_event("CSA-002"),
            make_logging_event("CSA-003")
        ],
        False
    )
]


print("\n" + "=" * 60)
print("MYCSA - EXACTLY ONE LOGGING VALIDATION")
print("=" * 60)

passed = 0

for name, messages, expected in tests:

    try:
        actual = validate_exactly_one(messages)
    except RuntimeError:
        actual = False

    status = "PASS" if actual == expected else "FAIL"

    print(f"{name}: {status}")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")

    if actual == expected:
        passed += 1


print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(f"Total Tests: {len(tests)}")
print(f"Passed: {passed}")
print(f"Failed: {len(tests) - passed}")
print("=" * 60)

if passed != len(tests):
    raise SystemExit(1)