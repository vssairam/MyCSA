import tempfile
from pathlib import Path

import idempotency_registry as registry


def run_tests():

    passed = 0
    total = 5

    with tempfile.TemporaryDirectory() as temp_dir:

        # Use a temporary registry instead of the real one
        registry.REGISTRY_FILE = (
            Path(temp_dir) / "test_registry.json"
        )

        request_id = "REQ-TEST001"
        ticket_id = "CSA-TEST001"

        # Test 1: Register a new ticket
        result = registry.register_ticket(
            request_id,
            ticket_id
        )

        if result == ticket_id:
            print("Test 1 - Register New Ticket: PASS")
            passed += 1
        else:
            print("Test 1 - Register New Ticket: FAIL")

        # Test 2: Retrieve existing ticket
        result = registry.get_existing_ticket(request_id)

        if result == ticket_id:
            print("Test 2 - Retrieve Existing Ticket: PASS")
            passed += 1
        else:
            print("Test 2 - Retrieve Existing Ticket: FAIL")

        # Test 3: Unknown request
        result = registry.get_existing_ticket("REQ-UNKNOWN")

        if result is None:
            print("Test 3 - Unknown Request: PASS")
            passed += 1
        else:
            print("Test 3 - Unknown Request: FAIL")

        # Test 4: Register same request and ticket again
        result = registry.register_ticket(
            request_id,
            ticket_id
        )

        if result == ticket_id:
            print("Test 4 - Repeated Registration: PASS")
            passed += 1
        else:
            print("Test 4 - Repeated Registration: FAIL")

        # Test 5: Reject conflicting ticket
        try:
            registry.register_ticket(
                request_id,
                "CSA-DIFFERENT"
            )
            print("Test 5 - Conflicting Registration: FAIL")

        except ValueError:
            print("Test 5 - Conflicting Registration: PASS")
            passed += 1

    print("\n" + "=" * 55)
    print("MYCSA - IDEMPOTENCY REGISTRY TEST SUMMARY")
    print("=" * 55)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print("=" * 55)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()