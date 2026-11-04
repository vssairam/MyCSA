import tempfile
from pathlib import Path

import request_status_registry as registry
from request_status import RequestStatus


def run_tests():

    passed = 0
    total = 6

    with tempfile.TemporaryDirectory() as temp_dir:

        # Use a temporary file instead of the real registry
        registry.STATUS_FILE = (
            Path(temp_dir) / "test_request_status.json"
        )

        request_id = "REQ-TEST001"

        # Test 1: Save PENDING status
        result = registry.save_request_status(
            request_id,
            RequestStatus.PENDING
        )

        if result["status"] == "PENDING":
            print("Test 1 - Save PENDING Status: PASS")
            passed += 1
        else:
            print("Test 1 - Save PENDING Status: FAIL")

        # Test 2: Update to PROCESSING
        result = registry.save_request_status(
            request_id,
            RequestStatus.PROCESSING
        )

        if result["status"] == "PROCESSING":
            print("Test 2 - Update PROCESSING Status: PASS")
            passed += 1
        else:
            print("Test 2 - Update PROCESSING Status: FAIL")

        # Test 3: Mark COMPLETED with ticket ID
        result = registry.save_request_status(
            request_id,
            RequestStatus.COMPLETED,
            ticket_id="CSA-TEST001"
        )

        if (
            result["status"] == "COMPLETED"
            and result["ticket_id"] == "CSA-TEST001"
        ):
            print("Test 3 - Save COMPLETED Status: PASS")
            passed += 1
        else:
            print("Test 3 - Save COMPLETED Status: FAIL")

        # Test 4: Save FAILED status
        result = registry.save_request_status(
            "REQ-FAILED",
            RequestStatus.FAILED
        )

        if result["status"] == "FAILED":
            print("Test 4 - Save FAILED Status: PASS")
            passed += 1
        else:
            print("Test 4 - Save FAILED Status: FAIL")

        # Test 5: Retrieve unknown request
        result = registry.get_request_status("REQ-UNKNOWN")

        if result is None:
            print("Test 5 - Unknown Request: PASS")
            passed += 1
        else:
            print("Test 5 - Unknown Request: FAIL")

        # Test 6: Reject invalid status
        try:
            registry.save_request_status(
                "REQ-INVALID",
                "INVALID"
            )
            print("Test 6 - Invalid Status: FAIL")

        except ValueError:
            print("Test 6 - Invalid Status: PASS")
            passed += 1

    print("\n" + "=" * 60)
    print("MYCSA - REQUEST STATUS REGISTRY TEST SUMMARY")
    print("=" * 60)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    print("=" * 60)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()