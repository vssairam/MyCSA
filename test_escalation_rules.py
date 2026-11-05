from agents.log_agent import determine_escalation


def run_test(name, inputs, expected_escalated, expected_reason=None):

    result = determine_escalation(**inputs)

    assert result["escalated"] == expected_escalated, (
        f"{name}: Expected escalation={expected_escalated}, "
        f"got {result['escalated']}"
    )

    if expected_reason:
        assert expected_reason in result["escalation_reasons"], (
            f"{name}: Missing reason '{expected_reason}'"
        )

    print(f"PASS: {name}")


def main():

    base = {
        "urgency": "Low",
        "sentiment": "Neutral",
        "confidence": 0.95,
        "kb_found": True,
        "requires_human_review": False,
        "complaint": "Where is my order?"
    }

    scenarios = [
        (
            "Normal complaint",
            {},
            False,
            None
        ),
        (
            "High urgency",
            {"urgency": "High"},
            True,
            "High urgency"
        ),
        (
            "Angry customer",
            {"sentiment": "Angry"},
            True,
            "Angry customer"
        ),
        (
            "Missing knowledge",
            {"kb_found": False},
            True,
            "No relevant knowledge base policy"
        ),
        (
            "Low confidence",
            {"confidence": 0.50},
            True,
            "Low resolution confidence"
        ),
        (
            "Human review requested",
            {"requires_human_review": True},
            True,
            "Resolution Agent requested human review"
        ),
        (
            "Payment dispute",
            {"complaint": "I have a payment dispute."},
            True,
            "Sensitive billing or legal complaint"
        ),
        (
            "Legal notice",
            {"complaint": "I received a legal notice."},
            True,
            "Sensitive billing or legal complaint"
        ),
        (
            "Confidence boundary - 0.70",
            {"confidence": 0.70},
            False,
            None
        ),
        (
            "Confidence below boundary",
            {"confidence": 0.69},
            True,
            "Low resolution confidence"
        ),
        (
            "String False - human review",
            {"requires_human_review": "False"},
            False,
            None
        ),
        (
            "Invalid confidence - NaN",
            {"confidence": float("nan")},
            True,
            "Low resolution confidence"
        ),
        (
            "Duplicate charge complaint",
            {"complaint": "I was charged twice for my order."},
            True,
            "Sensitive billing or legal complaint"
        )
    ]

    passed = 0
    failed = 0

    for name, overrides, expected, reason in scenarios:

        inputs = {**base, **overrides}

        try:
            run_test(name, inputs, expected, reason)
            passed += 1

        except AssertionError as error:
            print(f"FAIL: {error}")
            failed += 1

    print("\n" + "=" * 50)
    print("ESCALATION UNIT TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()