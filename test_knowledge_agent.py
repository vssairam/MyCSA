import json

from agents.knowledge_agent import search_company_knowledge


test_cases = [
    ("Shipping",
     "My package has not arrived and tracking has not changed.",
     "shipping_policy.md"),

    ("Refund",
     "I returned my product but have not received my refund.",
     "refund_policy.md"),

    ("Billing",
     "I was charged twice for the same order.",
     "billing_policy.md"),

    ("Account",
     "I cannot log into my account because I forgot my password.",
     "account_and_login.md"),

    ("Product",
     "The product I received is damaged and not working.",
     "product_faq.md"),

    ("Unrelated",
     "Can you help me book a flight from Chennai to Mumbai?",
     None)
]


passed = 0
failed = 0

print("\n" + "=" * 60)
print("MyCSA - KNOWLEDGE BASE AGENT TEST")
print("=" * 60)

for category, complaint, expected_source in test_cases:

    print(f"\nCategory: {category}")
    print(f"Complaint: {complaint}")

    try:
        response = search_company_knowledge(complaint)
        data = json.loads(response)

        found = data.get("found", False)
        sources = data.get("sources", [])
        matches = data.get("matches", [])

        print(f"Knowledge Found: {found}")
        print(f"Sources: {sources}")
        print(f"Matches: {len(matches)}")

        if matches:
            for match in matches:
                print(
                    f"  {match['source']} | "
                    f"{match['section']} | "
                    f"Similarity: {match['score']:.4f}"
                )

        if expected_source is None:
            success = not found and not matches and not sources
        else:
            success = (
                found
                and expected_source in sources
                and len(matches) > 0
                and all(
                    match["source"] in sources
                    for match in matches
                )
            )

        if success:
            print("Status: PASS")
            passed += 1
        else:
            print("Status: FAIL")
            failed += 1

    except Exception as error:
        print(f"Status: ERROR - {error}")
        failed += 1


print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Total Tests: {len(test_cases)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("=" * 60)