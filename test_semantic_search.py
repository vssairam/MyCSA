from semantic_search import search_semantic_kb

# Test complaints covering all five company policies
# and one unrelated question.

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
     None),

    (
        "Unrelated - Weather",
        "What will the weather be tomorrow in Chennai?",
        None
    ),
    (
        "Unrelated - Mathematics",
        "Can you solve a quadratic equation for me?",
        None
    ),
    (
        "Unrelated - Restaurant",
        "Please recommend a good restaurant nearby.",
        None
    ),
    (
        "Unrelated - Programming",
        "How do I create a Python Flask application?",
        None
    ),
    (
        "Unrelated - Travel",
        "What are the best tourist attractions in Paris?",
        None
    ),
]

"""
test_cases = [
    (
        "Billing",
        "I was charged twice for the same order.",
        "billing_policy.md"
    )
]
"""

passed = 0
failed = 0

print("\n" + "=" * 60)
print("MyCSA - FAISS SEMANTIC SEARCH TEST")
print("=" * 60)

for category, complaint, expected_source in test_cases:

    print(f"\nCategory: {category}")
    print(f"Complaint: {complaint}")

    try:
        results = search_semantic_kb(complaint, top_k=3)

        if results:
            print("\nRetrieved Sources:")

            for result in results:
                print(
                    f"  {result['source']} "
                    f"(Similarity: {result['score']:.4f})"
                )

            actual_source = results[0]["source"]

        else:
            print("\nNo relevant policy found.")
            actual_source = None

        if actual_source == expected_source:
            print("Status: PASS")
            passed += 1
        else:
            print("Status: FAIL")
            print(f"Expected: {expected_source}")
            print(f"Actual: {actual_source}")
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