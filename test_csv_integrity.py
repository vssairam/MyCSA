import csv
import json
import math
from datetime import datetime
from pathlib import Path

CSV_FILE = Path("logs/customer_tickets.csv")

REQUIRED_FIELDS = [
    "ticket_id",
    "timestamp",
    "complaint",
    "category",
    "urgency",
    "sentiment",
    "summary",
    "resolution",
    "confidence",
    "kb_found",
    "escalated",
]

errors = []

with CSV_FILE.open("r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

for row_number, row in enumerate(rows, start=2):

    # Check required fields
    for field in REQUIRED_FIELDS:
        if not str(row.get(field) or "").strip():
            errors.append(
                f"Row {row_number}: Missing {field}"
            )

    # Validate confidence
    try:
        confidence = float(row["confidence"])

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            errors.append(
                f"Row {row_number}: Invalid confidence"
            )

    except (ValueError, TypeError):
        errors.append(
            f"Row {row_number}: Confidence is not numeric"
        )

    # Validate boolean fields
    for field in ["kb_found", "escalated"]:
        if str(row.get(field)).lower() not in ["true", "false"]:
            errors.append(
                f"Row {row_number}: Invalid {field}"
            )

    # Validate timestamp
    try:
        datetime.fromisoformat(row["timestamp"])

    except (ValueError, TypeError):
        errors.append(
            f"Row {row_number}: Invalid timestamp"
        )

    # Validate JSON fields
    for field in ["sources", "escalation_reasons"]:
        try:
            value = json.loads(row[field])

            if not isinstance(value, list):
                errors.append(
                    f"Row {row_number}: {field} must be a list"
                )

        except (ValueError, TypeError):
            errors.append(
                f"Row {row_number}: Invalid JSON in {field}"
            )

print("=" * 55)
print("MYCSA - CSV DATA INTEGRITY VALIDATION")
print("=" * 55)

print(f"Total tickets checked: {len(rows)}")
print(f"Integrity issues found: {len(errors)}")

if errors:
    print("\nISSUES:")
    for error in errors:
        print(f" - {error}")

    print("\nRESULT: FAILED")

else:
    print("\nRESULT: PASSED")

print("=" * 55)