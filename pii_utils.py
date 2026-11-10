"""
PII Masking Utilities for MyCSA.

Masks:
1. Payment card numbers (13–19 digits, validated using Luhn).
2. Indian mobile numbers (10 digits, optionally prefixed with +91).

Preserves the last four digits for identification.
"""

import re


def is_valid_card_number(number: str) -> bool:
    """Validate a payment card number using the Luhn algorithm."""

    digits = re.sub(r"\D", "", number)

    if not 13 <= len(digits) <= 19:
        return False

    total = 0
    reverse_digits = digits[::-1]

    for index, digit in enumerate(reverse_digits):
        value = int(digit)

        if index % 2 == 1:
            value *= 2

            if value > 9:
                value -= 9

        total += value

    return total % 10 == 0


def mask_card_number(match: re.Match) -> str:
    """Mask a valid card number, retaining its last four digits."""

    original = match.group()
    digits = re.sub(r"\D", "", original)

    if not is_valid_card_number(digits):
        return original

    return "*" * (len(digits) - 4) + digits[-4:]


def mask_phone_number(match: re.Match) -> str:
    """Mask a phone number, retaining its last four digits."""

    original = match.group()
    digits = re.sub(r"\D", "", original)

    return "*" * (len(digits) - 4) + digits[-4:]


def mask_pii(text: str) -> str:
    """
    Mask payment card and Indian mobile numbers in text.

    Returns the original text if no supported PII is detected.
    """

    if not isinstance(text, str):
        return text

    # Match possible payment card numbers.
    card_pattern = (
        r"(?<!\d)"
        r"(?:\d[ -]?){12,18}\d"
        r"(?!\d)"
    )

    text = re.sub(
        card_pattern,
        mask_card_number,
        text
    )

    # Match Indian mobile numbers, with optional country code.
    phone_pattern = (
        r"(?<![\d+])"
        r"(?:\+91[\s-]?)?"
        r"[6-9]\d{4}[\s-]?\d{5}"
        r"(?!\d)"
    )

    text = re.sub(
        phone_pattern,
        mask_phone_number,
        text
    )

    return text