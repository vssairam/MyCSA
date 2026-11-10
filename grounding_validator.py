"""
Grounding Validator for MyCSA.

Validates whether policy sources cited by the Resolution Agent
were actually retrieved by the Knowledge Base Agent.

This checks source consistency, not factual correctness.
"""


def validate_grounding(
    retrieved_sources: list[str],
    cited_sources: list[str]
) -> dict:
    """
    Compare retrieved policy filenames against cited filenames.

    Returns validation status and unsupported citations.
    """

    retrieved = {
        source.strip()
        for source in retrieved_sources
        if isinstance(source, str) and source.strip()
    }

    cited = {
        source.strip()
        for source in cited_sources
        if isinstance(source, str) and source.strip()
    }

    unsupported_sources = sorted(cited - retrieved)

    is_valid = len(unsupported_sources) == 0

    return {
        "valid": is_valid,
        "retrieved_sources": sorted(retrieved),
        "cited_sources": sorted(cited),
        "unsupported_sources": unsupported_sources,
        "requires_human_review": not is_valid
    }