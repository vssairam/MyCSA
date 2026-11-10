import json
from unittest.mock import MagicMock, patch

from grounding_state import GroundingState
from agents.knowledge_agent import create_knowledge_agent


def test_knowledge_search_failure_tracking():

    test_cases = [
        (
            "Relevant policy found",
            json.dumps({
                "found": True,
                "sources": ["shipping_policy.md"],
                "matches": [{"source": "shipping_policy.md"}]
            }),
            False,
            ["shipping_policy.md"]
        ),
        (
            "No relevant policy found",
            json.dumps({
                "found": False,
                "sources": [],
                "matches": []
            }),
            False,
            []
        ),
        (
            "Search raises exception",
            RuntimeError("Search unavailable"),
            True,
            []
        ),
        (
            "Invalid JSON returned",
            "Invalid JSON",
            True,
            []
        )
    ]

    print("\n" + "=" * 60)
    print("MYCSA - KNOWLEDGE SEARCH FAILURE TRACKING TEST")
    print("=" * 60)

    for name, search_result, expected_failed, expected_sources in test_cases:

        state = GroundingState()

        mock_model = MagicMock()
        mock_model.model_info = {
            "function_calling": True,
            "json_output": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        }

        with patch(
            "agents.knowledge_agent.search_company_knowledge"
        ) as mock_search:

            if isinstance(search_result, Exception):
                mock_search.side_effect = search_result
            else:
                mock_search.return_value = search_result

            agent = create_knowledge_agent(
                model_client=mock_model,
                grounding_state=state
            )

            tool = agent._tools[0]
            tool_function = tool._func

            try:
                tool_function("test query")
            except (RuntimeError, ValueError, json.JSONDecodeError):
                pass

        actual_failed = state.has_search_failed()
        actual_sources = state.get_sources()

        passed = (
            actual_failed == expected_failed
            and actual_sources == expected_sources
        )

        print(f"{name}: {'PASS' if passed else 'FAIL'}")
        print(f"  Expected failure: {expected_failed}")
        print(f"  Actual failure:   {actual_failed}")
        print(f"  Expected sources: {expected_sources}")
        print(f"  Actual sources:   {actual_sources}")

        assert passed, f"Test failed: {name}"

    print("\nAll 4 knowledge search failure tracking tests passed.")


if __name__ == "__main__":
    test_knowledge_search_failure_tracking()