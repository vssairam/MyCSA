"""
Request-scoped grounding state for MyCSA.

Stores policy filenames returned by the Knowledge Base tool.
Each customer workflow should create its own instance.
"""


class GroundingState:

    def __init__(self):
        self.retrieved_sources = None
        self.search_failed = False

    def set_sources(self, sources: list[str]) -> None:
        """
        Store policy filenames retrieved by the Knowledge Base Agent.
        """
        self.retrieved_sources = list(sources)

    def get_sources(self) -> list[str] | None:
        """
        Return the retrieved policy filenames.
        """
        if self.retrieved_sources is None:
            return None

        return list(self.retrieved_sources)

    def set_search_failed(self, failed: bool) -> None:
        """
        Record whether the Knowledge Base search failed.
        """
        self.search_failed = bool(failed)


    def has_search_failed(self) -> bool:
        """
        Return the Knowledge Base search failure status.
        """
        return self.search_failed

       