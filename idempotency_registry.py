import json
from pathlib import Path

REGISTRY_FILE = Path("logs/idempotency_registry.json")


def load_registry():
    """
    Load existing request-to-ticket mappings.
    """

    if not REGISTRY_FILE.exists():
        return {}

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_existing_ticket(request_id):
    """
    Return the ticket ID if the request was processed before.
    """

    registry = load_registry()

    return registry.get(request_id)


def register_ticket(request_id, ticket_id):
    """
    Store the relationship between request ID and ticket ID.

    Reject conflicting mappings for an existing request ID.
    """

    registry = load_registry()

    existing_ticket = registry.get(request_id)

    if existing_ticket is not None:
        if existing_ticket != ticket_id:
            raise ValueError(
                f"Request {request_id} is already associated "
                f"with ticket {existing_ticket}"
            )

        return existing_ticket

    registry[request_id] = ticket_id

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REGISTRY_FILE, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    return ticket_id