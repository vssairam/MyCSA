import json
from pathlib import Path

from request_status import RequestStatus


STATUS_FILE = Path("logs/request_status_registry.json")


def load_request_statuses():
    """
    Load the saved request statuses.
    """

    if not STATUS_FILE.exists():
        return {}

    with open(STATUS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_request_status(request_id):
    """
    Retrieve the status information for a request.
    """

    statuses = load_request_statuses()

    return statuses.get(request_id)


def save_request_status(request_id, status, ticket_id=None):
    """
    Save the current processing status of a request.
    """

    if not isinstance(status, RequestStatus):
        raise ValueError("Invalid request status")

    statuses = load_request_statuses()

    statuses[request_id] = {
        "status": status.value,
        "ticket_id": ticket_id
    }

    STATUS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(statuses, file, indent=4)

    return statuses[request_id]