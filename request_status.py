from enum import Enum


class RequestStatus(str, Enum):
    """
    Possible processing states for a customer request.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"