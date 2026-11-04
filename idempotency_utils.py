import uuid


def generate_request_id():
    """
    Generate a unique ID for a new customer request.
    """
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


def is_same_request(request_id_1, request_id_2):
    """
    Check whether two request IDs are identical.
    """
    return request_id_1 == request_id_2