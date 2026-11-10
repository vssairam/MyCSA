import asyncio
import ast
import os
import json


from dotenv import load_dotenv

from idempotency_utils import generate_request_id
from idempotency_registry import get_existing_ticket, register_ticket

from request_status import RequestStatus
from request_status_registry import (
    get_request_status,
    save_request_status,
)

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import (
    FunctionalTermination,
    MaxMessageTermination
)

from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent
from agents.log_agent import create_log_agent
from grounding_validator import validate_grounding
from grounding_state import GroundingState

def ticket_logged(messages):
    """
    Return True only when the Log Agent's tool
    confirms successful ticket creation.
    """

    for message in messages:

        if message.source != "Log_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name not in (
                "log_customer_ticket",
                "log_ticket_with_grounding"
            ):
                continue

            if result.is_error:
                continue

            try:
                log_result = ast.literal_eval(result.content)

                if (
                    isinstance(log_result, dict)
                    and log_result.get("logged") is True
                    and log_result.get("ticket_id")
                ):
                    return True

            except (ValueError, SyntaxError, TypeError):
                continue

    return False

def count_successful_ticket_logs(messages):
    """
    Count successful ticket logging events.
    """

    count = 0

    for message in messages:

        if message.source != "Log_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name not in (
                "log_customer_ticket",
                "log_ticket_with_grounding"
            ):
                continue

            if result.is_error:
                continue

            try:
                log_result = ast.literal_eval(result.content)

                if (
                    isinstance(log_result, dict)
                    and log_result.get("logged") is True
                    and log_result.get("ticket_id")
                ):
                    count += 1

            except (ValueError, SyntaxError, TypeError):
                continue

    return count

def get_logged_ticket_id(messages):
    """
    Extract the ticket ID from a successful Log Agent result.
    """

    for message in messages:

        if message.source != "Log_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name not in (
                "log_customer_ticket",
                "log_ticket_with_grounding"
            ):
                continue

            if result.is_error:
                continue

            try:
                log_result = ast.literal_eval(result.content)

                if (
                    isinstance(log_result, dict)
                    and log_result.get("logged") is True
                    and isinstance(log_result.get("ticket_id"), str)
                    and log_result["ticket_id"].strip()
                ):
                    return log_result["ticket_id"]

            except (ValueError, SyntaxError, TypeError):
                continue

    return None

def select_next_agent(messages):
    """
    Enforce the mandatory four-agent workflow sequence.
    """

    agent_order = [
        "Customer_Query_Agent",
        "Knowledge_Base_Agent",
        "Resolution_Agent",
        "Log_Agent"
    ]

    completed_agents = []

    for message in messages:

        if message.source not in agent_order:
            continue

        # Ignore intermediate tool-call events.
        if type(message).__name__ in [
            "ToolCallRequestEvent",
            "ToolCallExecutionEvent"
        ]:
            continue

        if message.source not in completed_agents:
            completed_agents.append(message.source)

    for agent_name in agent_order:
        if agent_name not in completed_agents:
            return agent_name

    return "Log_Agent"

def knowledge_search_completed(messages):
    """
    Return True when the Knowledge Base Agent has completed
    a valid knowledge search, regardless of whether relevant
    policies were found.
    """

    for message in messages:

        if message.source != "Knowledge_Base_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name not in (
                "search_with_grounding",
                "search_company_knowledge"
            ):
                continue

            if result.is_error:
                continue

            try:
                data = json.loads(result.content)

                if not isinstance(data, dict):
                    continue

                if not isinstance(data.get("found"), bool):
                    continue

                if not isinstance(data.get("sources"), list):
                    continue

                if not isinstance(data.get("matches"), list):
                    continue

                return True

            except (ValueError, TypeError):
                continue

    return False

def count_knowledge_search_attempts(messages):
    """
    Count actual Knowledge Base search tool executions.

    Both successful and failed tool executions count
    toward the retry limit.
    """

    attempts = 0

    for message in messages:

        if message.source != "Knowledge_Base_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name in (
                "search_with_grounding",
                "search_company_knowledge"
            ):
                attempts += 1

    return attempts

def get_eligible_agents(messages):
    """
    Identify eligible agents based on completed workflow stages.

    Used by AutoGen's candidate_func to restrict selection
    to agents that satisfy workflow prerequisites.
    """

    # Never select another agent after successful ticket logging.
    if ticket_logged(messages):
        raise RuntimeError(
            "Agent selection requested after successful "
            "ticket logging. Workflow should have terminated."
        )

    agent_order = [
        "Customer_Query_Agent",
        "Knowledge_Base_Agent",
        "Resolution_Agent",
        "Log_Agent",
    ]

    completed_agents = set()

    for message in messages:

        if message.source not in agent_order:
            continue

        # Ignore intermediate tool execution events.
        if type(message).__name__ in (
            "ToolCallRequestEvent",
            "ToolCallExecutionEvent",
        ):
            continue

        completed_agents.add(message.source)

    for agent_name in agent_order:

        if agent_name not in completed_agents:

            # Do not proceed to resolution until the
            # Knowledge Base search has completed.
            if agent_name == "Resolution_Agent":

                if not knowledge_search_completed(messages):

                    attempts = count_knowledge_search_attempts(messages)

                    # Retry the Knowledge Base search once.
                    if attempts < 2:
                        return ["Knowledge_Base_Agent"]

                    # Both attempts failed.
                    # Continue only through the human-review path.
                    return ["Resolution_Agent"]

                return [agent_name]
            
            return [agent_name]
        
    # All four agents have responded.
    # A successful ticket should already trigger termination.
    if ticket_logged(messages):
        raise RuntimeError(
            "Agent selection requested after successful "
            "ticket logging. Workflow should have terminated."
        )

    # If logging has not succeeded, allow the Log Agent
    # another opportunity to complete the operation.
    return ["Log_Agent"]

def get_safe_customer_response(
    resolution: str,
    grounding_state: GroundingState
) -> str:
    """
    Return a customer-safe response based on knowledge-search status.

    A failed knowledge search must never produce an
    unsupported customer-facing resolution.
    """

    if grounding_state.has_search_failed():
        return (
            "Thank you for contacting The AI Demo Store. "
            "We are currently unable to verify the relevant "
            "company policy for your request. "
            "Your request requires review by our support team."
        )

    if not grounding_state.get_sources():
        return (
            "Thank you for contacting The AI Demo Store. "
            "We could not identify a relevant company policy "
            "for your request. "
            "Our support team will review your case."
        )

    return resolution

async def run_selector_customer_support(
    complaint: str,
    request_id: str = None
):
    load_dotenv()

    # Generate a request ID only for a new request
    if request_id is None:
        request_id = generate_request_id()

    print(f"\nRequest ID: {request_id}")

    # Check whether this request has already been processed
    existing_ticket = get_existing_ticket(request_id)

    if existing_ticket is not None:

        print("\n" + "=" * 60)
        print("MYCSA - IDEMPOTENCY CHECK")
        print("=" * 60)

        print("Request already processed.")
        print(f"Request ID: {request_id}")
        print(f"Existing Ticket ID: {existing_ticket}")

        print("Skipping AutoGen workflow.")

        return {
            "request_id": request_id,
            "ticket_id": existing_ticket,
            "duplicate_request": True
        }

    # Register a new customer request as PENDING
    save_request_status(
        request_id,
        RequestStatus.PENDING
    )

    print(f"Request Status: {RequestStatus.PENDING.value}")

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        # Mark the request as PROCESSING
        save_request_status(
            request_id,
            RequestStatus.PROCESSING
        )

        print(f"Request Status: {RequestStatus.PROCESSING.value}")

        # Create independent grounding evidence for this request.
        grounding_state = GroundingState()

        query_agent = create_query_agent(model_client)

        knowledge_agent = create_knowledge_agent(
            model_client,
            grounding_state=grounding_state
        )

        resolution_agent = create_resolution_agent(model_client)

        log_agent = create_log_agent(
            model_client,
            grounding_state=grounding_state
        )

        team = SelectorGroupChat(
            participants=[
                query_agent,
                knowledge_agent,
                resolution_agent,
                log_agent
            ],
            model_client=model_client,
            candidate_func=get_eligible_agents,
            selector_prompt="""
You are the workflow coordinator for The AI Demo Store.

Select exactly one agent to speak next from:
{participants}

Available agents:
{roles}

Conversation history:
{history}

MANDATORY WORKFLOW:

1. Customer_Query_Agent must classify the complaint first.
2. Knowledge_Base_Agent must retrieve relevant company policies second.
3. Resolution_Agent must generate a policy-grounded response third.
4. Log_Agent must create the customer ticket fourth.

SELECTION RULES:
- Select only from the currently eligible participants.
- Never skip complaint classification.
- Knowledge_Base_Agent must attempt policy retrieval before Resolution_Agent.
- Knowledge_Base_Agent may retry once if its first search fails.
- After two failed knowledge searches, proceed to Resolution_Agent for human-review handling.
- Never select Log_Agent before Resolution_Agent.
- Never repeat Customer_Query_Agent or Resolution_Agent.
- Once Log_Agent has successfully logged the ticket, stop.

Return only the exact agent name.
""",
            termination_condition=(
                FunctionalTermination(ticket_logged)
                | MaxMessageTermination(max_messages=20)
            )
        )

        task = f"""
You are processing a customer support request
for The AI Demo Store.

ORIGINAL CUSTOMER COMPLAINT:
{complaint}

Follow this workflow:

1. Customer_Query_Agent:
   Classify the complaint.

2. Knowledge_Base_Agent:
   Search company policies using the original complaint.

3. Resolution_Agent:
   Generate a policy-grounded response.

4. Log_Agent:
   Log the ticket and report escalation status.

IMPORTANT:
- Do not invent policies.
- Do not skip agents.
- Log the ticket exactly once.
"""

        result = await team.run(task=task)

        # Validate exactly one successful ticket logging event
        successful_logs = count_successful_ticket_logs(
            result.messages
        )

        if successful_logs == 0:
            raise RuntimeError(
                "MyCSA workflow ended without successfully "
                "logging a customer support ticket."
            )

        if successful_logs > 1:
            raise RuntimeError(
                "MyCSA detected duplicate ticket logging! "
                f"Successful logging events: {successful_logs}"
            )

        print(
            "\nTicket logging validation: PASSED "
            "(exactly one successful logging event)"
        )
        
        # NEW CODE STARTS HERE

        # Extract the successfully created ticket ID
        ticket_id = get_logged_ticket_id(result.messages)

        if ticket_id is None:
            raise RuntimeError(
                "MyCSA could not retrieve the logged ticket ID."
            )

        # Register the request-to-ticket mapping
        registered_ticket = register_ticket(
            request_id,
            ticket_id
        )

        print("\nIdempotency registration: PASSED")
        print(f"Request ID: {request_id}")
        print(f"Ticket ID: {registered_ticket}")

        # Mark the request as COMPLETED
        save_request_status(
            request_id,
            RequestStatus.COMPLETED,
            ticket_id=registered_ticket
        )

        print(f"Request Status: {RequestStatus.COMPLETED.value}")

        # NEW CODE ENDS HERE

        print("\n" + "=" * 60)
        print("MYCSA - SELECTOR GROUP CHAT")
        print("=" * 60)

        print(f"\nCustomer Complaint: {complaint}")

        for message in result.messages:

            if message.source == "Resolution_Agent":
                print("\nResolution_Agent:")

                # Display a safe response instead of the original
                # resolution when knowledge evidence is unavailable.
                safe_response = get_safe_customer_response(
                    str(message.content),
                    grounding_state
                )

                print(safe_response)

            else:
                print(f"\n{message.source}:")
                print(message.content)

        return result

    except Exception as error:
        # Record a failed request
        try:
            save_request_status(
                request_id,
                RequestStatus.FAILED
            )

            print(f"Request Status: {RequestStatus.FAILED.value}")

        except Exception as status_error:
            print(f"Unable to save FAILED status: {status_error}")

        print(f"Workflow Error: {error}")

        # Preserve the original error
        raise

    finally:
        await model_client.close()


if __name__ == "__main__":

    complaint = (
        "My order delivery is delayed by three days. "
        "I have not received any updates."
    )

    asyncio.run(
        run_selector_customer_support(complaint)
    )