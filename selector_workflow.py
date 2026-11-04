import asyncio
import ast
import os


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

            if result.name != "log_customer_ticket":
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

            if result.name != "log_customer_ticket":
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

            if result.name != "log_customer_ticket":
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

        query_agent = create_query_agent(model_client)
        knowledge_agent = create_knowledge_agent(model_client)
        resolution_agent = create_resolution_agent(model_client)
        log_agent = create_log_agent(model_client)

        team = SelectorGroupChat(
            participants=[
                query_agent,
                knowledge_agent,
                resolution_agent,
                log_agent
            ],
            model_client=model_client,
            selector_func=select_next_agent,
            allow_repeated_speaker=False,
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
- Never skip a required agent.
- Never select Log_Agent before Resolution_Agent.
- Never select Resolution_Agent before Knowledge_Base_Agent.
- Do not repeat an agent.
- Once the Log_Agent has logged the ticket, stop.

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