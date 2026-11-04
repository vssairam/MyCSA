import asyncio
import os

from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
    FunctionalTermination
)

from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent
from agents.log_agent import create_log_agent

def ticket_logged(messages):
    """
    Stop the workflow when the Log Agent's logging
    tool confirms that a ticket was saved.
    """

    for message in messages:

        if message.source != "Log_Agent":
            continue

        if type(message).__name__ != "ToolCallExecutionEvent":
            continue

        for result in message.content:

            if result.name == "log_customer_ticket":

                if (
                    not result.is_error
                    and "'logged': True" in result.content
                ):
                    return True

    return False

async def run_customer_support(complaint: str):
    """
    Run the four-agent customer support workflow.
    """

    load_dotenv()

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        # Create all four agents
        query_agent = create_query_agent(model_client)
        knowledge_agent = create_knowledge_agent(model_client)
        resolution_agent = create_resolution_agent(model_client)
        log_agent = create_log_agent(model_client)

        # Create the multi-agent team
        # Create the multi-agent team
        team = RoundRobinGroupChat(
            participants=[
                query_agent,
                knowledge_agent,
                resolution_agent,
                log_agent
            ],
           termination_condition=(
                FunctionalTermination(ticket_logged)
                | MaxMessageTermination(max_messages=20)
            )
        )

        # Instructions shared with the team
        task = f"""
You are processing a customer support request
for The AI Demo Store.

ORIGINAL CUSTOMER COMPLAINT:
{complaint}

Follow this workflow in the specified order:

1. Customer_Query_Agent:
   Classify the complaint.
   Return category, urgency, sentiment and summary.

2. Knowledge_Base_Agent:
   Search company policies using the original complaint.
   Report relevant policy sections and source filenames.

3. Resolution_Agent:
   Generate a professional customer response.
   Use only the retrieved company policies.
   Return JSON containing response, confidence,
   sources, requires_human_review and review_reason.

4. Log_Agent:
   Call log_customer_ticket using the complaint,
   classification, knowledge findings and resolution.
   Report the ticket ID and escalation status.

IMPORTANT:
- Do not invent company policies.
- Do not skip any agent.
- The Log Agent must execute its logging tool.
- Do not repeat the workflow.
"""

        print("=" * 60)
        print("MYCSA - MULTI-AGENT CUSTOMER SUPPORT")
        print("=" * 60)

        print("\nCustomer Complaint:")
        print(complaint)

        result = await team.run(task=task)

        print("\nMULTI-AGENT EXECUTION")
        print("-" * 60)

        for message in result.messages:
            print(f"\n{message.source}:")
            print(message.content)

        return result

    finally:
        await model_client.close()


if __name__ == "__main__":

    customer_complaint = (
        "My order delivery is delayed by three days. "
        "I am frustrated because I have not received any updates."
    )

    asyncio.run(
        run_customer_support(customer_complaint)
    )