import asyncio
import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Step 1: Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# Step 2: Create a simple AutoGen agent
async def main():

    if not API_KEY or API_KEY.startswith("your_"):
        print("ERROR: Configure OPENAI_API_KEY in your .env file.")
        return

    print("======================================")
    print(" CSA - AutoGen API Connectivity Test")
    print("======================================")

    print(f"Model: {MODEL_NAME}")
    print("Connecting to OpenAI API...\n")

    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=API_KEY
    )

    try:
        # Create AutoGen Assistant Agent
        test_agent = AssistantAgent(
            name="CSA_Test_Agent",
            model_client=model_client,
            system_message=(
                "You are a helpful customer support assistant. "
                "Respond clearly and professionally."
            )
        )

        # Send a test message
        result = await test_agent.run(
            task="Say hello and confirm that the CSA AI Agent is working."
        )

        print("Agent Response:")
        print("------------------------------")
        print(result.messages[-1].content)

        print("\nSUCCESS: AutoGen API connection is working!")

    except Exception as error:
        print("\nERROR: AutoGen API test failed.")
        print(f"Details: {error}")

    finally:
        await model_client.close()


# Step 3: Execute the program
if __name__ == "__main__":
    asyncio.run(main())