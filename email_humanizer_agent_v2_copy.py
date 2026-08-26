"""Travel Planner Agent - a LangChain agent that creates a trip plan and checklist.

Setup: pip install -r requirements.txt, copy .env.example to .env, add your key.
Run:   python email_humanizer_agent.py
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("email_humanizer")

load_dotenv()
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

CREATE_TRIP_PLAN_PROMPT = PromptTemplate(
    input_variables=["trip_request"],
    template="""You are an experienced travel planner.
Create a practical, well-structured trip plan from the travel request below.

Travel request:
{trip_request}

Include:
- Destination and the best season or suggested travel dates
- A brief overview of the trip
- A day-by-day itinerary with realistic activities
- Local transportation suggestions
- Accommodation guidance appropriate to the request
- Important practical notes, such as weather, safety, or entry requirements

If important information is missing, make reasonable assumptions and label them clearly.
Use clear headings and concise bullet points. Return only the trip plan.""",
)

CREATE_PACKING_CHECKLIST_PROMPT = PromptTemplate(
    input_variables=["trip_plan"],
    template="""You are an expert travel organizer.
Create a complete, categorized packing checklist based on the trip plan below.

Rules:
- Tailor clothing to the destination, weather, duration, and activities.
- Include useful quantities for clothing, toiletries, documents, electronics, and activity gear.
- Consider activities such as hiking, trekking, swimming, or rain exposure when relevant.
- Mark optional items as optional and avoid recommending unnecessary purchases.

Trip plan:
{trip_plan}

Return only the categorized checklist, with no extra commentary.""",
)


@tool
def create_trip_plan(trip_request: str) -> str:
    """Create a structured trip plan from a user's travel request. Use this first."""
    logger.info("Creating trip plan for: %s", trip_request)
    return llm.invoke(CREATE_TRIP_PLAN_PROMPT.format(trip_request=trip_request)).content


@tool
def create_packing_checklist(trip_plan: str) -> str:
    """Create a categorized packing checklist from a completed trip plan."""
    logger.info("Creating packing checklist")
    return llm.invoke(CREATE_PACKING_CHECKLIST_PROMPT.format(trip_plan=trip_plan)).content


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful travel-planning assistant.
Your job is to turn a user's travel request into a practical trip plan and a tailored packing checklist.

Always use both tools in this order:
1. Use create_trip_plan to produce the itinerary and practical travel details.
2. Pass the complete result to create_packing_checklist.
3. Return both the trip plan and the packing checklist with clear headings.

Do not skip either tool, even when the user's request is brief. Make reasonable assumptions when details
are missing, and keep those assumptions clear in the final response."""

agent = create_agent(
    model=llm,
    tools=[create_trip_plan, create_packing_checklist],
    system_prompt=SYSTEM_PROMPT,
)


def plan_trip(trip_request: str) -> str:
    """Run the travel-planning agent and return the final trip plan and checklist."""
    result = agent.invoke({"messages": [HumanMessage(content=trip_request)]})
    return result["messages"][-1].content


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    print("\nTRIP PLANNER AGENT (LangChain + OpenAI)")
    print("Describe your destination, dates, budget, group, and interests when known.\n")

    while True:
        trip_request = input("What trip would you like to plan? (type 'quit' to exit) ").strip()
        if not trip_request:
            continue
        if trip_request.lower() in ("quit", "exit", "q"):
            break

        try:
            trip_plan = plan_trip(trip_request)
            print("\n" + "=" * 60)
            print(trip_plan)
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error("Agent failed: %s", e)


if __name__ == "__main__":
    main()
