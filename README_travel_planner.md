# Travel Planner Agent

This project contains a LangChain agent that turns a user's travel request into:

1. A practical trip plan
2. A tailored packing checklist

The implementation is in `email_humanizer_agent_v2_copy.py`. The filename is retained from the original exercise, but the agent itself is travel-focused.

## What It Does

The agent uses two tools in a fixed order:

```text
Travel request
      |
      v
create_trip_plan
      |
      v
create_packing_checklist
      |
      v
Trip plan + packing checklist
```

The trip plan includes destination guidance, suggested dates or season, a day-by-day itinerary, transportation, accommodation suggestions, and practical travel notes. The packing checklist is organized by category and considers the destination, weather, duration, and activities.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- The packages listed in `requirements.txt`

## Setup

From the `Langchain_sample_project` directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file from the included example:

```powershell
Copy-Item .env.example .env
```

Add your API key to `.env`:

```text
OPENAI_API_KEY=your-openai-api-key
```

Do not commit `.env` or expose the API key in source control.

## Run

```powershell
python email_humanizer_agent_v2_copy.py
```

The program displays an interactive prompt:

```text
TRIP PLANNER AGENT (LangChain + OpenAI)
Describe your destination, dates, budget, group, and interests when known.

What trip would you like to plan? (type 'quit' to exit)
```

Enter a request such as:

```text
Plan a 5-day trip to Kyoto in October for two adults. We enjoy temples,
local food, and quiet neighborhoods. Our budget is moderate.
```

The agent returns the trip plan and packing checklist. Enter `quit`, `exit`, or `q` to close the program.

## Useful Request Details

The output becomes more useful when the request includes:

- Destination or destinations
- Travel dates or trip duration
- Number of travelers
- Budget level
- Interests and planned activities
- Accommodation preferences
- Mobility or dietary requirements

If details are missing, the agent makes reasonable assumptions and labels them in the trip plan.

## Main Components

- `email_humanizer_agent_v2_copy.py` - Travel planner agent and command-line interface
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template
- `README_travel_planner.md` - This guide

## Notes

- The script uses OpenAI's `gpt-4.1-mini` model.
- Each request makes model calls for the trip plan and the packing checklist.
- Travel information can change. Confirm current entry requirements, transportation details, weather, health guidance, and booking information with authoritative sources before traveling.
