# Debate Coach Agent - LangChain Single Agent Project

A project that builds on the single-agent pattern to teach **multi-step tool
chaining with a repair loop**. The agent takes a debate motion and an assigned
side, builds the strongest case for it, then attacks its own case from the
opposing side and writes ready-to-use replies.

## What You'll Learn

- How an agent chains **two tools in a fixed order**, passing one tool's full
  output into the next tool's input
- How `PromptTemplate` with `STRICT FORMAT` instructions controls exactly how
  the LLM structures its output
- How to detect and repair incomplete LLM output with a small, targeted
  follow-up call, instead of accepting a broken result
- How `re` (regular expressions) can find, extract, and reformat specific
  pieces of AI-generated text
- How the agent's tool-calling loop works (think -> act -> observe -> repeat)

## How It Works

```
User's motion + assigned side
       |
       v
  [Agent thinks: "I need to build the strongest case first"]
       |
       v
  [Tool: build_arguments] --> 4-5 numbered arguments, each with
                               Reasoning + Example
       |
       v
  [Agent thinks: "Now I need to attack this case and prepare replies"]
       |
       v
  [Tool: prepare_rebuttals] --> 5-6 predicted attacks, each with
                                 They say / You answer
       |
       v
  [format_debate_output] --> checks every argument for a missing
                              Reasoning line; if one is missing, asks
                              the LLM a small follow-up question and
                              inserts the real answer; adds spacing
                              before every sub-heading
       |
       v
  Final coaching brief returned to user
```

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rohanjs94-dev/my-agentic-ai-projects.git
cd my-agentic-ai-projects
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and add your real key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Run

```bash
python Debate_Coach_Agent.py
```

You'll see an interactive prompt:

```
DEBATE COACH AGENT (LangChain + OpenAI)
Provide a motion and the side you must argue. Type 'quit' to exit.

Motion:
```

Type your motion (e.g., `Remote work should be permanent for all office jobs`),
then the side you've been assigned (`for` or `against`). The agent will build
your case, attack it, and hand you both.

The prompt then loops — you can prepare as many motions as you like in one
session. Type `quit`, `exit`, or `q` at either prompt to end the program.

## Example

**Input:**
```
Motion: Remote work should be permanent for all office jobs
Your side (for/against): against
```

**Output:**
```
============================================================
1. Remote work weakens team cohesion and company culture.

   Reasoning: Spontaneous collaboration and informal mentorship rely
   heavily on in-person presence, and losing that erodes shared identity
   over time.

   Example: A 2023 Microsoft study found hybrid teams reported a 25% drop
   in cross-team collaboration compared to fully in-office teams.

...

Opponent Attacks

1. ATTACKS ARGUMENT 1

   They say: Culture is built through shared purpose, not proximity -
   plenty of remote-first companies have strong cultures.

   You answer: Purpose alone doesn't replace the spontaneous
   problem-solving and trust-building that happens when teammates share
   a room every day.
============================================================
```

## Project Structure

```
.
├── Debate_Coach_Agent.py      # Main agent code
├── requirements.txt           # Python dependencies
├── .env.example                # API key template
├── .gitignore                  # Keeps secrets and venv out of git
└── README.md                   # This file
```

## Tech Stack

- [LangChain](https://python.langchain.com/) - Framework for building LLM applications
- [OpenAI GPT-4.1-mini](https://platform.openai.com/) - The LLM powering the agent
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management
- `re` (Python standard library) - Parsing and repairing the agent's raw output
