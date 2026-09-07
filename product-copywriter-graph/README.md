# Product Copywriter Graph - LangGraph Parallel + Conditional Routing Project

A project that teaches **fan-out / fan-in parallel execution and conditional
routing** in LangGraph. The graph takes a product name and its features,
runs 3 specialist analyses at the same time, uses a decision node to pick a
tone, then routes to exactly one of 2 final writer nodes.

## What You'll Learn

- How to design a graph's `State` **before** writing any node code (state ->
  nodes -> edges)
- How to **fan out** - run several independent nodes in parallel by giving
  them all the same `START` edge
- How to **fan in** - make a node wait for several parallel nodes to finish
  by pointing all of their edges into it
- How `add_conditional_edges` works: a source node, a plain decision
  function (passed without `()`), and a dictionary mapping the function's
  returned string to the next node
- Why a decision function returns a **plain string**, not a dict, while
  every node function returns a **dict** of state updates
- How to give a graph 2 separate final output nodes instead of 1 node with
  an if/else inside it

## How It Works

```
                         product_name + product_features
                                      |
              ------------------------------------------------
              |                      |                        |
              v                      v                        v
   [write_feature_summary]  [write_benefit_summary]  [identify_target_customer]
              |                      |                        |
              ------------------------------------------------
                                      |
                                      v
                              [decide_tone]
                 (reads all 3 summaries, replies "formal"
                          or "marketing")
                                      |
                        ---------------------------
                        |                          |
                        v                          v
        [formal_product_description]   [marketing_product_copy]
                        |                          |
                        ---------------------------
                                      |
                                      v
                              final_copy returned
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
python Product_Copywriter_Graph.py
```

You'll see two prompts:

```
What is the product name?
What are the product features?
```

Type a product name (e.g., `EcoBottle`) and its features (e.g.,
`insulated, BPA-free, keeps drinks cold for 24 hours`). The graph runs the
3 specialists in parallel, decides on a tone, and prints the final copy.

## Example

**Input:**
```
What is the product name? SunTime
What are the product features? analog clock face, built-in wristwatch strap, solar powered
```

**Output (marketing tone was chosen):**
```
Meet SunTime - the clock that never asks for a battery. Wear it on your
wrist or set it on your desk, and let the sun keep you on time, every
single day...
```

## Project Structure

```
.
├── Product_Copywriter_Graph.py   # Main graph code
├── requirements.txt               # Python dependencies
├── .env.example                    # API key template
├── .gitignore                       # Keeps secrets and venv out of git
└── README.md                         # This file
```

## Tech Stack

- [LangGraph](https://langchain-ai.github.io/langgraph/) - Framework for building stateful, multi-step graphs
- [LangChain](https://python.langchain.com/) - Used for the `ChatOpenAI` model wrapper
- [OpenAI GPT-4o-mini](https://platform.openai.com/) - The LLM powering every node
- [Pydantic](https://docs.pydantic.dev/) - Defines and validates the graph's `State`
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management
