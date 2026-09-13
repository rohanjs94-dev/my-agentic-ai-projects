# BrightSmile Dental Clinic — Guardrailed Agent

A LangGraph-based defense-in-depth wrapper around **Mia**, an intentionally
insecure dental-clinic appointment chatbot. Mia's system prompt is left
completely unmodified (that's the assignment rule); all protection against
prompt injection and data leakage lives in guard **nodes** placed around her
in the graph.

## Why this exists

Mia's baseline system prompt (`system_prompt.txt`) has access to a full
clinic dataset — patient records, employee HR data, salaries, passwords,
API tokens — with no instruction to withhold any of it. The goal of this
project is to prove that a chatbot with a leaky prompt can still be made
safe **without touching the prompt**, by sandwiching it between input and
output guardrails.

## Architecture

```
START
  |
  v
input_guard_regex   -- 3-way: block / allow (fast-track) / review
  |            |                  |
  v            v                  v
refuse       agent          input_guard_llm  -- allow / block
                                  |     |
                                  v     v
                                agent  refuse
                                  |
                                  v
                            output_guard      -- exact-match secrets,
                                  |               shape/format detectors,
                                  v               LLM leak check
                                 END
```

All guard decisions and reasons are carried in the shared `GuardState` object
that flows through every node, so a caller (or the Streamlit UI) can inspect
*why* something was blocked or allowed.

## Files

- [dental_guardrail_agent.py](dental_guardrail_agent.py) — the guarded LangGraph pipeline + CLI
- [system_prompt.txt](system_prompt.txt) — Mia's unmodified system prompt
- [streamlit_app.py](streamlit_app.py) — chat UI over the same graph
- [run_tests.py](run_tests.py) — batch test runner
- [test_prompts.csv](test_prompts.csv) — test cases (prompt, expected)
- [test_results.csv](test_results.csv) — latest test run output
- [RESULTS.md](RESULTS.md) — results write-up
- [requirements.txt](requirements.txt) — Python dependencies
- [.env.example](.env.example) — env file template

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy the env template and add your OpenAI API key:

   ```bash
   copy .env.example .env
   ```

   Then edit `.env` and set:

   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

## Usage

### CLI

```bash
python dental_guardrail_agent.py
```

Type messages at the `You:` prompt; type `quit`, `exit`, or `q` to leave.

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Opens a chat interface with sidebar buttons for representative
"should work" and "should be blocked" prompts, plus a toggle to reveal each
guard's decision and reasoning for a given turn.

### Batch tests

```bash
python run_tests.py
```

Runs every prompt in `test_prompts.csv` through the compiled graph, prints a
PASS/FAIL summary (with full detail for failures), and writes per-prompt
results to `test_results.csv`.

## Results

- **Attack prompts blocked:** 70/70 (100%), stable across runs.
- **Legitimate prompts allowed:** ~28–30/31 (~90–97%); the small variance
  comes from gpt-4o-mini's non-determinism on "find me an available slot"
  questions, not from any guard weakness — see [RESULTS.md](RESULTS.md) for
  the full explanation.
- **System prompt modified?** No — `system_prompt.txt` is unchanged from the
  assignment spec; all defense is external to it.

See [RESULTS.md](RESULTS.md) for the full breakdown and rationale.
