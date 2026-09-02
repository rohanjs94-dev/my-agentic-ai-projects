import logging
import os
import re
import sys
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Debate_Coach_Agent")

load_dotenv()
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

create_debate_coach_prompt = PromptTemplate(
    input_variables=["debate_topic", "which_side"],
    template="""You are a skilled debate coach.
Provide guidance and strategies for the debate on the topic below.

Motion: {debate_topic}
Side to argue: {which_side}

The side is assigned, not chosen. Build that case only — never both sides, never hedge.

Give the 4-5 strongest arguments, ordered so argument 1 is the one you'd open with.

STRICT FORMAT — no other text is allowed anywhere in a numbered block:
1. [claim — one short sentence, no preamble]
   Reasoning: [exactly 1-2 sentences, no filler words, straight to the point]
   Example: [one short, concrete example — no extra commentary]

The claim line must be followed IMMEDIATELY by "Reasoning:" on the next line —
never write any unlabeled explanation between the claim and "Reasoning:".
Do not skip Reasoning or Example for any argument.
Do not add any text outside this exact 3-line structure per argument.

Use a statistic or named study only if you're sure it's real; otherwise use an
everyday illustration. A fabricated number loses the round.

Return only the numbered arguments.""",
)

create_counterargument_prompt = PromptTemplate(
    input_variables=["debate_topic", "which_side", "arguments"],
    template="""You are a championship debate coach preparing a student for attack.

Motion: {debate_topic}
Your student is arguing: {which_side}

Their case:
{arguments}

Become the strongest debater on the opposing side and find the sharpest attacks
on this case. Do not go easy — a weak attack leaves your student unprepared.
Then switch back to coach and write their reply to each.

Give 5-6 attacks. Most should target a specific argument by number; at least one
should attack the case as a whole.

STRICT FORMAT — no other text allowed:
1. ATTACKS ARGUMENT 2   (or: ATTACKS THE WHOLE CASE)
   They say: [the attack, in the opponent's voice — 1 sentence, sharp]
   You answer: [the reply — 1-2 sentences, short enough to say out loud under pressure]

Return only the numbered list — nothing before or after it.
Do not include an Example line in the counter-arguments.""",
)


@tool
def build_arguments(debate_topic: str, which_side: str) -> str:
    """Build the 4-5 strongest arguments for the assigned side of a debate motion.
    Call this FIRST, before anything else.

    Args:
        debate_topic: The debate motion.
        which_side: The side the student must argue — "for" or "against".
    """
    logger.info("[build_arguments] topic=%s | side=%s", debate_topic, which_side)
    prompt = create_debate_coach_prompt.format(
        debate_topic=debate_topic,
        which_side=which_side,
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


@tool
def prepare_rebuttals(debate_topic: str, which_side: str, arguments: str) -> str:
    """Predict the opponent's attacks and write the student's reply to each.
    Call this SECOND, after build_arguments.

    Args:
        debate_topic: The debate motion, worded exactly as the user gave it.
        which_side: The side the student is arguing.
        arguments: The FULL numbered list returned by build_arguments. Paste it
            in exactly — do not summarise. The replies must defend these specific
            numbered arguments.
    """
    logger.info("[prepare_rebuttals] received %d chars of arguments", len(arguments))
    prompt = create_counterargument_prompt.format(
        debate_topic=debate_topic,
        which_side=which_side,
        arguments=arguments,
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

system_message = """You are a championship debate coach. You prepare students to
argue any assigned side persuasively and stay unshaken under fire.

When the user gives you a motion and a side:
1. Call build_arguments to build their strongest arguments.
2. Call prepare_rebuttals, passing the motion, the side, and the FULL
   argument list from step 1 — not a summary.
3. Show the student both: their case first, then the attacks and replies.

The side is assigned, never chosen. Never give balanced both-sides answers.
Always use both tools in order. Never answer from memory.
For the main case, include a separate Example line for each numbered point. Do not skip examples in later points.
For the counter-arguments, use only the They say / You answer format and do not add Example lines."""

agent = create_agent(
    model=llm,
    tools=[build_arguments, prepare_rebuttals],
    system_prompt=system_message,
)


def get_missing_reasoning(claim: str) -> str:
    """Ask the AI for a short reasoning, when the main response skipped it."""
    prompt = f"In 2-3 sentences, explain why this claim is true and why it matters:\n{claim}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def format_debate_output(raw_output: str) -> str:
    """Ensure the numbered debate case has a separate Reasoning/Example line for
    every point, with a blank line before each sub-heading for readability.
    Also spaces out the counter-argument blocks the same way."""
    text = raw_output.strip()

    if "Opponent Attacks" in text or "Predicted Attacks" in text or "Rebuttals" in text:
        if "Opponent Attacks" in text:
            split_marker = "Opponent Attacks"
        elif "Predicted Attacks" in text:
            split_marker = "Predicted Attacks"
        else:
            split_marker = "Rebuttals"
        case_text, tail_text = text.split(split_marker, 1)
        case_text = case_text.strip()
        tail_text = split_marker + tail_text
    else:
        case_text = text
        tail_text = ""

    def normalize_case_block(block: str) -> str:
        block = block.strip()
        if not block:
            return block

        if "Reasoning:" not in block:
            # AI skipped Reasoning entirely -> ask for it, then insert BOTH
            # Reasoning and Example with a blank line before each
            claim_line = block.split("\n")[0]
            real_reasoning = get_missing_reasoning(claim_line)
            block = re.sub(
                r"\n\s*Example:",
                lambda match: f"\n\n   Reasoning: {real_reasoning}\n\n   Example:",
                block,
            )
        else:
            # Reasoning already present -> just add a blank line before it
            block = re.sub(r"\n\s*Reasoning:", r"\n\n   Reasoning:", block, count=1)

        if "Example:" in block:
            block = re.sub(
                r"(\n\s*)(Example:\s*)", r"\n\n   Example:", block, flags=re.DOTALL, count=1
            )
        else:
            block = f"{block}\n   "

        return block

    def normalize_counter_block(block: str) -> str:
        """Same spacing idea, but for 'They say:' / 'You answer:' blocks."""
        block = block.strip()
        if not block:
            return block
        block = re.sub(r"\n\s*They say:", r"\n\n   They say:", block, count=1)
        block = re.sub(r"\n\s*You answer:", r"\n\n   You answer:", block, count=1)
        return block

    numbered_blocks = re.findall(r"(?ms)(^\d+\..*?)(?=(?:\n\s*\d+\.)|\Z)", case_text)
    fixed_case = "\n\n".join(normalize_case_block(block) for block in numbered_blocks if block.strip())

    if tail_text:
        header_match = re.search(r"(?m)^\d+\.", tail_text)          # find where the first numbered attack starts
        if header_match:
            header = tail_text[: header_match.start()].strip()        # everything before it, e.g. "Opponent Attacks"
            body = tail_text[header_match.start():]                    # the numbered attacks themselves
            counter_blocks = re.findall(r"(?ms)(^\d+\..*?)(?=(?:\n\s*\d+\.)|\Z)", body)
            fixed_counter = "\n\n".join(
                normalize_counter_block(b) for b in counter_blocks if b.strip()
            )
            return fixed_case + "\n\n" + header + "\n\n" + fixed_counter
        return fixed_case + "\n\n" + tail_text.strip()
    return fixed_case


def run_debate_coach(debate_topic: str, which_side: str) -> str:
    """Run the agent on a motion and side, and return the full coaching brief."""
    user_message = (
        f"Motion: {debate_topic}\n"
        f"I have been assigned to argue: {which_side}\n\n"
        f"Prepare me."
    )
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    formatted_output = format_debate_output(result["messages"][-1].content)
    return formatted_output


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    print("\nDEBATE COACH AGENT (LangChain + OpenAI)")
    print("Provide a motion and the side you must argue. Type 'quit' to exit.\n")

    while True:
        debate_topic = input("Motion: ").strip()
        if debate_topic.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!\n")
            break
        if not debate_topic:
            print("Motion cannot be empty.\n")
            continue

        which_side = input("Your side (for/against): ").strip().lower()
        if which_side in ("quit", "exit", "q"):
            print("\nGoodbye!\n")
            break
        if which_side not in ("for", "against"):
            print("Please type 'for' or 'against'.\n")
            continue

        try:
            brief = run_debate_coach(debate_topic, which_side)
            print("\n" + "=" * 60)
            print(brief)
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error("Agent failed: %s", e)


if __name__ == "__main__":
    main()