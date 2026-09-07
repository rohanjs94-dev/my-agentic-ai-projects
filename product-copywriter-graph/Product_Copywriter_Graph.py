"""Use case: a product copywriter assistant

Real scenario: someone gives the app a product name and its features
(e.g., "EcoBottle - insulated, BPA-free, 24hr cold retention"). Instead of
just writing ONE generic description, the app:

Gets 3 different types of analysis AT THE SAME TIME - like consulting
3 specialists simultaneously instead of one after another:
  A feature-summary specialist - condenses the raw features
  A benefit-summary specialist - turns features into customer benefits
  A target-customer specialist - identifies the likely buyer persona

Decides on the right tone - based on all 3 analyses combined, decides:
does this product need a FORMAL description, or MARKETING-heavy copy?

Delivers the right output - either the formal version or the
marketing version, depending on that decision.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

load_dotenv()  # load OPENAI_API_KEY from .env

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ProductCopyState(BaseModel):
    """The data that flows through the whole graph."""

    product_name: str = ""          # input
    product_features: str = ""       # input
    feature_summary: str = ""          # from write_feature_summary
    benefit_summary: str = ""            # from write_benefit_summary
    target_customer: str = ""              # from identify_target_customer
    marketing_tone: str = ""                 # from decide_tone: "formal" or "marketing"
    final_copy: str = ""                       # from whichever final node runs


def write_feature_summary(state: ProductCopyState) -> dict:
    """Specialist node: summarize the raw product features."""
    prompt = f"Summarize the following product features in a concise manner:\n{state.product_features}"
    response = model.invoke(prompt)
    return {"feature_summary": response.content}


def write_benefit_summary(state: ProductCopyState) -> dict:
    """Specialist node: convert product features into customer benefits."""
    prompt = f"Convert the following product features into customer benefits:\n{state.product_features}"
    response = model.invoke(prompt)
    return {"benefit_summary": response.content}


def identify_target_customer(state: ProductCopyState) -> dict:
    """Specialist node: identify the likely buyer persona."""
    prompt = f"Identify the likely buyer persona for a product with these features:\n{state.product_features}"
    response = model.invoke(prompt)
    return {"target_customer": response.content}


def decide_tone(state: ProductCopyState) -> dict:
    """Decision node (fan-in): reads all 3 specialist outputs, picks formal vs marketing."""
    prompt = (
        f"Based on these summaries, decide if this product needs a FORMAL description "
        f"or MARKETING-heavy copy.\n"
        f"Feature Summary: {state.feature_summary}\n"
        f"Benefit Summary: {state.benefit_summary}\n"
        f"Target Customer: {state.target_customer}\n\n"
        f'Reply with EXACTLY one word: "formal" or "marketing".'
    )
    response = model.invoke(prompt)
    tone = response.content.strip().lower()
    return {"marketing_tone": "marketing" if "marketing" in tone else "formal"}  # safe fallback


def formal_product_description(state: ProductCopyState) -> dict:
    """Final node 1: a formal, professional product description."""
    prompt = f"Write a formal product description for {state.product_name} with features:\n{state.product_features}"
    response = model.invoke(prompt)
    return {"final_copy": response.content}


def marketing_product_copy(state: ProductCopyState) -> dict:
    """Final node 2: marketing-heavy, benefit-driven product copy."""
    prompt = f"Write marketing-heavy product copy for {state.product_name}, highlighting:\n{state.benefit_summary}"
    response = model.invoke(prompt)
    return {"final_copy": response.content}


def route_after_tone(state: ProductCopyState) -> str:
    """Decision function (not a node) - tells the conditional edge where to go next."""
    return "marketing" if state.marketing_tone == "marketing" else "formal"


graph = StateGraph(ProductCopyState)

graph.add_node("write_feature_summary", write_feature_summary)
graph.add_node("write_benefit_summary", write_benefit_summary)
graph.add_node("identify_target_customer", identify_target_customer)
graph.add_node("decide_tone", decide_tone)
graph.add_node("formal_product_description", formal_product_description)
graph.add_node("marketing_product_copy", marketing_product_copy)

# Fan-out: all 3 specialists run in parallel, none depend on each other
graph.add_edge(START, "write_feature_summary")
graph.add_edge(START, "write_benefit_summary")
graph.add_edge(START, "identify_target_customer")

# Fan-in: decide_tone waits for all 3 specialists to finish
graph.add_edge("write_feature_summary", "decide_tone")
graph.add_edge("write_benefit_summary", "decide_tone")
graph.add_edge("identify_target_customer", "decide_tone")

# Conditional edge: routes to exactly one of the 2 final nodes
graph.add_conditional_edges(
    "decide_tone",
    route_after_tone,
    {"marketing": "marketing_product_copy", "formal": "formal_product_description"},
)

graph.add_edge("marketing_product_copy", END)
graph.add_edge("formal_product_description", END)

app = graph.compile()


def run_product_copywriter(product_name: str, product_features: str) -> str:
    """Run the graph on a product and return the final copy."""
    result = app.invoke({"product_name": product_name, "product_features": product_features})
    return result["final_copy"]


if __name__ == "__main__":
    name = input("What is the product name? ")
    features = input("What are the product features? ")
    print(run_product_copywriter(name, features))
