"""Day 6: LangGraph routing for AutoMate AI.
Requires: pip install -U langgraph langchain-groq pydantic
Set GROQ_API_KEY in your environment before running.
"""
import csv
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

DATA_FILE = Path(__file__).resolve().parent / "inventory_template.csv"
MODEL_NAME = os.getenv("AUTOMATE_MODEL", "openai/gpt-oss-20b")


def get_key():
    if os.getenv("GROQ_API_KEY"):
        return
    try:
        from google.colab import userdata
        token = userdata.get("GROQ_API_KEY")
        if token:
            os.environ["GROQ_API_KEY"] = token
    except ImportError:
        pass
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Provide GROQ_API_KEY using environment variables or Colab Secrets.")


get_key()
llm = ChatGroq(model=MODEL_NAME, temperature=0)
        
class RouteLabel(BaseModel):
    intent: Literal["inventory", "faq", "other"] = Field(
        description="Choose the appropriate dealership specialist."
    )


class AgentState(TypedDict, total=False):
    message: str
    intent: Literal["inventory", "faq", "other"]
    reply: str
    sources: list[str]


router = llm.with_structured_output(RouteLabel)


def classify(state: AgentState) -> AgentState:
    question = state.get("message", "").strip()
    prompt = (
        "Classify this request into one category only: "
        "inventory = searching available dealer stock; "
        "faq = dealer warranty, inspection, transfer or policy; "
        "other = general advice and greetings.\n"
        f"Customer: {question}"
    )
    try:
        label = router.invoke(prompt).intent
        return {"intent": label}
    except Exception:
        return {"intent": "other", "reply": "Unable to classify this request. Please rephrase."}


def choose_node(state: AgentState) -> Literal["inventory", "faq", "other"]:
    return state.get("intent", "other")


def read_inventory():
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def detect_budget(text):
    result = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", text)
    if result:
        return int(float(result.group(1)) * 100000)
    result = re.search(r"(?:rs\.?\s*|₹)(\d{4,})", text)
    return int(result.group(1)) if result else None


def inventory(state: AgentState) -> AgentState:
    text = state["message"].lower()
    rows = read_inventory()
    if not rows:
        return {
            "reply": "The inventory file is empty or missing. Confirm available vehicles with dealership staff.",
            "sources": ["inventory_template.csv"],
        }
    maximum = detect_budget(text)
    fuel = next((f for f in ("petrol", "diesel", "electric", "hybrid", "cng") if f in text), "")
    gear = next((g for g in ("automatic", "manual") if g in text), "")
    body = next((b for b in ("hatchback", "sedan", "suv", "mpv") if b in text), "")
    valid = []
    for car in rows:
        if str(car.get("status", "")).upper() != "AVAILABLE":
            continue
        try:
            price = int(float(car.get("asking_price_inr", "")))
        except (TypeError, ValueError):
            continue
        if maximum is not None and price > maximum:
            continue
        if fuel and fuel not in str(car.get("fuel_type", "")).lower():
            continue
        if gear and gear not in str(car.get("transmission", "")).lower():
            continue
        if body and body not in str(car.get("body_type", "")).lower():
            continue
        valid.append(car)
    if not valid:
        return {"reply": "There are no matching AVAILABLE vehicles in the supplied CSV.", "sources": ["inventory_template.csv"]}
    lines = [
        f"{v.get('vehicle_id')}: {v.get('year')} {v.get('make')} {v.get('model')} "
        f"{v.get('variant')} | INR {v.get('asking_price_inr')} | "
        f"{v.get('fuel_type')} | {v.get('transmission')} | {v.get('location')}"
        for v in valid[:5]
    ]
    prompt = (
        "Summarize the following approved inventory rows for this customer. "
        "Do not invent details. Mention that availability needs staff confirmation.\n"
        + "Customer: " + state["message"] + "\nRows:\n" + "\n".join(lines)
    )
    reply = llm.invoke(prompt).content
    return {"reply": str(reply), "sources": ["inventory_template.csv"]}


FAQ = (
    "Inspection: buyers should arrange an independent vehicle check and review service records.\n"
    "Warranty: any warranty is dealer- and vehicle-specific; only written terms confirm coverage.\n"
    "Transfer: exact ownership procedures vary by jurisdiction; confirm with transport authorities.\n"
    "Availability: older advertisements do not guarantee a vehicle is still for sale."
)


def faq(state: AgentState) -> AgentState:
    prompt = (
        "Answer using only the following FAQ notes. If information is missing, "
        "say dealership staff must confirm. Avoid inventing a policy.\n"
        + FAQ + "\nQuestion: " + state["message"]
    )
    return {"reply": str(llm.invoke(prompt).content), "sources": ["FAQ notes in task.py"]}


def general(state: AgentState) -> AgentState:
    if state.get("reply"):
        return {"reply": state["reply"], "sources": []}
    prompt = (
        "You are AutoMate AI. Answer general vehicle questions clearly in English. "
        "Do not fabricate live pricing, availability, or policies.\n"
        "User: " + state["message"]
    )
    return {"reply": str(llm.invoke(prompt).content), "sources": []}


workflow = StateGraph(AgentState)
workflow.add_node("classify", classify)
workflow.add_node("inventory", inventory)
workflow.add_node("faq", faq)
workflow.add_node("other", general)
workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify", choose_node, {"inventory": "inventory", "faq": "faq", "other": "other"}
)
for node in ("inventory", "faq", "other"):
    workflow.add_edge(node, END)
app = workflow.compile()


def ask(question):
    result = app.invoke({"message": question})
    return result


def main():
    print("AutoMate AI: LangGraph specialists (inventory / faq / other).")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit", "bye"):
            break
        if not text:
            continue
        try:
            result = ask(text)
            print("Route:", result.get("intent", "other"))
            print("AutoMate AI:", result.get("reply", ""))
            print("Sources:", ", ".join(result.get("sources", [])))
        except Exception as exc:
            print("Request failed:", type(exc).__name__)


if __name__ == "__main__":
    main()
