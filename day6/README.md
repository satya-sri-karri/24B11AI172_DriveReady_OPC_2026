# Day 6 — AutoMate AI Multi-Route Agent with LangGraph

## Objective
Route customer enquiries to the right specialist using a LangGraph workflow.

## Features
1. An LLM-based classifier categorises each message as `inventory`, `faq`, or `other`.
2. The inventory node searches a local CSV (only rows marked `AVAILABLE`).
3. The FAQ node uses a small set of approved dealership reference notes.
4. The general node handles greetings and general automotive questions.
5. The graph returns a reply, the selected route, and source information.

## Project files
- `task.py`: classification, graph nodes, routing and interactive chat
- `inventory_template.csv`: blank inventory sheet with required columns
- `test.csv`: example questions with expected route and behaviour
- `README.md`: setup and usage guide

## Running the task
Install `langgraph`, `langchain-groq`, and `pydantic` in Python/Colab:
```bash
pip install -U langgraph langchain-groq pydantic
```
Set your `GROQ_API_KEY` as an environment variable (or use Colab Secrets).
Keep `inventory_template.csv` beside `task.py` and run `python task.py`.
In Colab, upload both files and run `%run task.py`.
Type `exit` to end the session.

## Inventory schema
`vehicle_id,make,model,variant,year,asking_price_inr,fuel_type,transmission,body_type,odometer_km,location,status,last_verified_at`

The CSV is intentionally empty. Add fictional practice records or authorised real inventory. A missing or empty file must not result in fabricated stock.

## Learning outcomes
Explore explicit routing with `StateGraph`, `START`, `END`, conditional edges, typed state, and three specialist nodes. Observe how structured LLM outputs help with classification.

## Limitations
This exercise does not connect to a live dealership system and does not provide verified warranties, quotes or current market prices. Rerun with a valid Groq API key to check the live behaviour.
