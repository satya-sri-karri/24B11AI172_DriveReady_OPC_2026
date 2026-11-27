# Drive Ready OPC 2026 – Agentic AI Training

This repository contains my day-wise practical assignments for Drive Ready OPC 2026. The exercises cover prompt engineering, LangChain, LLM APIs, tools, conversation history, evaluation, and external integrations.

## Assignments

| Day | Focus | Implementation |
| --- | --- | --- |
| [Day 1](day1/) | Groq, system prompts, prompt-injection checks | Sweet Crumbs Home Bakery assistant |
| [Day 2](day2/) | Few-shot prompting, JSON extraction, decorators, tools, memory | AutoMate AI automotive assistant |
| [Day 3](day3/) | Agents, Gemini/OpenAI, memory, CSV-based testing | AutoMate AI dealership assistant |
| [Day 4](day4/) | Google Sheets integration, inventory, EMI and customer leads | AutoMate AI dealership CRM assistant |

## Repository structure
```text
day1/
  README.md
  task.ipynb
day2/
  README.md
  task.py
day3/
  README.md
  task.py
  test.csv
day4/
  README.md
  task.py
  test.csv
```

Each day has its own README explaining the task, prerequisites and execution steps. Use the appropriate Python packages, authorise API access and run the code in Google Colab when indicated.

**Security:** Store API keys in Colab Secrets or environment variables, never in GitHub. Day 4 can save customer information to a Google Sheet, so use only authorised data and obtain explicit consent.
