# Day 3 – Drive Ready OPC

## AutoMate AI: Dealership Assistant
A basic LangChain agent for handling used-car enquiries, qualifying leads and preparing customer reply drafts.

## Concepts practiced
- Creating an agent using `create_agent`
- Selecting Gemini or OpenAI as the model provider
- Retaining eight recent conversation turns
- Reading test questions using Pandas
- Evaluating answers against groups of expected keywords
- Reporting PASS, REVIEW and ERROR results

## Execution
1. In Google Colab run `%pip install -U langchain langchain-google-genai langchain-openai pandas`.
2. Configure `GOOGLE_API_KEY` or `OPENAI_API_KEY` in Colab Secrets and enable access.
3. Upload `task.py` and `test.csv` to the same directory.
4. Select the provider in `task.py`. Adjust the model ID if required by your account.
5. Run `%run task.py`. Enter `reset` to clear memory or `exit` to stop.

## Evaluation
Each semicolon-separated group must match at least one alternative separated by `|`. Keyword matching is a basic check, not proof of factual correctness. Test cases run independently of chat history.

No external inventory, CRM or messaging tools are connected. Replies are drafts only.
