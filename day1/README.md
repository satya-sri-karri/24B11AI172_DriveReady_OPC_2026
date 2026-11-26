# Day 1 – Drive Ready OPC

## Objective
Explore large language models through the Groq API and build a simple bakery customer support assistant with LangChain.

## Task overview
- Configure the Groq API key securely using `getpass`.
- Initialize the language model with `ChatGroq`.
- Define bakery policies through a system prompt.
- Test customer questions about eggless cakes, advance orders and delivery.
- Experiment with Prompt Guard using normal and prompt injection inputs.

## Requirements
Python 3, Jupyter Notebook or Google Colab, a Groq API key, `langchain_groq` and `groq`.

## Running the notebook
1. Open `task.ipynb`.
2. Run the installation and setup cells.
3. Enter your Groq API key when prompted.
4. Execute the assistant and Prompt Guard examples.

## File
`task.ipynb` – Day 1 implementation and test queries.

## Learning outcome
System prompts guide model behavior, while Prompt Guard can help identify suspicious inputs. The guard demonstration is separate from the assistant.
