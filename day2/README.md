# Day 2 – Drive Ready OPC

## AutoMate AI
An English-language automotive assistant built using Groq and LangChain.

## Concepts practiced
- Few-shot classification of automotive enquiries
- Extracting enquiry details into JSON
- Python functions, docstrings, dictionaries and decorators
- Creating and invoking a LangChain tool
- Maintaining the latest eight conversation turns
- Resetting and exiting an interactive chat

## Execution
In Google Colab, install `langchain` and `langchain-groq` using `%pip install -U langchain langchain-groq`. Add `GROQ_API_KEY` to Colab Secrets and enable access. Upload `task.py` and execute `%run task.py`.

The vehicle guide is demonstrated separately; it is not automatically called by the chat model. The assistant has no live pricing or inventory access.
