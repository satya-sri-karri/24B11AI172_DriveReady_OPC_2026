# Day 5 — PDF Question Answering with AutoMate AI

## Objective
Build a dealership document assistant that can retrieve passages from PDF files and remember messages separately for each conversation.

## Concepts
- Upload and parse approved dealership PDFs with `PyPDFLoader`
- Divide long pages into overlapping passages using a text splitter
- Convert passages to Hugging Face embeddings and index them with Chroma
- Expose document retrieval as a LangChain tool for a Groq-powered agent
- Keep separate conversation histories with LangGraph's `InMemorySaver` and thread IDs

## Colab setup
1. Add `GROQ_API_KEY` in **Colab → Secrets** and enable notebook access.
2. Install the dependencies:
   ```python
   %pip install -U langchain langchain-groq langchain-community langchain-chroma langchain-huggingface langchain-text-splitters langgraph pypdf sentence-transformers
   ```
3. Upload `task.py` and run `%run task.py`.
4. Upload at least one dealership-owned PDF when asked.
5. Use `switch` to change to another thread, or `exit` to finish.

## How it works
The model can call `search_documents` to search the uploaded documents. Retrieved excerpts include the PDF filename and page number. The assistant should say when a dealership policy is not supported by the documents rather than filling in missing details.

The two memory-demo threads are independent; conversation history is kept only while the Colab session is active.

## Notes
Use only PDFs you are authorised to upload. Do not upload sensitive customer records. Documents are treated as reference data, not instructions. Retrieval scores are heuristic and do not guarantee factual accuracy. The API workflow requires a valid Groq key and has not been live-tested here.
