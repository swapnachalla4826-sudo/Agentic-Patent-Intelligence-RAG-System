# 🔬 PatentLens — Explainable Multi-Agent RAG

PatentLens is an explainable **Multi-Agent Retrieval-Augmented Generation system** for analyzing uploaded patent documents. Specialized agents can search, rewrite queries, analyze evidence, compare inventions, verify evidence, critique findings, and synthesize a grounded final response.

## Overview

```text
Patent PDFs
      ↓
PDF Processing
      ↓
Chunking + Metadata
      ↓
Embeddings
      ↓
Qdrant
      ↓
Supervisor
      ↓
Specialized Agents
      ↓
Evidence Verification
      ↓
Critic
      ↓
Synthesis
      ↓
Final Answer + Sources
```

## Why Agentic RAG?

Traditional RAG commonly follows:

```text
Question → Retrieve → Generate
```

Patent analysis can require several different operations. PatentLens separates these responsibilities into specialized agents and exposes the workflow so users can inspect how the result was produced.

## Supported Agent Roles

The application can display:

- Supervisor Agent
- Search Agent
- Query Rewriter Agent
- Analysis Agent
- Research Analysis Agent
- Similarity Agent
- Comparison Agent
- Evidence Verification Agent
- Critic Agent
- Synthesis Agent

The UI dynamically displays additional agent outputs as well. fileciteturn2file4L343-L395

## Patent Processing

Uploaded PDFs are loaded with `PyPDFLoader`. Each page receives patent ID and source-file metadata before recursive chunking.

Current configuration:

```text
chunk_size = 1000
chunk_overlap = 150
```

The resulting chunks are stored in an in-memory Qdrant collection named `patentlens`, and the retriever uses `k = 6`. fileciteturn2file4L237-L297

## LLM

The system uses:

```text
Groq
Model: openai/gpt-oss-20b
Temperature: 0
```

The Streamlit application builds the agent with the retriever and LLM and then invokes it for each user question. fileciteturn2file4L568-L580 fileciteturn2file4L627-L654

## Explainability

The application exposes more than the final answer.

### Final Answer
The synthesized response.

### Sources
Patent IDs and page references.

### Multi-Agent Workflow
Outputs from the agents involved.

### Execution Steps
The sequence of agent operations.

### Retrieved Evidence
The patent chunks used during reasoning.

### Evidence Verification
Whether the retrieved evidence is considered sufficient, together with the verification reason when available.

These components are explicitly rendered by the application. fileciteturn2file4L397-L454 fileciteturn2file4L847-L889

## UI Metrics

The dashboard displays:

```text
Files Indexed
Pages
Chunks
```

For each result it can also display:

```text
Intent
Search Attempts
Sources
```

## Tech Stack

- Python
- Streamlit
- LangChain
- Agent orchestration
- Groq
- Qdrant
- Hugging Face Embeddings
- Sentence Transformers
- PyPDF
- Recursive Character Text Splitter
- RAG
- Multi-Agent AI
- Vector Search

## Project Structure

```text
PatentLens-Agentic-RAG/
├── app.py
├── agent.py
├── requirements.txt
├── .env
└── README.md
```

The UI imports:

```python
from agent import build_agent, run_agent
```

so the agent logic is separated from the Streamlit presentation layer. fileciteturn2file4L8-L16

## Installation

```bash
pip install streamlit langchain langchain-community langchain-groq langchain-huggingface langchain-qdrant qdrant-client pypdf sentence-transformers
```

Install any additional packages required by `agent.py`.

## API Key

Set:

```text
GROQ_API_KEY=your_api_key_here
```

or enter it through the Streamlit sidebar.

Never commit API keys.

## Run

```bash
streamlit run app.py
```

## Usage

1. Start the application.
2. Enter the Groq API key.
3. Upload one or more patent PDFs.
4. Click **Process documents**.
5. Wait for indexing and agent initialization.
6. Ask a patent-related question.
7. Inspect the final answer, sources, agent workflow, execution steps, evidence, and verification.

## Applications

- Patent intelligence
- Similar-invention discovery
- Patent comparison
- Technical document analysis
- Prior-art exploration
- Invention ideation support
- Explainable patent Q&A

## Future Improvements

- Explicit LangGraph state transitions
- Conditional routing by intent
- Retrieval grading
- Automatic query-rewrite retries
- Hybrid BM25 + vector search
- Cross-encoder reranking
- Persistent Qdrant collections
- Langfuse tracing
- Citation-level evaluation
- Structured outputs from specialist agents
- Automated evaluation datasets

## Disclaimer

PatentLens is an AI-assisted document analysis system. It does not provide legal advice, determine patentability, or replace professional patent examination.

## Author

**Challa Swapna**
