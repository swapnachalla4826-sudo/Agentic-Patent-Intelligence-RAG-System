from fastapi import FastAPI
from pydantic import BaseModel

from agent import build_agent, run_agent
from rag_pipeline import llm, retriever

app = FastAPI(
    title="PatentLens API",
    description="AI-Powered Patent Intelligence - Agentic RAG",
    version="2.0",
)

# Build the agent once at startup
agent = build_agent(retriever, llm)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "PatentLens Agentic RAG API is running"}


@app.post("/ask")
def ask_patentlens(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return {
            "question": question,
            "answer": "Please provide a question.",
            "sources": [],
            "steps": [],
        }

    result = run_agent(agent, question)

    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "steps": result["steps"],  # what the agent did, for debugging/UI
    }
