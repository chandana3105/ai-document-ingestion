# FastAPI REST interface for the RAG pipeline.

import asyncio
import json
import os
from typing import AsyncGenerator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from ingest import ingest
from search import SearchEngine

load_dotenv()

app = FastAPI(
    title="AI Document Ingestion API",
    description="RAG pipeline with OpenAI and Claude, MMR retrieval, and streaming.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        if not os.path.exists(settings.vector_db_dir):
            raise HTTPException(
                status_code=400,
                detail="Vector DB not found. POST /ingest first.",
            )
        _engine = SearchEngine()
    return _engine


# Models

class IngestResponse(BaseModel):
    chunks_created: int
    documents_dir: str


class ChatRequest(BaseModel):
    question: str
    provider: Literal["openai", "claude"] | None = None
    k: int | None = Field(default=None, ge=1, description="Number of chunks to retrieve (min 1)")


# Routes

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_db_ready": os.path.exists(settings.vector_db_dir),
        "default_provider": settings.llm_provider,
    }


@app.post("/ingest", response_model=IngestResponse)
def run_ingest():
    "Embed all documents in the configured documents directory."
    global _engine
    _engine = None  
    chunks = ingest()
    return IngestResponse(chunks_created=chunks, documents_dir=settings.documents_dir)


class ChatResponse(BaseModel):
    text: str


@app.post(
    "/chat",
    responses={
        200: {
            "description": "Server-Sent Events stream. Each event is `data: {\"text\": \"...\"}`. Ends with `data: [DONE]`.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def chat(req: ChatRequest):
    "Stream an answer using MMR retrieval + the selected LLM provider."
    engine = get_engine()
    provider = req.provider or settings.llm_provider

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()

        def generate():
            return list(engine.ask(req.question, provider=provider, k=req.k))

        chunks = await loop.run_in_executor(None, generate)
        for chunk in chunks:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sources")
def list_sources():
    " List all ingested document sources in the vector store."
    engine = get_engine()
    collection = engine._store.get()
    sources = sorted({
        m.get("source", "unknown")
        for m in collection.get("metadatas", [])
    })
    return {"sources": sources, "count": len(sources)}
