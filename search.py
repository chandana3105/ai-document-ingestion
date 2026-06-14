from __future__ import annotations

import os
from typing import Generator

import anthropic
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import settings

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided document context. If the answer is not found in the context, say: "
    "'I could not find that information in the uploaded documents.' "
    "Always cite the source document when possible."
)

CONTEXT_TEMPLATE = "Source: {source}\n{content}"


class SearchEngine:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._store = Chroma(
            persist_directory=settings.vector_db_dir,
            embedding_function=embeddings,
        )

    def retrieve(self, query: str, k: int | None = None) -> list[tuple]:
        k = k or settings.retrieval_k
        # MMR for diversity — avoids returning near-duplicate chunks
        results = self._store.max_marginal_relevance_search(query, k=k)
        results = [(doc, 0.0) for doc in results]
        return results  # list of (Document, score)

    def _build_context(self, results: list[tuple]) -> str:
        parts = []
        for doc, score in results:
            parts.append(
                CONTEXT_TEMPLATE.format(
                    source=doc.metadata.get("source", "unknown"),
                    content=doc.page_content,
                )
            )
        return "\n\n---\n\n".join(parts)

    def ask_openai(self, question: str, k: int | None = None) -> Generator[str, None, None]:
        results = self.retrieve(question, k=k)
        context = self._build_context(results)
        sources = list({doc.metadata.get("source", "?") for doc, _ in results})

        llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0, streaming=True)
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        for chunk in llm.stream(messages):
            yield chunk.content

        yield f"\n\nSources: {', '.join(sources)}"

    def ask_claude(self, question: str, k: int | None = None) -> Generator[str, None, None]:
        results = self.retrieve(question, k=k)
        context = self._build_context(results)
        sources = list({doc.metadata.get("source", "?") for doc, _ in results})

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text

        yield f"\n\nSources: {', '.join(sources)}"

    def ask(self, question: str, provider: str | None = None, k: int | None = None) -> Generator[str, None, None]:
        provider = provider or settings.llm_provider
        if provider == "claude":
            yield from self.ask_claude(question, k=k)
        else:
            yield from self.ask_openai(question, k=k)


def main():
    if not os.path.exists(settings.vector_db_dir):
        print("Vector DB not found. Run ingest.py first.")
        return

    engine = SearchEngine()
    provider = settings.llm_provider
    print(f"AI Document Search ready (provider: {provider}). Type 'exit' to quit.")

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        print("\nAnswer:")
        for chunk in engine.ask(question):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
