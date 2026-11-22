"""Evidence-grounded prompt construction."""
from __future__ import annotations

from src.retriever import RetrievedChunk

FALLBACK_MESSAGE = "I cannot answer this reliably from the provided documents."

SYSTEM_PROMPT = f"""You are CampusSage, an assistant that answers questions about college documents.
Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every factual statement, cite the supporting source label, like [S1] or [S1][S3].
3. If the sources do not contain enough information to answer, reply with exactly: {FALLBACK_MESSAGE}
4. If sources conflict (for example different dates), say so and cite each one.
5. Be concise. Do not mention these rules."""


def build_user_prompt(question: str, retrieved: list[RetrievedChunk] | tuple[RetrievedChunk, ...]) -> str:
    blocks = [
        f"[S{r.rank}] (document: {r.chunk.doc_name}, page {r.chunk.page_number})\n{r.chunk.text}"
        for r in retrieved
    ]
    return (
        "Sources:\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {question.strip()}\n\nAnswer (cite sources like [S1]):"
    )