"""Retrieve -> prompt -> generate -> validate citations. Returns a fully recorded Answer."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import config
from src.prompt_templates import FALLBACK_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from src.retriever import RetrievedChunk

STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_GATED = "gated_low_retrieval_score"
STATUS_INVALID_CITATIONS = "invalid_citations"
STATUS_UNCITED = "uncited"
STATUS_NO_OUTPUT = "no_output"

_BRACKET_GROUP = re.compile(r"\[([^\[\]]*)\]")
_SOURCE_ID = re.compile(r"\bS(\d+)\b")


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]    # every source number the answer references, in order of appearance
    invalid: tuple[int, ...]  # references to sources that were never supplied

    @property
    def valid(self) -> tuple[int, ...]:
        return tuple(n for n in self.cited if n not in self.invalid)


def check_citations(text: str, num_sources: int) -> CitationCheck:
    """Structural check only: do cited labels exist? It does NOT verify the claim is supported."""
    cited: list[int] = []
    for group in _BRACKET_GROUP.findall(text):
        for raw in _SOURCE_ID.findall(group):
            n = int(raw)
            if n not in cited:
                cited.append(n)
    invalid = tuple(n for n in cited if n < 1 or n > num_sources)
    return CitationCheck(tuple(cited), invalid)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    status: str
    retrieved: tuple[RetrievedChunk, ...]
    citation_check: CitationCheck
    model: str | None
    retrieval_ms: float
    generation_s: float
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt: str

    @property
    def supporting(self) -> list[RetrievedChunk]:
        """Retrieved chunks the answer cites (valid labels only). Cited, not verified."""
        valid = set(self.citation_check.valid)
        return [r for r in self.retrieved if r.rank in valid]


def _is_fallback(text: str) -> bool:
    return FALLBACK_MESSAGE.lower().rstrip(".") in text.lower()


class AnswerGenerator:
    def __init__(self, retriever, engine, k: int = config.TOP_K, min_score: float | None = config.MIN_RETRIEVAL_SCORE):
        self.retriever = retriever
        self.engine = engine
        self.k = k
        self.min_score = min_score

    def answer(self, question: str) -> Answer:
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        t0 = time.perf_counter()
        retrieved = tuple(self.retriever.search(question, self.k))
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if self.min_score is not None and (not retrieved or retrieved[0].score < self.min_score):
            return Answer(
                question=question, text=FALLBACK_MESSAGE, status=STATUS_GATED, retrieved=retrieved,
                citation_check=CitationCheck((), ()), model=None, retrieval_ms=retrieval_ms,
                generation_s=0.0, total_s=retrieval_ms / 1000, prompt_tokens=None,
                completion_tokens=None, prompt="",
            )

        prompt = build_user_prompt(question, retrieved)
        response = self.engine.generate(SYSTEM_PROMPT, prompt)
        text = response.text.strip()
        check = check_citations(text, len(retrieved))

        if not text:
            status = STATUS_NO_OUTPUT
        elif _is_fallback(text):
            status = STATUS_INSUFFICIENT
        elif check.invalid:
            status = STATUS_INVALID_CITATIONS
        elif not check.cited:
            status = STATUS_UNCITED
        else:
            status = STATUS_ANSWERED

        return Answer(
            question=question, text=text, status=status, retrieved=retrieved, citation_check=check,
            model=response.model, retrieval_ms=retrieval_ms, generation_s=response.latency_s,
            total_s=retrieval_ms / 1000 + response.latency_s, prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens, prompt=prompt,
        )