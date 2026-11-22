import pytest

from src.answer_generator import (STATUS_ANSWERED, STATUS_GATED, STATUS_INSUFFICIENT,
                                  STATUS_INVALID_CITATIONS, STATUS_NO_OUTPUT, STATUS_UNCITED,
                                  AnswerGenerator, check_citations)
from src.llm_engine import LLMError, LLMResponse
from src.prompt_templates import FALLBACK_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from src.retriever import RetrievedChunk
from src.text_chunker import Chunk


def make_hit(rank, score, doc, page, text):
    return RetrievedChunk(rank, score, Chunk(f"{doc}::p{page}::c0", doc, page, 0, text, 0, len(text)))


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=4):
        return self.hits[:k]


class FakeEngine:
    model = "fake-llm"

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return LLMResponse(self.reply, self.model, 0.01, 10, 5)


@pytest.fixture
def hits():
    return [make_hit(1, 0.30, "rules.pdf", 3, "Students need 75 percent attendance."),
            make_hit(2, 0.20, "notice.pdf", 1, "Hostel fees are due before semester.")]


def test_user_prompt_contains_labelled_sources_and_question(hits):
    prompt = build_user_prompt("What is the attendance rule?", hits)
    assert "[S1] (document: rules.pdf, page 3)" in prompt
    assert "[S2] (document: notice.pdf, page 1)" in prompt
    assert "Students need 75 percent attendance." in prompt
    assert "Question: What is the attendance rule?" in prompt


def test_check_citations_valid_and_invalid():
    check = check_citations("A [S1]. B [S5]. C [S0].", num_sources=2)
    assert check.cited == (1, 5, 0)
    assert check.invalid == (5, 0)
    assert check.valid == (1,)


def test_check_citations_grouped_and_ignores_other_brackets():
    check = check_citations("Due [S1, S2] and [S2][S3]. See [note] [Sample 3] (S1)", num_sources=2)
    assert check.cited == (1, 2, 3)
    assert check.invalid == (3,)


def test_answered_with_valid_citations(hits):
    generator = AnswerGenerator(StubRetriever(hits), FakeEngine("75 percent attendance is required [S1]."))
    answer = generator.answer("attendance rule?")
    assert answer.status == STATUS_ANSWERED
    assert [r.chunk.doc_name for r in answer.supporting] == ["rules.pdf"]


def test_invalid_citation_flagged(hits):
    answer = AnswerGenerator(StubRetriever(hits), FakeEngine("Something [S9].")).answer("q")
    assert answer.status == STATUS_INVALID_CITATIONS
    assert answer.supporting == []


def test_missing_citation_flagged(hits):
    answer = AnswerGenerator(StubRetriever(hits), FakeEngine("75 percent is required.")).answer("q")
    assert answer.status == STATUS_UNCITED


def test_fallback_detected(hits):
    answer = AnswerGenerator(StubRetriever(hits), FakeEngine(FALLBACK_MESSAGE)).answer("q")
    assert answer.status == STATUS_INSUFFICIENT


def test_score_gate_skips_llm(hits):
    engine = FakeEngine("should not be used [S1]")
    answer = AnswerGenerator(StubRetriever(hits), engine, min_score=0.5).answer("q")
    assert answer.status == STATUS_GATED
    assert answer.text == FALLBACK_MESSAGE
    assert engine.calls == [] and answer.model is None


def test_gate_disabled_still_calls_llm(hits):
    engine = FakeEngine("Fees are due before semester [S2].")
    answer = AnswerGenerator(StubRetriever(hits), engine, min_score=None).answer("q")
    assert len(engine.calls) == 1 and answer.status == STATUS_ANSWERED


def test_engine_error_propagates(hits):
    generator = AnswerGenerator(StubRetriever(hits), FakeEngine(error=LLMError("boom")))
    with pytest.raises(LLMError, match="boom"):
        generator.answer("q")


def test_empty_output_status(hits):
    answer = AnswerGenerator(StubRetriever(hits), FakeEngine("   ")).answer("q")
    assert answer.status == STATUS_NO_OUTPUT


def test_timing_and_token_fields_recorded(hits):
    answer = AnswerGenerator(StubRetriever(hits), FakeEngine("Yes [S1].")).answer("q")
    assert answer.model == "fake-llm"
    assert (answer.prompt_tokens, answer.completion_tokens) == (10, 5)
    assert answer.generation_s == 0.01
    assert answer.retrieval_ms >= 0 and answer.total_s >= answer.generation_s


def test_engine_receives_system_prompt_and_context(hits):
    engine = FakeEngine("Yes [S1].")
    AnswerGenerator(StubRetriever(hits), engine).answer("attendance rule?")
    system_prompt, user_prompt = engine.calls[0]
    assert system_prompt == SYSTEM_PROMPT
    assert "rules.pdf" in user_prompt and "attendance rule?" in user_prompt


def test_empty_question_raises(hits):
    with pytest.raises(ValueError):
        AnswerGenerator(StubRetriever(hits), FakeEngine()).answer("   ")