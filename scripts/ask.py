"""Ask CampusSage from the terminal.
Usage: python scripts/ask.py "question" ["another"] [--k 4] [--model llama3.2:3b] [--show-prompt]
With no questions, starts an interactive prompt (blank line to quit).
Needs an index: run scripts/retrieval_preview.py --rebuild first."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.answer_generator import AnswerGenerator
from src.embeddings import Embedder
from src.llm_engine import LLMError, OllamaEngine
from src.retriever import Retriever


def show(answer, show_prompt: bool) -> None:
    print(f"\nQ: {answer.question}")
    print(f"A: {answer.text}")
    print(f"Status: {answer.status}")
    if answer.citation_check.invalid:
        print(f"WARNING: cites non-existent sources: {list(answer.citation_check.invalid)}")
    if answer.supporting:
        print("Cited sources (cited, not verified):")
        for r in answer.supporting:
            snippet = r.chunk.text.replace("\n", " ")[:120]
            print(f"  [S{r.rank}] {r.chunk.doc_name} p.{r.chunk.page_number} (score {r.score:.3f}): {snippet!r}")
    print(
        f"Timing: retrieval {answer.retrieval_ms:.0f} ms, generation {answer.generation_s:.1f} s, "
        f"tokens in/out {answer.prompt_tokens}/{answer.completion_tokens}"
    )
    if show_prompt:
        print("\n--- prompt sent to model ---\n" + answer.prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    engine = OllamaEngine(model=args.model)
    try:
        if not engine.is_ready():
            print(f"Model '{args.model}' is not installed. Run: ollama pull {args.model}")
            return 1
    except LLMError as exc:
        print(exc)
        return 1
    try:
        retriever = Retriever.load(config.INDEX_DIR, Embedder())
    except FileNotFoundError:
        print("No index found. Run: python scripts/retrieval_preview.py --rebuild")
        return 1

    generator = AnswerGenerator(retriever, engine, k=args.k)
    print("Note: the first answer includes model loading time.")
    try:
        if args.questions:
            for q in args.questions:
                show(generator.answer(q), args.show_prompt)
        else:
            while (q := input("\nQuestion (blank to quit): ").strip()):
                show(generator.answer(q), args.show_prompt)
    except LLMError as exc:
        print(f"LLM error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())