"""Rank sentences with BM25 and record the top-sentence margin."""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from split_sentences import split_sentences

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SINGLE_SENTENCE_MARGIN = 999_999.0


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def rank_passage(question: str, passage: str) -> dict:
    sentences = split_sentences(passage)
    tokenized = [tokenize(sentence) for sentence in sentences]
    scores = BM25Okapi(tokenized).get_scores(tokenize(question))
    order = sorted(range(len(sentences)), key=lambda i: float(scores[i]), reverse=True)
    top_sentence = sentences[order[0]]
    if len(order) == 1:
        margin = SINGLE_SENTENCE_MARGIN
    else:
        margin = float(scores[order[0]] - scores[order[1]])
    return {
        "top_sentence": top_sentence,
        "margin": margin,
        "n_sentences": len(sentences),
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def rank_split(name: str) -> None:
    source = read_jsonl(DATA_DIR / f"{name}.jsonl")
    ranked = []
    for row in source:
        ranked.append({**row, **rank_passage(row["question"], row["passage"])})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"bm25_{name}.jsonl"
    write_jsonl(destination, ranked)
    print(f"Wrote {len(ranked)} rows to results/bm25_{name}.jsonl")


def main() -> None:
    rank_split("dev")
    rank_split("test")


if __name__ == "__main__":
    main()
