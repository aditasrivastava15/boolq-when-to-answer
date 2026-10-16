"""Choose the abstention cutoff τ on the development set."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_REPLIES = ROOT / "results" / "replies" / "full_passage_dev.jsonl"
CUTOFF_PATH = ROOT / "results" / "cutoff.json"

PERCENTILES = np.arange(10, 95, 5)
MIN_COVERAGE = 0.50
MAX_COVERAGE = 0.90
TARGET_COVERAGE = 0.70


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def gold_label(row: dict) -> str:
    return "yes" if row["answer"] else "no"


def evaluate(rows: list[dict], tau: float) -> dict:
    answered = 0
    correct = 0
    retrieval_refusals = 0
    model_refusals = 0
    n = len(rows)
    for row in rows:
        if row["margin"] < tau:
            retrieval_refusals += 1
            continue
        parsed = row["parsed"]
        if parsed == "unknown":
            model_refusals += 1
            continue
        answered += 1
        if parsed == gold_label(row):
            correct += 1
    coverage = answered / n
    accuracy = (correct / answered) if answered else 0.0
    return {
        "tau": float(tau),
        "accuracy": accuracy,
        "coverage": coverage,
        "answered": answered,
        "correct": correct,
        "retrieval_refusals": retrieval_refusals,
        "model_refusals": model_refusals,
    }


def choose(rows: list[dict]) -> dict:
    margins = np.array([row["margin"] for row in rows], dtype=float)
    candidates = []
    for p in PERCENTILES:
        tau = float(np.percentile(margins, p))
        result = evaluate(rows, tau)
        result["percentile"] = int(p)
        candidates.append(result)

    in_band = [
        c
        for c in candidates
        if MIN_COVERAGE <= c["coverage"] <= MAX_COVERAGE
    ]
    if not in_band:
        raise RuntimeError(
            "No cutoff produced development coverage between 0.50 and 0.90."
        )

    best_accuracy = max(c["accuracy"] for c in in_band)
    tied = [c for c in in_band if c["accuracy"] == best_accuracy]
    chosen = min(tied, key=lambda c: abs(c["coverage"] - TARGET_COVERAGE))
    return {"chosen": chosen, "candidates": candidates}


def main() -> None:
    rows = read_jsonl(DEV_REPLIES)
    result = choose(rows)
    chosen = result["chosen"]
    CUTOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CUTOFF_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(
        f"Chose tau={chosen['tau']:.6f} "
        f"(p{chosen['percentile']}, "
        f"accuracy={chosen['accuracy']:.4f}, "
        f"coverage={chosen['coverage']:.4f})"
    )
    print(f"Wrote {CUTOFF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
