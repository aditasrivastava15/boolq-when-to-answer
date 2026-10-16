"""Score the four methods on the frozen test set."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPLIES_DIR = RESULTS_DIR / "replies"
CUTOFF_PATH = RESULTS_DIR / "cutoff.json"
N_TEST = 1000
ERROR_LIMIT = 30


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def gold_label(row: dict) -> str:
    return "yes" if row["answer"] else "no"


def is_correct_answer(parsed: str | None, gold: str) -> bool:
    return parsed in {"yes", "no"} and parsed == gold


def score_always_yes(rows: list[dict]) -> dict:
    correct = sum(1 for row in rows if gold_label(row) == "yes")
    return {
        "method": "always_yes",
        "accuracy": correct / N_TEST,
        "coverage": 1.0,
        "answered": N_TEST,
        "correct": correct,
        "retrieval_refusals": 0,
        "model_refusals": 0,
    }


def score_always_answer(rows: list[dict], method: str) -> dict:
    correct = sum(
        1 for row in rows if is_correct_answer(row["parsed"], gold_label(row))
    )
    return {
        "method": method,
        "accuracy": correct / N_TEST,
        "coverage": 1.0,
        "answered": N_TEST,
        "correct": correct,
        "retrieval_refusals": 0,
        "model_refusals": 0,
    }


def score_abstain(rows: list[dict], tau: float) -> tuple[dict, list[dict]]:
    answered = 0
    correct = 0
    retrieval_refusals = 0
    model_refusals = 0
    decisions = []
    for row in rows:
        gold = gold_label(row)
        if row["margin"] < tau:
            decision = "retrieval_refusal"
            retrieval_refusals += 1
        elif row["parsed"] == "unknown":
            decision = "model_refusal"
            model_refusals += 1
        else:
            decision = "answer"
            answered += 1
            if is_correct_answer(row["parsed"], gold):
                correct += 1
        decisions.append({**row, "tau": tau, "decision": decision})
    coverage = answered / N_TEST
    accuracy = (correct / answered) if answered else 0.0
    metrics = {
        "method": "answer_or_abstain",
        "accuracy": accuracy,
        "coverage": coverage,
        "answered": answered,
        "correct": correct,
        "retrieval_refusals": retrieval_refusals,
        "model_refusals": model_refusals,
    }
    return metrics, decisions


def collect_errors(full_rows: list[dict], top_rows: list[dict]) -> list[dict]:
    top_by_index = {row["index"]: row for row in top_rows}
    errors = []
    for full in full_rows:
        if len(errors) >= ERROR_LIMIT:
            break
        top = top_by_index[full["index"]]
        gold = gold_label(full)
        full_ok = is_correct_answer(full["parsed"], gold)
        top_ok = is_correct_answer(top["parsed"], gold)
        if full_ok and not top_ok:
            errors.append(
                {
                    "index": full["index"],
                    "question": full["question"],
                    "gold": gold,
                    "top_sentence": full["top_sentence"],
                    "raw_reply": top["raw_reply"],
                    "error_label": "",
                }
            )
    return errors


def write_scores(path: Path, rows: list[dict]) -> None:
    fields = [
        "method",
        "accuracy",
        "coverage",
        "answered",
        "correct",
        "retrieval_refusals",
        "model_refusals",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_errors(path: Path, rows: list[dict]) -> None:
    fields = [
        "index",
        "question",
        "gold",
        "top_sentence",
        "raw_reply",
        "error_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main() -> None:
    with CUTOFF_PATH.open(encoding="utf-8") as file:
        tau = json.load(file)["chosen"]["tau"]

    full_rows = read_jsonl(REPLIES_DIR / "full_passage.jsonl")
    top_rows = read_jsonl(REPLIES_DIR / "top_sentence.jsonl")

    always_yes = score_always_yes(full_rows)
    full_passage = score_always_answer(full_rows, "full_passage")
    top_sentence = score_always_answer(top_rows, "top_sentence")
    abstain_metrics, abstain_rows = score_abstain(full_rows, tau)
    errors = collect_errors(full_rows, top_rows)

    write_scores(RESULTS_DIR / "scores.csv", [
        always_yes,
        full_passage,
        top_sentence,
        abstain_metrics,
    ])
    write_jsonl(REPLIES_DIR / "abstain.jsonl", abstain_rows)
    write_errors(RESULTS_DIR / "errors.csv", errors)

    print(f"Used tau={tau:.6f}")
    for row in [always_yes, full_passage, top_sentence, abstain_metrics]:
        print(
            f"{row['method']}: accuracy={row['accuracy']:.4f} "
            f"coverage={row['coverage']:.4f} "
            f"answered={row['answered']}"
        )
    print(f"Wrote {len(errors)} error candidates to results/errors.csv")


if __name__ == "__main__":
    main()
