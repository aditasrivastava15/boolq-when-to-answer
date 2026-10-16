"""Freeze the BoolQ validation split into dev and test JSONL files."""

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
TEST_SIZE = 1000
DEV_SIZE = 200

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    dataset = load_dataset("google/boolq", split="validation")

    records = []
    for index, row in enumerate(dataset):
        records.append(
            {
                "index": index,
                "question": row["question"],
                "passage": row["passage"],
                "answer": bool(row["answer"]),
            }
        )

    random.seed(SEED)
    random.shuffle(records)

    test_records = records[:TEST_SIZE]
    dev_records = records[TEST_SIZE : TEST_SIZE + DEV_SIZE]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA_DIR / "test.jsonl", test_records)
    write_jsonl(DATA_DIR / "dev.jsonl", dev_records)

    unused = len(records) - TEST_SIZE - DEV_SIZE
    print(f"Wrote {len(test_records)} rows to data/test.jsonl")
    print(f"Wrote {len(dev_records)} rows to data/dev.jsonl")
    print(f"Left {unused} rows unused")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
