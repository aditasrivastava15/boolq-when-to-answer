"""Run the Qwen reader on full passages and top sentences."""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPLIES_DIR = RESULTS_DIR / "replies"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 10
PARSE_RE = re.compile(r"\b(yes|no|unknown)\b", re.IGNORECASE)

PROMPT = """Passage:
{text}

Question: {question}
Answer yes or no. If the passage does not decide the question, answer unknown.
Reply with one word: yes, no, or unknown.
"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def parse_reply(text: str) -> str | None:
    match = PARSE_RE.search(text)
    if match is None:
        return None
    return match.group(1).lower()


def load_reader():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate(tokenizer, model, device, passage_text: str, question: str) -> tuple[str, str | None]:
    user_text = PROMPT.format(text=passage_text, question=question)
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, parse_reply(raw)


def run_job(
    rows: list[dict],
    context_field: str,
    destination: Path,
    tokenizer,
    model,
    device,
) -> None:
    replies = []
    for i, row in enumerate(rows, start=1):
        raw, parsed = generate(tokenizer, model, device, row[context_field], row["question"])
        replies.append(
            {
                **row,
                "context_field": context_field,
                "raw_reply": raw,
                "parsed": parsed,
            }
        )
        if i % 50 == 0 or i == len(rows):
            print(f"{destination.name}: {i}/{len(rows)}")
    write_jsonl(destination, replies)
    print(f"Wrote {len(replies)} rows to {destination.relative_to(ROOT)}")


def main() -> None:
    tokenizer, model, device = load_reader()
    print(f"Loaded {MODEL_NAME} on {device}")

    test_rows = read_jsonl(RESULTS_DIR / "bm25_test.jsonl")
    dev_rows = read_jsonl(RESULTS_DIR / "bm25_dev.jsonl")

    run_job(test_rows, "passage", REPLIES_DIR / "full_passage.jsonl", tokenizer, model, device)
    run_job(test_rows, "top_sentence", REPLIES_DIR / "top_sentence.jsonl", tokenizer, model, device)
    run_job(dev_rows, "passage", REPLIES_DIR / "full_passage_dev.jsonl", tokenizer, model, device)


if __name__ == "__main__":
    main()
