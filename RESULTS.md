# Results in plain language

On 1,000 yes/no test questions, answering only when the system was more sure helped. It got more of its answers right, but it answered fewer questions.

## What was compared

| Method | What it did | Accuracy | How many it answered |
|---|---|---|---|
| Always Yes | Always say yes | 61.5% (615 of 1,000) | All 1,000 |
| Full Passage | Give the model the whole article | 58.9% (589 of 1,000) | All 1,000 |
| Top Sentence | Give the model only the best-matching sentence | 49.2% (492 of 1,000) | All 1,000 |
| Answer or Abstain | Answer only if one sentence clearly ranked above the next | 77.6% (401 of 517) | 517 of 1,000 |

Accuracy here means: of the questions it actually answered, how many were right.

## Main results

- Most true answers are yes. 615 of 1,000 gold labels are yes, so “always say yes” already scores 61.5%.
- The model is weaker than that baseline when it must always answer. With the whole article it scores 58.9%. With only the best sentence it scores 49.2%.
- The best-matching sentence is often not the deciding sentence. That drop from 58.9% to 49.2% is the main problem the project measured.
- Refusing the shaky cases helps. Answer or Abstain scores 77.6%, but only by answering 517 questions and skipping 483.

## How the 1,000 test questions were split (Answer or Abstain)

- Answered: 517. Of these, 401 were right and 116 were wrong.
- Refused: 483. These do not count in accuracy.

The 483 refusals split into two kinds:

- 314 retrieval refusals. The top two sentences were too close (margin below 0.191741), so the system did not ask the model.
- 169 model refusals. The margin was large enough, but the model said `unknown`.

So about half the questions get an answer (51.7% coverage), and about three quarters of those answers are right.

## Why the best sentence fails (30 example mistakes)

These are the first 30 test cases where the whole article was answered right and the best sentence was answered wrong:

- 24 wrong sentence. The fact that decides the question was in another sentence. Search picked the wrong one.
- 4 model contradicted. The best sentence already had the fact. The model still got it wrong.
- 2 not decidable. Even the full article does not clearly settle yes or no.

Most of those mistakes are search, not reading.

## One-line takeaway

Forcing an answer on every question is worse than always saying yes. Letting the system say “I don’t know” when two sentences look equally relevant raises accuracy from 58.9% to 77.6%, at the cost of answering only about half the questions.
