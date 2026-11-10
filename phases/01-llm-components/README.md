# Phase 1 — LLMs as engineering components

To the rest of a system, an LLM is an API that is unreliable, metered and stochastic.
This phase covers the properties that shape every agent built on top of one, then wraps the
API in a client the rest of the repo can rely on.

## 1.1 Tokens and cost

### Tokens

- Models read **tokens**: subword chunks from a fixed vocabulary of roughly 100k–200k entries,
  built with algorithms such as byte-pair encoding (BPE). Common words are one token; rare words
  split into several.
- English prose averages about 4 characters (0.75 words) per token. Code, JSON and non-English
  text use more tokens per character.
- Each provider has its own tokenizer, so the same text yields different counts. Use the
  provider's own token counter whenever the number matters.

### Billing

- Prices are per million tokens, with separate rates for **input** (everything sent: system
  prompt, tool definitions, history, documents) and **output** (everything generated, including
  hidden reasoning tokens).
- Output costs several times more per token than input. Input is processed in one parallel pass
  (*prefill*); output is generated one token at a time, each needing its own pass (*decode*).

### Why agent loops get expensive

The API is stateless: every call re-sends the full conversation. In an agent loop, each step
appends a tool call and its result, and the next call sends all of it again.

With **S** = fixed prefix per call (system prompt + tool definitions), **k** = tokens appended
per step, and **n** = steps:

```
total input tokens ≈ n·S + k·n(n+1)/2
```

The history term grows with n².

**Example.** S = 3,000, k = 1,500, 300 output tokens per step. Illustrative prices: $3 per
million input tokens, $15 per million output.

| Scenario | Input tokens | Output tokens | Cost |
| --- | --- | --- | --- |
| 25 steps (baseline) | 562,500 | 7,500 | $1.80 |
| 50 steps | 2,062,500 | 15,000 | $6.41 |
| 25 steps, S halved | 525,000 | 7,500 | $1.69 |
| 25 steps, k halved | 318,750 | 7,500 | $1.07 |

What the numbers show:

- **Input dominates.** Output is 5× pricier per token, yet input is 94% of the baseline bill.
- **Step count is the main cost driver.** Doubling steps multiplies cost by 3.6× here: linear
  terms double, the history term quadruples. Longer runs approach 4×.
- **Shrink what accumulates before what is fixed.** Halving k saves 6.5× more than halving S,
  because k multiplies n² and S only n. Compact tool results and history compaction are the
  biggest levers. Prompt caching (1.7) changes the calculation for repeated tokens.
- **Compare models on cost per completed task, not price per token.** A model that finishes in
  fewer steps or writes shorter outputs can cost less even at a higher rate.