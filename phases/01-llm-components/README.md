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

## 1.2 Context windows and degradation

### The window

- The **context window** is the maximum tokens per request, shared by input and output.
  A separate, much smaller limit caps output tokens.
- Exceeding it fails the request or, in some frameworks, silently drops content. Agent history
  grows every step (1.1), so long-running agents reach the limit unless something manages it.

### Effective length is shorter than advertised

- **Lost in the middle** ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)): recall is best
  for information at the start and end of the context, worst in the middle.
- **Needle-in-a-haystack overstates ability.** It tests finding one distinctive phrase by
  keyword. Benchmarks with multi-hop questions, aggregation and no keyword overlap
  ([RULER](https://arxiv.org/abs/2404.06654), [NoLiMa](https://arxiv.org/abs/2502.05167))
  show effective length well below the advertised window.
- **Context rot** ([Chroma, 2025](https://research.trychroma.com/context-rot)): quality drops
  as input grows, even on simple tasks, and faster when the context contains distractors —
  content similar to the answer but wrong.
- Intuition: attention is a fixed budget spread across all tokens, similar content competes for
  it, and models see far fewer very long examples in training.

### Rules of thumb

- **Fewer, higher-signal tokens beat more tokens.** Extra context costs money, adds latency
  (prefill time grows with input) and lowers quality.
- **Placement matters.** Put long documents first and the question last. Wrap each document in
  tags with an ID (`<doc id="7">…</doc>`) so the model can separate and cite them.
- **Distractors hurt more than noise.** Ten near-duplicates do more damage than ten unrelated
  documents; retrieve less, but more precisely.
- **The agent's own history dilutes its instructions.** Long histories weaken constraints given
  at the start. Mitigate by re-injecting key instructions near the end and compacting history.
- **Enforce hard constraints in code, not in the prompt.** If an action must never happen, the
  tool should refuse it. The prompt explains the rule; the code guarantees it.

## 1.3 Sampling and non-determinism

### Decoding

At each step the model produces a score (logit) for every vocabulary token; softmax turns these
into probabilities, and a decoding strategy picks one token.

- **Greedy** takes the most likely token. **Sampling** draws from the distribution.
- **Temperature** divides the logits: below 1 sharpens the distribution, above 1 flattens it,
  0 is effectively greedy.
- **top-p** samples from the smallest set of tokens whose probability sums to p; **top-k**
  from the k most likely. Some reasoning models fix these parameters or ignore them.

### Why temperature 0 is not deterministic

- **Floating-point addition is not associative**, and GPU kernels change summation order with
  batch composition, so a request's numerics depend on concurrent traffic
  ([Thinking Machines, 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
  Mixture-of-experts routing can depend on the batch too.
- **Small differences compound.** A flip between near-tied tokens changes everything after it.
- **Models change behind aliases.** Pin dated snapshot IDs. A `seed` parameter, where offered,
  is best-effort.

### Engineering consequences

- **Assert properties, not exact strings:** parses, matches the schema, required fields
  present, values in allowed sets.
- **Separate testing code from evaluating the model.** Unit tests replay recorded responses:
  fast, free, deterministic, and they cover parsing, errors and retries. Evals run many inputs,
  several times each, and report rates with uncertainty.
- **Input variation dominates.** Even a deterministic model needs many eval cases; one run on
  one input says nothing about the rest.
- **Reproduce through logging.** Record model snapshot, parameters, full prompt and response for
  every call.
- **Use variance deliberately.** Diverse samples enable voting, best-of-n selection, exploration
  after failures, synthetic data generation and RL rollouts.
- **Temperature:** low for extraction, classification and tool calls; higher for generating
  diverse candidates. Tune only against an eval.