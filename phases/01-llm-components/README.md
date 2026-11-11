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

## 1.4 Messages, system prompts and prompting

### Anatomy of a request

- **Model** (a dated snapshot), **system prompt** (standing instructions), **messages**
  (alternating `user`/`assistant` turns made of content blocks: text, images, tool calls, tool
  results), **tools** (Phase 2) and **parameters** (`max_tokens`, temperature, stop sequences).
- The response carries content blocks, a **stop reason** and **usage** (token counts).
- Providers differ in shape: where the system prompt goes, how tool calls and results are
  structured. A shared client normalizes this.
- The provider renders everything into one token sequence using the model's **chat template**.
  Roles carry trained authority — system above user above tool results — an **instruction
  hierarchy** that matters for security (Phase 8).
- The API is stateless: the application stores the conversation and decides what each call
  includes.

### The system prompt as policy

Cover role and context, the task and what success looks like, constraints with their reasons,
output format, and when to use tools. Aim between brittle if/else rules and vague slogans:
clear principles plus the reasoning behind them.

### Prompting principles

- **Be clear and direct.** Brief the model like a capable colleague new to the project.
- **Give the reason for each rule.** The model applies reasons to cases the rule never names,
  and they supply an honest alternative to the forbidden action.
- **State what to do, not only what to avoid.** Naming the unwanted behavior can prime it.
- **Use varied examples.** Models copy every shared feature of their examples — length, tone,
  format, label balance — not just the intended pattern. Include hard cases; shuffle order.
- **Structure long prompts** with XML tags or headings to keep instructions, data and examples
  separate.
- **Use structured outputs** (1.5) for anything code will parse.
- **Treat prompts as code.** Version them, change one thing at a time, judge changes with evals.

Example. Weak: "Don't be verbose. Don't use bullet points. Don't mention competitors."
Stronger:

> You answer customers of Acme Billing in a chat widget shown on phone screens. Reply in two to
> four short sentences of plain prose; long or list-formatted replies are hard to read in the
> widget. Help only with Acme's products. If a customer asks about another vendor, say you can
> only speak to Acme and point them to the Acme feature that meets their need, because
> comparisons we can't verify may be inaccurate.

## 1.5 Structured outputs

Code that acts on model output needs that output in a known shape.

### Levels of guarantee

| Approach | Guarantees | Does not guarantee |
| --- | --- | --- |
| Prompt only ("return JSON") | Nothing | Parsing, shape |
| JSON mode | Valid JSON | Shape: fields may be missing or extra |
| Schema-constrained decoding | Valid JSON matching the schema | Correct values |
| Forced tool call | Arguments matching the tool schema | Correct values |

Prefer schema-constrained decoding where the provider supports it.

### How constrained decoding works

The JSON Schema is compiled into a grammar; at each step the provider masks every token that
would violate it. Limits:

- Only a subset of JSON Schema is supported; unsupported keywords may be rejected or
  silently ignored.
- A new schema adds compilation latency on first use; providers cache it.
- It guarantees shape, not truth.
- Truncation (`max_tokens`) or a refusal still yields incomplete output. Check the stop
  reason before parsing.

### One schema definition

Define the shape once as a Pydantic model: generate the JSON Schema for the request, validate
the response into the same model. Validate what the schema cannot express (cross-field rules,
values that must exist elsewhere) in code, and on failure run a bounded repair loop: send the
validation error back and retry once or twice.

### Schema design

- **Descriptions are prompt text.** Write them as instructions.
- **Use enums for categories**, lowercase, each value described. Include a value for cases
  that don't fit cleanly (e.g. `mixed`) instead of forcing a choice.
- **Represent "not present" as `null`, not a sentinel string.** `due_date: date | None` keeps
  the type honest; a string like `"unknown"` breaks parsing and leaks into stored data.
  Required fields without an escape hatch invite fabrication.
- **Order fields deliberately.** Generation follows schema order; put evidence or reasoning
  before the answer it should inform.
- **Keep schemas small and flat.**
- **Rigid formats can reduce reasoning quality**
  ([Tam et al., 2024](https://arxiv.org/abs/2408.02442)). For hard tasks, reason first, then
  emit the structure.

## 1.6 Reasoning models

### What changes

Reasoning models generate a chain of thought before answering, trained with reinforcement
learning on verifiable tasks. They trade test-time compute for quality.

- **Thinking tokens are billed as output** and add latency; they can far exceed the visible answer.
- **Effort is controllable** through a token budget or effort level, depending on the provider;
  some models adapt it automatically.
- **Visibility varies:** full reasoning, a summary, or nothing.
- **Tool loops carry reasoning across turns.** Models can think between tool calls, and
  providers may require earlier reasoning to be passed back unchanged. Follow each provider's
  rules and log usage to see what is billed.

### Cost impact

The 1.1 agent (25 steps) with 2,000 thinking tokens per step:

| Scenario | Input | Output | Total |
| --- | --- | --- | --- |
| No thinking | $1.69 | $0.11 | $1.80 |
| Thinking, not re-sent | $1.69 | $0.86 | $2.55 |
| Thinking re-sent as history | $3.64 | $0.86 | $4.50 |

Output grows from 6% to a third of the bill, and reasoning that accumulates in history joins
the n² term.

### When to use one

| Worth it | Usually not |
| --- | --- |
| Multi-step logic, math, code | Extraction, classification, formatting |
| Planning, task decomposition | Latency-critical steps |
| Ambiguous tool choice | High-volume repetitive steps |
| Complex or conflicting instructions | Tasks a small model already handles |

- **Route per step.** Reasoning model for judgment, smaller models for routine steps, plain
  code for anything deterministic such as lookups.
- **Make judgments checkable.** Ask for structured decisions that cite their sources.
- **Set effort with an eval**, not intuition.

### Prompting

- Give the goal, constraints and success criteria; don't script the steps. Detailed procedures
  lock the model into one method and spend its thinking on compliance. Keep mandatory steps only
  where required, stated with their reason.
- "Think step by step" is redundant. Fewer examples are needed.
- **Visible reasoning is not guaranteed to be faithful** to what drove the answer
  ([Anthropic, 2025](https://www.anthropic.com/research/reasoning-models-dont-say-think)).
  Use it for debugging, not as an explanation.

## 1.7 Prompt caching and latency

### Prompt caching

Providers can reuse the prefill computation of a prompt prefix seen recently. Reused tokens are
much cheaper and faster.

- **Anthropic:** explicit cache breakpoints. Writes cost a premium (about 1.25× base input for
  the default 5-minute lifetime, more for longer lifetimes), reads about 0.1×. Hits refresh the
  lifetime.
- **OpenAI:** automatic above a minimum length; the discount varies by model.
- **Matching is by exact prefix.** Reuse stops at the first differing token, so one changing
  token early in the request prevents caching everything after it.

Order requests from most to least stable:

```
tool definitions → system prompt → long-lived context → conversation history → newest message
```

Keep timestamps, request IDs and per-user details late, and at the coarsest precision the task
needs. Keep tool order fixed and serialize JSON deterministically.

### Caching in agent loops

Agent history is append-only, so each call's prompt is the previous call's plus one step: most
input becomes cache reads. The 1.1 agent with reads at 0.1× and writes at 1.25×:

| | Tokens | Cost |
| --- | --- | --- |
| Cache reads | 522,000 | $0.16 |
| Cache writes | 40,500 | $0.15 |
| Output | 7,500 | $0.11 |
| **Total** | | **$0.42** (vs $1.80 uncached) |

- The fixed prefix S becomes nearly free; new per-step content k still matters.
- Pauses longer than the cache lifetime (human approval) force a full rewrite. Options: a longer
  lifetime at a higher write price, compacting before pausing, fewer approval points, or
  accepting the rewrite when pauses are rare.

### Latency

```
latency ≈ time to first token (queue + prefill) + output tokens ÷ output speed
```

- Time to first token grows with input; caching reduces it.
- **Output length usually dominates.** 800 tokens at 80 tokens/s is 10 s of decoding against
  well under a second of prefill.
- Agent steps run in sequence, so total time is the sum of calls plus tool time.
- Levers, roughly by impact: fewer steps, shorter outputs (including thinking), faster models
  for routine steps, parallel independent calls, caching, and streaming for perceived latency.
- Measure p50 and p95, not means. Batch APIs trade hours of delay for about 50% off — suited to
  evals and offline jobs.