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

### In the code

The client turns each provider's usage report into one shape (`core/src/core/llm/records.py`,
condensed):

```python
class Usage(BaseModel):
    input_tokens: NonNegativeInt = 0  # base input rate
    cache_read_tokens: NonNegativeInt = 0  # cache-read rate
    cache_write_tokens: NonNegativeInt = 0  # cache-write rate
    output_tokens: NonNegativeInt = 0  # output rate, reasoning included
    reasoning_tokens: NonNegativeInt = 0  # the reasoning share of output_tokens
```

The three input counts never overlap, so each input token gets exactly one rate. Reasoning is
billed as output, so it sits inside `output_tokens` instead of being added on top.

Providers report this differently. OpenAI counts cached tokens inside `input_tokens`, so its
adapter takes them out (`core/src/core/llm/openai_provider.py`):

```python
uncached = usage.input_tokens - details.cached_tokens - details.cache_write_tokens
```

Pricing is the per-million arithmetic from Billing (`core/src/core/llm/pricing.py`):

```python
total = usage.input_tokens * self.input + usage.output_tokens * self.output
for tokens, rate in [
    (usage.cache_read_tokens, self.cache_read),
    (usage.cache_write_tokens, self.cache_write),
]:
    if tokens == 0:
        continue
    if rate is None:
        return None
    total += tokens * rate
return total / TOKENS_PER_UNIT
```

A missing rate returns `None`, not 0, so an unknown cost never passes for a cheap one.

Models are compared per completed task (`phases/01-llm-components/src/p01/analysis.py`):

```python
"cost_per_1k_usd": group["cost_usd"].mean() * 1000,
```

Each row is one finished extraction, with the cost of all its calls summed, repair turns
included. That makes this cost per task, not price per token.


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

### In the code

Output has its own cap, set on every request (`core/src/core/llm/types.py`):

```python
max_tokens: PositiveInt = 1024
```

The benchmark raises it, because reasoning spends the same budget
(`phases/01-llm-components/src/p01/run.py`):

```python
MAX_TOKENS = 8192
"""Generous, because current models think before answering and thinking counts as output."""
```

Hitting a limit is never silent. Each adapter maps its provider's reasons to one normalized
stop reason; Anthropic reports a full context window separately, and it lands in the same
bucket (`core/src/core/llm/anthropic_provider.py`):

```python
"max_tokens": "max_tokens",
"model_context_window_exceeded": "max_tokens",
```

A cut-off answer is refused rather than half-parsed (`core/src/core/llm/client.py`):

```python
if response.stop_reason in UNPARSEABLE_STOP_REASONS:  # {"max_tokens", "refusal"}
    raise StructuredOutputError(...)
```

The document goes in tags, with the instructions kept apart in the system prompt
(`phases/01-llm-components/src/p01/run.py`):

```python
document = f"<document>\n{case.text}\n</document>"
return Request(
    model=model.id, system=SYSTEM_PROMPT, messages=[Message.user(document)], max_tokens=MAX_TOKENS
)
```

Hard constraints are enforced in code: the prompt asks for the schema, and the client accepts
only what validates against it (`core/src/core/llm/client.py`):

```python
value = output_type.model_validate_json(response.text)
```

Every request here is one short document, so nothing in this phase gets near the window.
Degradation with length shows up once agents accumulate history.

## 1.3 Sampling and non-determinism

### Decoding

At each step the model produces a score (logit) for every vocabulary token; softmax turns these
into probabilities, and a decoding strategy picks one token.

- **Greedy** takes the most likely token. **Sampling** draws from the distribution.
- **Temperature** divides the logits: below 1 sharpens the distribution, above 1 flattens it,
  0 is effectively greedy.
- **top-p** samples from the smallest set of tokens whose probability sums to p; **top-k**
  from the k most likely.

### Sampling controls in current APIs

- Anthropic has removed `temperature`, `top_p` and `top_k`. Claude 4.7 and later models
  reject non-default values, and API version 2026-10-21 rejects them on every model; requests
  sample at the defaults, temperature 1.0
  ([docs](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)).
- Reasoning models from other providers often fix or restrict these parameters too. Check
  each model's documentation.
- The remaining levers are prompting (style, variety) and effort settings (how much the model
  reasons, 1.6). The mechanics above still apply to open-weight models you serve yourself.

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
- **Steer with prompts and effort where temperature is unavailable.** Where it is available:
  low for extraction, classification and tool calls; higher for diverse candidates. Tune only
  against an eval.

### In the code

`Request` has no sampling parameters, and it rejects unknown fields, so passing one fails
before anything is sent (`core/src/core/llm/types.py`):

```python
class Request(BaseModel):
    """What to send. Sampling parameters are deliberately absent: current models reject them."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)
```

`Request(model="m", messages=[...], temperature=0)` raises a `ValidationError`.

Models change behind aliases, so the response keeps the ID the provider actually served:

```python
model: str
"""Model ID reported by the provider, which may differ from the requested alias."""
```

Unit tests replay recorded provider responses instead of calling the API
(`core/tests/test_llm_anthropic_provider.py`):

```python
def load_message(name: str, **overrides) -> AnthropicMessage:
    data = json.loads((FIXTURES / name).read_text())
    return AnthropicMessage.model_validate(data | overrides)
```

The benchmark is the eval: every case runs several times
(`phases/01-llm-components/src/p01/run.py`):

```python
jobs = [(model, case, r) for model in models for case in cases for r in range(1, repeats + 1)]
```

Scoring asserts properties, not exact strings: names compare on letters and digits only, and
totals within half a cent (`phases/01-llm-components/src/p01/scoring.py`):

```python
"vendor": normalize_name(predicted.vendor) == normalize_name(expected.vendor),
"total": abs(predicted.total - expected.total) < CENT,
```

Rates come with uncertainty, counted over cases rather than calls, because repeats of one case
are correlated and input variation dominates (`phases/01-llm-components/src/p01/analysis.py`):

```python
low, high = wilson_interval(group["exact"].mean(), n=per_case.ngroups)
...
"all_repeats_correct": per_case.all().mean(),   # reliable every time (pass^k)
"any_repeat_correct": per_case.any().mean(),    # right at least once (pass@k)
```

Not yet done: the call ledger records the model, usage and stop reason of every call, but not
the full prompt and response, so a single call can't be replayed from the logs.

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