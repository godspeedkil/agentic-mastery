# Agentic Mastery

A hands-on path from LLM fundamentals to production-grade, evaluated and RL-trained agents.
Each phase is a self-contained guide (its folder's README) followed by a lab built step by step.
The git history is the record: one branch and PR per phase, tagged `phase-NN` when complete.

## Progress

| # | Phase | Lab deliverable | Status |
| --- | --- | --- | --- |
| 0 | Repo setup | Workspace, tooling, CI, hooks | Done |
| 1 | LLMs as components | Robust multi-provider client with cost ledger | Done |
| 2 | Tool use and the agent loop | Framework-free agent in under 200 lines | Not started |
| 3 | Context engineering and memory | Research agent with compaction and memory | Not started |
| 4 | Workflow patterns | One task, three architectures, compared | Not started |
| 5 | Protocols and toolchains | MCP server + A2A-exposed agent | Not started |
| 6 | Multi-agent systems | Orchestrator with parallel subagents | Not started |
| 7 | Evaluation and observability | Eval harness with calibrated judge + CI gate | Not started |
| 8 | Reliability and security | Red-team report with before/after attack rates | Not started |
| 9 | Production engineering | Deployed, resumable, cost-monitored service | Not started |
| 10 | Training and optimizing agents | RL-tuned small model vs. prompted frontier model | Not started |
| — | Capstone | Public write-up | Not started |

## Repo layout

```
core/       shared code that grows across phases (LLM client, tracing, utilities)
phases/     one folder per phase: README.md, src/, tests/, results/
capstone/   final project
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages        # creates .venv with Python 3.12 and all workspace packages
uv run pre-commit install     # once per clone: ruff on commit, tests on push
cp .env.example .env          # then add your API keys
uv run poe check              # lint, format check and tests — exactly what CI runs
```

## Everyday commands

| Command | What it does |
| --- | --- |
| `uv run poe` | List all tasks |
| `uv run poe check` | Everything CI runs |
| `uv run poe fix` | Auto-fix lint issues and format |
| `uv run poe test` | Tests only |

## Conventions

- Branch per phase: `phase-NN-name`, merged via PR (merge commit, not squash), tagged `phase-NN`.
- Conventional Commits with a phase scope, e.g. `feat(p02): add loop detection`.
- One job per commit. A typical phase: `docs` (guide sections) → `feat` (lab core) →
  `test` (tests + results) → `refactor`/`perf` (improvement round) → `docs` (results and findings).
- Code reused by later phases moves into `core/` in its own `refactor(core)` commit.

## License

[MIT](LICENSE)