# Phases

Each phase lives in its own folder (`01-llm-components/`, `02-agent-loop/`, ...) and is a uv
workspace member with its own `pyproject.toml`, so its dependencies stay isolated.

Layout of a phase folder:

```
NN-name/
├── pyproject.toml
├── NOTES.md      # concepts in your own words + checkpoint answers
├── src/          # lab code
├── tests/
└── results/      # benchmarks, eval outputs, charts (summaries, not raw dumps)
```