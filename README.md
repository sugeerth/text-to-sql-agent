# Text-to-SQL Agent

An interactive CLI that turns plain-English questions into SQL, runs it against a SQLite
database, and shows you the query and the results. Ask a follow-up ("now only for Brazil")
and it refines the last query.

One idea drives the design: **the LLM does the reasoning; deterministic code does the
safety and control.** Works with **any OpenAI-compatible LLM endpoint** — a hosted
provider, OpenAI, or a local server (vLLM, SGLang, Ollama).

**Walkthrough page:** https://sugeerth.github.io/text-to-sql-agent/ — the pipeline, how
results are graded, what moved accuracy from 2/10 to 10/10, retries, latency and cost.

```
question ─> prompt: schema + FK map + 3 sample rows/table + history
         ─> LLM writes one SQL query            (1 call on the happy path)
         ─> guard: exactly one SELECT / WITH    (plain code)
         ─> execute on a read-only connection   (plain code)
         ─> SQL + result table + timing
              ↑ on guard rejection or SQLite error, the exact error text goes back
                to the LLM to repair — max 2 repairs, then it fails honestly
```

## Results

10 questions over the [Chinook](https://github.com/lerocha/chinook-database) sample database
(difficulty tiers 1–3), graded by executing the SQL and comparing result sets:

| Configuration | Accuracy | P50 latency |
|---|---|---|
| Naive one-line prompt ("Convert this question to SQL"), no schema | 2/10 | 1.1 s |
| This agent · DeepSeek V4 Flash, reasoning off · two consecutive runs | **10/10** | 0.86–1.30 s |
| This agent · Nemotron Lightning 30B (A/B) | **10/10** | 0.31 s |

About 1,935 input + 55 output tokens per question, one model call per question in the final
runs. Ten questions are enough to steer, not to prove.

What closed the gap: schema-aware prompting removed the hallucinated tables; turning hidden
reasoning off cut P50 from 8.9 s to 1.3 s and stopped it eating the output budget; and two
output-convention rules (never select id columns; listings keep raw columns, per-person
aggregations use one name column) removed the remaining column-shape mismatches.

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync                  # install dependencies
uv run setup-db          # download the Chinook sample database to data/Chinook.db
cp .env.example .env     # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
uv run cli               # interactive session
```

Inside the CLI: ask questions in plain English; `/schema [table]`, `/reset`, `/help`, `exit`.
`uv run cli --model <id>` tries another model; `uv run cli --db path/to.db` points it at another
SQLite database (the prompt's output rules were tuned on Chinook).

## Evaluation

```bash
uv run python -m src.eval              # agent on all 10 questions -> dev_answers.json; accuracy, P50/P90, tokens
uv run python -m src.eval --self-test  # gold SQL through the same harness — validates the grader, no API key
uv run python -m src.eval --baseline   # the naive one-line prompt, for comparison
uv run python -m src.eval --model <id> # A/B another model
uv run python -m src.eval --pause 6    # pace requests for rate-limited keys
```

Result sets are compared as unordered multisets (column names ignored, floats to 2 dp),
because the gold answers contain rank ties and two different playlists named "Music".
`dev_answers.json` is the output of a 10/10 run.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `LLM_BASE_URL` | yes* | OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1` |
| `LLM_API_KEY` | yes | API key (any non-empty string for local servers) |
| `LLM_MODEL` | yes | model id |
| `LLM_REASONING_EFFORT` | no | sent as `reasoning_effort` for reasoning models; the measured runs used `none` |

\* `OPENAI_BASE_URL` / `OPENAI_API_KEY` work as fallbacks.

## Project structure

```
src/
  cli.py        terminal REPL: SQL, result table, timing; /schema /reset /help
  agent.py      the loop: generate -> guard -> execute -> repair; conversation memory
  prompts.py    system prompt + schema context (columns, FK arrows, sample rows)
  db.py         SQLite helpers: read-only connection, queries, schema introspection
  eval.py       grader: executes SQL, compares result sets, reports latency + tokens
  setup_db.py   downloads the Chinook sample database
data/           10 questions with gold SQL and expected results
docs/           the walkthrough page (published at the link above)
```

## Why not a tool-calling agent?

Same observe-and-repair behavior, but every question would cost 2–3 model round trips
(~3–5 s) instead of one (~1 s). The four stages are already the tool boundaries: turning
execution into a model-invoked tool for large schemas is a small change.

## License

MIT (see [LICENSE](LICENSE)). The Chinook sample database is downloaded at setup from
[lerocha/chinook-database](https://github.com/lerocha/chinook-database) and is not
redistributed here. Built with Claude (Anthropic) as a coding assistant.
