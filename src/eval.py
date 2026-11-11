"""Evaluation harness for the 10 dev questions.

Runs the agent on each question, executes the generated SQL, and compares the
executed RESULT SET against the expected result (each question's
"sql_result_match" evaluation) — SQL text is never string-compared.

Usage:
  uv run python -m src.eval                # full run, writes dev_answers.json
  uv run python -m src.eval --model <id>   # A/B another model
  uv run python -m src.eval --self-test    # run GOLD SQL through the executor
                                           # to validate the harness (no API key)
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.agent import AgentConfig, SqlAgent
from src.db import query

QUESTIONS_PATH = Path("data/dev_questions_with_answers.json")
ANSWERS_PATH = Path("dev_answers.json")
MAX_ANSWER_ROWS = 10


# ------------------------------------------------------------- comparison

def _normalize(value):
    """Make cell values comparable: round floats to 2dp, strip strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        return value.strip()
    return value


def _row_key(row: dict) -> tuple:
    """A row as an order- and column-name-insensitive fingerprint."""
    return tuple(sorted((repr(_normalize(v)) for v in row.values())))


def results_match(actual_rows: list[dict], expected_rows: list[dict]) -> bool:
    """Compare result sets as multisets of rows. Column names and row order are
    ignored: gold queries use ORDER BY with ties (q_008, q_009), so ordered
    comparison would false-fail correct answers; content is what matters."""
    return Counter(map(_row_key, actual_rows)) == Counter(map(_row_key, expected_rows))


# ------------------------------------------------------------- reporting

def format_answer(rows: list[dict]) -> str:
    """Human-readable one-line summary of a result set for dev_answers.json."""
    if not rows:
        return "No results."

    def fmt(v):
        return f"{v:,.2f}" if isinstance(v, float) else str(v)

    parts = []
    for row in rows[:MAX_ANSWER_ROWS]:
        values = list(row.values())
        if len(values) == 1:
            parts.append(fmt(values[0]))
        else:
            head, *rest = values
            parts.append(f"{fmt(head)} ({', '.join(fmt(v) for v in rest)})")
    summary = ", ".join(parts)
    if len(rows) > MAX_ANSWER_ROWS:
        summary += f", ... and {len(rows) - MAX_ANSWER_ROWS} more"
    return summary


# ------------------------------------------------------------- harness

BASELINE_PROMPT = "Convert this question to SQL: put the final SQLite query in a ```sql fence."


def run(self_test: bool = False, model: str | None = None, pause_s: float = 0.0,
        baseline: bool = False) -> int:
    questions = json.loads(QUESTIONS_PATH.read_text())
    config = AgentConfig(**({"model": model} if model else {}))

    answers: dict[str, dict] = {}
    records = []
    for q in questions:
        # Fresh agent per question: eval measures single-question accuracy,
        # so no conversation state may bleed between questions.
        agent = SqlAgent(config=config)
        if baseline:
            # Naive baseline: a one-line prompt, no schema, no repair.
            agent.system_prompt = BASELINE_PROMPT
            agent.config.max_repairs = 0
        if self_test:
            sql = q["gold_sql"]
            start = time.perf_counter()
            df = query(agent.conn, sql)
            latency, attempts = time.perf_counter() - start, 0
            rows, error = df.to_dict("records"), None
        else:
            result = agent.ask(q["question"])
            sql, latency, attempts, error = (
                result.sql, result.latency_s, result.attempts, result.error,
            )
            rows = result.df.to_dict("records") if result.ok else []

        passed = error is None and results_match(rows, q["expected_result"])
        records.append({"q": q, "passed": passed, "latency": latency,
                        "attempts": attempts, "error": error,
                        "tokens": (0, 0) if self_test else
                                  (result.prompt_tokens, result.completion_tokens)})
        answers[q["id"]] = {"sql": " ".join(sql.split()),
                            "answer": format_answer(rows) if error is None
                                      else f"FAILED: {error}"}
        status = "PASS" if passed else "FAIL"
        print(f"{q['id']}  tier {q['tier']}  {status}  {latency:5.2f}s  "
              f"attempts={attempts}  {q['question'][:60]}")
        if pause_s:
            time.sleep(pause_s)

    # ---- summary
    latencies = sorted(r["latency"] for r in records)
    n = len(latencies)
    p50 = statistics.median(latencies)
    p90 = latencies[min(n - 1, int(0.9 * n))]
    passed_n = sum(r["passed"] for r in records)
    print("\n" + "=" * 72)
    in_tok = sum(r["tokens"][0] for r in records)
    out_tok = sum(r["tokens"][1] for r in records)
    print(f"Accuracy: {passed_n}/{n}   Latency P50: {p50:.2f}s  P90: {p90:.2f}s  "
          f"max: {latencies[-1]:.2f}s   Model: {config.model.split('/')[-1]}")
    if in_tok:
        print(f"Tokens: {in_tok} in / {out_tok} out total  "
              f"(avg {in_tok//n} in + {out_tok//n} out per question)")
    for r in records:
        if not r["passed"]:
            why = r["error"] or "result mismatch vs expected_result"
            print(f"  FAIL {r['q']['id']}: {why}")

    if not self_test and not baseline:
        ANSWERS_PATH.write_text(json.dumps(answers, indent=2) + "\n")
        print(f"Wrote {ANSWERS_PATH} ({len(answers)} answers)")
    return passed_n


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run gold SQL through the executor to validate the harness")
    parser.add_argument("--model", help="model id override (default: LLM_MODEL)")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to sleep between questions (free-tier rate limits)")
    parser.add_argument("--baseline", action="store_true",
                        help="run a naive one-line prompt (no schema, no repair) for comparison")
    args = parser.parse_args()
    run(self_test=args.self_test, model=args.model, pause_s=args.pause,
        baseline=args.baseline)


if __name__ == "__main__":
    main()
