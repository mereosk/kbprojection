"""Baseline LangPro proving pass, WordNet-only KB, no injected relations.

Reads esnli_sample.csv (id, premise, hypothesis, explanation) and calls the
remote LangPro API once per row with kb=None. Results are cached by
kbprojection's own LangPro cache, so reruns are cheap. Resumable: existing
rows in the output CSV are skipped.
"""
import argparse
import asyncio
import csv
from pathlib import Path

from kbprojection.langpro import langpro_api_call

PILOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PILOT_DIR / "data" / "esnli_sample.csv"
DEFAULT_OUTPUT = PILOT_DIR / "data" / "baseline.csv"


def read_rows(input_csv: Path) -> list[dict]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_done_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


async def prove_one(row: dict) -> dict:
    result = await langpro_api_call(
        premises=[row["premise"]],
        hypothesis=row["hypothesis"],
        kb=None,
    )
    return {
        "id": row["id"],
        "result": result.label.value,
        "error": result.error or "",
    }


async def run(input_csv: Path, output_csv: Path, limit: int | None) -> None:
    rows = read_rows(input_csv)
    if limit is not None:
        rows = rows[:limit]
    done_ids = read_done_ids(output_csv)
    todo = [row for row in rows if row["id"] not in done_ids]

    print(f"{len(rows)} total rows, {len(done_ids)} already done, {len(todo)} to run")

    write_header = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "result", "error"])
        if write_header:
            writer.writeheader()

        tasks = [asyncio.create_task(prove_one(row)) for row in todo]
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            outcome = await task
            writer.writerow(outcome)
            handle.flush()
            print(f"[{completed}/{len(todo)}] {outcome['id']} -> {outcome['result']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    args = parser.parse_args()
    asyncio.run(run(args.input_csv, args.output_csv, args.limit))


if __name__ == "__main__":
    main()
