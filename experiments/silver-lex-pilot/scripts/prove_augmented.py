"""Re-run LangPro with each row's silver LEX relations injected.

Joins esnli_sample.csv (premise/hypothesis) with a silver_lex_<model>.csv
(relations from generate_silver_lex.py) on id, then calls the remote LangPro
API once per row with kb=<that row's silver relations>. Rows with an empty
silver KB hit the same cache key as the baseline call, so they resolve
instantly at no extra API cost. Resumable: existing rows in the output CSV
are skipped. Output defaults to data/augmented_<model-slug>.csv, derived from
--silver-csv's own name, so different models' outputs never collide.
"""
import argparse
import asyncio
import csv
from pathlib import Path

from kbprojection.langpro import langpro_api_call

PILOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_CSV = PILOT_DIR / "data" / "esnli_sample.csv"
DEFAULT_SILVER_CSV = PILOT_DIR / "data" / "silver_lex_gemini-3.1-flash-lite.csv"
RELATION_SEP = " ;; "


def output_name_for_silver_csv(silver_csv: Path) -> str:
    """'silver_lex_<slug>.csv' -> 'augmented_<slug>.csv'."""
    stem = silver_csv.stem
    slug = stem.removeprefix("silver_lex_")
    return f"augmented_{slug}.csv"


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_done_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


def join_sample_with_silver(sample_rows: list[dict], silver_rows: list[dict]) -> list[dict]:
    silver_by_id = {row["id"]: row for row in silver_rows}
    joined = []
    for row in sample_rows:
        silver = silver_by_id.get(row["id"])
        if silver is None:
            kb = []
            source_note = "no matching row for this id in the silver_lex CSV"
        elif silver["error"]:
            kb = []
            source_note = f"silver_lex generation error: {silver['error']}"
        else:
            kb = [r.strip() for r in silver["lex_relations"].split(RELATION_SEP) if r.strip()]
            source_note = ""
        joined.append(
            {
                "id": row["id"],
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "kb": kb,
                "source_note": source_note,
            }
        )
    return joined


async def prove_one(row: dict) -> dict:
    result = await langpro_api_call(
        premises=[row["premise"]],
        hypothesis=row["hypothesis"],
        kb=row["kb"],
    )
    return {
        "id": row["id"],
        "result": result.label.value,
        "kb_used": RELATION_SEP.join(row["kb"]),
        "error": result.error or row["source_note"],
    }


async def run(sample_csv: Path, silver_csv: Path, output_csv: Path, limit: int | None) -> None:
    sample_rows = read_rows(sample_csv)
    silver_rows = read_rows(silver_csv)
    joined = join_sample_with_silver(sample_rows, silver_rows)
    if limit is not None:
        joined = joined[:limit]

    done_ids = read_done_ids(output_csv)
    todo = [row for row in joined if row["id"] not in done_ids]

    print(f"{len(joined)} total rows, {len(done_ids)} already done, {len(todo)} to run")

    write_header = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "result", "kb_used", "error"])
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
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--silver-csv", type=Path, default=DEFAULT_SILVER_CSV)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to data/augmented_<model-slug>.csv, derived from --silver-csv.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    args = parser.parse_args()
    output_csv = args.output_csv or PILOT_DIR / "data" / output_name_for_silver_csv(args.silver_csv)
    asyncio.run(run(args.sample_csv, args.silver_csv, output_csv, args.limit))


if __name__ == "__main__":
    main()
