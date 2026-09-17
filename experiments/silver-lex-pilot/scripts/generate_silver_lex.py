"""Generate silver LEX relations from e-SNLI human explanations.

For each row in esnli_sample.csv, calls an LLM (Gemini 3.1 Flash-Lite by
default, via OpenRouter) with a prompt that includes the premise, hypothesis,
and human explanation, asking it to extract the lexical entailment relations
the explanation implies (see ../prompts/silver_lex_extraction.txt). Requires
OPENROUTER_API_KEY in the environment (never pass it on the command line or
commit it). Output defaults to data/silver_lex_<model-slug>.csv so different
models' outputs never collide.
"""
import argparse
import asyncio
import csv
from pathlib import Path

from dotenv import load_dotenv

from kbprojection.async_runtime import resolve_async_run_context
from kbprojection.llm import AsyncGenericAIClient, _extract_validated_kb_from_output

PILOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PILOT_DIR / ".env")
DEFAULT_INPUT = PILOT_DIR / "data" / "esnli_sample.csv"
DEFAULT_PROMPT_TEMPLATE = PILOT_DIR / "prompts" / "silver_lex_extraction.txt"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"

RELATION_SEP = " ;; "


def model_slug(model: str) -> str:
    """'google/gemini-3.1-flash-lite' -> 'gemini-3.1-flash-lite' for filenames."""
    return model.rsplit("/", 1)[-1]


def read_rows(input_csv: Path) -> list[dict]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_done_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return {row["id"] for row in csv.DictReader(handle)}


async def generate_one(client: AsyncGenericAIClient, context, template: str, model: str, row: dict) -> dict:
    prompt = template.format(
        premise=row["premise"],
        hypothesis=row["hypothesis"],
        explanation=row["explanation"],
    )
    try:
        async with context.llm_semaphore:
            raw_output = await client.generate(prompt=prompt, model=model)
        relations = _extract_validated_kb_from_output(raw_output)
        return {
            "id": row["id"],
            "lex_relations": RELATION_SEP.join(relations),
            "raw_llm_output": raw_output,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - persist any provider/parsing failure as a row
        return {
            "id": row["id"],
            "lex_relations": "",
            "raw_llm_output": "",
            "error": str(exc),
        }


async def run(
    input_csv: Path,
    output_csv: Path,
    prompt_template_path: Path,
    provider: str,
    model: str,
    limit: int | None,
) -> None:
    rows = read_rows(input_csv)
    if limit is not None:
        rows = rows[:limit]
    done_ids = read_done_ids(output_csv)
    todo = [row for row in rows if row["id"] not in done_ids]

    print(f"{len(rows)} total rows, {len(done_ids)} already done, {len(todo)} to run")

    template = prompt_template_path.read_text(encoding="utf-8")
    client = AsyncGenericAIClient(provider=provider)
    context = resolve_async_run_context()

    write_header = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "lex_relations", "raw_llm_output", "error"])
        if write_header:
            writer.writeheader()

        tasks = [asyncio.create_task(generate_one(client, context, template, model, row)) for row in todo]
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            outcome = await task
            writer.writerow(outcome)
            handle.flush()
            status = outcome["error"] or outcome["lex_relations"] or "(empty KB)"
            print(f"[{completed}/{len(todo)}] {outcome['id']} -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to data/silver_lex_<model-slug>.csv, derived from --model.",
    )
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    args = parser.parse_args()
    output_csv = args.output_csv or PILOT_DIR / "data" / f"silver_lex_{model_slug(args.model)}.csv"
    asyncio.run(run(args.input_csv, output_csv, args.prompt_template, args.provider, args.model, args.limit))


if __name__ == "__main__":
    main()
