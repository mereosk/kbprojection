"""Ablation: do e-SNLI human explanations help the paper's LEX prompt?

Runs the paper's Lasha Plus Precision prompt (`lasha`) on the 280 gold items
that have an e-SNLI explanation, in three conditions:

  A  lasha              premise + hypothesis (the paper's prompt, unchanged)
  B  lasha_explanation  same prompt + one label-neutral sentence allowing use
                        of the human explanation, the explanation in the input
                        block (as "human justification", since the examples
                        already use "explanation:" for unwanted-output notes)
  B2 lasha_explanation_examples
                        B with a stricter instruction (extract and reformat,
                        don't copy the justification's wording) + a
                        hand-written "human justification" line in few-shot
                        examples 2, 6 and 8, so the model sees the field used
  C  lasha_highlights   same prompt + one label-neutral sentence allowing use
                        of the human highlights, and annotator 1's highlighted
                        premise/hypothesis words in the input block

B and C each differ from A only in the final input block and one sentence
before it, and B2 additionally in its three example lines; build_templates()
refuses to run if anything else differs. Existing outputs are reused, so adding a
condition only calls the API for that condition.

Protocol matches the paper's intrinsic LEX evaluation
(scripts/experiments/run_repeated_multi_reference_experiment.py): temperature
0, repeated runs, relations scored as generated (no KB filtering), and the
multi-reference micro-F1 with non-entailment answers excluded. Generation,
output format, consistency metrics and F1 scoring are reused from that script.

Extra outputs: F1 with non-entailment answers counted as wrong, a paired
bootstrap over items for the B - A difference, and the paper's saved A outputs
for the same model rescored on the same 280 items (no API calls).

--dry-run uses a canned response instead of the API (free, offline).
"""
import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PILOT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from dotenv import load_dotenv  # noqa: E402

from calculate_multi_reference_f1 import evaluate_prediction_column, model_prediction_exclusion  # noqa: E402
from kbprojection.filtering import FILTERING_PIPELINE_VERSION  # noqa: E402
from kbprojection.llm import AsyncGenericAIClient, _extract_validated_kb_from_output  # noqa: E402
from kbprojection.prompts import _insert_before_marker, get_prompt  # noqa: E402
from run_repeated_multi_reference_experiment import (  # noqa: E402
    DEFAULT_REFERENCE_COLUMNS,
    OUTPUT_FIELDNAMES,
    build_output_rows,
    call_with_retries,
    format_kb,
    is_done,
    load_existing_outputs,
    model_slug,
    read_rows,
    write_f1_metrics,
    write_metrics,
    write_rows,
)

load_dotenv(PILOT_DIR / ".env")

# gold362_esnli_matched.csv + annotator-1 highlights (scripts/add_esnli_highlights.py)
DEFAULT_INPUT = PILOT_DIR / "data" / "gold362_esnli_highlights.csv"
PAPER_OUTPUTS = (
    REPO_ROOT / "experiment_results" / "lasha_all362_5runs"
    / "small_medium_lasha_all362_5runs_no_filter_outputs.csv"
)
MARKER = "Now process the following input while strictly following the above instructions and formatting."
EXPLANATION_INSTRUCTION = (
    "The input also includes a human-written explanation of the premise-hypothesis "
    "pair. You may use it to identify the lexical relations needed, but all the rules "
    "above still apply."
)
STRICT_EXPLANATION_INSTRUCTION = (
    "The input also includes a human-written justification of the premise-hypothesis "
    "pair: a human's informal reasoning. Extract the lexical relation(s) it implies, "
    "but reformat them to satisfy every rule above; do not copy wording from the "
    "justification directly."
)
HIGHLIGHTS_INSTRUCTION = (
    "The input also includes the words a human annotator highlighted in the premise "
    "and the hypothesis as important for the premise-hypothesis pair. You may use them "
    "to identify the lexical relations needed, but all the rules above still apply."
)
INPUT_BLOCK = "\thypothesis: ${HYPOTHESIS}\ncorrect output:"
EXPLANATION_FIELDS = "\thuman justification: ${EXPLANATION}\n"
HIGHLIGHTS_FIELDS = (
    "\thighlighted premise words: ${PREMISE_HIGHLIGHTS}\n"
    "\thighlighted hypothesis words: ${HYPOTHESIS_HIGHLIGHTS}\n"
)
# Hand-written (by the user) for B2, keyed by each example's hypothesis line.
EXAMPLE_JUSTIFICATIONS = {
    "\t\thypothesis: A woman gets out of the pool.\n": "a female swimmer is a type of woman",
    "\t\thypothesis: The man in a hat is climbing a rope.\n": "a cap is a hat and a cord is basically a rope",
    "\t\thypothesis: A fat person is holding a small animal.\n": "obese means fat and a chinchilla is a small animal",
}
CONDITION_A, CONDITION_B, CONDITION_C = "lasha", "lasha_explanation", "lasha_highlights"
CONDITION_B2 = "lasha_explanation_examples"
CONDITIONS = [CONDITION_A, CONDITION_B, CONDITION_B2, CONDITION_C]
FILE_LABELS = {CONDITION_A: "A", CONDITION_B: "B", CONDITION_B2: "B2", CONDITION_C: "C"}
# v3 "+label": the gold label (entailment) is given, the model only outputs relations.
LABEL_SUFFIX = "_label"
# Control: the A+label prompt unchanged, sampled at temperature 0.7 (vs --temperature, default 0).
CONDITION_A_LABEL_T07 = CONDITION_A + LABEL_SUFFIX + "_t0.7"
CONDITION_TEMPERATURES = {CONDITION_A_LABEL_T07: 0.7}
LABEL_CONDITIONS = [c + LABEL_SUFFIX for c in (CONDITION_A, CONDITION_B, CONDITION_C)] + [CONDITION_A_LABEL_T07]
LABEL_FILE_LABELS = {c + LABEL_SUFFIX: f"v3_{FILE_LABELS[c]}_label" for c in (CONDITION_A, CONDITION_B, CONDITION_C)}
LABEL_FILE_LABELS[CONDITION_A_LABEL_T07] = "v3_A_label_t0.7"
# (old, new, expected count) applied to the paper's lasha prompt; each must match exactly.
LABEL_EDITS = [
    (
        "You will reason carefully and decide whether the premise entails the hypothesis, which means that",
        "The premise entails the hypothesis, which means that",
        1,
    ),
    ('If the answer is "entailment", output a structured explanation', "Output a structured explanation", 1),
    (
        "Example 3:\n\tinput:\n\t\tpremise: A young girl wearing a pink coat plays with a yellow toy.\n"
        "\t\thypothesis: A kid is swinging a toy golf club.\n\tcorrect output:\n\t\tanswer: non-entailment\n\n",
        "",
        1,
    ),
    ("\t\tanswer: entailment\n", "", 8),
    (
        "Output exactly in one of the following formats:\n\tanswer: entailment\n\trelations: { ... }\n"
        "\tor\n\tanswer: non-entailment\n",
        "Output exactly in the following format:\n\trelations: { ... }\n",
        1,
    ),
]
NO_HIGHLIGHTS = "(none)"
DRY_RUN_RESPONSE = "answer: entailment\nrelations: { }"
LABEL_DRY_RUN_RESPONSE = "relations: { entails(field, grass) }"
NO_FILTER_CONFIG = json.dumps(None)


def derive(base: str, instruction: str, fields: str) -> str:
    """Base prompt + one sentence before MARKER + extra lines after the target hypothesis.

    Raises if removing exactly those additions does not give back the base prompt,
    so a condition can never touch the few-shot examples or anything else.
    """
    if base.count(INPUT_BLOCK) != 1:
        raise ValueError("Unexpected lasha input block; refusing to build a variant.")
    added_sentence = f"{instruction}\n\n{MARKER}"
    variant = _insert_before_marker(base, MARKER, instruction)
    variant = variant.replace(INPUT_BLOCK, INPUT_BLOCK.replace("\ncorrect", f"\n{fields}correct"))
    restored = variant.replace(added_sentence, MARKER).replace(fields, "")
    if variant.count(added_sentence) != 1 or variant.count(fields) != 1 or restored != base:
        raise ValueError("Variant differs from the base prompt beyond its two additions.")
    return variant


def add_example_justifications(prompt: str) -> str:
    """Add one "human justification" line after each listed example's hypothesis.

    Raises if removing exactly those lines does not give back the input prompt.
    """
    variant = prompt
    added = []
    for hypothesis_line, justification in EXAMPLE_JUSTIFICATIONS.items():
        if variant.count(hypothesis_line) != 1:
            raise ValueError(f"Few-shot example not found once: {hypothesis_line.strip()}")
        line = f"\t\thuman justification: {justification}\n"
        variant = variant.replace(hypothesis_line, hypothesis_line + line)
        added.append(line)
    restored = variant
    for line in added:
        restored = restored.replace(line, "", 1)
    if restored != prompt:
        raise ValueError("Example justifications changed more than their own lines.")
    return variant


def label_base(base: str) -> str:
    """v3: the paper prompt with the label given and the answer line removed (LABEL_EDITS)."""
    for old, new, count in LABEL_EDITS:
        if base.count(old) != count:
            raise ValueError(f"Expected {count} match(es) of {old[:60]!r}; refusing to build v3.")
        base = base.replace(old, new)
    if "answer:" in base or "non-entailment" in base:
        raise ValueError("v3 prompt still mentions an answer line.")
    return base


def build_templates() -> dict[str, str]:
    base = get_prompt("lasha")
    labelled = label_base(base)
    return {
        CONDITION_A: base,
        CONDITION_B: derive(base, EXPLANATION_INSTRUCTION, EXPLANATION_FIELDS),
        CONDITION_B2: add_example_justifications(
            derive(base, STRICT_EXPLANATION_INSTRUCTION, EXPLANATION_FIELDS)
        ),
        CONDITION_C: derive(base, HIGHLIGHTS_INSTRUCTION, HIGHLIGHTS_FIELDS),
        CONDITION_A + LABEL_SUFFIX: labelled,
        CONDITION_B + LABEL_SUFFIX: derive(labelled, EXPLANATION_INSTRUCTION, EXPLANATION_FIELDS),
        CONDITION_C + LABEL_SUFFIX: derive(labelled, HIGHLIGHTS_INSTRUCTION, HIGHLIGHTS_FIELDS),
        CONDITION_A_LABEL_T07: labelled,
    }


def fill(template: str, row: dict) -> str:
    return (
        template.replace("${PREMISE}", str(row["premise"]))
        .replace("${HYPOTHESIS}", str(row["hypothesis"]))
        .replace("${EXPLANATION}", str(row.get("explanation_1", "")))
        .replace("${PREMISE_HIGHLIGHTS}", str(row.get("premise_highlights_1") or NO_HIGHLIGHTS))
        .replace("${HYPOTHESIS_HIGHLIGHTS}", str(row.get("hypothesis_highlights_1") or NO_HIGHLIGHTS))
    )


async def generate(rows, *, templates, source_by_id, output_csv, args) -> None:
    client = None if args.dry_run else AsyncGenericAIClient(provider="openrouter")
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    jobs = [row for row in rows if not is_done(row)]
    if len(jobs) > args.max_calls:
        raise SystemExit(
            f"{len(jobs)} calls needed but --max-calls is {args.max_calls}; nothing was sent."
        )
    print(f"Running {len(jobs)} calls ({len(rows) - len(jobs)} already done).", flush=True)
    completed = 0

    async def run_one(row) -> None:
        nonlocal completed
        async with semaphore:
            row["raw_response"] = row["KB"] = row["error"] = ""
            try:
                prompt_text = fill(templates[row["prompt"]], source_by_id[row["ID"]])
                if args.dry_run:
                    raw = LABEL_DRY_RUN_RESPONSE if args.label else DRY_RUN_RESPONSE
                else:
                    raw = await call_with_retries(
                        client,
                        prompt_text=prompt_text,
                        model=args.model,
                        temperature=CONDITION_TEMPERATURES.get(row["prompt"], args.temperature),
                        max_retries=args.max_retries,
                        request_timeout=args.request_timeout,
                    )
                row["raw_response"] = raw
                row["KB"] = format_kb(_extract_validated_kb_from_output(raw))
            except Exception as exc:  # noqa: BLE001 - persisted as an error row
                row["error"] = str(exc)
        async with write_lock:
            completed += 1
            if completed % args.write_every_jobs == 0 or completed == len(jobs):
                write_rows(output_csv, OUTPUT_FIELDNAMES, rows)
                print(f"{completed}/{len(jobs)} calls done", flush=True)

    await asyncio.gather(*(run_one(row) for row in jobs))
    write_rows(output_csv, OUTPUT_FIELDNAMES, rows)


def item_counts(rows, source_by_id, prompt, policy, scratch: Path) -> dict:
    """{(item, repeat): (tp, fp, fn)} for one condition, via the repo scorer's details."""
    counts = {}
    for repeat in sorted({int(r["repeat"]) for r in rows if r["prompt"] == prompt}):
        group = []
        for r in rows:
            if r["prompt"] == prompt and int(r["repeat"]) == repeat:
                joined = {k: str(v) for k, v in r.items()}
                for column in DEFAULT_REFERENCE_COLUMNS:
                    joined[column] = source_by_id[r["ID"]].get(column, "")
                group.append(joined)
        details = scratch / f"details_{prompt}_{policy}_r{repeat}.csv"
        evaluate_prediction_column(
            group, "KB", DEFAULT_REFERENCE_COLUMNS,
            empty_prediction_is_no_relation=False,
            details_path=details, non_entailment_policy=policy,
        )
        with details.open(newline="", encoding="utf-8-sig") as handle:
            for d in csv.DictReader(handle):
                counts[(d["ID"], repeat)] = (int(d["tp"]), int(d["fp"]), int(d["fn"]))
    return counts


def micro_f1(values) -> float:
    tp = sum(v[0] for v in values)
    fp = sum(v[1] for v in values)
    fn = sum(v[2] for v in values)
    return 2 * tp / (2 * tp + fp + fn) if (tp or fp or fn) else float("nan")


def paired_bootstrap(a: dict, b: dict, resamples: int = 5000, seed: int = 0) -> dict:
    """Condition (b) - A micro-F1 over (item, repeat) pairs scored in both; resamples items."""
    shared = sorted(set(a) & set(b))
    by_item: dict[str, list] = {}
    for key in shared:
        by_item.setdefault(key[0], []).append(key)
    items = sorted(by_item)

    def diff(sample_items):
        keys = [k for item in sample_items for k in by_item[item]]
        return micro_f1([b[k] for k in keys]) - micro_f1([a[k] for k in keys])

    rng = random.Random(seed)
    diffs = sorted(diff([rng.choice(items) for _ in items]) for _ in range(resamples))
    return {
        "items_compared": len(items),
        "f1_A": round(micro_f1([a[k] for k in shared]), 4),
        "f1_condition": round(micro_f1([b[k] for k in shared]), 4),
        "f1_diff_minus_A": round(diff(items), 4),
        "ci95_low": round(diffs[int(0.025 * resamples)], 4),
        "ci95_high": round(diffs[int(0.975 * resamples) - 1], 4),
        "share_of_resamples_condition_better": round(sum(d > 0 for d in diffs) / resamples, 4),
    }


def rescore_paper_run(model: str, source_rows, output_dir: Path) -> None:
    ids = {r["ID"] for r in source_rows}
    _, saved = read_rows(PAPER_OUTPUTS)
    subset = [
        {**r, "prompt": "lasha_paper_saved"}
        for r in saved
        if r["model"] == model and r["prompt"] == "lasha" and r["ID"] in ids
    ]
    if not subset:
        print(f"No saved paper outputs for {model}; skipping paper-run rescore.")
        return
    write_f1_metrics(
        subset, source_rows, DEFAULT_REFERENCE_COLUMNS,
        output_dir / "paper_saved_A_f1_by_run.csv",
        output_dir / "paper_saved_A_f1_summary.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--write-every-jobs", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    parser.add_argument("--max-calls", type=int, default=None, help="refuse to start if more calls are needed (default: full grid)")
    parser.add_argument("--dry-run", action="store_true", help="canned responses, no API calls")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--conditions", default=None, help=f"comma-separated subset of {CONDITIONS} (or {LABEL_CONDITIONS} with --label); A is always included")
    parser.add_argument("--label", action="store_true", help="v3 prompts: label given, relations-only output (A/B/C + label)")
    parser.add_argument("--show-prompts", action="store_true", help="print the filled prompts for the first item and exit")
    args = parser.parse_args()

    _, source_rows = read_rows(args.input_csv)
    if args.limit is None and len(source_rows) != 280:
        raise SystemExit(f"Expected the 280 matched items, found {len(source_rows)} in {args.input_csv}")
    source_rows = source_rows[: args.limit]
    source_by_id = {r["ID"]: r for r in source_rows}
    all_conditions = LABEL_CONDITIONS if args.label else CONDITIONS
    file_labels = LABEL_FILE_LABELS if args.label else FILE_LABELS
    baseline = all_conditions[0]
    requested = [c.strip() for c in (args.conditions or ",".join(all_conditions)).split(",") if c.strip()]
    unknown = set(requested) - set(all_conditions)
    if unknown:
        raise SystemExit(f"Unknown conditions {sorted(unknown)}; choose from {all_conditions}")
    conditions = [c for c in all_conditions if c == baseline or c in requested]
    if any(c.startswith(CONDITION_C) for c in conditions) and "premise_highlights_1" not in source_rows[0]:
        raise SystemExit(f"{args.input_csv} has no highlight columns; run scripts/add_esnli_highlights.py")
    templates = {name: t for name, t in build_templates().items() if name in conditions}

    if args.show_prompts:
        for name, template in templates.items():
            print(f"===== {name} =====\n{fill(template, source_rows[0])}\n")
        return

    run_name = model_slug(args.model) + ("_label" if args.label else "") + ("_dryrun" if args.dry_run else "")
    output_dir = args.output_dir or PILOT_DIR / "data" / "explanation_ablation" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, label in file_labels.items():
        if name in templates:
            (output_dir / f"prompt_{label}_{name}.txt").write_text(templates[name], encoding="utf-8")
    output_csv = output_dir / "outputs.csv"

    rows = build_output_rows(
        source_rows, conditions, [args.model], args.repeats,
        DEFAULT_REFERENCE_COLUMNS, load_existing_outputs(output_csv),
    )
    for row in rows:
        row["filtering_config"] = NO_FILTER_CONFIG
        row["filtering_version"] = FILTERING_PIPELINE_VERSION
    args.max_calls = args.max_calls if args.max_calls is not None else len(rows)
    asyncio.run(generate(rows, templates=templates, source_by_id=source_by_id, output_csv=output_csv, args=args))

    write_metrics(rows, output_dir / "consistency.csv", args.repeats)
    for policy in ("exclude", "always-wrong"):
        write_f1_metrics(
            rows, source_rows, DEFAULT_REFERENCE_COLUMNS,
            output_dir / f"f1_by_run_{policy}.csv",
            output_dir / f"f1_summary_{policy}.csv",
            non_entailment_policy=policy,
        )

    scratch = output_dir / "item_details"
    scratch.mkdir(exist_ok=True)
    str_rows = [{k: str(v) for k, v in r.items()} for r in rows]
    non_entailment = {
        prompt: sum(
            r["prompt"] == prompt and model_prediction_exclusion(r, "KB") == "non_entailment"
            for r in str_rows
        )
        for prompt in conditions
    }
    comparison = []
    for policy in ("exclude", "always-wrong"):
        a = item_counts(str_rows, source_by_id, baseline, policy, scratch)
        for condition in conditions[1:]:
            b = item_counts(str_rows, source_by_id, condition, policy, scratch)
            comparison.append({
                "condition": condition,
                "non_entailment_policy": policy,
                **paired_bootstrap(a, b),
                "non_entailment_answers_A": non_entailment[baseline],
                "non_entailment_answers_condition": non_entailment[condition],
            })
    if comparison:
        write_rows(output_dir / "comparison_vs_A.csv", list(comparison[0]), comparison)

    if not args.dry_run and args.limit is None and not args.label:
        rescore_paper_run(args.model, source_rows, output_dir)

    for policy in ("exclude", "always-wrong"):
        with (output_dir / f"f1_summary_{policy}.csv").open(encoding="utf-8-sig") as handle:
            for s in csv.DictReader(handle):
                print(f"[{policy}] {s['prompt']:18s} mean F1 {float(s['mean_micro_f1']):.4f} "
                      f"± {float(s['sample_stddev_micro_f1'] or 'nan'):.4f}")
    for c in comparison:
        print(f"[{c['non_entailment_policy']}] {c['condition']} - A = {c['f1_diff_minus_A']:+.4f} "
              f"(95% CI {c['ci95_low']:+.4f} .. {c['ci95_high']:+.4f}), items {c['items_compared']}")
    counts = ", ".join(f"{name} {n}" for name, n in non_entailment.items())
    print(f"Non-entailment answers (all gold items are entailment; "
          f"{len(rows) // len(conditions)} per condition): {counts}")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
