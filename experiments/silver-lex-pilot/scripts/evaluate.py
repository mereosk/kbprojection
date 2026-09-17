"""Evaluate augmented-proving runs against the WordNet-only baseline.

For each run (a silver_lex_<model>.csv + its augmented_<model>.csv from
prove_augmented.py), this:

1. Joins baseline.csv with that run's augmented_<model>.csv by id, excluding
   ids whose baseline result is "-" (prover errors, see README).
2. Reports baseline/augmented entailment counts, delta, and any regressions
   (entailment -> not entailment; should be zero under a sound prover, so any
   regression is itself worth reporting).
3. For every neutral -> entailment flip, runs a minimal-subset ablation
   against the live LangPro API: it tries increasing-size subsets of that
   item's injected relations and records the smallest subset that alone still
   proves entailment. This answers "which relation(s) actually did the work,"
   rather than assuming every offered relation mattered.
4. Flags relations whose arguments are bare function words (prepositions,
   determiners, auxiliaries) as suspected proof-hacking artifacts rather than
   genuine lexical facts -- a cheap, explicit, reproducible heuristic, not a
   claim of certainty. Manual judgment (see the CLAUDE.md decisions log) still
   has the final word.

Writes three files to --output-dir (default: ../data/evaluation/):
  summary.csv        one row per run: counts, delta, flips, regressions
  flips.csv          one row per flip per run: full context + ablation result
  report.md          human-readable writeup combining both, meant to be read
                      directly (e.g. to summarize the pilot to a professor)

Makes live LangPro API calls (for the ablation step only); everything else is
computed from already-saved CSVs.
"""
import argparse
import asyncio
import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path

from kbprojection.langpro import langpro_api_call

PILOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PILOT_DIR / "data"
DEFAULT_BASELINE = DATA_DIR / "baseline.csv"
DEFAULT_SAMPLE = DATA_DIR / "esnli_sample.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "evaluation"
RELATION_SEP = " ;; "

# Bare function words: if a relation argument is just one of these, it is not
# a real content phrase and can't be a genuine lexical entailment relation.
FUNCTION_WORDS = {
    "in", "on", "at", "of", "to", "for", "with", "by", "from", "as", "into",
    "onto", "over", "under", "through", "about", "against", "between",
    "among", "during", "without", "within", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "the", "a",
    "an", "and", "or", "but", "not", "up", "down", "out", "off",
}

DEFAULT_RUNS = [
    ("gemini-3.1-flash-lite", DATA_DIR / "silver_lex_gemini-3.1-flash-lite.csv", DATA_DIR / "augmented_gemini-3.1-flash-lite.csv"),
    ("claude-sonnet-5", DATA_DIR / "silver_lex_claude-sonnet-5.csv", DATA_DIR / "augmented_claude-sonnet-5.csv"),
]


@dataclass
class RunSummary:
    label: str
    n_valid: int
    baseline_entailment: int
    augmented_entailment: int
    flips: list = field(default_factory=list)
    regressions: list = field(default_factory=list)

    @property
    def delta(self) -> int:
        return self.augmented_entailment - self.baseline_entailment

    @property
    def baseline_rate(self) -> float:
        return self.baseline_entailment / self.n_valid if self.n_valid else 0.0

    @property
    def augmented_rate(self) -> float:
        return self.augmented_entailment / self.n_valid if self.n_valid else 0.0


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_function_word_relation(relation: str) -> bool:
    inner = relation.strip()
    if inner.startswith("isa_wn(") and inner.endswith(")"):
        inner = inner[len("isa_wn("):-1]
    args = [a.strip().lower() for a in inner.split(",", 1)]
    return any(arg in FUNCTION_WORDS for arg in args)


async def find_minimal_subset(premise: str, hypothesis: str, relations: list[str]) -> list[str] | None:
    """Try increasing-size subsets of relations; return the first that alone proves entailment."""
    for size in range(1, len(relations) + 1):
        for subset in itertools.combinations(relations, size):
            result = await langpro_api_call(premises=[premise], hypothesis=hypothesis, kb=list(subset))
            if result.label.value == "entailment":
                return list(subset)
    return None


async def evaluate_run(
    label: str,
    silver_csv: Path,
    augmented_csv: Path,
    baseline: dict[str, str],
    sample_by_id: dict[str, dict],
) -> tuple[RunSummary, list[dict]]:
    silver_by_id = {row["id"]: row for row in read_rows(silver_csv)}
    augmented_by_id = {row["id"]: row for row in read_rows(augmented_csv)}

    valid_ids = [id_ for id_, b in baseline.items() if b in ("entailment", "neutral")]
    baseline_entailment = sum(1 for id_ in valid_ids if baseline[id_] == "entailment")
    augmented_entailment = sum(
        1 for id_ in valid_ids if augmented_by_id.get(id_, {}).get("result") == "entailment"
    )

    flip_ids = [
        id_ for id_ in valid_ids
        if baseline[id_] == "neutral" and augmented_by_id.get(id_, {}).get("result") == "entailment"
    ]
    regression_ids = [
        id_ for id_ in valid_ids
        if baseline[id_] == "entailment" and augmented_by_id.get(id_, {}).get("result") != "entailment"
    ]

    summary = RunSummary(
        label=label,
        n_valid=len(valid_ids),
        baseline_entailment=baseline_entailment,
        augmented_entailment=augmented_entailment,
        flips=flip_ids,
        regressions=regression_ids,
    )

    flip_rows = []
    for id_ in flip_ids:
        sample = sample_by_id[id_]
        offered_relations = [
            r.strip() for r in silver_by_id[id_]["lex_relations"].split(RELATION_SEP) if r.strip()
        ]
        minimal = await find_minimal_subset(sample["premise"], sample["hypothesis"], offered_relations)
        suspect_relations = [r for r in offered_relations if is_function_word_relation(r)]
        minimal_is_suspect = bool(minimal) and any(is_function_word_relation(r) for r in minimal)
        flip_rows.append(
            {
                "run": label,
                "id": id_,
                "premise": sample["premise"],
                "hypothesis": sample["hypothesis"],
                "explanation": sample["explanation"],
                "offered_relations": RELATION_SEP.join(offered_relations),
                "minimal_sufficient_subset": RELATION_SEP.join(minimal) if minimal else "(none closed the proof alone)",
                "function_word_relations_offered": RELATION_SEP.join(suspect_relations),
                "flagged_as_proof_hacking_artifact": minimal_is_suspect,
            }
        )

    return summary, flip_rows


def write_summary_csv(summaries: list[RunSummary], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run", "n_valid", "baseline_entailment", "baseline_rate",
                "augmented_entailment", "augmented_rate", "delta",
                "n_flips", "n_regressions",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "run": s.label,
                    "n_valid": s.n_valid,
                    "baseline_entailment": s.baseline_entailment,
                    "baseline_rate": f"{s.baseline_rate:.4f}",
                    "augmented_entailment": s.augmented_entailment,
                    "augmented_rate": f"{s.augmented_rate:.4f}",
                    "delta": s.delta,
                    "n_flips": len(s.flips),
                    "n_regressions": len(s.regressions),
                }
            )


def write_flips_csv(flip_rows: list[dict], path: Path) -> None:
    if not flip_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flip_rows)


def write_report_md(summaries: list[RunSummary], flip_rows: list[dict], path: Path) -> None:
    lines = ["# Silver-LEX pilot: evaluation report", ""]
    lines.append(
        "Baseline: LangPro + built-in WordNet only, no injected relations. "
        "Augmented: baseline + silver LEX relations extracted from e-SNLI "
        "human explanations by the named model. Denominators exclude items "
        "where the baseline call itself failed (prover server errors)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Run | n | Baseline | Augmented | Delta | Flips | Regressions |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.n_valid} "
            f"| {s.baseline_entailment} ({s.baseline_rate:.1%}) "
            f"| {s.augmented_entailment} ({s.augmented_rate:.1%}) "
            f"| {'+' if s.delta >= 0 else ''}{s.delta} "
            f"| {len(s.flips)} | {len(s.regressions)} |"
        )
    lines.append("")

    for s in summaries:
        lines.append(f"## {s.label} -- flip detail")
        lines.append("")
        if s.regressions:
            lines.append(f"**Regressions found:** {s.regressions} -- investigate before trusting this run's delta.")
            lines.append("")
        run_flips = [r for r in flip_rows if r["run"] == s.label]
        if not run_flips:
            lines.append("No neutral -> entailment flips in this run.")
            lines.append("")
            continue
        for r in run_flips:
            flag = " **(flagged: proof-hacking artifact, not a genuine lexical fact)**" if r["flagged_as_proof_hacking_artifact"] else ""
            lines.append(f"### `{r['id']}`{flag}")
            lines.append("")
            lines.append(f"- Premise: {r['premise']}")
            lines.append(f"- Hypothesis: {r['hypothesis']}")
            lines.append(f"- Human explanation: {r['explanation']}")
            lines.append(f"- Relations offered: `{r['offered_relations']}`")
            lines.append(f"- Minimal sufficient subset (ablation result): `{r['minimal_sufficient_subset']}`")
            if r["function_word_relations_offered"]:
                lines.append(f"- Function-word (suspect) relations offered: `{r['function_word_relations_offered']}`")
            lines.append("")

    n_genuine = sum(1 for r in flip_rows if not r["flagged_as_proof_hacking_artifact"])
    n_flagged = sum(1 for r in flip_rows if r["flagged_as_proof_hacking_artifact"])
    lines.append("## Headline, adjusted for flagged artifacts")
    lines.append("")
    flagged_verb = "does" if n_flagged == 1 else "do"
    lines.append(
        f"Across all runs: {len(flip_rows)} total flips, of which {n_genuine} have no "
        f"function-word relation in their minimal sufficient subset, and {n_flagged} "
        f"{flagged_verb} and should not be counted as evidence the method identified a "
        "genuine missing lexical fact."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(runs: list[tuple[str, Path, Path]], baseline_csv: Path, sample_csv: Path, output_dir: Path) -> None:
    baseline = {row["id"]: row["result"] for row in read_rows(baseline_csv)}
    sample_by_id = {row["id"]: row for row in read_rows(sample_csv)}

    summaries = []
    all_flip_rows = []
    for label, silver_csv, augmented_csv in runs:
        print(f"Evaluating run: {label}")
        summary, flip_rows = await evaluate_run(label, silver_csv, augmented_csv, baseline, sample_by_id)
        summaries.append(summary)
        all_flip_rows.extend(flip_rows)
        print(
            f"  n={summary.n_valid} baseline={summary.baseline_entailment} "
            f"augmented={summary.augmented_entailment} delta={summary.delta} "
            f"flips={len(summary.flips)} regressions={len(summary.regressions)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summaries, output_dir / "summary.csv")
    write_flips_csv(all_flip_rows, output_dir / "flips.csv")
    write_report_md(summaries, all_flip_rows, output_dir / "report.md")
    print(f"\nWrote summary.csv, flips.csv, report.md to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL:SILVER_CSV:AUGMENTED_CSV",
        help="Add a run to evaluate (repeatable). If omitted, evaluates the two known pilot runs.",
    )
    args = parser.parse_args()

    if args.run:
        runs = []
        for spec in args.run:
            label, silver, augmented = spec.split(":")
            runs.append((label, Path(silver), Path(augmented)))
    else:
        runs = DEFAULT_RUNS

    asyncio.run(run(runs, args.baseline_csv, args.sample_csv, args.output_dir))


if __name__ == "__main__":
    main()
