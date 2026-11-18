"""Sample entailment pairs from e-SNLI train for the large silver-LEX run.

Reads the local e-SNLI GitHub CSVs (data/esnli_github/esnli_train_{1,2}.csv),
keeps entailment pairs, and drops every pair whose premise matches the premise
of any of the 362 gold LEX items (after whitespace/quote normalization). SNLI
reuses each premise for several hypotheses, so excluding by premise (not only
by exact pair) stops the gold set from leaking into the silver resource. Exact
duplicate premise+hypothesis pairs are kept once. A random sample (fixed seed)
is written with the e-SNLI explanation and annotator-1 highlights for reference.
No network or API calls.
"""
import argparse
import csv
import random
import re
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PILOT.parents[1]
TRAIN_CSVS = [PILOT / "data/esnli_github/esnli_train_1.csv", PILOT / "data/esnli_github/esnli_train_2.csv"]
GOLD_CSV = REPO_ROOT / "data/all_usable_items_362.csv"
FIELDS = ["id", "pairID", "premise", "hypothesis", "explanation",
          "premise_highlighted", "hypothesis_highlighted"]


def normalize(text: str) -> str:
    # same rule as match_gold_esnli.py: unify quote characters, collapse whitespace
    text = re.sub(r"[\"'“”‘’`]", "'", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path,
                        default=PILOT / "data/esnli_train_entailment_10k.csv")
    args = parser.parse_args()

    with GOLD_CSV.open(newline="", encoding="utf-8-sig") as handle:
        gold = list(csv.DictReader(handle))
    assert len(gold) == 362, f"expected 362 gold items, got {len(gold)}"
    gold_premises = {normalize(r["premise"]) for r in gold}

    seen_pairs: set[tuple[str, str]] = set()
    candidates: list[dict] = []
    stats = {"rows": 0, "entailment": 0, "blank": 0, "gold_premise": 0, "duplicate_pair": 0}
    for path in TRAIN_CSVS:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stats["rows"] += 1
                if row["gold_label"] != "entailment":
                    continue
                stats["entailment"] += 1
                premise, hypothesis = normalize(row["Sentence1"]), normalize(row["Sentence2"])
                if not premise or not hypothesis:
                    stats["blank"] += 1
                    continue
                if premise in gold_premises:
                    stats["gold_premise"] += 1
                    continue
                if (premise, hypothesis) in seen_pairs:
                    stats["duplicate_pair"] += 1
                    continue
                seen_pairs.add((premise, hypothesis))
                candidates.append({
                    "pairID": row["pairID"],
                    "premise": row["Sentence1"].strip(),
                    "hypothesis": row["Sentence2"].strip(),
                    "explanation": row["Explanation_1"].strip(),
                    "premise_highlighted": row["Sentence1_Highlighted_1"],
                    "hypothesis_highlighted": row["Sentence2_Highlighted_1"],
                })

    if args.sample_size > len(candidates):
        raise ValueError(f"sample_size={args.sample_size} > {len(candidates)} candidates")
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(candidates, args.sample_size), key=lambda r: r["pairID"])
    for i, row in enumerate(sample):
        row["id"] = f"esnli_train10k_{i:05d}"

    # final leakage check against the gold premises
    assert not any(normalize(r["premise"]) in gold_premises for r in sample)
    assert len({r["pairID"] for r in sample}) == len(sample)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sample)

    print(", ".join(f"{k}={v}" for k, v in stats.items()))
    print(f"candidates={len(candidates)}; wrote {len(sample)} rows (seed {args.seed}) to {args.output_csv}")


if __name__ == "__main__":
    main()
