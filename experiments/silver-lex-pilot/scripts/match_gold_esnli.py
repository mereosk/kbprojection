"""Step 2a: match the 362 gold LEX problems against e-SNLI by premise+hypothesis.

Searches all three e-SNLI splits (train/validation/test) for rows whose
premise and hypothesis text equal a gold problem's, after whitespace and
quote-character normalization. SICK gold items cannot match (e-SNLI only covers SNLI) and are
reported as unmatched. When a pair occurs more than once in e-SNLI, the
candidate whose split and label agree with the gold row is preferred.

Writes the matched subset (gold columns + e-SNLI explanations) and a
per-item match log covering all 362 rows.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_DATA = Path(__file__).resolve().parent.parent / "data"

ESNLI_LABELS = ["entailment", "neutral", "contradiction"]
ESNLI_SPLITS = ["train", "validation", "test"]
# gold `split` values -> e-SNLI split names
GOLD_TO_ESNLI_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


def normalize(text: str) -> str:
    # quote style differs between sources (gold has "London 2012", e-SNLI 'London 2012')
    text = re.sub(r"[\"'“”‘’`]", "'", str(text))
    return re.sub(r"\s+", " ", text).strip()


def build_esnli_index() -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for split in ESNLI_SPLITS:
        dataset = load_dataset("esnli", split=split, revision="refs/convert/parquet")
        columns = dataset.to_dict()
        for idx in range(len(dataset)):
            key = (normalize(columns["premise"][idx]), normalize(columns["hypothesis"][idx]))
            index.setdefault(key, []).append(
                {
                    "esnli_id": f"esnli_{split}_{idx}",
                    "esnli_split": split,
                    "esnli_label": ESNLI_LABELS[columns["label"][idx]]
                    if columns["label"][idx] in (0, 1, 2)
                    else "",
                    "explanation_1": columns["explanation_1"][idx],
                    "explanation_2": columns["explanation_2"][idx],
                    "explanation_3": columns["explanation_3"][idx],
                }
            )
    return index


def pick_candidate(candidates: list[dict], gold_row: pd.Series) -> dict:
    wanted_split = GOLD_TO_ESNLI_SPLIT.get(gold_row["split"])

    def score(candidate: dict) -> tuple[int, int]:
        return (
            candidate["esnli_split"] == wanted_split,
            candidate["esnli_label"] == gold_row["gold_label"],
        )

    return max(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-csv", type=Path, default=REPO_ROOT / "data" / "all_usable_items_362.csv"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=PILOT_DATA / "gold362_esnli_matched.csv"
    )
    parser.add_argument(
        "--log-csv", type=Path, default=PILOT_DATA / "gold362_esnli_match_log.csv"
    )
    args = parser.parse_args()

    gold = pd.read_csv(args.gold_csv, encoding="utf-8-sig")
    if len(gold) != 362:
        raise ValueError(f"Expected 362 gold rows, got {len(gold)} in {args.gold_csv}")

    index = build_esnli_index()

    matched_rows = []
    log_rows = []
    for _, row in gold.iterrows():
        candidates = index.get((normalize(row["premise"]), normalize(row["hypothesis"])), [])
        log = {
            "ID": row["ID"],
            "dataset": row["dataset"],
            "split": row["split"],
            "gold_label": row["gold_label"],
            "n_candidates": len(candidates),
        }
        if not candidates:
            log["status"] = "sick_not_in_esnli" if row["dataset"] == "sick" else "no_match"
            log_rows.append(log)
            continue

        chosen = pick_candidate(candidates, row)
        split_agrees = chosen["esnli_split"] == GOLD_TO_ESNLI_SPLIT.get(row["split"])
        label_agrees = chosen["esnli_label"] == row["gold_label"]
        log.update(
            status="matched",
            esnli_id=chosen["esnli_id"],
            split_agrees=split_agrees,
            label_agrees=label_agrees,
        )
        log_rows.append(log)
        matched_rows.append(
            {**row.to_dict(), **chosen, "split_agrees": split_agrees, "label_agrees": label_agrees}
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matched_rows).to_csv(args.output_csv, index=False)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(args.log_csv, index=False)

    print(f"Gold rows: {len(gold)}")
    print(log_df.groupby(["dataset", "status"]).size().to_string())
    matched = log_df[log_df["status"] == "matched"]
    if len(matched):
        print(f"Matched with >1 e-SNLI candidate: {(matched['n_candidates'] > 1).sum()}")
        print(f"Matched but split disagrees: {(~matched['split_agrees'].astype(bool)).sum()}")
        print(f"Matched but label disagrees: {(~matched['label_agrees'].astype(bool)).sum()}")
    print(f"Wrote {len(matched_rows)} matched rows to {args.output_csv}")
    print(f"Wrote match log to {args.log_csv}")


if __name__ == "__main__":
    main()
