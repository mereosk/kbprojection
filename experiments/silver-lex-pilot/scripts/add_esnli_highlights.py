"""Add e-SNLI annotator-1 highlights to the 280 gold items matched to e-SNLI.

The Hugging Face copy of e-SNLI has no highlights, so this reads the original
GitHub CSVs (OanaMariaCamburu/e-SNLI, dataset/esnli_{dev,test,train_1,train_2}.csv),
joins on SNLI pairID (our `ID` column), and keeps annotator 1 only: that is the
annotator whose Explanation_1 the explanation ablation uses, and train items have
only one annotator. Highlighted words are the *starred* tokens in
Sentence{1,2}_marked_1, with surrounding punctuation stripped.

Download the CSVs first (~200 MB, not committed):
  mkdir -p data/esnli_github && cd data/esnli_github && for f in esnli_dev.csv \
    esnli_test.csv esnli_train_1.csv esnli_train_2.csv; do curl -sSfLO \
    https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/$f; done
"""
import argparse
import csv
import re
from pathlib import Path

PILOT_DATA = Path(__file__).resolve().parent.parent / "data"
GITHUB_FILES = ["esnli_dev.csv", "esnli_test.csv", "esnli_train_1.csv", "esnli_train_2.csv"]
NEW_COLUMNS = [
    "esnli_github_file",
    "premise_marked_1",
    "hypothesis_marked_1",
    "premise_highlights_1",
    "hypothesis_highlights_1",
]


def highlighted_words(marked: str) -> list[str]:
    words = (re.sub(r"^\W+|\W+$", "", w) for w in re.findall(r"\*([^*]+)\*", marked or ""))
    return [w for w in words if w]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path, default=PILOT_DATA / "gold362_esnli_matched.csv")
    parser.add_argument("--github-dir", type=Path, default=PILOT_DATA / "esnli_github")
    parser.add_argument("--output-csv", type=Path, default=PILOT_DATA / "gold362_esnli_highlights.csv")
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        items = list(reader)
    wanted = {row["ID"] for row in items}

    found: dict[str, tuple[str, dict]] = {}
    for name in GITHUB_FILES:
        with (args.github_dir / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["pairID"] in wanted:
                    found[row["pairID"]] = (name, row)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} items not found in the GitHub CSVs, e.g. {sorted(missing)[:3]}")

    for item in items:
        name, row = found[item["ID"]]
        if row["Explanation_1"].strip() != item["explanation_1"].strip():
            raise SystemExit(f"Explanation_1 differs for {item['ID']}; annotator 1 may not be the same person.")
        item["esnli_github_file"] = name
        item["premise_marked_1"] = row["Sentence1_marked_1"].strip()
        item["hypothesis_marked_1"] = row["Sentence2_marked_1"].strip()
        item["premise_highlights_1"] = ", ".join(highlighted_words(row["Sentence1_marked_1"]))
        item["hypothesis_highlights_1"] = ", ".join(highlighted_words(row["Sentence2_marked_1"]))

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + NEW_COLUMNS)
        writer.writeheader()
        writer.writerows(items)
    empty_p = sum(not i["premise_highlights_1"] for i in items)
    empty_h = sum(not i["hypothesis_highlights_1"] for i in items)
    print(f"Wrote {len(items)} items to {args.output_csv} "
          f"(no premise highlights: {empty_p}, no hypothesis highlights: {empty_h})")


if __name__ == "__main__":
    main()
