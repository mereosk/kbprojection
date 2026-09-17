"""Step A: sample entailment-labeled pairs from e-SNLI for the silver-LEX pilot.

Pulls the test split of the HuggingFace `esnli` dataset (auto-converted
parquet revision, since the dataset-script version is no longer supported by
recent `datasets`), keeps only entailment-labeled rows, takes a random sample
(not the first N), and writes id/premise/hypothesis/explanation to CSV.
"""
import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset

ESNLI_ENTAILMENT_LABEL = 0  # datasets.ClassLabel(['entailment', 'neutral', 'contradiction'])


def sample_pilot_data(sample_size: int, seed: int, split: str) -> list[dict]:
    dataset = load_dataset("esnli", split=split, revision="refs/convert/parquet")

    entailment_indices = [
        i for i, label in enumerate(dataset["label"]) if label == ESNLI_ENTAILMENT_LABEL
    ]
    if sample_size > len(entailment_indices):
        raise ValueError(
            f"Requested sample_size={sample_size} exceeds available entailment "
            f"rows ({len(entailment_indices)}) in esnli split '{split}'."
        )

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(entailment_indices, sample_size))

    rows = []
    for idx in sampled_indices:
        row = dataset[idx]
        rows.append(
            {
                "id": f"esnli_{split}_{idx}",
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "explanation": row["explanation_1"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "esnli_sample.csv",
    )
    args = parser.parse_args()

    rows = sample_pilot_data(args.sample_size, args.seed, args.split)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "premise", "hypothesis", "explanation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
