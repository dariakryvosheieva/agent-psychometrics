#!/usr/bin/env python3
"""Convert IRT items.csv to question_difficulties.csv format.

Daria's predict_question_difficulty.py expects a CSV with columns:
  item_ix, item_id, diff

The IRT training output (items.csv) has format:
  - 1PL: (index), b, b_std
  - 2PL: (index), a, b, a_std, b_std

This script extracts the 'b' (difficulty) column and renumbers items.

Usage:
    python convert_irt_to_difficulties.py -i clean_data/.../items.csv
    python convert_irt_to_difficulties.py -i items.csv -o /custom/path/difficulties.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert IRT items.csv to question_difficulties.csv format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to IRT items.csv (index column contains item_id)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: question_difficulties.csv in same directory as input)"
    )
    parser.add_argument(
        "--model",
        choices=["1pl", "2pl"],
        default="1pl",
        help="IRT model type (both have 'b' column, just for documentation)"
    )
    args = parser.parse_args()

    # Load items.csv with first column as index (item_id)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, index_col=0)

    if "b" not in df.columns:
        raise ValueError(f"Expected 'b' column in {input_path}. Found columns: {list(df.columns)}")

    # Create output DataFrame with expected schema
    out = pd.DataFrame({
        "item_ix": range(len(df)),
        "item_id": df.index,
        "diff": df["b"]
    })

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "question_difficulties.csv"

    # Write output
    out.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"  Items: {len(out)}")
    print(f"  Difficulty range: [{out['diff'].min():.3f}, {out['diff'].max():.3f}]")
    print(f"  Difficulty mean: {out['diff'].mean():.3f}")


if __name__ == "__main__":
    main()
