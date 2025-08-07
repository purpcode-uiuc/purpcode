# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def split_dataset(dataset, num_splits, output_dir):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    if dataset.endswith(".jsonl"):
        with open(dataset, "r") as f:
            data = [json.loads(line) for line in f]
    else:
        data = load_dataset(dataset, split="test")
        data = list(data)

    # Calculate items per split
    total_items = len(data)
    items_per_split = total_items // num_splits
    remainder = total_items % num_splits

    # Create splits
    start_idx = 0
    for i in range(num_splits):
        # Add one extra item to first 'remainder' splits
        current_split_size = items_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size

        # Save split
        split_data = data[start_idx:end_idx]
        output_file = output_dir / f"split_{i+1:03d}.jsonl"

        with open(output_file, "w") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Split {i+1}: {len(split_data)} items -> {output_file}")
        start_idx = end_idx

    print(f"\nTotal: {total_items} items split into {num_splits} files")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into multiple files")
    parser.add_argument("--dataset", help="Input JSONL file")
    parser.add_argument(
        "-n", "--num-splits", type=int, required=True, help="Number of splits"
    )
    parser.add_argument("-o", "--output-dir", default="splits", help="Output directory")

    args = parser.parse_args()
    split_dataset(args.dataset, args.num_splits, args.output_dir)


if __name__ == "__main__":
    main()
