# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path

from datasets import Dataset


def analyze_records(records):
    """Analyze filtered records and return statistics."""
    if not records:
        return None
    print(f"Total records: {len(records)}")
    print(
        f"{len({record['original_prompt']['additional_context']['cwe_id'] for record in records})} unique CWEs"
    )

    # distribution per language
    languages = [
        record["original_prompt"]["additional_context"]["language"]
        for record in records
    ]
    language_counts = Counter(languages)
    print("Language distribution:")
    for lang, count in language_counts.items():
        print(f"  {lang}: {count}")


def process_directory(root):
    root = Path(root)

    if not root.exists():
        print(f"Error: Directory '{root}' does not exist.")
        return

    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        return

    # Find all JSONL files in the directory
    jsonl_files = list(root.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in '{root}'")
        return
    print(f"Found {len(jsonl_files)} JSONL files to process...")

    num_all_records = 0
    final_records = []
    for file_path in jsonl_files:
        records = [
            json.loads(line.strip()) for line in open(file_path, "r") if line.strip()
        ]
        num_all_records += len(records)
        filtered = [
            record
            for record in records
            if (
                record.get("malicious") == "disagree"
                and record.get("unnatural") == "disagree"
                and record.get("too_simple") == "disagree"
            )
        ]

        print(
            f"{len(filtered)} / {len(records)} records selected from {file_path.name}"
        )
        final_records.extend(filtered)

    print(
        f"Total {len(final_records)} / {num_all_records} records selected across all files."
    )

    # Analyze and display statistics
    analyze_records(final_records)

    # push this to huggingface
    # transform rows
    dataset_rows = []
    for record in final_records:
        row = {
            "task_id": record["task_id"],
            "messages": record["original_prompt"]["messages"],
            "additional_context": record["original_prompt"]["additional_context"],
        }
        dataset_rows.append(row)
    dataset = Dataset.from_list(dataset_rows, split="test")
    dataset.push_to_hub("purpcorn/xscode-annotation-merge", private=True)


if __name__ == "__main__":
    from fire import Fire

    Fire(process_directory)
