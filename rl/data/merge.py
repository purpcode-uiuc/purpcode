# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from utils import SYSTEM_PROMPT


def align_system_prompt(example, system_prompt):
    if "system" == example["prompt"][0]["role"]:
        example["prompt"][0]["content"] = system_prompt
    else:
        example["prompt"].insert(0, {"role": "system", "content": system_prompt})
    return example


def merge_datasets(datasets, root, skip: str = None):
    skip_ids = []
    if skip is not None:  # skip some task_ids
        with open(skip, "r") as f:
            skip_ids = [
                line.strip().rsplit(":", maxsplit=1)[0]
                for line in f.readlines()
                if line.strip()
            ]

    output_ds_name = "-".join([name.split("/")[-1] for name in datasets])

    filtered_datasets = []

    # extend example["extra_info"]["task_id"] if nonexistent
    for ds_name in datasets:
        print("=" * 28)
        print("Processing", ds_name)
        if os.path.isdir(ds_name):
            dataset = load_dataset(
                "parquet",
                data_files={
                    "train": os.path.join(ds_name, "train.parquet"),
                    "test": os.path.join(ds_name, "test.parquet"),
                },
            )
        else:
            dataset = load_dataset(ds_name)
        if "task_id" not in dataset["train"]["extra_info"][0]:
            dataset["train"] = dataset["train"].map(
                lambda x: {"extra_info": {**x["extra_info"], "task_id": None}},
                num_proc=64,
            )
            dataset["test"] = dataset["test"].map(
                lambda x: {"extra_info": {**x["extra_info"], "task_id": None}},
                num_proc=64,
            )
        print(f"Before skipping {dataset['train'] = }")
        dataset["train"] = dataset["train"].filter(
            lambda x: x["extra_info"]["task_id"] not in skip_ids
        )
        print(f"After skipping {dataset['train'] = }")

        filtered_datasets.append(dataset)

    train_datasets = [dataset["train"] for dataset in filtered_datasets]
    test_datasets = [
        dataset["test"].select(range(min(800, len(dataset["test"]))))
        for dataset in filtered_datasets
    ]
    align_system_prompt_fn = partial(align_system_prompt, system_prompt=SYSTEM_PROMPT)
    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    train_dataset = train_dataset.map(align_system_prompt_fn)
    test_dataset = concatenate_datasets(test_datasets).map(align_system_prompt_fn)

    print(f"{train_dataset = }")
    print(f"{test_dataset = }")

    local_dir = os.path.join(root, output_ds_name)
    print(f"Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--root-dir", default="./local_data/")
    parser.add_argument("--skip", default=None, help="Path to skip file")
    args = parser.parse_args()

    merge_datasets(datasets=args.datasets, root=args.root_dir, skip=args.skip)
