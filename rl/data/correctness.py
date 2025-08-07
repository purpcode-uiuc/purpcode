# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import rich
from datasets import DatasetDict, concatenate_datasets, load_dataset
from rich.rule import Rule

from rl.reward_utility.sandbox_fusion import code_exec_sandbox_fusion

N_SFT_SIZE = 1024 * 5
KOD_SIZE = 1024 * 50
N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


def kodcode():
    dataset_name = "KodCode/KodCode-V1-SFT-R1"
    rich.print(Rule(f"Loading {dataset_name}..."))
    dataset = load_dataset(dataset_name, split="train")

    block_libs = [
        "fake-useragent",
        "keras",
        "socket",
        "torch",
        "scipy",
        "sklearn",
        "cv2",
        "imageio",
        "sphinx-pyproject",
        "xgboost",
        "tweepy",
        "flask",
        "matplotlib",
        "pillow",
        "seaborn",
        "smtpd",
    ]

    def make_map_fn(split):

        def process_fn(example, idx):
            reference_solution = example["solution"]
            test_code = "from solution import *\n" + example["test_code"].strip()
            # skip it if reference solution requires libs from block_libs
            if any(lib in reference_solution for lib in block_libs):
                return _EMPTY_RETURN_
            if any(lib in test_code for lib in block_libs):
                return _EMPTY_RETURN_
            prompt = f"Please solve the programming task below in Python. Code should wrapped in a markdown code block.\n\n{example['question'].strip()}"
            if example["test_entry_point"] and example["test_entry_point"].strip():
                prompt += f"\n\nNote that the output function should be {example['test_entry_point'].strip()}."

            succ, err = code_exec_sandbox_fusion(
                code=reference_solution, pytest=test_code
            )
            if not succ:
                rich.print(
                    f"[bold red]Test code failed for {example['conversation_id']}"
                )
                print(reference_solution)
                print(err)
                return _EMPTY_RETURN_

            return {
                "data_source": "purpcode:code",
                "prompt": [
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"pytest": test_code}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference_solution,
                    "prompt": prompt,
                    "oracles": ["correctness"],
                    "dataset": dataset_name,
                },
            }

        return process_fn

    # filter by difficulty: 0.3 - 0.9
    dataset = dataset.filter(lambda x: 0.3 <= x["gpt_pass_percentage"] < 0.9)
    dataset = dataset.shuffle(seed=666).select(range(KOD_SIZE))
    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=96,
        remove_columns=dataset.column_names,
    ).filter(lambda x: x != _EMPTY_RETURN_)
    splits = dataset.train_test_split(test_size=N_TESTSET_PER_DATASET, seed=666)
    train_dataset = splits["train"].shuffle(seed=666)
    test_dataset = splits["test"]
    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset(
        "json",
        data_files="code-r1/LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl",
    )["train"]
    print("Test set:", test_dataset)

    train_dataset = concatenate_datasets(
        [
            load_dataset(
                "json",
                data_files="code-r1/LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl",
            )["train"],
            load_dataset(
                "json",
                data_files="code-r1/LeetCodeDataset/data/LeetCodeDataset-v2-sft-problems.jsonl",
            )["train"],
        ]
    ).filter(
        lambda example: example["meta"]["question_id"]
        not in set([d["question_id"] for d in test_dataset["meta"]])
    )
    print("Before deduplication - Training set:", train_dataset)

    first_time_idx = []
    seen_question_ids = set()
    for i, example in enumerate(train_dataset):
        if example["meta"]["question_id"] not in seen_question_ids:
            first_time_idx.append(i)
            seen_question_ids.add(example["meta"]["question_id"])
    train_dataset = train_dataset.select(first_time_idx)

    print("After deduplication - Training set:", train_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['meta']['query'].strip()}"
            return {
                "data_source": "purpcode:code",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            "functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
                        }
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],
                    "prompt": prompt,
                    "oracles": ["correctness"],
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./local_data/")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    root_dir = args.root_dir

    train_datasets = []
    test_datasets = []

    dataset_makes = [leetcode2k, kodcode]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    splits = train_dataset.train_test_split(test_size=N_SFT_SIZE, seed=666)
    train_dataset = splits["train"]
    sft_dataset = splits["test"]
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    dataset_name = f"code-r1-{round(len(train_dataset) / 1000)}k-{names}"
    local_dir = os.path.join(root_dir, dataset_name)
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # combined dataset
    if args.push_to_hub:
        print(f"Pushing to hub: purpcode/{dataset_name}")
        DatasetDict(
            {
                "train": train_dataset,
                "test": test_dataset,
                "sft": sft_dataset,
            }
        ).push_to_hub(repo_id=f"purpcode/{dataset_name}", private=True)
