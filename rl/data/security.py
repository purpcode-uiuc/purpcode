# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os

import rich
from datasets import DatasetDict, concatenate_datasets, load_dataset
from rich.rule import Rule

from rl.data.correctness import _EMPTY_RETURN_


def vulcode(dataset_name):
    rich.print(Rule(f"Loading {dataset_name}..."))
    dataset = load_dataset(dataset_name)

    def make_map_fn(split):

        def process_fn(example, idx):
            # NOTE: Multi-turn conversations are skipped until Yuxiang's exploration on veRL support
            messages = example["messages"]

            if len(messages) != 1:
                return _EMPTY_RETURN_

            assert (
                messages[0]["role"] == "user"
            ), f"First message must be from user, but got {messages[0]['role']}"

            example_prompt = messages[0]["content"]
            reference = None

            label = "codesec"
            return {
                "data_source": f"purpcode:{label}",
                "prompt": messages,
                "ability": "vulcode",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "",
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference,
                    "prompt": example_prompt,
                    "oracles": [label],
                    "dataset": dataset_name,
                    "task_id": f"{dataset_name}:{example['task_id']}",
                },
            }

        return process_fn

    train_dataset = (
        dataset["train"]
        .map(
            function=make_map_fn("train"),
            with_indices=True,
            num_proc=64,
            remove_columns=dataset["train"].column_names,
        )
        .filter(lambda x: x != _EMPTY_RETURN_)
        .shuffle(seed=666)
    )
    print(f"{train_dataset = }")
    return train_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./local_data/")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    train_datasets = []

    dataset_makes = [vulcode]

    for ds in [
        "purpcode/vul2prompt-general-oss",
        "purpcode/vul2prompt-benign2vul-oss",
        "purpcode/vul2prompt-vul2vul-oss",
        "purpcode/vul2prompt-jailbreaking-oss-11k",
    ]:
        train = vulcode(ds)
        train_datasets.append(train)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    splits = train_dataset.train_test_split(test_size=1024, shuffle=True, seed=666)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    dataset_name = f"rl-codesec-{round(len(train_dataset) / 1000)}k"
    local_dir = os.path.join(args.root_dir, dataset_name)
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.push_to_hub:
        print(f"Pushing to hub: purpcode/{dataset_name}")
        DatasetDict(
            {
                "train": train_dataset,
                "test": test_dataset,
            }
        ).push_to_hub(f"purpcode/{dataset_name}", private=True)
