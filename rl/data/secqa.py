# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os

import rich
from datasets import DatasetDict, concatenate_datasets, load_dataset
from rich.rule import Rule

from rl.data.correctness import _EMPTY_RETURN_


def secqa(dataset_name):
    rich.print(Rule(f"Loading {dataset_name}..."))
    dataset = load_dataset(dataset_name)

    def make_map_fn(split):

        def process_fn(example, idx):
            # NOTE: Multi-turn conversations are skipped until Yuxiang's exploration on veRL support
            if len(example["messages"]) != 1:
                return _EMPTY_RETURN_

            messages = example["messages"]
            assert (
                messages[0]["role"] == "user"
            ), f"First message must be from user, but got {messages[0]['role']}"

            example_prompt = messages[0]["content"]
            reference = None

            label = "noreject"
            return {
                "data_source": f"purpcode:{label}",
                "prompt": messages,
                "ability": "secqa",
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
    test_dataset = (
        dataset["test"]
        .map(
            function=make_map_fn("test"),
            with_indices=True,
            num_proc=64,
            remove_columns=dataset["test"].column_names,
        )
        .filter(lambda x: x != _EMPTY_RETURN_)
        .shuffle(seed=666)
    )
    return train_dataset, test_dataset


def main(root_dir: str = "./local_data", push_to_hub: bool = False):
    train_datasets = []
    test_datasets = []

    for ds in ["purpcode/secqa_utility_train"]:
        train, test = secqa(ds)
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    print("Combined training set:", train_dataset)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    dataset_name = f"rl-secqa-{round(len(train_dataset) / 1000)}k"
    local_dir = os.path.join(root_dir, dataset_name)
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if push_to_hub:
        print(f"Pushing to hub: purpcode/{dataset_name}")
        DatasetDict({"train": train_dataset, "test": test_dataset}).push_to_hub(
            f"purpcode/{dataset_name}", private=True
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
