# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os

import rich
from datasets import DatasetDict, concatenate_datasets, load_dataset
from rich.rule import Rule

from rl.data.correctness import _EMPTY_RETURN_

MAX_MAL_EVENT_SIZE = 1024 * 8  # 8k prompts


def safety(dataset_name, single_turn=False):
    rich.print(Rule(f"Loading {dataset_name}..."))
    dataset = load_dataset(dataset_name)

    def make_map_fn(split):

        def process_fn(example, idx):
            messages = example["messages"]
            assert (
                messages[0]["role"] == "user"
            ), f"First message must be from user, but got {messages[0]['role']}"
            assert all(
                msg["role"] == "user" for msg in messages
            ), f"User only in 'role'"
            # NOTE: Multi-turn conversations are skipped until Yuxiang's exploration on veRL support
            if single_turn and len(example["messages"]) > 1:
                return _EMPTY_RETURN_

            example_prompt = messages[0]["content"]
            reference = None

            label = "general-safety"
            return {
                "data_source": f"purpcode:{label}",
                "prompt": messages,
                "ability": "safety",
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
        .shuffle(seed=886)
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
        .shuffle(seed=886)
    )
    return train_dataset, test_dataset


def main(root_dir="./local_data/", push_to_hub=False):
    train_datasets = []
    test_datasets = []

    for ds in [
        "purpcode/mal-event-jailbreak-single-oss-16k",
        "purpcode/mal-event-seed-attack-oss-24k",
    ]:
        train, test = safety(ds, single_turn=True)
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = (
        concatenate_datasets(train_datasets)
        .shuffle(seed=666)
        .select(range(MAX_MAL_EVENT_SIZE))
    )
    print("Combined training set:", train_dataset)
    test_dataset = concatenate_datasets(test_datasets).select(range(256))

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    dataset_name = f"rl-safety-{round(len(train_dataset) / 1000)}k" + "-single-turn"
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
