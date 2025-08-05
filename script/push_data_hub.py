# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


# Example: python script/push_data_hub.py --path [data_path]  --split [train/test] --dataset [target_dataset_hf_name]
def push_model(path: str, split: str, dataset: str = None):
    print(f"-- Loading: `{path}:{split}`")
    try:
        conversations = load_dataset("json", data_files=path, split=split)
    except FileNotFoundError:
        conversations = load_dataset(path, split=split)

    if not dataset:
        dataset = "purpcode/" + path.split("/")[-1].replace(".jsonl", "")

    print(f"-- Target hub location: `{dataset}`")
    conversations.push_to_hub(dataset, private=True)
    print(f"-- Dataset `{dataset}` pushed to the hub")

    print("Please check:")
    print(f"https://huggingface.co/datasets/{dataset}")


if __name__ == "__main__":
    from fire import Fire

    Fire(push_model)
