# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Example: python script/push_model_hub.py --path [model_path]
def push_model(path: str, model_name: str = None):
    candidates = [f for f in path.split("/")[-2:] if "checkpoint" not in f]
    model_name = model_name or candidates[-1]
    assert "/" not in model_name, "Model name should not contain slashes"
    repo_id = f"purpcode/{model_name}"
    print(f"-- Pushing `{repo_id}` to the hub")

    # tokenizer
    AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
    ).push_to_hub(repo_id, private=True)
    print("-- Tokenizer pushed")

    # model
    AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
    ).push_to_hub(repo_id, private=True)
    print("-- Model pushed")

    print("Please check:")
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    from fire import Fire

    Fire(push_model)
