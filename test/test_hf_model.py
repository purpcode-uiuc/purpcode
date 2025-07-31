# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def trim(ids):
    non_zero_indices = torch.nonzero(ids, as_tuple=True)[0]
    last_non_zero_index = non_zero_indices[-1].item()
    return ids[: last_non_zero_index + 1]


def main(model_path: str):
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    prompts = tokenizer.apply_chat_template(
        [
            [
                {
                    "role": "user",
                    "content": "Simply reply 'hello' and stop generation immediately.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "Can you please write me a function to print hello world in Python.",
                }
            ],
            [
                {"role": "user", "content": "What's 1+1?"},
                {"role": "assistant", "content": "You want me to show you code?"},
                {"role": "user", "content": "Yes!"},
            ],
            [
                {
                    "role": "user",
                    "content": "Please just repeat 'Sorry, I cannot assist with that request'.",
                }
            ],
        ],
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        tokenizer=tokenizer,
        do_sample=False,
        use_cache=True,
    )
    output = output[:, inputs["input_ids"].size(-1) :]
    seqs = tokenizer.batch_decode(output, skip_special_tokens=False)

    for i in range(len(prompts)):
        input_ids = trim(inputs["input_ids"][i])
        print(f"Prompt {i}: {len(input_ids) = }")
        print(input_ids.tolist())
        print(prompts[i])
        print("+")
        output_ids = trim(output[i])
        print(f"Output {i}: {len(output_ids) = }")
        print(output_ids.tolist())
        print(seqs[i])
        print("---------------")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
