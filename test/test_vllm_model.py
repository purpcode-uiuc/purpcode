# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, RequestOutput


def main(model_path: str):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    inputs = tokenizer.apply_chat_template(
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
        add_generation_prompt=True,
        tokenize=False,
    )
    llm = LLM(model=model_path, generation_config="auto", trust_remote_code=True)

    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = 0.0
    sampling_params.max_tokens = 512
    sampling_params.skip_special_tokens = False

    outputs: List[RequestOutput] = llm.generate(inputs, sampling_params)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i}: {len(output.prompt_token_ids) = }")
        print(output.prompt_token_ids)
        print(prompt)
        print("+")
        print(f"Output {i}: {len(output.outputs[0].token_ids) = }")
        print(output.outputs[0].token_ids)
        print(generated_text)
        print("---------------")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
