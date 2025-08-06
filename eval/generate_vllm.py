# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizer
from vllm import LLM


def generate_vllm(
    model: LLM,
    tokenizer: PreTrainedTokenizer,
    messages_batch: list,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
):
    sampling_params = model.get_default_sampling_params()
    sampling_params.temperature = temperature
    sampling_params.max_tokens = max_new_tokens
    sampling_params.skip_special_tokens = True

    return [
        o.outputs[0].text
        for o in model.chat(messages_batch, sampling_params, use_tqdm=False)
    ]
