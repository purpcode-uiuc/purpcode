# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from typing import Callable, Dict, List

from dotenv import load_dotenv
from litellm import completion_with_retries
from termcolor import cprint
from tqdm import tqdm
from vllm import LLM

from utils import split_batch


def log_costs(completions):
    costs = [r._hidden_params["response_cost"] for r in completions]
    if len(costs) == 0 or None in costs:
        return
    cprint(f"{len(costs)} requests costs ${sum(costs):.3f}", "yellow")


def mini_batch_completion(messages, parallel: int = 32, **kwargs):
    batches = split_batch(messages, n=parallel)
    outputs = []
    for minibatch in tqdm(batches):
        with ThreadPoolExecutor(max_workers=len(minibatch)) as executor:
            futures = []
            for sample in minibatch:
                future = executor.submit(
                    completion_with_retries,
                    messages=sample["messages"],
                    num_retries=32,
                    retry_strategy="exponential_backoff_retry",
                    **kwargs,
                )
                futures.append(future)

            for future in futures:
                outputs.append(future.result())

    return outputs


def configure_openai_api(model: str) -> dict:
    return {
        "api_key": (
            os.getenv("OPENAI_API_KEY", "none") if model.count("/") == 1 else "none"
        ),
        "api_base": (
            os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")
            if model.count("/") == 1
            else "http://0.0.0.0:8000/v1"
        ),
    }


def is_o_series_model(model: str) -> bool:
    return (
        model.startswith("openai/o1-")
        or model.startswith("openai/o3-")
        or model.startswith("openai/o4-")
    )


def run_batched_inference(
    batched_rows: List,  # each row includes at least "messages"
    row_transform: Callable[[Dict], Dict] = lambda x: x,
    max_new_tokens: int = None,
    temperature: float = None,
    model: str = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    parallel: int = 12,
    **kwargs,
):
    assert batched_rows and "messages" in batched_rows[0]
    batched_rows = [row_transform(row) for row in batched_rows]
    print("Running batched completion for LLM judge")

    if model.startswith("openai/"):
        kwargs.update(configure_openai_api(model))
    elif model.startswith("bedrock/"):
        load_dotenv()
    else:
        model = LLM(
            model=model,
            generation_config="auto",
            trust_remote_code=True,
            tensor_parallel_size=8,
        )
        sampling_params = model.get_default_sampling_params()
        sampling_params.temperature = temperature if temperature is not None else 0.0
        sampling_params.max_tokens = (
            max_new_tokens if max_new_tokens is not None else 2048
        )
        sampling_params.skip_special_tokens = True

        prompts = [row["messages"] for row in batched_rows]
        vllm_outputs = model.chat(prompts, sampling_params, use_tqdm=True)

        outputs = [SimpleNamespace(content=o.outputs[0].text) for o in vllm_outputs]

        output_rows = []
        for row, ext in zip(batched_rows, outputs):
            row = deepcopy(row)
            reasoning_content = (
                "<think>\n" + ext.reasoning_content + "\n</think>\n"
                if hasattr(ext, "reasoning_content")
                and ext.reasoning_content
                or "thinking" in kwargs
                else ""
            )
            row["messages"].append(
                {"role": "assistant", "content": reasoning_content + ext.content}
            )
            output_rows.append(row)
        return output_rows

    parameters = {
        "model": model,
        "parallel": parallel,
        "messages": batched_rows,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        **kwargs,
    }
    if "thinking" in kwargs:
        assert parameters["max_tokens"] is None
        assert parameters["temperature"] is None
    else:
        if is_o_series_model(model):
            if "temperature" in parameters:
                del parameters["temperature"]
        elif parameters["temperature"] is None:
            parameters["temperature"] = 0.0

    outputs = mini_batch_completion(**parameters)
    log_costs(outputs)
    outputs = [item.choices[0].message for item in outputs]

    output_rows = []
    for row, ext in zip(batched_rows, outputs):
        row = deepcopy(row)
        reasoning_content = (
            "<think>\n" + ext.reasoning_content + "\n</think>\n"
            if hasattr(ext, "reasoning_content")
            and ext.reasoning_content
            or "thinking" in kwargs
            else ""
        )
        row["messages"].append(
            {"role": "assistant", "content": reasoning_content + ext.content}
        )
        output_rows.append(row)
    return output_rows
