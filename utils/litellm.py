# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Callable, Dict, List

from dotenv import load_dotenv
from litellm import completion_with_retries
from termcolor import cprint
from tqdm import tqdm

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

    if model.startswith("openai"):
        kwargs["api_key"] = (
            os.getenv("OPENAI_API_KEY", "none") if model.count("/") == 1 else "none"
        )
        kwargs["api_base"] = (
            os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")
            if model.count("/") == 1
            else "http://0.0.0.0:8000/v1"
        )
    elif model.startswith("bedrock"):
        load_dotenv()

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
        if (
            model.startswith("openai/o1-")
            or model.startswith("openai/o3-")
            or model.startswith("openai/o4-")
        ):
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
