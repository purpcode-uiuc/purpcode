# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import Dataset, load_dataset
from litellm import completion_with_retries
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from utils import SYSTEM_PROMPT, split_batch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_SPLIT_MARKER = "In each assistant turn, you should respond in the format below"
assert (
    _SPLIT_MARKER in SYSTEM_PROMPT
), f"SYSTEM_PROMPT does not contain expected split marker: '{_SPLIT_MARKER}'"
SYSTEM_PROMPT_FOR_GENERAL_MODEL = SYSTEM_PROMPT.split(_SPLIT_MARKER)[0].strip()


def validate_message_fmt(dataset: Dataset) -> None:
    for row in dataset:
        assert "messages" in row, "Each row must have a 'messages' key"
        assert "task_id" in row, "Each row must have a 'task_id' key"
        for message in row["messages"]:
            assert isinstance(message, dict), "Each message must be a dictionary"
            assert "role" in message, "Each message must have a 'role' key"
            assert "content" in message, "Each message must have a 'content' key"
            assert message["role"] in [
                "user",
                "assistant",
            ], "Role must be 'user' or 'assistant'"


def _get_end_of_turn(tokenizer: PreTrainedTokenizer) -> str:
    _SPLIT_1 = r"%@^1983"
    _SPLIT_2 = r"(n^4m)"
    output = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": _SPLIT_1},
            {"role": "user", "content": _SPLIT_2},
        ],
        tokenize=False,
    )

    return output.split(_SPLIT_1)[1].split(_SPLIT_2)[0].strip()


def trim_thinking(response: str) -> str:
    # trim special tokens first
    response = (
        response.replace("<|assistant|>", "")
        .replace("<|user|>", "")
        .replace("<|system|>", "")
    )
    return (
        response.split("</think>")[-1]  # compatible for xml
        .strip()
        .removeprefix("<answer>")
        .removesuffix("</answer>")
        .strip()
        .split("# Answer", maxsplit=1)[-1]
        .split("## Proposed Safe Response", maxsplit=1)[
            -1
        ]  # for eval within ctx distill
        .strip()  # compatible for md
    )


def converse_trim_thinking(messages: list) -> list:
    messages = [dict(row) for row in messages]
    for i, row in enumerate(messages):
        if row["role"] == "assistant":
            messages[i]["content"] = trim_thinking(row["content"])
    return messages


def preprocess_generation(
    generation_path: str, valid_fields: list = ["task_id", "messages"]
) -> str:
    data = []
    with open(generation_path, "r") as f:
        for line in f:
            item = json.loads(line)
            item = {k: v for k, v in item.items() if k in valid_fields}
            item["messages"] = converse_trim_thinking(item["messages"])
            data.append(item)

    generation_path = generation_path.replace(".jsonl", ".trimmed.jsonl")
    with open(generation_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return generation_path


# huggingface version
def generate_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages_batch: list,
    temperature: float = 0.0,
    max_new_tokens: int = 8192,
) -> list:
    end_of_turn = _get_end_of_turn(tokenizer)
    prompts = tokenizer.apply_chat_template(
        messages_batch, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to("cuda")
    end_strings = [end_of_turn, "<end_of_turn>"]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stop_strings=end_strings,
        tokenizer=tokenizer,
        temperature=temperature,
    )
    gen_strs = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].size(-1) :], skip_special_tokens=True
    )
    for i, gen_str in enumerate(gen_strs):
        for ends in end_strings:
            gen_strs[i] = gen_str.split(ends)[0].strip()
    return gen_strs


def generate_bedrock(
    model: str,
    tokenizer: PreTrainedTokenizer,
    messages_batch: list,
    temperature: float = 0.0,
    max_new_tokens: int = 8192,
):
    from dotenv import load_dotenv

    load_dotenv()
    outputs = []
    with ThreadPoolExecutor(max_workers=len(messages_batch)) as executor:
        futures = []
        for sample in messages_batch:
            future = executor.submit(
                completion_with_retries,
                messages=sample,
                num_retries=16,
                retry_strategy="exponential_backoff_retry",
                max_tokens=max_new_tokens,
                stop=["<end_of_turn>"],
                model=model,
                temperature=temperature,
            )
            futures.append(future)

        for future in futures:
            outputs.append(future.result().choices[0].message.content)

    return outputs


def generate_openai(
    model: str,
    tokenizer: PreTrainedTokenizer,
    messages_batch: list,
    temperature: float = 0.0,
    max_new_tokens: int = 8192,
):
    assert model.startswith("openai/"), (
        "If running openai backend, model name must start with 'openai/'. "
        "For example, 'deepseek-ai/DeepSeek-R1' should be 'openai/deepseek-ai/DeepSeek-R1'"
    )

    outputs = []
    with ThreadPoolExecutor(max_workers=len(messages_batch)) as executor:
        futures = []
        for sample in messages_batch:
            kwargs = {
                "messages": sample,
                "num_retries": 16,
                "retry_strategy": "exponential_backoff_retry",
                "max_tokens": max_new_tokens,
                "model": model,
                "api_key": (
                    os.getenv("OPENAI_API_KEY", "none")
                    if model.count("/") == 1
                    else "none"
                ),
                "api_base": (
                    os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")
                    if model.count("/") == 1
                    else "http://0.0.0.0:8000/v1"
                ),
                "temperature": temperature,
                "stop": ["<end_of_turn>"],
            }

            if (
                model.startswith("openai/o1-")
                or model.startswith("openai/o3-")
                or model.startswith("openai/o4-")
            ):
                # O-series models don't support customized temperature. Only default temperature=1 is supported.
                del kwargs["temperature"]
                del kwargs["stop"]

            future = executor.submit(completion_with_retries, **kwargs)
            futures.append(future)

        for future in futures:
            outputs.append(future.result().choices[0].message.content)

    return outputs


def get_model_id(model_path: str) -> str:
    candidates = [
        f
        for f in model_path.rstrip(os.sep).split("/")[-2:]
        if f.strip() and "checkpoint" not in f
    ]
    model_name = candidates[-1]
    return model_name


def run_llm_conversation(
    id2messages: dict,
    generation_fn,
    model,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 128,
    temperature: float = 0.0,
    use_preset_assistant_response: bool = False,
    trim_thinking: bool = True,
    answer_token_budget: int = 8192,
    sys_prompt: bool = False,
):
    # sanity check
    assert all(
        [len(messages) > 0 for messages in id2messages.values()]
    ), "All messages must have at least one turn"

    # spliting system messages
    id2system = {}
    for k, msgs in id2messages.items():
        if msgs[0]["role"] == "system":
            id2system[k] = [msgs[0]]  # for list addition
            id2messages[k] = msgs[1:]

    # Add system prompt if requested
    if sys_prompt:
        for k in id2messages.keys():
            if k not in id2system:
                id2system[k] = [
                    {"role": "system", "content": SYSTEM_PROMPT_FOR_GENERAL_MODEL}
                ]
            # If there's already a system message, don't override it

    if not use_preset_assistant_response:
        for messages in id2messages.values():
            # assert all turns are "role": "user"
            assert all(
                m["role"] == "user" for m in messages
            ), "All turns must be user turns -- assistant responses must be generated not hardcoded"
        # insert None as place holders for assistant responses
        for messages in id2messages.values():
            for i in range(len(messages)):
                messages.insert(i * 2 + 1, {"role": "assistant", "content": None})

    # sanity check: user / assistant alternating
    for messages in id2messages.values():
        for i in range(len(messages)):
            assert messages[i]["role"] == "user" if i % 2 == 0 else "assistant"

    max_turn = max(
        map(lambda x: sum(m["role"] == "user" for m in x), id2messages.values())
    )
    for turn_id in range(max_turn):
        print(f"Running inference for turn {turn_id + 1}/{max_turn} ({turn_id = })")
        responses = []
        prompts = [
            id2system.get(k, [])
            + (
                converse_trim_thinking(v[: (2 * turn_id + 1)])
                if trim_thinking
                else v[: (2 * turn_id + 1)]
            )
            for k, v in id2messages.items()
        ]

        for batch in tqdm(split_batch(prompts, batch_size)):
            generations = generation_fn(
                model,
                tokenizer,
                batch,
                temperature=temperature,
                max_new_tokens=answer_token_budget + 1024,
            )

            responses.extend(generations)

        finished_task_ids = []
        for (task_id, messages), response in zip(id2messages.items(), responses):
            if not use_preset_assistant_response:
                messages[2 * turn_id + 1] = {"role": "assistant", "content": response}

            if len(messages) == 2 * (turn_id + 1):  # finished
                finished_task_ids.append(task_id)

            yield {
                "task_id": task_id if len(messages) == 2 else f"{task_id}:{turn_id}",
                "messages": messages[: (2 * turn_id + 1)]
                + [{"role": "assistant", "content": response}],
            }

        # Remove finished tasks
        for task_id in finished_task_ids:
            del id2messages[task_id]


def generate_main(
    task: str,
    model: str,
    bs: int = 64,
    backend: str = "vllm",
    model_id: str = None,
    tp: int = 1,
    transform_conversation: str = None,  # a Python file with `transform_conversation` function
    trim_thinking: bool = True,
    answer_token_budget: int = 8192,
    temperature: float = 0.0,
    sys_prompt: bool = False,
) -> str:
    dataset = load_dataset(task, split="test")
    validate_message_fmt(dataset)
    print(f"Loaded {len(dataset)} examples from {task}")

    os.makedirs(os.path.join("results", task.split("/")[-1]), exist_ok=True)
    model_id = model_id or get_model_id(model)
    target_result_path = os.path.join(
        "results", task.split("/")[-1], model_id + f".jsonl"
    )
    if os.path.exists(target_result_path):
        print(
            f"Result file {target_result_path} already exists. Skipping... (Delete it to regenerate)"
        )
        return target_result_path

    if transform_conversation:
        print(f"Importing {transform_conversation}::transform_conversation")
        spec = importlib.util.spec_from_file_location(
            "transform_conversation", transform_conversation
        )
        transform_conversation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transform_conversation)
        dataset = dataset.map(
            transform_conversation.transform_conversation,
            remove_columns=dataset.column_names,
        )

    tokenizer = None
    generation_fn = None
    print(f"{backend = }")

    if backend == "hf":
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to("cuda")

        generation_fn = generate_hf
    elif backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM

        from eval.generate_vllm import generate_vllm

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model)
        model = LLM(
            model=model,
            generation_config="auto",
            trust_remote_code=True,
            tensor_parallel_size=tp,
        )
        generation_fn = generate_vllm
    elif backend == "openai":
        generation_fn = generate_openai
    elif backend == "bedrock":
        generation_fn = generate_bedrock
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # For a K-turn conversation, "messages" should have K roles of {"role": "user", "content": "..."}
    # A K-turn conversation will lead to K output samples where the task id is {task_id}:{turn_id} and includes the top (2*turn_id + 2) rows in the messages field
    id2messages = {row["task_id"]: row["messages"] for row in dataset}

    user_only_tasks = {}
    preset_assistant_tasks = {}

    for task_id, messages in id2messages.items():
        if any(m["role"] == "assistant" for m in messages):
            preset_assistant_tasks[task_id] = messages
        else:
            user_only_tasks[task_id] = messages

    assert (
        len(user_only_tasks) > 0 or len(preset_assistant_tasks) > 0
    ), "No tasks to run"

    if user_only_tasks:
        for output in run_llm_conversation(
            user_only_tasks,
            generation_fn,
            model,
            tokenizer,
            bs,
            temperature=temperature,
            trim_thinking=trim_thinking,
            answer_token_budget=answer_token_budget,
            sys_prompt=sys_prompt,
        ):
            with open(target_result_path, "a+") as f:
                f.write(json.dumps(output) + "\n")

    if preset_assistant_tasks:
        for output in run_llm_conversation(
            preset_assistant_tasks,
            generation_fn,
            model,
            tokenizer,
            bs,
            temperature=temperature,
            use_preset_assistant_response=True,
            trim_thinking=trim_thinking,
            answer_token_budget=answer_token_budget,
            sys_prompt=sys_prompt,
        ):
            with open(target_result_path, "a+") as f:
                f.write(json.dumps(output) + "\n")

    return target_result_path


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_main)
