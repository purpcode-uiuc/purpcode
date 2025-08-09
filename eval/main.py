# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from eval.evaluate import DEFAULT_LLM_JUDGE, evaluate_main
from eval.generate import generate_main


def main(
    task: str,
    model: str,
    bs: int = 64,
    backend: str = "vllm",
    model_id: str = None,
    llm_judge: str = DEFAULT_LLM_JUDGE,
    purplellama_path: str = None,
    cweval_path: str = None,
    tp: int = 1,
    transform_conversation: str = None,
    oracle: str = None,
    trim_thinking: bool = True,
    answer_token_budget: int = 2048,
    temperature: float = 0.0,
    sys_prompt: bool = False,
):
    generation_path = generate_main(
        task,
        model,
        bs,
        backend,
        model_id,
        tp=tp,
        transform_conversation=transform_conversation,
        trim_thinking=trim_thinking,
        answer_token_budget=answer_token_budget,
        temperature=temperature,
        sys_prompt=sys_prompt,
    )
    evaluate_main(
        task,
        generation_path,
        oracle=oracle,
        llm_judge=llm_judge,
        purplellama_path=purplellama_path,
        cweval_path=cweval_path,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
