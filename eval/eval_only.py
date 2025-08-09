# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from eval.evaluate import DEFAULT_LLM_JUDGE, evaluate_main


def main(
    task: str,
    generation_path: str,
    oracle: str = None,
    llm_judge: str = DEFAULT_LLM_JUDGE,
    purplellama_path: str = None,
    cweval_path: str = None,
):
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
