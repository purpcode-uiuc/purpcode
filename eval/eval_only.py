# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from eval.evaluate import DEFAULT_LLM_JUDGE, evaluate_main


def main(
    task: str,
    generation_path: str,
    oracle: str = None,
    llm_judge: str = DEFAULT_LLM_JUDGE,
    reference_results_path: str = None,
    purplellama_path: str = None,
):
    evaluate_main(
        task,
        generation_path,
        oracle=oracle,
        llm_judge=llm_judge,
        reference_results_path=reference_results_path,
        purplellama_path=purplellama_path,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
