# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

from evalplus.sanitize import sanitize

from eval.generate import preprocess_generation

DEFAULT_LLM_JUDGE = "openai/gpt-4o"


def to_evalplus_format(generation_path: str) -> str:
    with open(generation_path, "r") as f:
        data = [json.loads(line) for line in f]

    evalplus_path = generation_path.replace(".jsonl", ".evalplus.jsonl")
    with open(evalplus_path, "w") as f:
        for item in data:
            f.write(
                json.dumps(
                    {
                        "task_id": item["task_id"],
                        "solution": sanitize(item["messages"][-1]["content"]),
                    }
                )
                + "\n"
            )
    return evalplus_path


def evaluate_main(
    task: str,
    generation_path: str,
    oracle: str = None,
    llm_judge: str = None,
    purplellama_path: str = None,
    cweval_path: str = None,
):
    if oracle is None:  # Guessing oracle
        print(f"Guessing oracle for task {task}...")
        if task in ["purpcode/mbppplus", "purpcode/humanevalplus"]:
            oracle = "evalplus"
        elif task.startswith("purpcode/CyberSecEval"):
            field = task.split("/")[-1].split("-")[-1]
            if field == "SCG":
                oracle = "cyberseceval"
            elif field == "MITRE":
                oracle = "malevent"
            elif field == "FRR":
                oracle = "overrefusal"
            else:
                raise ValueError(f"Unknown dataset: {task}")
        elif task.startswith("purpcode/CodeLMSec"):
            oracle = "codeql"
        elif task.startswith("purpcode/XSCode"):
            oracle = "xscode"
        elif task.startswith("purpcode/XSTest"):
            oracle = "overrefusal"
        elif task.startswith("purpcode/PHTest"):
            oracle = "phtest"
        elif task.startswith("purpcode/CWEval"):
            oracle = "cweval"
        else:
            raise ValueError(f"Unknown oracle for {task = }. Please specify.")
        print(f"Guessed oracle: {oracle}")

    generation_path = preprocess_generation(generation_path)
    if oracle == "evalplus":
        generation_path = to_evalplus_format(generation_path)
        dataset = task.split("/")[-1].removesuffix("plus")
        assert dataset in ["mbpp", "humaneval"]
        subprocess.run(
            ["evalplus.evaluate", "--dataset", dataset, "--samples", generation_path]
        )
    elif oracle == "safety":
        from eval.oracles.compound_safety_check import evaluate_compound_safety

        evaluate_compound_safety(
            task=task,
            generation_path=generation_path,
            model=llm_judge or DEFAULT_LLM_JUDGE,
            min_severity_level="MEDIUM",
        )
    elif oracle == "cyberseceval":
        from eval.cyberseceval import evaluate_cyberseceval

        evaluate_cyberseceval(
            generation_path=generation_path,
            task=task,
            purplellama_path=purplellama_path,
        )
    elif oracle == "codeguru":
        from eval.oracles.secure_code_oracles import evaluate_secure_code_gen

        evaluate_secure_code_gen(
            generation_path=generation_path,
            min_severity_level="MEDIUM",
            analyzers=["codeguru"],
        )
    elif oracle == "codeql":
        from eval.oracles.secure_code_oracles import evaluate_secure_code_gen

        evaluate_secure_code_gen(
            generation_path=generation_path,
            min_severity_level="MEDIUM",
            analyzers=["codeql"],
        )
    elif oracle == "malevent":
        from eval.oracles.compound_safety_check import (
            evaluate_malicious_event_assistance,
        )

        evaluate_malicious_event_assistance(
            generation_path=generation_path,
            model=llm_judge or DEFAULT_LLM_JUDGE,
        )
    elif oracle == "overrefusal":
        from eval.oracles.check_secqa import evaluate_secqa_answers

        evaluate_secqa_answers(
            generation_path=generation_path, model=llm_judge or DEFAULT_LLM_JUDGE
        )
    elif oracle == "xscode":
        from eval.oracles.xscode_overrefuse import evaluate_xscode_answers

        evaluate_xscode_answers(
            generation_path=generation_path, model=DEFAULT_LLM_JUDGE
        )

    elif oracle == "phtest":
        from eval.phtest import evaluate_phtest

        evaluate_phtest(
            generation_path=generation_path, model=llm_judge or DEFAULT_LLM_JUDGE
        )
    elif oracle == "cweval":
        from eval.cweval import evaluate_cweval

        evaluate_cweval(
            generation_path=generation_path, task=task, cweval_path=cweval_path
        )
    else:
        raise ValueError(f"Unknown oracle: {oracle}")


if __name__ == "__main__":
    from fire import Fire

    Fire(evaluate_main)
