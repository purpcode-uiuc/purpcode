# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

from datasets import load_dataset


def evaluate_cyberseceval(generation_path, task, purplellama_path=None):
    output_path = os.path.abspath(generation_path.replace(".jsonl", ".reformat.json"))
    task_dataset = load_dataset(task)["test"]
    task_dict = {item["task_id"]: item for item in task_dataset}

    with open(generation_path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    results = []
    for idx, item in enumerate(data):
        task_id = item["task_id"]
        assistant_content = None
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                assistant_content = msg["content"]
                break

        if task_id in task_dict:
            task_data = task_dict[task_id]
            model = generation_path.split("/")[-1].split(".")[0]

            result = {
                "prompt_id": idx,
                "pass_id": 0,
                "test_case_prompt": task_data["test_case_prompt"],
                "response": assistant_content,
                "cwe_identifier": task_data["cwe_identifier"],
                "language": task_data["language"],
                "line_text": task_data["line_text"],
                "origin_code": task_data["origin_code"],
                "variant": task_data.get("variant", ""),
                "rule": task_data.get("rule", ""),
                "repo": task_data.get("repo", ""),
                "model": model,
                "original_code": task_data["origin_code"],
            }
            results.append(result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    import subprocess

    original_pythonpath = os.environ.get("PYTHONPATH", "")
    current_dir = os.getcwd()

    if purplellama_path:
        os.chdir(purplellama_path)
    else:
        os.chdir("..")
        os.chdir("PurpleLlama")
    os.environ["PYTHONPATH"] = original_pythonpath + ":" + os.getcwd()

    stat_path = output_path.replace(".reformat.json", ".stat.json")

    cmd = [
        "python3",
        "-m",
        "CybersecurityBenchmarks.benchmark.run",
        "--benchmark=instruct",
        "--use-precomputed-responses",
        f"--response-path={output_path}",
        f"--stat-path={stat_path}",
    ]

    subprocess.run(cmd)
    os.chdir(current_dir)
    os.environ["PYTHONPATH"] = original_pythonpath
