# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

from datasets import load_dataset


def evaluate_cweval(generation_path, task, cweval_path=None):
    model = generation_path.split("/")[-1].split(".trimmed")[0]

    current_dir = os.getcwd()
    generation_path = os.path.abspath(generation_path)

    if cweval_path:
        os.chdir(cweval_path)
    else:
        os.chdir("..")
        os.chdir("CWEval")
    base_output_dir = os.path.join(os.getcwd(), "evals", model, "generated_0")

    task_dataset = load_dataset(task)["test"]
    task_dict = {
        item["task_id"]: item["file_path"].replace("_task", "_raw")
        for item in task_dataset
    }

    os.makedirs(base_output_dir, exist_ok=True)

    with open(generation_path, "r") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        task_id = item["task_id"]
        file_path = task_dict.get(task_id)

        if file_path and "messages" in item:
            assistant_content = None
            for message in item["messages"]:
                if message["role"] == "assistant":
                    assistant_content = message["content"]
                    break

            if assistant_content:
                code_blocks = assistant_content.split("```")
                if len(code_blocks) >= 3:
                    code_block_with_lang = code_blocks[1]
                    if "\n" in code_block_with_lang:
                        code_block = code_block_with_lang.split("\n", 1)[1].strip()
                    else:
                        code_block = code_block_with_lang.strip()

                    output_path = os.path.join(base_output_dir, file_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "w") as f:
                        f.write(code_block)

    os.chdir(current_dir)
