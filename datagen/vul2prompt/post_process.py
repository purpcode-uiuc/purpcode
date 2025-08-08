# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
from collections import defaultdict

import fire


def process_files(
    input_path="outputs/vul2prompt/vul2prompt.jsonl",
    seed_code_path="outputs/rule2code/cwe2code.processed.jsonl",
):
    if input_path is None:
        print("Please provide your input file path")
        return

    base_output = input_path.replace(".jsonl", "")

    seed_data = {}
    with open(seed_code_path, "r") as f:
        for line in f:
            data = json.loads(line)
            seed_data[data["id"]] = {
                "analyzer_results": data["analyzer_results"],
                "seed_code": data["parent_content"],
                "source": data["source"],
            }

    with open(input_path, "r") as f_in:
        prompts_data = []

        for line in f_in:
            data = json.loads(line)
            assistant_count = 0
            for message in data["conversation"]:
                if message.get("role") == "assistant":
                    assistant_count += 1
                    content = message.get("content", "").strip()
                    prompt_match = re.search(
                        r"--- BEGIN OF PROMPT ---(.*?)--- END OF PROMPT ---",
                        content,
                        re.DOTALL,
                    )
                    if prompt_match:
                        prompt = prompt_match.group(1).strip()
                        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                        messages = [{"role": "user", "content": prompt}]
                        output = {
                            "task_id": prompt_hash,
                            "seed_code_id": data["id"],
                            "strategy": data["strategy"],
                            "strategy_name": data["strategy_name"],
                            "messages": messages,
                        }
                        prompts_data.append(output)

    strategy_data = defaultdict(list)

    for data in prompts_data:
        seed_code_id = data.get("seed_code_id")

        if seed_code_id and seed_code_id in seed_data:
            data["analyzer_results"] = seed_data[seed_code_id]["analyzer_results"]
            data["seed_code"] = seed_data[seed_code_id]["seed_code"]
            data["source"] = seed_data[seed_code_id]["source"]

        strategy = data["strategy"]
        strategy_data[strategy].append(data)

    for strategy, data_list in strategy_data.items():
        output_file = f"{base_output}.{strategy}.jsonl"
        with open(output_file, "w") as f:
            for data in data_list:
                f.write(json.dumps(data) + "\n")
        print(f"Generated {output_file} with {len(data_list)} entries")


if __name__ == "__main__":
    fire.Fire(process_files)
