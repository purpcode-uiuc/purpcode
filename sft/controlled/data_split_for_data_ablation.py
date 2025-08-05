# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
from collections import Counter

from datasets import load_dataset

data = load_dataset(
    "purpcode/ctxdistill-verified-ablation-Qwen2.5-14B-Instruct-1M-73k", split="train"
)
datasets = [r["task_id"].split(":")[0] for r in data]
print(Counter(datasets))

safety_data = [
    r
    for r in data
    if r["task_id"].split(":")[0]
    in [
        "purpcorn/vul2prompt-jailbreaking-oss-11k",
        "purpcorn/vul2prompt-benign2vul-oss",
        "purpcorn/mal-event-fitd-multi-turn-oss-2k",
        "purpcorn/vul2prompt-multi-oss-5k",
        "purpcorn/mal-event-seed-attack-oss-24k",
        "purpcorn/mal-event-jailbreak-single-oss-16k",
        "purpcorn/vul2prompt-general-oss",
        "purpcorn/vul2prompt-vul2vul-oss",
    ]
]

utility_data = [
    r
    for r in data
    if r["task_id"].split(":")[0]
    in [
        "purpcorn/secqa_utility_train",
        "KodCode/KodCode-V1-SFT-R1",
    ]
]

print(f"{len(safety_data) = }")
print(f"{len(utility_data) = }")

# shuffle
random.seed(42)  # For reproducibility

random.shuffle(safety_data)
random.shuffle(utility_data)

with open("safety_ratio_1_3.jsonl", "w") as f:
    data = safety_data[: len(safety_data) // 4] + utility_data
    random.shuffle(data)  # Shuffle again after combining
    for r in data:
        f.write(json.dumps(r) + "\n")
    print(f"{len(data) = } for 1/3 safety ratio")

with open("safety_ratio_1_2.jsonl", "w") as f:
    data = safety_data[: len(safety_data) // 2] + utility_data
    random.shuffle(data)  # Shuffle again after combining
    for r in data:
        f.write(json.dumps(r) + "\n")
    print(f"{len(data) = } for 1/2 safety ratio")

with open("safety_ratio_2_3.jsonl", "w") as f:
    data = safety_data + utility_data
    random.shuffle(data)  # Shuffle again after combining
    for r in data:
        f.write(json.dumps(r) + "\n")
    print(f"{len(data) = } for 2/3 safety ratio")
