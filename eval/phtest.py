# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json

from datasets import load_dataset
from rich import print as rprint

from eval.oracles.check_secqa import evaluate_secqa_answers, is_bad_answer


def evaluate_phtest(
    generation_path: str,
    model: str,
):
    dataset = load_dataset("purpcode/PHTest", split="test")
    task_id_to_harmfulness = {row["task_id"]: row["Harmfulness"] for row in dataset}

    evaluate_secqa_answers(generation_path, model)

    intermediate_path = generation_path.replace(".jsonl", ".reject-intermediate.jsonl")
    with open(intermediate_path, "r") as f:
        intermediate_results = [json.loads(line) for line in f]

    harmless_rejected = 0
    harmless_total = 0
    controversial_rejected = 0
    controversial_total = 0

    for row in intermediate_results:
        task_id = row["task_id"]
        is_rejected = is_bad_answer(row["messages"][-1]["content"])
        harmfulness = task_id_to_harmfulness.get(task_id, "unknown")

        if harmfulness == "harmless":
            harmless_total += 1
            if is_rejected:
                harmless_rejected += 1
        elif harmfulness == "controversial":
            controversial_total += 1
            if is_rejected:
                controversial_rejected += 1

    rprint(
        f"[green]Harmless rejection rate:{harmless_rejected} / {harmless_total} = {100 * harmless_rejected / harmless_total:.1f}%[/green]"
    )
    rprint(
        f"[green]Controversial rejection rate:{controversial_rejected} / {controversial_total} = {100 * controversial_rejected / controversial_total:.1f}%[/green]"
    )
    rprint(
        f"[green]Total rejection rate:{harmless_rejected + controversial_rejected} / {harmless_total + controversial_total} = {100 * (harmless_rejected + controversial_rejected) / (harmless_total + controversial_total):.1f}%[/green]"
    )
