# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List

from rl.reward import (
    DEFAULT_FMT_REWARD,
    _wrap_log,
    check_fmt_with_idx,
    extract_code_from_string,
    sync_group_compute_answer_score_with_idx,
    try_extract_answer,
)
from rl.reward_utility.code_analyzer import analyze_code_security_v2

PRINT_RATIO = 0.05

SOFT_CODESEC_REWARD = float(os.environ.get("SOFT_CODESEC_REWARD", "0.8"))


def compute_score(
    responses: List[str], ground_truth, data_sources: List[str], extra_info
) -> List[float]:
    t_start = time.time()
    elapsed = lambda: f"[⏰ GROUP TIME :: {time.time() - t_start:.1f}s] :: "

    counter = 0

    def _print_log(score, reward_log, max_chars=None):
        nonlocal counter
        counter += 1
        if random.random() > PRINT_RATIO:
            return
        reward_log = _wrap_log(score, reward_log, max_chars=max_chars)
        reward_log += f"\n{elapsed()} :: Finished sample {counter} / {len(responses)}"
        print(re.sub(r"\n+", "\n", reward_log) + "\n\n")

    return_scores = [None] * len(responses)
    answers = [None] * len(responses)

    oracle2idx = defaultdict(list)

    # Step 1: Grouped format checking
    with ProcessPoolExecutor(max_workers=min(64, cpu_count())) as executor:
        fs = [
            executor.submit(check_fmt_with_idx, index=i, response=response)
            for i, response in enumerate(responses)
        ]

        for f in as_completed(fs):
            i, (fmt_ok, fmt_logs) = f.result()
            return_scores[i] = DEFAULT_FMT_REWARD if fmt_ok else 0.0
            if not fmt_ok:
                _print_log(score=return_scores[i], reward_log=fmt_logs, max_chars=2048)
                continue

            # TODO(@ganler): consider multiple orales in the future
            answers[i] = try_extract_answer(responses[i])
            oracles = extra_info[i]["oracles"]
            assert len(oracles) == 1, "Assume one oracle"
            oracle2idx[oracles[0]].append(i)

    print(f"{elapsed()} Finished fmt-check for {len(responses)} samples")
    print("Post-fmt-check tasks:", {k: len(v) for k, v in oracle2idx.items()})

    # Step 2: Dispatching
    # - Correctness & LLM Judge

    def emit_tasks(executor, indices, max_concurrency=None):
        return executor.submit(
            sync_group_compute_answer_score_with_idx,
            indices=indices,
            answers=[answers[i] for i in indices],
            ground_truth=[ground_truth[i] for i in indices],
            extra_info=[extra_info[i] for i in indices],
            max_concurrency=max_concurrency,
        )

    task_groups = [
        {"oracles": ["correctness"], "max_concurrency": round(1.5 * cpu_count())},
        {"oracles": ["general-safety", "reject", "noreject"], "max_concurrency": 1024},
    ]

    with ProcessPoolExecutor(max_workers=len(task_groups)) as executor:
        subgroup_futures = []
        for group in task_groups:
            if indices := sum([oracle2idx[oracle] for oracle in group["oracles"]], []):
                subgroup_futures.append(
                    emit_tasks(
                        executor, indices, max_concurrency=group["max_concurrency"]
                    )
                )

        if safety_indices := oracle2idx["codesec"]:  # - Code security analysis
            sec_tstart = time.time()
            print(f"{elapsed()} STARTING code analysis...")
            issue_list = analyze_code_security_v2(
                [extract_code_from_string(answers[i]) for i in safety_indices]
            )
            has_python = {i: "```python" in answers[i] for i in safety_indices}
            for idx, issues in zip(safety_indices, issue_list):
                fense = lambda s: "-" * 16 + s + "-" * 16
                logs = [fense("Prompt"), extra_info[idx]["prompt"]]
                logs.extend([fense("Code to Analyze"), answers[idx]])
                if not issues:
                    return_scores[idx] = 1.0 if has_python[idx] else SOFT_CODESEC_REWARD
                    logs.append("💪" * 16 + "No vulnerability detected" + "💪" * 16)
                else:
                    logs.append("🐞" * 16 + "Vulnerabilities detected" + "🐞" * 16)
                    logs.append(json.dumps(issues, indent=4))
                _print_log(return_scores[idx], "\n".join(logs), max_chars=2048)
            print(f"{elapsed()} Code analysis done in {time.time() - sec_tstart:.1f}s")

        for f in as_completed(subgroup_futures):
            for group_result in f.result():
                idx, (ok, log) = group_result
                if ok:
                    return_scores[idx] = 1.0
                _print_log(score=return_scores[idx], reward_log=log, max_chars=2048)

    # Step 3: Return
    assert all(score is not None for score in return_scores)
    return return_scores
