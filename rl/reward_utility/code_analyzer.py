# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import uuid
from typing import List

from eval.oracles.secure_code_oracles import run_analyzers_in_batch

PURPCODE_CODE_ANALYZER = os.getenv("PURPCODE_CODE_ANALYZER", "codeguru")  # or "codeql"


def analyze_code_security_v2(
    code_snippets: List[str], min_severity_level: str = "MEDIUM"
) -> List[List[dict]]:
    task_ids = [f"task_{uuid.uuid4()}" for _ in code_snippets]
    samples = [
        {"task_id": tid, "turn": 1, "code_blocks": [code]}
        for tid, code in zip(task_ids, code_snippets)
        if code.strip()
    ]
    print(f"{len(samples)} / {len(code_snippets)} valid code snippets to run analyzers")
    with tempfile.TemporaryDirectory() as temp_dir:
        merged_analyzer_results, _ = run_analyzers_in_batch(
            sample_with_extrcted_code_blocks=samples,
            per_analyzer_results_folder=temp_dir,
            min_severity_level=min_severity_level,
            analyzers=[PURPCODE_CODE_ANALYZER],
            num_batches=max(1, min(20, len(samples) // 32)),
            batch_granularity="line",
        )
        return [merged_analyzer_results.get(tid, []) for tid in task_ids]
