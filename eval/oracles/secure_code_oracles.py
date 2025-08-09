# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Literal, Optional, Tuple

import rich
from tqdm import tqdm

from eval.oracles.codeguru_oracle import (
    execute_codeguru,
    parse_and_filter_codeguru_results,
)
from eval.oracles.codeql_oracle import (
    check_codeql_installation,
    execute_codeql,
    parse_and_filter_codeql_results,
)


# -------------- Data Processing -----------------#
def amz_code_extract(text: str) -> list:
    code_snippets = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    return [snippet for snippet in code_snippets if snippet.strip()]


def extract_code_blocks(messages):
    return [
        {
            "turn": ((i + 1) // 2),
            "code_blocks": amz_code_extract(message["content"]),
        }
        for i, message in enumerate(messages)
        if message["role"] == "assistant"
    ]


def add_code_blocks(item):
    return [
        {**entry, "task_id": item["task_id"]}
        for entry in extract_code_blocks(item["messages"])
    ]


def parallel_extract_and_annotate(data, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(add_code_blocks, data), total=len(data)))
    return [solution for solutions in results for solution in solutions]


# -------------- Evaluation -----------------#
def run_static_analyzer(
    samples: List[Dict],
    output_folder: str,
    analyzer: str = "codeguru",
    min_severity_level: str = "MEDIUM",
    verbose: str = "debug",
) -> Dict:

    # Set temp directory for codeql and codeguru
    tmp_results_directory = output_folder

    if verbose:
        rich.print(f"[red]Temporary directory = {tmp_results_directory}")

    output_path = f"{output_folder}/{analyzer}.json"
    os.makedirs(output_folder, exist_ok=True)

    analyzer_results = {}

    if os.path.exists(output_path):
        rich.print(
            f"[red]{analyzer} results already present ..... Delete it to regenerate"
        )
        with open(output_path, "r") as file:
            analyzer_results = json.load(file)
    else:
        if analyzer == "codeql":
            check_codeql_installation()
            analyzer_results = execute_codeql(samples, tmp_results_directory)
        elif analyzer == "codeguru":
            analyzer_results = execute_codeguru(samples, tmp_results_directory)
        else:
            raise ValueError(f"Unsupported analyzer: {analyzer}")

        with open(output_path, "w") as f:
            f.write(json.dumps(analyzer_results, indent=2) + "\n")

    analysis_fn = {
        "codeql": parse_and_filter_codeql_results,
    }.get(analyzer, parse_and_filter_codeguru_results)

    analyzer_vul_samples = analysis_fn(
        analyzer_results, min_severity_level=min_severity_level
    )

    return analyzer_vul_samples


def process_batch(
    batch_index: int,
    batch_data: List[Dict],
    analyzer: str,
    min_severity: str,
    analyzer_temp_dir: str,
) -> Optional[Dict]:  # Return type changed to Optional[Dict]

    batch_results = None
    max_retries = 10  # Keep existing retry logic
    delay = 1

    for attempt in range(max_retries):
        try:
            with tempfile.TemporaryDirectory(
                dir=analyzer_temp_dir, prefix=f"{analyzer}_batch_{batch_index}_"
            ) as tmpdir:
                batch_results = run_static_analyzer(
                    batch_data,
                    output_folder=tmpdir,
                    analyzer=analyzer,
                    min_severity_level=min_severity,
                )
            break  # Success
        except Exception as e:
            rich.print(
                f"[red] Batch {batch_index} ({analyzer}) failed attempt {attempt + 1}/{max_retries}. Error: {e}. Retrying..."
            )
            if attempt == max_retries - 1:
                print(
                    f"[bold red]Batch {batch_index} ({analyzer}) failed permanently after {max_retries} attempts. Error: {e}[/bold red]",
                    file=sys.stderr,
                )
            time.sleep(delay)
            delay *= 2
    return batch_results


def split_batch_equal_code_snippets(
    items: List[Dict], num_batches: int
) -> List[List[Dict]]:
    total = sum(len(item.get("code_blocks", [])) for item in items)
    target = total / max(num_batches, 1)
    batches, current, count = [], [], 0

    for item in items:
        c = len(item.get("code_blocks", []))
        if current and count + c > target and len(batches) < num_batches - 1:
            batches.append(current)
            current, count = [], 0
        current.append(item)
        count += c
    if current:
        batches.append(current)
    return batches


def count_lines_in_item(item: Dict) -> int:
    count = 0
    for snippet in item.get("code_blocks", []):
        count += len([l for l in snippet.splitlines() if l.strip()])
    return count


def split_batch_equal_lines_of_code(
    items: List[Dict], num_batches: int
) -> List[List[Dict]]:
    total_lines = sum(count_lines_in_item(item) for item in items)
    target_lines_per_batch = total_lines / max(num_batches, 1)

    batches = []
    current_batch = []
    current_batch_lines = 0

    for item in items:
        item_lines = count_lines_in_item(item)

        if (
            current_batch
            and current_batch_lines + item_lines > target_lines_per_batch
            and len(batches) < num_batches - 1
        ):
            batches.append(current_batch)
            current_batch = []
            current_batch_lines = 0

        current_batch.append(item)
        current_batch_lines += item_lines

    if current_batch:
        batches.append(current_batch)

    return batches


def count_code_units(batch, batch_granularity):
    if batch_granularity == "block":
        return sum(len(item.get("code_blocks", [])) for item in batch)
    elif batch_granularity == "line":
        return sum(
            len(cb.splitlines()) for item in batch for cb in item.get("code_blocks", [])
        )
    else:
        raise ValueError(f"Unknown granularity: {batch_granularity}")


def get_batches(samples, granularity, num_batches):
    if granularity == "block":
        return split_batch_equal_code_snippets(samples, num_batches)
    elif granularity == "line":
        return split_batch_equal_lines_of_code(samples, num_batches)
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")


def prepare_batches(
    samples: List[Dict], granularity: str, num_batches: int = 20
) -> List[List[Dict]]:
    if granularity == "block":
        return split_batch_equal_code_snippets(samples, num_batches)
    elif granularity == "line":
        return split_batch_equal_lines_of_code(samples, num_batches)
    else:
        raise ValueError(f"Unsupported batch_granularity '{granularity}'")


def process_batches_parallel(batches, analyzer, min_severity, temp_dir, granularity):
    results = {}
    failed_batches = []
    futures_map = {}

    max_workers = min(len(batches), 8)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, batch in enumerate(batches):
            num_units = count_code_units(batch, granularity)
            if num_units == 0:
                rich.print(
                    f"[yellow]Skipping batch with {i}: 0 {granularity}(s)[/yellow]"
                )
                continue

            rich.print(f"[cyan]Batch {i}: {num_units} {granularity}(s)[/cyan]")
            future = executor.submit(
                process_batch, i, batch, analyzer, min_severity, temp_dir
            )
            futures_map[future] = {
                "index": i,
                "analyzer": analyzer,
                "num_snippets": num_units,
            }

        for future in tqdm(
            as_completed(futures_map),
            total=len(futures_map),
            desc=f"Analyzing {analyzer} batches",
        ):
            info = futures_map[future]
            try:
                batch_results = future.result()
                if batch_results is None:
                    failed_batches.append(
                        {
                            "batch_index": info["index"],
                            "analyzer": analyzer,
                            "num_snippets": info["num_snippets"],
                            "reason": "Processing failed after retries.",
                        }
                    )
                else:
                    results.update(batch_results)
            except Exception as exc:
                failed_batches.append(
                    {
                        "batch_index": info["index"],
                        "analyzer": analyzer,
                        "num_snippets": info["num_snippets"],
                        "reason": f"Execution error: {exc}",
                    }
                )

    return results, failed_batches


def cleanup_temp_dir(path):
    if os.path.exists(path):
        time.sleep(0.2)
        rich.print(f"[blue]Cleaning up temporary directory: {path}[/blue]")
        shutil.rmtree(path)


def merge_analyzer_results(per_analyzer_results):
    merged = defaultdict(list)
    for _, results in per_analyzer_results.items():
        for task_id, vulns in results.items():
            merged[task_id].extend(vulns)
    return dict(merged)


def log_failed_batches(failures):
    if failures:
        rich.print("\n[bold yellow]Summary of Failed Batches:[/bold yellow]")
        for f in failures:
            rich.print(
                f"  - Analyzer: {f['analyzer']}, Batch: {f['batch_index']}, "
                f"Units: {f['num_snippets']}, Reason: {f['reason']}"
            )


def run_analyzers_in_batch(
    sample_with_extrcted_code_blocks: List[Dict],
    per_analyzer_results_folder: str,
    num_batches: int = 20,
    min_severity_level: str = "MEDIUM",
    analyzers: List[str] = ["codeguru"],
    batch_granularity: Literal["line", "block"] = "block",
) -> Tuple[Dict, Dict]:

    assert {"code_blocks", "task_id", "turn"}.issubset(
        sample_with_extrcted_code_blocks[0]
    )

    per_analyzer_results = {}
    failed_batches_summary = []

    for analyzer in analyzers:
        # Prepare output directory and results file
        results_path = os.path.join(per_analyzer_results_folder, f"{analyzer}.json")
        temp_dir = os.path.join(
            per_analyzer_results_folder, f"{analyzer}_batch_results"
        )
        os.makedirs(temp_dir, exist_ok=True)

        rich.print(f"[purple] Analyzer: {analyzer}")

        if not os.path.exists(results_path):

            # Prepare batches based on the granularity
            batches = prepare_batches(
                sample_with_extrcted_code_blocks, batch_granularity, num_batches
            )

            start_time = time.time()

            # Parallel processing of batches
            analyzer_results, failures = process_batches_parallel(
                batches, analyzer, min_severity_level, temp_dir, batch_granularity
            )

            failed_batches_summary.extend(failures)

            cleanup_temp_dir(temp_dir)

            rich.print(
                f"[purple] Analyzer {analyzer} finished. "
                f"Total Time: {time.time() - start_time:.2f}s, "
                f"#Batches processed: {len(batches)}"
            )

            with open(results_path, "w") as f:
                json.dump(analyzer_results, f, indent=2)
        else:
            with open(results_path, "r") as f:
                analyzer_results = json.load(f)
            rich.print(f"[yellow]Using cached results from {results_path}[/yellow]")

        per_analyzer_results[analyzer] = analyzer_results
        rich.print(f"[red] #Vulnerable samples by {analyzer}: {len(analyzer_results)}")
        rich.print("-" * 20)

    merged_results = merge_analyzer_results(per_analyzer_results)
    log_failed_batches(failed_batches_summary)

    return merged_results, per_analyzer_results


def evaluate_secure_code_gen(
    generation_path: str,
    min_severity_level: str = "MEDIUM",
    analyzers: List[str] = ["codeguru"],
    num_batches: int = 20,
) -> Tuple[Dict, Dict]:

    # Store the results of the analyzer a subfolder of the directory by model name
    per_analyzer_results_folder = (
        f"{generation_path.rsplit('.', maxsplit=1)[0]}_analyzer_results"
    )

    with open(generation_path, "r") as f:
        gen_rspns = [json.loads(line) for line in f]

    sample_with_extrcted_code_blocks = parallel_extract_and_annotate(
        gen_rspns, num_workers=32
    )
    unique_task_ids = list(
        set(
            [
                code_snippet["task_id"]
                for code_snippet in sample_with_extrcted_code_blocks
            ]
        )
    )

    mrged_analyzer_results, per_analyzer_results = run_analyzers_in_batch(
        sample_with_extrcted_code_blocks,
        per_analyzer_results_folder=per_analyzer_results_folder,
        num_batches=num_batches,
        min_severity_level=min_severity_level,
        analyzers=analyzers,
    )

    n_total = len(unique_task_ids)
    n_vul = len(mrged_analyzer_results)

    rich.print(
        f"[yellow u b]🐞 Vulnerability Detection on Any Analyzer: {n_vul} / {n_total} ({n_vul / n_total * 100:.1f}%)"
    )

    for a in per_analyzer_results:
        n_total = len(unique_task_ids)
        n_vul = len(per_analyzer_results[a])
        rich.print(
            f"[yellow]|- {a}: {n_vul} / {n_total} ({n_vul / n_total * 100:.1f}%)"
        )

    cwes = {
        session_id: [cwe for vuln in vulns for cwe in vuln["cwes"]]
        for session_id, vulns in mrged_analyzer_results.items()
    }

    with open(f"{per_analyzer_results_folder}/static_analyzer_results.json", "w") as f:
        json.dump(mrged_analyzer_results, f, indent=2)

    return mrged_analyzer_results, cwes


if __name__ == "__main__":
    from fire import Fire

    Fire(evaluate_secure_code_gen)
