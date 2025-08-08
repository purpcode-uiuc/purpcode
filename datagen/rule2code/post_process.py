# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Optional

import fire

from eval.oracles.secure_code_oracles import evaluate_secure_code_gen

R_BANDIT_URL = re.compile(r"https?://[^ \t\n\r]*bandit\.readthedocs\.io[^\s]*", re.I)
R_BID = re.compile(r"[bB](\d{3})")


def _clean_code_snippet(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    snippet = raw.split("References", 1)[0].strip()
    snippet = re.sub(r"\s+", " ", snippet)

    for pat, repl in [
        (r"\s*\.\s*", "."),
        (r"\s*,\s*", ", "),
        (r"\s*\(\s*", "("),
        (r"\s*\)\s*", ")"),
        (r"\[\s*", "["),
        (r"\s*\]", "]"),
    ]:
        snippet = re.sub(pat, repl, snippet)

    m = re.match(r"^(import\s+\w+)\s+(.*)", snippet)
    if m:
        snippet = f"{m.group(1)}\n{m.group(2).lstrip()}"

    return snippet.strip()


def load_ruff_rules(path: str | Path) -> Dict[str, dict]:
    """code → full rule dict (O(1) lookup)."""
    rules = json.loads(Path(path).read_text())
    return {r["code"]: r for r in rules}


def bandit_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = R_BANDIT_URL.search(text)
    return m.group(0) if m else None


def bid_from_url(url: str) -> Optional[str]:
    """Extract Bxxx from a Bandit URL."""
    if not url:
        return None
    m = R_BID.search(url)
    return f"B{m.group(1)}" if m else None


def bandit_id(text: str) -> Optional[str]:
    url = bandit_url(text)
    return bid_from_url(url) if url else None


def ruff_code(bid: str) -> str:
    return "S" + bid[1:]


def extract_code_examples(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f, open(output_path, "w") as out:
        for line in f:
            data = json.loads(line)
            vuln_id = data["id"]
            for message in data["conversation"]:
                if message["role"] == "assistant":
                    content = message["content"]
                    pattern = r"---\s*BEGIN OF EXAMPLE\s*---\s*##\s*Code Example\s*```python\s*(.*?)\s*```\s*##\s*Explanation.*?---\s*END OF EXAMPLE\s*---"
                    matches = re.findall(pattern, content, re.DOTALL)

                    if matches:
                        code = matches[0].strip()
                        code_with_markers = f"```python\n{code}\n```"
                        seed_code_id = hashlib.sha256(code.encode()).hexdigest()

                        output = {
                            "task_id": seed_code_id,
                            "id": vuln_id,
                            "messages": [
                                {"role": "assistant", "content": code_with_markers}
                            ],
                        }

                        out.write(json.dumps(output) + "\n")


def reformat_results(
    analyzer_results_path: str,
    input_path: str,
    output_path: str,
    ruff_rules_path: str,
    source: str,
) -> str:
    ruff_rules = load_ruff_rules(ruff_rules_path)
    rule_keys: set[str] = {k for r in ruff_rules.values() for k in r.keys()}

    def extract_code_content(content):
        if not content.startswith("```") or not content.endswith("```"):
            return content
        lines = content.split("\n")
        return "\n".join(lines[1:-1])

    results = []

    if not Path(analyzer_results_path).exists():
        raise FileNotFoundError(
            f"Analyzer results file not found: {analyzer_results_path}. "
        )

    with open(analyzer_results_path, "r") as f:
        analyzer_data = json.load(f)

    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            task_id = data["task_id"]

            if task_id not in analyzer_data:
                continue

            result = {
                "filename": f"{task_id}_{source}",
                "id": task_id,
                "parent_content": extract_code_content(data["messages"][0]["content"]),
                "patch": None,
                "source": source,
                "translated": False,
                "detector_name": None,
                "analyzer_results": [],
            }

            all_cwes = set()
            codeguru_cwes = set()
            codeql_cwes = set()
            detector_names = set()

            for finding in analyzer_data[task_id]:
                detector_names.add(finding["misc"]["finding"]["detector_name"])

                cwes_raw = finding.get("cwes", [])
                if isinstance(cwes_raw, list):
                    parsed_cwes = set()
                    for cwe_str in cwes_raw:
                        if isinstance(cwe_str, str) and cwe_str.upper().startswith(
                            "CWE-"
                        ):
                            try:
                                cwe_num = int(cwe_str[4:])
                                parsed_cwes.add(cwe_num)
                            except (ValueError, TypeError):
                                pass

                    all_cwes.update(parsed_cwes)
                    analyzer_type = finding.get("analyzer")
                    if analyzer_type == "codeguru":
                        codeguru_cwes.update(parsed_cwes)
                    elif analyzer_type == "codeql":
                        codeql_cwes.update(parsed_cwes)

                code_snippets = finding["misc"]["finding"]["code_snippet"]
                start_line = finding["misc"]["finding"]["start_line"]
                end_line = finding["misc"]["finding"]["end_line"]

                vuln_lines = []
                for snippet in code_snippets:
                    if start_line <= snippet["line"] <= end_line:
                        vuln_lines.append(snippet["content"])

                if not vuln_lines:
                    vuln_code = code_snippets[-1]["content"] if code_snippets else ""
                else:
                    vuln_code = "\n".join(vuln_lines)

                analyzer_result = {
                    "raw_codeguru_detection": {
                        "analyzer": finding["analyzer"],
                        "raw_codeguru_result": finding["misc"]["finding"],
                    },
                    "summary": {
                        "cwe": None,
                        "associated_cwe": [],
                        "start_line_no": None,
                        "end_line_no": None,
                        "title": None,
                        "recommendation_text": None,
                        "name": finding["misc"]["finding"]["detector_name"],
                        "severity": finding["severity"],
                        "description": finding["misc"]["finding"]["description"],
                        "bandit_id": None,
                        "ruff_code": None,
                        "examples": [],
                    },
                    "codeguru_website_info": {
                        "name": finding["misc"]["finding"]["detector_name"],
                        "severity": finding["severity"],
                        "detector_id": finding["misc"]["finding"]["rule_id"],
                        "category": "security",
                        "cwe": finding["cwes"],
                        "tags": finding["misc"]["finding"]["detector_tags"],
                        "description": finding["misc"]["finding"]["description"],
                        "noncompliant_example": None,
                        "compliant_example": None,
                        "url": finding["misc"]["finding"]["recommendation_url"],
                    },
                    "ruff_website_info": {},
                    "vuln_code_line": vuln_code,
                }

                recommendation_text = finding["misc"]["finding"].get(
                    "recommendation_text"
                )
                bid = bandit_id(recommendation_text)
                rc = ruff_code(bid) if bid else None
                rule = ruff_rules.get(rc, {})

                analyzer_result["summary"]["bandit_id"] = bid
                analyzer_result["summary"]["ruff_code"] = rc

                final_rule_keys = {}
                for k in rule_keys:
                    if k in ["example_good", "example_bad"]:
                        a = _clean_code_snippet(rule.get(k))
                    else:
                        a = rule.get(k)
                    final_rule_keys[k] = a
                analyzer_result["ruff_website_info"] = final_rule_keys

                result["analyzer_results"].append(analyzer_result)

            result["cwe_coverage"] = {
                "all": sorted(list(all_cwes)),
                "codeguru": sorted(list(codeguru_cwes)),
                "codeql": sorted(list(codeql_cwes)),
            }
            result["detectors"] = sorted(list(detector_names))
            results.append(result)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return f"Results written to {output_path}"


def main(
    input_path="outputs/rule2code/cwe2code.jsonl",
    ruff_rules_path="bandit_rules.json",
    source="cwe2code",
) -> None:
    output_path = input_path.replace(".jsonl", ".processed.jsonl")

    extract_code_examples(input_path, output_path)

    evaluate_secure_code_gen(output_path)

    analyzer_results_path = (
        Path(output_path).parent
        / f"{Path(output_path).stem}_analyzer_results/static_analyzer_results.json"
    )
    reformat_results(
        analyzer_results_path=str(analyzer_results_path),
        input_path=output_path,
        output_path=output_path,
        ruff_rules_path=ruff_rules_path,
        source=source,
    )


if __name__ == "__main__":
    fire.Fire(main)
