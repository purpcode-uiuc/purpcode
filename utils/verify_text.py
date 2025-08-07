# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Tuple

# Legacy thinking format based on XML
XML_THINK_PATTERN = re.compile(r"<think>(.{16,})</think>.*", re.DOTALL)


def xml_validate_response_structure(response: str) -> bool:
    if match := XML_THINK_PATTERN.match(response):
        thinking = f"<think>{match.group(1)}</think>"
        response = (
            response.split(thinking)[-1]
            .strip()
            .removeprefix("<answer>")
            .removesuffix("</answer>")
            .strip()
        )
        # <answer>, </answer>, <think>, </think> should not be in the response
        return (
            "<answer>" not in response
            and "</answer>" not in response
            and "<think>" not in response
            and "</think>" not in response
        )

    return False


def try_extract_answer_from_xml(response: str) -> str:
    answer_pattern = r"<think>(.*?)</think>"
    if matches := list(re.finditer(answer_pattern, response, re.DOTALL)):
        thinking = f"<think>{matches[-1].group(1)}</think>"
        response = response.split(thinking)[-1].strip()
    return response.removeprefix("<answer>").removesuffix("</answer>").strip()


# Thinking format based on Markdown
MD_THINK_PATTERN = re.compile(r"^## Analysis\n(.{16,})\n## Answer\n(.{8,})$", re.DOTALL)


def markdown_validate_response_structure(response: str) -> bool:
    prefill = "## Analysis\n"
    response = prefill + response.strip().lstrip(prefill)
    return bool(MD_THINK_PATTERN.match(response.strip()))


def try_extract_answer_from_markdown(response: str) -> str:
    return response.split("\n## Answer", maxsplit=1)[-1].strip()


# Select mode
validate_response_structure = markdown_validate_response_structure
try_extract_answer = try_extract_answer_from_markdown


def _has_repetition(s: str, rep_length_thresh: int = 32, rep_count_thresh: int = 8):
    if not s or len(s) < rep_length_thresh:
        return False, ""

    subsequence_count = {}
    length = rep_length_thresh

    for i in range(len(s) - length + 1):
        subseq = s[i : i + length]
        subsequence_count[subseq] = subsequence_count.get(subseq, 0) + 1
        if subsequence_count[subseq] >= rep_count_thresh:
            return (
                True,
                "-" * 16
                + "Repitition Check"
                + "-" * 16
                + f"\nRepeated {subsequence_count[subseq]} times: {subseq}",
            )

    return False, ""


CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


def extract_code_from_string(solution_str):
    code_blocks = CODE_PATTERN.findall(solution_str)
    return "\n".join(code_blocks).strip()


def check_fmt(response: str) -> Tuple[bool, str]:
    reward_log = (
        "-" * 16
        + "Bad format detected -- Original Model Output"
        + "-" * 16
        + "\n"
        + response
    )

    if not validate_response_structure(response):
        return False, reward_log

    if (rep_res := _has_repetition(response))[0]:
        return False, rep_res[1]

    if len(try_extract_answer(response)) == 0:
        return False, reward_log

    return True, ""
