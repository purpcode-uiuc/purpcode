# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import format_exc
from typing import Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

from eval.oracles.check_secqa import *
from eval.oracles.malicious_assistance_detection import *
from rl.reward_utility.sandbox_fusion import _ERROR_MSG_PREFIX
from rl.reward_utility.sandbox_fusion import code_exec_sandbox_fusion as code_exec
from utils.verify_text import check_fmt, extract_code_from_string, try_extract_answer

_MAX_CHAR_DISPLAY = 4096
LLM_JUDGE_OPENAI_URL = os.getenv("LLM_JUDGE_OPENAI_URL")

print(f"{LLM_JUDGE_OPENAI_URL = }")


def get_llm_judge_name():
    try:
        client = OpenAI(api_key="none", base_url=LLM_JUDGE_OPENAI_URL)
        models = client.models.list()
        return models.data[0].id
    except Exception:
        print("Failed to get LLM judge model name, using `default`.")
        print(format_exc())
        return "default"


LLM_JUDGE_MODEL_ID = get_llm_judge_name()


def check_fmt_with_idx(index, response):
    return index, check_fmt(response)


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def _check_correctness(solution_str, ground_truth, extra_info) -> Tuple[bool, str]:
    reward_log = []

    solution_code = extract_code_from_string(solution_str)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    reward_log.append(solution_code)

    t_start = time.time()

    ground_truth = json.loads(ground_truth)
    if "pytest" in ground_truth or "functional" in ground_truth:
        if "functional" in ground_truth:
            succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])
        else:  # pytest
            succ, output = code_exec(solution_code, pytest=ground_truth["pytest"])
        if not succ:
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"])
            reward_log.append(
                "!" * 16
                + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                + "!" * 16
            )
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            return False, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(16, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
                    reward_log.append(
                        "!" * 16
                        + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                        + "!" * 16
                    )
                    reward_log.append(f"🔎Input: {repr(stdin)}")
                    reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                    reward_log.append(
                        f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}"
                    )
                    return False, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output)
    return True, "\n".join(reward_log)


async def _check_malicious(actor_response, prompt):
    reward_log = []
    prompt_check_malicious = malicious_assistance_detection_prompt(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": actor_response},
            ]
        }
    )

    client = AsyncOpenAI(api_key="none", base_url=LLM_JUDGE_OPENAI_URL)
    completion = await client.chat.completions.create(
        model=LLM_JUDGE_MODEL_ID,
        messages=prompt_check_malicious["messages"],
        max_completion_tokens=256,
        stop=["PV_MAL_CODE: TRUE", "PV_MAL_EXPLN: TRUE"],
    )
    judge_response = completion.choices[0].message.content

    reward_log.append("-" * 16 + "Prompt" + "-" * 16)
    reward_log.append(prompt)
    reward_log.append("-" * 16 + "Response" + "-" * 16)
    reward_log.append(actor_response)

    if ": TRUE" in judge_response:
        reward_log.append("-" * 16 + "💥 Malicious Assistance Detected!" + "-" * 16)
        reward_log.append("-" * 16 + "LLM Judge Output Below" + "-" * 16)
        reward_log.append(judge_response)
        return False, "\n".join(reward_log)

    reward_log.append("+" * 16 + "🔒 No Malicious Assistance Detected!" + "+" * 16)
    reward_log.append(judge_response)
    return True, "\n".join(reward_log)


async def _check_secqa(actor_response, prompt, expect_bad=True):
    reward_log = []
    prompt_check_malicious = check_secqa_prompt(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": actor_response},
            ]
        }
    )

    client = AsyncOpenAI(api_key="none", base_url=LLM_JUDGE_OPENAI_URL)
    completion = await client.chat.completions.create(
        model=LLM_JUDGE_MODEL_ID,
        messages=prompt_check_malicious["messages"],
        max_completion_tokens=256,
    )
    judge_response = completion.choices[0].message.content

    reward_log.append("-" * 16 + "Prompt" + "-" * 16)
    reward_log.append(prompt)
    reward_log.append("-" * 16 + "Response" + "-" * 16)
    reward_log.append(actor_response)
    is_bad = is_bad_answer(judge_response)
    hatch = "🚨" if is_bad != expect_bad else "🔒"
    if is_bad:
        reward_log.append("-" * 16 + f"{hatch} Bad SecQA Answer Detected" + "-" * 16)
    else:
        reward_log.append("-" * 16 + f"{hatch} Good SecQA Answer" + "-" * 16)
    reward_log.append(judge_response)
    return is_bad == expect_bad, "\n".join(reward_log)


DEFAULT_FMT_REWARD = 0.0  # set it to zero now as models can now better follow the fmt


def _wrap_log(score: float, reward_log: str, max_chars: int = None):
    if max_chars is not None and len(reward_log) > max_chars:
        half = max_chars // 2
        omitted_count = len(reward_log) - (2 * half)
        reward_log = (
            reward_log[:half]
            + f"\n... omitted {omitted_count} chars ... \n"
            + reward_log[-half:]
        )

    fense = ("✅" if score >= 1.0 else "❌") * 16
    return (
        fense
        + " Reward Calculation "
        + fense
        + "\n"
        + reward_log
        + "\n"
        + fense
        + f" Final Reward = {score} "
        + fense
    )


async def compute_answer_score(answer, ground_truth, extra_info):
    reward_log_list = []
    for o in extra_info["oracles"]:
        tstart = time.time()
        if "correctness" == o:
            ok, reward_log = await asyncio.to_thread(
                _check_correctness, answer, ground_truth, extra_info=extra_info
            )
        elif "general-safety" == o:
            ok, reward_log = await _check_malicious(answer, extra_info["prompt"])
        elif o in ["reject", "noreject"]:
            ok, reward_log = await _check_secqa(
                answer, extra_info["prompt"], expect_bad="reject" == o
            )
        elif o == "security":
            raise ValueError(
                "Please use grouped reward manager for security-related oracle."
            )
        else:
            raise ValueError(f"Unknown oracle: {o}")

        reward_log_list.append(reward_log)
        reward_log_list.append(
            f"⏰ Running oracle {o} took {time.time() - tstart:.1f}s"
        )
        if not ok:
            return False, "\n".join(reward_log_list)

    return True, "\n".join(reward_log_list)


def compute_score(data_source, response, ground_truth, extra_info):
    answer_reward = 1.0
    t_start = time.time()
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    def _return_score(score, reward_log, marker="-", max_chars=None):
        reward_log = _wrap_log(
            score, "\n".join(reward_log), marker=marker, max_chars=max_chars
        )
        reward_log += f"\n{data_source = } :: computed in {time.time() - t_start}s"
        return score

    reward_logs = []

    # Format
    fmt_ok, fmt_logs = check_fmt(response)
    reward_logs.append(fmt_logs)
    if not fmt_ok:
        return _return_score(0.0, reward_logs, marker="❌", max_chars=2048)

    answer = try_extract_answer(response)
    ok, reward_log = asyncio.run(compute_answer_score(answer, ground_truth, extra_info))
    reward_logs.append(reward_log)
    if not ok:
        return _return_score(DEFAULT_FMT_REWARD, reward_logs, marker="❌")

    return _return_score(answer_reward, reward_logs, marker="✅")


def sync_group_compute_answer_score_with_idx(
    indices, answers, ground_truth, extra_info, max_concurrency=None
):
    tstart = time.time()

    async def main():
        async def run(index, ans, gt, ei):
            return index, await compute_answer_score(ans, gt, ei)

        sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def run_with_concurrency(index, ans, gt, ei):
            async with sem:
                return index, await compute_answer_score(ans, gt, ei)

        run_fn = run_with_concurrency if sem else run

        return await asyncio.gather(
            *[
                asyncio.create_task(run_fn(i, a, g, e))
                for i, a, g, e in zip(indices, answers, ground_truth, extra_info)
            ]
        )

    ret = asyncio.run(main())  # sync here
    stat = defaultdict(dict)
    for (_, (ok, _)), ei in zip(ret, extra_info):
        for o in ei["oracles"]:
            stat[o]["ok"] = stat[o].get("ok", 0) + int(ok)
            stat[o]["total"] = stat[o].get("total", 0) + 1

    for o, v in stat.items():
        v["succ_rate"] = f'{v["ok"] / v["total"]:.1%}'
        v["throughput"] = v["total"] / (time.time() - tstart)
        v["throughput"] = f'{v["throughput"]:.2f} response/s'

    # append more info to each log
    for i in range(len(ret)):
        idx, (ok, reward_log) = ret[i]
        reward_log += f"\nGroup {stat = }"
        ret[i] = (idx, (ok, reward_log))

    return ret
