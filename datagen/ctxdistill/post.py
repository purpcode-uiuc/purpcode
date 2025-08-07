# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

# Dataset post-processing: Merge conversations, convert file to markdown format, and filter out invalid conversations

import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import fire
from datasets import Dataset, DatasetDict
from markdown_it import MarkdownIt
from rich import print as rprint
from tqdm import tqdm

from utils.verify_text import validate_response_structure


def _parse_analysis_text(markdown_text: str) -> dict[str, str]:
    md = MarkdownIt()
    tokens = md.parse(markdown_text)
    lines = markdown_text.split("\n")

    # --- Pass 1: Find the minimum header level in the document ---
    min_level = min(
        [float("inf")]
        + [int(token.tag[1:]) for token in tokens if token.type == "heading_open"]
    )
    if min_level == float("inf"):
        return {}

    # --- Pass 2: Extract sections based on the minimum header level found ---
    top_level_tag = f"h{min_level}"
    parsed_dict = {}
    top_level_sections = []

    # Find all header tokens that match the determined top-level tag
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == top_level_tag:
            section_title = tokens[i + 1].content
            # The map gives [start_line, end_line] of the header itself.
            # Content starts on the line immediately following the header.
            content_start_line = token.map[1]

            # The section's start line (for finding the end boundary later)
            section_start_line = token.map[0]

            top_level_sections.append(
                {
                    "title": section_title,
                    "section_start_line": section_start_line,
                    "content_start_line": content_start_line,
                }
            )

    # Determine the content for each top-level section
    for i, section in enumerate(top_level_sections):
        # The content for the current section starts after its header
        content_start = section["content_start_line"]

        # And ends just before the next top-level header begins
        if (i + 1) == len(top_level_sections):  # is_last_section
            content_end = len(lines)
        else:
            content_end = top_level_sections[i + 1]["section_start_line"]

        content_block = "\n".join(lines[content_start:content_end]).strip()
        parsed_dict[section["title"]] = content_block

    return {k.lower(): v for k, v in parsed_dict.items()}


def transform_md_thinking(response: str) -> str | None:
    sections = _parse_analysis_text(response)
    thinking = ["## Analysis"]
    answer = ["## Answer"]
    expected_titles = ["intent analysis", "safety analysis", "proposed safe response"]
    if len(sections) != len(expected_titles):
        return None

    for i, (title, content) in enumerate(sections.items()):
        if title != expected_titles[i]:
            return None

        if i in [0, 1]:
            thinking.append(content)
        elif i == 2:
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            if content.startswith("'") and content.endswith("'"):
                content = content[1:-1]
            if content.startswith("```plaintext"):
                content = content[12:]
                content = content.removesuffix("```")
            answer.append(content)

    if "" in thinking:
        rprint("[red]Error: Empty content in thinking sections[/]")
        return None
    if "" in answer:
        rprint("[red]Error: Empty content in answer sections[/]")
        return None
    final_response = "\n".join(thinking) + "\n" + "\n".join(answer)
    if not validate_response_structure(final_response):
        rprint("[red]Error: Invalid response structure[/]")
        return None
    return final_response


def extract_safe_answer(response: str) -> str:
    response = transform_md_thinking(response) or response
    return response.split("## Answer")[-1].strip()


def _transform_generation(generation):
    generation = deepcopy(generation)
    for row in generation["messages"]:
        if row["role"] == "assistant":
            if new_content := transform_md_thinking(row["content"]):
                row["content"] = new_content
            else:
                # Signal that this was a "bad conversation"
                return None
    return generation


def thinking_format(generations):
    transformed = []

    with ProcessPoolExecutor(max_workers=48) as executor:
        futures = [
            executor.submit(_transform_generation, generation)
            for generation in generations
        ]

        for future in tqdm(
            as_completed(futures), desc="Transforming generations", total=len(futures)
        ):
            if (generation := future.result()) is not None:
                transformed.append(generation)

    rprint(
        f"[green]Transformed {len(transformed)} / {len(generations)} = {len(transformed) / len(generations):.2%} conversations![/]"
    )
    return transformed


SCORE_THRESHOLD = 0.5


def check_rewards(generations, reward_cache_file, batch_size=4096):
    from rl.grouped_reward import compute_score

    hash2score = {}
    if os.path.exists(reward_cache_file):
        with open(reward_cache_file, "r") as f:
            for line in f:
                item = json.loads(line)
                hash2score[item["hash"]] = item["score"]
        rprint(
            f"[green]Loaded {len(hash2score)} cached rewards from {reward_cache_file}[/]"
        )

    hash_fn = lambda gen: hashlib.sha256(
        json.dumps(gen, sort_keys=True).encode()
    ).hexdigest()
    cached_verified_generations = [
        {**gen, "score": hash2score[hash_fn(gen)]}
        for gen in generations
        if hash2score.get(hash_fn(gen), 0) >= SCORE_THRESHOLD
    ]
    todo_generations = [gen for gen in generations if hash_fn(gen) not in hash2score]

    batches = [
        todo_generations[i : i + batch_size]
        for i in range(0, len(todo_generations), batch_size)
    ]

    failed_conv_ids = set()
    verified_generations = []
    for batch in tqdm(batches):
        responses = [gen["messages"][-1]["content"] for gen in batch]
        extra_info = [
            {"oracles": None, "prompt": gen["messages"][-2]["content"]} for gen in batch
        ]
        ground_truth = [None for _ in batch]
        for i, gen in enumerate(batch):
            if gen["test_code"]:
                ground_truth[i] = json.dumps({"pytest": gen["test_code"]})
                extra_info[i]["oracles"] = ["correctness"]
            elif "secqa" in gen["task_id"].split(":")[0]:
                extra_info[i]["oracles"] = ["noreject"]
            elif "mal-event" in gen["task_id"].split(":")[0]:
                extra_info[i]["oracles"] = ["general-safety"]
            elif (
                "cwe2inst" in gen["task_id"].split(":")[0]
                or "code2inst" in gen["task_id"].split(":")[0]
                or "guru2inst" in gen["task_id"].split(":")[0]
                or "vul-code" in gen["task_id"].split(":")[0]
                or "vul2prompt" in gen["task_id"].split(":")[0]
            ):
                extra_info[i]["oracles"] = ["codesec"]
            else:
                raise ValueError(
                    f"Unknown task_id: {gen['task_id']}, please check the task_id format."
                )

        scores = compute_score(responses, ground_truth, None, extra_info)
        for i, score in enumerate(scores):
            if score >= SCORE_THRESHOLD:
                verified_generations.append({**batch[i], "score": score})
            else:
                failed_conv_ids.add(batch[i]["task_id"].rsplit(":", maxsplit=1)[0])
            with open(reward_cache_file, "a") as f:
                f.write(json.dumps({"hash": hash_fn(batch[i]), "score": score}) + "\n")

    # Filter out failed conversations
    verified_generations = verified_generations + cached_verified_generations
    n_verified = len(verified_generations)
    rprint(
        f"[green]Verified {n_verified} / {len(generations)} = {n_verified / len(generations):.2%} conversations![/]"
    )
    return verified_generations


def split_into_prefix_subturns(generations):
    def _split(messages):
        # assert all user-assistant turns
        assert len(messages) % 2 == 0
        assert all(m["role"] in ["assistant", "user"] for m in messages)
        result = []
        if len(messages) < 2:
            return []
        for i in range(2, len(messages) + 1, 2):
            if i % 2 == 0:
                result.append(messages[:i])
        return result

    new_generations = []
    for gen in generations:
        subturns = _split(gen["messages"])
        for sub in subturns:
            new_gen = deepcopy(gen)
            new_gen["messages"] = sub
            new_gen["task_id"] += f":{len(sub)}"
            new_generations.append(new_gen)
    return new_generations


def main(generation_path: str, push_to_hub: bool = False):
    with open(generation_path, "r") as f:
        generations = [json.loads(line) for line in f]

    sample_per_prompt = 1
    for gen in generations:
        if "@" in gen["task_id"]:
            sample_per_prompt = max(
                sample_per_prompt, int(gen["task_id"].split("@")[-1])
            )

    generations = thinking_format(generations)
    generations = split_into_prefix_subturns(
        generations
    )  # necessary as we will cut thinking parts in earilier turns
    reward_cache_file = generation_path.replace(".jsonl", ".reward_cache.jsonl")
    print(f"{len(generations)} samples after splitting into prefix subturns")
    verified_generations = []

    for generation in check_rewards(generations, reward_cache_file=reward_cache_file):
        generation = deepcopy(generation)
        # only preserve content and role
        generation["messages"] = [
            {"role": m["role"], "content": m["content"], "train": False}
            for m in generation["messages"]
        ]

        # pop system prompt
        if generation["messages"][0]["role"] == "system":
            generation["messages"].pop(0)

        # set trainable turns and remove thinking parts of past turns
        for m in generation["messages"][:-1]:
            if m["role"] == "assistant":
                m["content"] = m["content"].split("# Answer")[-1].strip()
        generation["messages"][-1]["train"] = True
        verified_generations.append(
            {
                "task_id": generation["task_id"],
                "messages": generation["messages"],
                "score": generation["score"],
            }
        )

    # original_task_id -> {sample_id}
    task_id_to_vsample_ids = defaultdict(list)
    for vgen in verified_generations:
        vtask_id = vgen["task_id"]  # {tasl_id}@{sample_id}:{turn_id}
        task_id = vtask_id.rsplit("@", maxsplit=1)[0]
        sample_id = vtask_id.rsplit(":", maxsplit=1)[0].rsplit("@", maxsplit=1)[-1]
        turn_id = vtask_id.rsplit(":", maxsplit=1)[-1]
        task_id = f"{task_id}:{turn_id}"  # augment task_id with turn_id
        task_id_to_vsample_ids[task_id].append(
            {
                "sample_id": sample_id,
                "messages": vgen["messages"],
                "score": vgen["score"],
            }
        )

    # write task_ids that have 70%+ verfified samples
    with open(
        f"{generation_path.split('/')[-1].split('.distill')[0]}.ez_task_ids.txt", "w"
    ) as f:
        for task_id, vsample_ids in task_id_to_vsample_ids.items():
            if len(vsample_ids) >= sample_per_prompt * 0.7:
                f.write(task_id + "\n")

    # select one sample per task_id
    verified_generations = []
    for task_id, vsamples in task_id_to_vsample_ids.items():
        selected = vsamples[0]
        for vsample in vsamples[1:]:
            if vsample["score"] > selected["score"] or (
                vsample["score"] == selected["score"]
                and len(vsample["messages"][-1]["content"])
                < len(selected["messages"][-1]["content"])
            ):
                selected = vsample
        verified_generations.append(
            {
                "task_id": task_id,
                "messages": selected["messages"],
                "score": selected["score"],
            }
        )

    print(
        f"Selected {len(verified_generations)} BoN samples from {len(task_id_to_vsample_ids)} task_id:turn"
    )

    dataset_name = f"ctxdistill-verified-{generation_path.split('/')[-1].split('.distill')[0]}-{len(verified_generations) // 1000}k"
    with open(f"{dataset_name}.jsonl", "w") as f:
        for generation in verified_generations:
            f.write(json.dumps(generation, default=str) + "\n")

    if push_to_hub:
        dataset = DatasetDict({"train": Dataset.from_list(verified_generations)})
        dataset.push_to_hub(f"purpcode/{dataset_name}", private=True)
        rprint(f"[green]Pushed dataset to Hugging Face Hub: {dataset_name}[/]")


if __name__ == "__main__":
    fire.Fire(main)
