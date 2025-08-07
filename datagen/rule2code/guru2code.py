# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import batch_completion
from openai import OpenAI
from rich import print as rprint
from tqdm import tqdm


def generate_initial_prompt(seed_data):
    name = seed_data["name"]
    description = seed_data["description"]

    example_section = ""
    if not seed_data.get("code_snippets"):
        if seed_data.get("noncompliant_example"):
            example_section += f"""### Noncompliant Example
```
{seed_data["noncompliant_example"]}
```

"""
        if seed_data.get("compliant_example"):
            example_section += f"""### Compliant Example
```
{seed_data["compliant_example"]}
```
"""

    snippets_section = ""
    if code_snippets := seed_data.get("code_snippets", []):

        selected_snippets = random.sample(code_snippets, min(len(code_snippets), 5))
        snippets_section = (
            "\n## Code Snippets that Contain the Vulnerability for Inspiration\n"
        )
        for i, snippet in enumerate(selected_snippets, 1):
            snippets_section += f"""### Example {i}
{snippet}

"""

    example_section = (
        f"\n## Code Examples\n{example_section}" if example_section else ""
    )

    return f"""Given the following information about the "{name}", please demonstrate complete PYTHON code examples that demonstrate this vulnerability or violate this security pattern:

--- Information for "{name}" ---
## Description of "{name}"
{description}
{example_section}{snippets_section}--- End of Information ---
Now, please create a code example that demonstrates this vulnerability or violates this security pattern.

Notes:
* Create a code example in PYTHON
* The code example you generate **MUST** contain **this** vulnerability or violate this security pattern, and the vulnerability in the code example MUST actually be detected by CodeGuru.
* Format: An example output is demonstrate below:

--- BEGIN OF EXAMPLE ---
## Code Example

```python
[your code here]
```
## Explanation

[Explain why this code contains this vulnerability or violates this security pattern.]
--- END OF EXAMPLE ---"""


def generate_followup_prompt(seed_data):
    name = seed_data["name"]
    description = seed_data["description"]

    example_section = ""
    if not seed_data.get("code_snippets"):
        if seed_data.get("noncompliant_example"):
            example_section += f"""### Noncompliant Example
```
{seed_data["noncompliant_example"]}
```

"""
        if seed_data.get("compliant_example"):
            example_section += f"""### Compliant Example
```
{seed_data["compliant_example"]}
```
"""

    snippets_section = ""
    if code_snippets := seed_data.get("code_snippets", []):

        selected_snippets = random.sample(code_snippets, min(len(code_snippets), 5))
        snippets_section = (
            "\n## Code Snippets that Contain the Vulnerability for Inspiration\n"
        )
        for i, snippet in enumerate(selected_snippets, 1):
            snippets_section += f"""### Example {i}
{snippet}

"""

    example_section = (
        f"\n## Code Examples\n{example_section}" if example_section else ""
    )

    return f"""Now, please create a NEW code example that demonstrates this vulnerability or violates this security pattern.

--- Information for "{name}" ---
## Description of "{name}"
{description}
{example_section}{snippets_section}--- End of Information ---

Notes:
* The code example you generate **MUST** contain this vulnerability or violate this security pattern, and the vulnerability in the code example MUST actually be detected by CodeGuru.
* Format: An example output is demonstrate below:

--- BEGIN OF EXAMPLE ---
## Code Example

```python
[your code here]
```
## Explanation

[Explain why this code contains this vulnerability or violates this security pattern.]
--- END OF EXAMPLE ---"""


def _create_client(remote_api=False):
    if remote_api:
        load_dotenv()
        return None, "bedrock/converse/us.deepseek.r1-v1:0"
    return (
        OpenAI(api_key="none", base_url="http://localhost:30000/v1"),
        "default",
    )


def datagen_for_one_seed(
    seed_data,
    output_file,
    finished_pairs,
    depth=1,
    remote_api=False,
):
    client, model = _create_client(remote_api=remote_api)
    common_args = {
        "model": model,
        "temperature": 0.8,
    }

    if seed_data["name"] in finished_pairs:
        return True

    rprint(f"[bold yellow]Processing: Seed ID: {seed_data['name']}[/bold yellow]")

    messages = [
        {
            "role": "user",
            "content": generate_initial_prompt(seed_data),
        }
    ]

    for i in range(depth):
        if remote_api:
            response = batch_completion(
                model=model,
                messages=[messages],
            )[0]
        else:
            response = client.chat.completions.create(messages=messages, **common_args)

        if response.choices[0].finish_reason == "length":
            break

        content = response.choices[0].message.content.split("</think>")[-1].strip()
        messages.append({"role": "assistant", "content": content})

        if i < depth - 1:
            messages.append(
                {
                    "role": "user",
                    "content": generate_followup_prompt(seed_data),
                }
            )

        if i == depth - 1 or response.choices[0].finish_reason == "length":
            result = {
                "id": seed_data["name"],
                "conversation": messages,
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            finished_pairs.add(seed_data["name"])
            rprint(f"[bold green]Completed: Seed ID: {seed_data['name']}[/bold green]")

    return True


def main(
    parallel=256,
    output_path="outputs/rule2code/guru2code-raw.jsonl",
    depth=1,
    remote_api=False,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    finished_pairs = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                finished_pairs.add(data["id"])
        print(f"Found {len(finished_pairs)} already processed seed_code_ids")

    dataset = load_dataset("purpcode/codeguru-python-detectors", split="test")
    seed_data_list = dataset.to_list()

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for seed_data in seed_data_list:
            if seed_data["name"] not in finished_pairs:
                futures.append(
                    executor.submit(
                        datagen_for_one_seed,
                        seed_data,
                        output_path,
                        finished_pairs,
                        depth,
                        remote_api,
                    )
                )

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
