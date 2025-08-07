# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import html
import json
import os
import re
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from traceback import print_exc

import requests
from openai import OpenAI
from rich import print as rprint
from tqdm import tqdm


def transform_xml_to_code(root):

    def get_indent(style):
        """Return extra indent based on a style like 'margin-left:1em'."""
        match = re.search(r"margin-left\s*:\s*([\d.]+)em", style)
        if match:
            return "  " * int(float(match.group(1)))
        return ""

    def rec(node, current_indent=""):
        parts = []
        if node.text and node.text.strip():
            text = html.unescape(node.text).strip()
            parts.append(current_indent + text)
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "br":
                parts.append("\n" + current_indent)
                if child.tail and child.tail.strip():
                    parts.append(html.unescape(child.tail).strip())
            elif tag == "div":
                extra = get_indent(child.attrib.get("style", ""))
                new_indent = current_indent + extra
                child_text = rec(child, new_indent)
                parts.append("\n" + child_text)
                if child.tail and child.tail.strip():
                    parts.append(
                        "\n" + current_indent + html.unescape(child.tail).strip()
                    )
            else:
                parts.append(rec(child, current_indent))
        return "".join(parts)

    code = rec(root, "")
    code = textwrap.dedent(code)
    lines = code.splitlines()
    cleaned = "\n".join(line.rstrip() for line in lines if line.strip() != "")
    return cleaned


def parse_examples(examples, ns):
    if examples is None:
        return []
    markdown_output = []
    examples = [example for example in examples if example is not None]
    for example in examples:
        text = f"#### Example {len(markdown_output) + 1}\n"
        for elem in example:
            if (
                elem.tag.endswith("Intro_Text") or elem.tag.endswith("Body_Text")
            ) and elem.text:
                text += elem.text.strip() + "\n"

            elif elem.tag.endswith("Example_Code"):
                language = elem.attrib.get("Language", "").lower()
                xhtml_div = elem.find("xhtml:div", ns)
                if xhtml_div is None:
                    continue
                block = transform_xml_to_code(xhtml_div)
                text += f"```{language}\n{block}\n```\n"
        markdown_output.append(text.strip())

    return markdown_output


def create_cwe_information(skip_high_level=True):
    collection = {}  # return value

    cache_file = "cwec_latest.xml.zip"
    if os.path.exists(cache_file):
        print(f"Using cached CWE ZIP file: {cache_file}")
        z = zipfile.ZipFile(cache_file)
    else:
        url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
        print(f"Fetching CWE data from: {url}")
        response = requests.get(url)
        assert response.status_code == 200, f"Failed to fetch {url}"

        with open(cache_file, "wb") as f:
            f.write(response.content)
        print(f"Cached CWE ZIP file to: {cache_file}")

        z = zipfile.ZipFile(cache_file)

    xml_files = [f for f in z.namelist() if f.endswith(".xml")]
    assert len(xml_files) == 1, "Expected exactly one XML file in the zip"
    tree = ET.parse(z.open(xml_files[0]))

    root = tree.getroot()

    # Extract the namespace dynamically
    ns = {
        "cwe": root.tag.split("}")[0].strip("{"),
        "xhtml": "http://www.w3.org/1999/xhtml",
    }
    weaknesses = root.findall(".//cwe:Weakness", ns)
    for weakness in weaknesses:
        markdown_output = []
        weakness_id = weakness.attrib.get("ID")
        name = weakness.attrib.get("Name")
        if weakness.attrib.get("Status") == "Deprecated":
            continue

        # skip Mapping_Notes - Usage == "Prohibited"
        if skip_high_level:
            if (mapping_notes := weakness.find(".//cwe:Mapping_Notes", ns)) is not None:
                usage = mapping_notes.find(".//cwe:Usage", ns)
                if usage is not None and usage.text.lower() != "allowed":
                    print(
                        f"Skipping CWE-{weakness_id} due to prohibited usage: {usage.text}"
                    )
                    continue

        # CWE title
        markdown_output.append(f"## CWE-{weakness_id}: {name}")

        # Description
        markdown_output.append("### Description")
        markdown_output.append(weakness.find(".//cwe:Description", ns).text)
        edesc = weakness.find(".//cwe:Extended_Description", ns)
        if edesc is not None and edesc.text and edesc.text.strip():
            markdown_output.append(edesc.text)

        # Examples
        examples = weakness.findall(".//cwe:Demonstrative_Example", ns)
        if expmkd := parse_examples(examples, ns):
            markdown_output.append("### Examples")
            markdown_output.extend(expmkd)

        # Notes
        if (notes := weakness.find(".//cwe:Notes", ns)) is not None:
            note_text = "### Notes\n"
            for note in notes:
                if note.text and note.text.strip():
                    note_text += "\n* " + note.text.strip()
            markdown_output.append(note_text)

        # Common_Consequences
        if (conseq := weakness.find(".//cwe:Common_Consequences", ns)) is not None:
            conseq_text = ""
            for csq in conseq:
                csq = csq.find(".//cwe:Note", ns)
                if csq is not None and csq.text and csq.text.strip():
                    conseq_text += "\n* " + csq.text.strip()
            if conseq_text.strip():
                markdown_output.append("### Common Consequences\n" + conseq_text)

        markdown_result = "\n\n".join(markdown_output)

        collection[weakness_id] = markdown_result

    return collection


def generate_initial_prompt(cwe_id, markdown):
    return (
        f"""Given the following markdown text for CWE-{cwe_id}, please demonstrate scenarios (either abstract or realistic) and complete PYTHON code examples that demonstrate this vulnerability:

--- Markdown Content for CWE-{cwe_id} ---
{markdown}
--- End of Markdown Content ---
"""
        + r"""Now, please create a scenario and PYTHON code example that demonstrates this vulnerability.

Notes:
* Create a scenario under applicable platforms
* Format: An example output is demonstrate below:

--- BEGIN OF EXAMPLE ---
## Code Example

```python
{code}
```
## Explanation

Explain why this code is vulnerable.
--- END OF EXAMPLE ---"""
    )


def generate_followup_prompt():
    return r"""Now, please create a NEW scenario and PYTHON code example that demonstrates this vulnerability.

Notes:
* Creativity & Diversity: Create new scenarios and use different platforms (if applicable)
* Format: An example output is demonstrate below:

--- BEGIN OF EXAMPLE ---
## Code Example

```python
{code}
```
## Explanation

Explain why this code is vulnerable.
--- END OF EXAMPLE ---
"""


def _create_client(remote_api=False):
    if remote_api:
        return OpenAI(base_url="https://api.deepseek.com"), "deepseek-reasoner"
    # Otherwise sglang
    return OpenAI(api_key="none", base_url="http://0.0.0.0:30000/v1"), "default"


def datagen_for_one_cwe(cwe_id, markdown, depth, remote_api=False):
    assert depth > 0

    client, model = _create_client(remote_api=remote_api)
    common_args = {"model": model, "temperature": 0.6}

    rprint(f"[bold yellow]Processing: CWE ID: {cwe_id}[/bold yellow]")

    messages = [
        {
            "role": "user",
            "content": generate_initial_prompt(cwe_id, markdown)
            + generate_followup_prompt(),
        }
    ]
    choice = client.chat.completions.create(messages=messages, **common_args).choices[0]

    if choice.finish_reason == "length":
        return {"id": cwe_id, "conversation": messages}

    messages.append(
        {
            "role": "assistant",
            "content": choice.message.content.split("</think>")[-1].strip(),
        }
    )

    for i in range(depth - 1):
        messages.append({"role": "user", "content": generate_followup_prompt()})
        choice = client.chat.completions.create(
            messages=messages, **common_args
        ).choices[0]

        if choice.finish_reason == "length":
            break

        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content.split("</think>")[-1].strip(),
            }
        )

    rprint(f"[bold green]Completed: CWE ID: {cwe_id}[/bold green]")
    return {"id": cwe_id, "conversation": messages}


def main(
    parallel=256,
    output_path="outputs/rule2code/cwe2code-raw.jsonl",
    depth=1,
    remote_api=False,
):
    collection = create_cwe_information()
    # each line: cwe_id, conversation

    finished = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                finished.add(json.loads(line)["id"])

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for cwe_id, markdown in collection.items():
            if cwe_id in finished:
                continue
            futures.append(
                executor.submit(
                    datagen_for_one_cwe, cwe_id, markdown, depth, remote_api
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            try:
                result_json = json.dumps(result, ensure_ascii=False)
            except Exception:
                print("Error found -- ")
                print(f"{result = }")
                print_exc()
                continue

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(result_json + "\n")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
