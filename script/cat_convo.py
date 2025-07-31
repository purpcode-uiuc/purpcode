# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import json
import random

from datasets import load_dataset
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


def visualize_conversation(conversation):
    table = Table(title="Conversation", show_lines=True)
    table.add_column("Role", style="bold cyan", no_wrap=True)
    table.add_column("Content", style="")

    for message in conversation:
        role = message.get("role", "unknown")
        content = message.get("content", "").encode("utf-8", "replace").decode()
        # Optionally style based on role
        if role == "assistant":
            role_style = "green"
        elif role == "user":
            role_style = "magenta"
        elif role == "system":
            role_style = "yellow"
        else:
            role_style = "white"

        table.add_row(f"[{role_style}]{role}[/{role_style}]", escape(content.strip()))

    console.print(table)


def main(
    path,
    shuffle: bool = False,
    multi: bool = False,
    prefix: str = None,
    split: str = "train",
    include_kw: str = "",
):
    try:
        with open(path, "r") as file:
            conversations = [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        conversations = load_dataset(path, split=split)

    print(f"{len(conversations)} messages in the conversation:")

    if shuffle:
        random.shuffle(conversations)

    for data in conversations:
        if prefix and not data.get("task_id", "").startswith(prefix):
            continue
        if include_kw not in "".join([r["content"] for r in data["messages"]]):
            continue

        conversation = data.get("conversation", data.get("messages"))
        if multi and len([m for m in conversation if m["role"] != "system"]) <= 2:
            continue
        if "task_id" in data:
            console.print(f"[bold]Task ID:[/bold] {data['task_id']}")
        if conversation:
            visualize_conversation(conversation)
            console.print("\n" + "=" * 50 + "\n")
        else:
            console.print("[yellow]No conversation found in this line.[/yellow]")

        input()  # Wait for user input to continue to the next line


# Example:
# python scripts/cat_convo.py --path ./cwe2code_raw.jsonl
if __name__ == "__main__":
    from fire import Fire

    Fire(main)
