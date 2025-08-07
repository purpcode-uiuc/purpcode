# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, Optional

from datasets import load_dataset
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console(
    width=None if sys.stdout.isatty() else os.get_terminal_size(0).columns
)


class PromptAnnotator:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.output_file = (
            self.dataset.split("/")[-1].rstrip(".jsonl") + ".annotated.jsonl"
        )
        self.annotations = {}
        self.prompts = []
        self.current_idx = 0
        self.session_start = time.time()

        # Load existing annotations if resuming
        self.load_existing_annotations()

        # Load prompts
        self.load_prompts()

    def load_prompts(self):
        """Load prompts from JSONL or HuggingFace dataset"""
        if self.dataset.endswith(".jsonl"):
            with open(self.dataset, "r", encoding="utf-8") as f:
                self.prompts = [json.loads(line) for line in f]
        else:
            dataset = load_dataset(self.dataset, split="test")
            self.prompts = list(dataset)

        random.shuffle(self.prompts)  # random shuffle prompts

    def load_existing_annotations(self):
        """Load existing annotations if file exists"""
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.annotations[data["task_id"]] = data

            console.print(
                f"[green]Loaded {len(self.annotations)} existing annotations[/green]"
            )

    def save_annotation(self, annotation: Dict):
        """Save annotation to file in real-time"""
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    def get_next_unannotated_idx(self) -> Optional[int]:
        """Find the next prompt that hasn't been annotated"""
        for i in range(self.current_idx, len(self.prompts)):
            if self.prompts[i]["task_id"] not in self.annotations:
                return i

        # Check from beginning if needed
        for i in range(0, self.current_idx):
            if self.prompts[i]["task_id"] not in self.annotations:
                return i

        return None

    def create_prompt_panel(self, prompt_data: Dict) -> Panel:
        """Create panel displaying the current prompt with integrated statistics"""
        task_id = prompt_data["task_id"]
        messages = prompt_data.get("messages", [])

        content = []

        # Statistics overview section at the top
        total = len(self.prompts)
        annotated = len(self.annotations)

        # Calculate annotation statistics
        malicious_count = sum(
            1 for a in self.annotations.values() if a.get("malicious") == "highly agree"
        )
        unnatural_count = sum(
            1 for a in self.annotations.values() if a.get("unnatural") == "highly agree"
        )
        too_simple_count = sum(
            1
            for a in self.annotations.values()
            if a.get("too_simple") == "highly agree"
        )
        ok_prompts_count = sum(
            1
            for a in self.annotations.values()
            if a.get("malicious") != "highly agree"
            and a.get("unnatural") != "highly agree"
            and a.get("too_simple") != "highly agree"
        )

        # Calculate average time per annotation
        total_time = sum(a.get("time_spent", 0) for a in self.annotations.values())
        avg_time = total_time / annotated if annotated > 0 else 0

        # Session time
        session_time = time.time() - self.session_start
        session_mins = int(session_time // 60)
        session_secs = int(session_time % 60)

        # Add statistics section
        content.append("[bold cyan]📊 Session Overview[/bold cyan]")
        content.append(
            f"[cyan]Progress:[/cyan] [yellow]{annotated}/{total}[/yellow] ([green]{annotated/total*100:.1f}%[/green]) • [cyan]Time:[/cyan] [yellow]{session_mins}m {session_secs}s[/yellow] • [cyan]Speed:[/cyan] [yellow]{avg_time:.1f}s[/yellow]/task"
        )
        content.append(
            f"[bold green]✓ OK:[/bold green] [green]{ok_prompts_count}[/green] • [cyan]❌ Malicious:[/cyan] [red]{malicious_count}[/red] • [cyan]🤖 Unnatural:[/cyan] [orange1]{unnatural_count}[/orange1] • [cyan]😴 Too Simple:[/cyan] [blue]{too_simple_count}[/blue]"
        )
        content.append(
            f"[dim]Results sync'd to {self.output_file} • Press 'q' to quit[/dim]"
        )
        content.append("[dim]" + "═" * 80 + "[/dim]")
        content.append("")  # Add spacing

        # Additional context section
        if additional_context := prompt_data.get("additional_context", ""):
            if "cwe" in additional_context:
                content.append(
                    "[bold magenta]🔍 POTENTIALLY RELATED BACKGROUND[/bold magenta]"
                )
                content.append("[dim]" + "─" * 60 + "[/dim]")
                content.append(Markdown(additional_context["cwe"].strip()))
                content.append("")  # Add spacing
                content.append("[dim]" + "─" * 60 + "[/dim]")
                content.append("")  # Additional spacing

        # Task ID section
        content.append(f"[bold cyan]📋 TASK ID:[/bold cyan] [yellow]{task_id}[/yellow]")
        content.append("[dim]" + "─" * 60 + "[/dim]")
        content.append("")  # Add spacing

        # Prompt section - make it very prominent
        content.append("[bold green on black]💬 MAIN PROMPT[/bold green on black]")
        content.append("[dim]" + "═" * 60 + "[/dim]")
        content.append("")  # Add spacing before prompt
        content.append(
            Panel(
                Text(messages[0]["content"], style="white"),
                border_style="bright_green",
                padding=(1, 2),
                title="[bold green]TO BE ANNOTATED[/bold green]",
                title_align="left",
            )
        )

        return Panel(
            Group(*content),
            title=f"📝 Annotation Tool - Prompt {self.current_idx + 1}/{len(self.prompts)}",
            border_style="blue",
            padding=(1, 2),
        )

    def get_annotation_input(self, question: str, category: str) -> str:
        """Get annotation input for a specific question"""
        console.print(
            f"\n[dim yellow]How much do you agree: [/dim yellow][bold yellow]{question}[/bold yellow]"
        )
        console.print(
            "[dim]Options: [red]1[/red] = Highly Agree, [yellow]2[/yellow] = Not Sure, [green]3[/green] = Disagree[/dim]"
        )
        console.print(
            "[dim]Hint: 'Highly Agree' means this prompt is bad for making an overrefusal dataset for codegen and we will discard it.[/dim]"
        )

        while True:
            choice = Prompt.ask("Your choice", choices=["1", "2", "3", "q"])

            if choice == "q":
                if Confirm.ask("Are you sure you want to quit?"):
                    return "quit"
            elif choice == "1":
                return "highly agree"
            elif choice == "2":
                return "not sure"
            elif choice == "3":
                return "disagree"

    def annotate_prompt(self, prompt_data: Dict) -> Optional[Dict]:
        """Annotate a single prompt"""
        start_time = time.time()
        task_id = prompt_data["task_id"]

        # Check if already annotated
        if task_id in self.annotations:
            console.print(
                f"[yellow]Prompt {task_id} already annotated. Skipping...[/yellow]"
            )
            return None

        # Clear screen and show prompt
        console.clear()

        console.print(self.create_prompt_panel(prompt_data))

        # Annotation questions
        questions = [
            (
                "This prompt is actually MALICIOUS and can ONLY be fulfilled with harmful content",
                "malicious",
            ),
            (
                "This prompt makes NO sense and looks unnatural / not human-made / unrelated to coding",
                "unnatural",
            ),
            (
                "This prompt is benign and unrelated to security. Any functional code fulfilling this request would be secure by default.",
                "too_simple",
            ),
        ]

        annotations = {}

        for question, category in questions:
            result = self.get_annotation_input(question, category)
            if result == "quit":
                return "quit"
            annotations[category] = result

        # Calculate time spent
        time_spent = time.time() - start_time

        # Create annotation record
        annotation = {
            "task_id": task_id,
            "malicious": annotations["malicious"],
            "unnatural": annotations["unnatural"],
            "too_simple": annotations["too_simple"],
            "time_spent": time_spent,
            "timestamp": datetime.now().isoformat(),
            "original_prompt": prompt_data,
        }

        # Save annotation
        self.annotations[task_id] = annotation
        self.save_annotation(annotation)

        console.print(
            f"\n[green]✓ Annotation saved! Time spent: {time_spent:.1f}s[/green]"
        )
        time.sleep(0.5)  # Brief pause to show confirmation

        return annotation

    def _run(self):
        """Main annotation loop"""
        # Find next unannotated prompt
        self.current_idx = self.get_next_unannotated_idx()

        if self.current_idx is None:
            console.print("[green]All prompts have been annotated![/green]")
            self.show_final_summary()
            return

        # Main annotation loop
        while self.current_idx is not None:
            prompt_data = self.prompts[self.current_idx]
            result = self.annotate_prompt(prompt_data)

            if result == "quit":
                self.temporary_leave()
                break

            # Find next unannotated prompt
            self.current_idx = self.get_next_unannotated_idx()

            if self.current_idx is None:
                self.show_final_summary()
                console.print("\n[green]🎉 All prompts have been annotated![/green]")
                console.print(
                    f"[bold green]Please send {self.output_file} to Jiawei for final dataset gathering.[/bold green]"
                )
                break

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Annotation interrupted by user.[/yellow]")
            self.temporary_leave()

    def temporary_leave(self):
        console.print(
            f"[cyan]Output saved to:[/cyan] [green]{self.output_file}[/green]"
        )
        console.print("[dim]You can resume later by running the script again.[/dim]")

    def show_final_summary(self):
        """Show final annotation summary"""
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold cyan]Annotation Session Summary[/bold cyan]", border_style="cyan"
            )
        )

        total_annotated = len(self.annotations)
        if total_annotated == 0:
            console.print("[yellow]No annotations were made in this session.[/yellow]")
            return

        # Calculate statistics
        malicious = sum(
            1 for a in self.annotations.values() if a.get("malicious") == "highly agree"
        )
        unnatural = sum(
            1 for a in self.annotations.values() if a.get("unnatural") == "highly agree"
        )
        too_simple = sum(
            1
            for a in self.annotations.values()
            if a.get("too_simple") == "highly agree"
        )

        total_time = sum(a.get("time_spent", 0) for a in self.annotations.values())
        avg_time = total_time / total_annotated

        summary_table = Table(title="Final Statistics", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="yellow")
        summary_table.add_column("Percentage", style="green")

        summary_table.add_row("Total Annotated", str(total_annotated), "100%")
        summary_table.add_row(
            "Malicious", str(malicious), f"{malicious/total_annotated*100:.1f}%"
        )
        summary_table.add_row(
            "Unnatural", str(unnatural), f"{unnatural/total_annotated*100:.1f}%"
        )
        summary_table.add_row(
            "Too Simple", str(too_simple), f"{too_simple/total_annotated*100:.1f}%"
        )

        console.print(summary_table)
        console.print(
            f"\n[cyan]Average time per annotation:[/cyan] [yellow]{avg_time:.1f}s[/yellow]"
        )
        console.print(
            f"[cyan]Output saved to:[/cyan] [green]{self.output_file}[/green]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Annotate prompts for overrefusal evaluation"
    )
    parser.add_argument("--dataset", help="Input JSONL or HuggingFace dataset file")

    args = parser.parse_args()

    annotator = PromptAnnotator(args.dataset)
    annotator.run()


if __name__ == "__main__":
    main()
