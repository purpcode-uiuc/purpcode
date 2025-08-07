# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0


from cwe2ovrf import cwe2ovrf_main
from dedup import dedup_main
from post_filter import post_filter_main
from pre_filter import pre_filter_main

GENERATION_MODEL = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
PRE_FILTER_JUDGE_MODEL = "bedrock/us.meta.llama3-3-70b-instruct-v1:0"
POST_FILTER_JUDGE_MODEL = "bedrock/us.deepseek.r1-v1:0"
DEFAULT_OUTPUT_DIRECTORY = "eval/compile_xscode/results/"


def main(
    vuln_rules: str = "cwe",
    output_directory: str = DEFAULT_OUTPUT_DIRECTORY,
    num_questions_per_gen: int = 5,
    gen_model: str = GENERATION_MODEL,
    annotated_filepath: str = None,
    keep_unsure: bool = False,
):

    # Step 1: Generate CWE2ovrf prompts
    generation_path = cwe2ovrf_main(
        vuln_rules_type=vuln_rules,
        output_directory=output_directory,
        num_questions_per_gen=num_questions_per_gen,
        gen_model=gen_model,
    )

    # Step 2: Deduplicate the generated prompts
    dedup_filepath = dedup_main(
        generation_path=generation_path,
    )

    # Step 3: Pre-filter the deduplicated prompts
    # Pre Annotation Filtering
    pre_filtered_filepath = pre_filter_main(
        generation_path=dedup_filepath,
        keep_unsure=keep_unsure,
        judge_model=PRE_FILTER_JUDGE_MODEL,
    )

    # Step 4: Annotation
    #### Step Annotation: this step is not included in the script,
    #### Keep the annotated prompts in the annotated_filepath in the same exact format as the pre-filtered prompts.

    # Step 4: Post-filter annotation Filtering
    post_filter_main(
        generation_path=pre_filtered_filepath,
        annotated_filepath=annotated_filepath,
        keep_unsure=keep_unsure,
        judge_model=POST_FILTER_JUDGE_MODEL,
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
