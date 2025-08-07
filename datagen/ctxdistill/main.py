# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from datagen.ctxdistill.ctxdistill import run_distillation

DEFAULT_SAMPLE_RATIO_MEDIUM = 0.25

# Core logic behind:
# Pick a subset of data {D} total prompts -> {N} prompts
# Ctx based sampling -> {N * S} where S is the sample size
# Verfication and pick best of S -> getting {K} prompts with verified responses
# SFT over {K} prompt-response pairs
# RL on {D} - {N where most responses are right} prompts, i.e., rm very easy prompts


def main(**kwargs):
    single_turn_datasets = [
        # mal event / single-turn
        (
            "purpcode/mal-event-jailbreak-single-oss-16k",
            DEFAULT_SAMPLE_RATIO_MEDIUM,
            4096,
        ),
        (
            "purpcode/mal-event-seed-attack-oss-24k",
            DEFAULT_SAMPLE_RATIO_MEDIUM,
            4096,
        ),
        # vul code / single-turn
        ("purpcode/vul2prompt-general-oss-26k", DEFAULT_SAMPLE_RATIO_MEDIUM, 4096),
        ("purpcode/vul2prompt-benign2vul-oss-21k", DEFAULT_SAMPLE_RATIO_MEDIUM, 4096),
        ("purpcode/vul2prompt-vul2vul-oss-21k", DEFAULT_SAMPLE_RATIO_MEDIUM, 2048),
        (
            "purpcode/vul2prompt-jailbreaking-oss-11k",
            DEFAULT_SAMPLE_RATIO_MEDIUM,
            2048,
        ),
        # utility
        ("purpcode/secqa_utility_train", DEFAULT_SAMPLE_RATIO_MEDIUM, 4096),
        ("KodCode/KodCode-V1-SFT-R1", DEFAULT_SAMPLE_RATIO_MEDIUM, 8192),
    ]
    multi_turn_datasets = [
        # vul code / multi-turn
        ("purpcode/vul2prompt-multi-oss-5k", 1.0),
        # mal event / multi-turn
        ("purpcode/mal-event-fitd-multi-turn-oss-2k", 1.0),
    ]

    datasets = single_turn_datasets + multi_turn_datasets
    run_distillation(datasets=datasets, **kwargs)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
