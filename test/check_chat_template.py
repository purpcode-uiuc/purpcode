# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer


def print_tokens(encoded, tokenizer):
    """Print the tokens and their IDs in a readable format."""
    tokens = []
    for token_id in encoded.input_ids:
        token = tokenizer.decode([token_id])
        tokens.append(f"{token_id}: '{token}'")
    return tokens


def check_chat_template(model_name, messages):
    print("=" * 60)
    print(f"Checking chat template for: {model_name}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"{tokenizer.chat_template}")
    print("=" * 60)

    print("\n\n")
    print("=" * 60)
    print("Example chat messages:")
    print("=" * 60)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(formatted_prompt)
    print("=" * 60)


def main(model_path: str):
    example_messages = [
        {"role": "system", "content": "[...SYSTEM_MESSAGES...]"},
        {"role": "user", "content": "[...USER_MESSAGES...]"},
        {"role": "assistant", "content": "[...ASSISTANT_MESSAGES...]"},
        {"role": "user", "content": "[...USER_MESSAGES...]"},
    ]
    check_chat_template(model_path, example_messages)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
