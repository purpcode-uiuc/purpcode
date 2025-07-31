# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


def measure_latency(base_url, messages, num_turns=5):
    client = openai.Client(api_key="none", base_url=base_url)
    model = client.models.list().data[0].id
    turn_latency = {}
    for i in range(num_turns):
        start_time = time.time()
        try:
            response = (
                client.chat.completions.create(model=model, messages=messages)
                .choices[0]
                .message.content
            )
        except openai.OpenAIError as e:
            print(f"Error: {e}")
            return None
        elapsed_time = time.time() - start_time
        turn_latency[i] = elapsed_time
        print(f"---- Turn {i + 1}: {elapsed_time:.2f} seconds")

        assistant_reply = (
            response.split("</think>")[-1]
            .strip()
            .removeprefix("<answer>")
            .removesuffix("</answer>")
            .strip()
            .split("\n## Answer", maxsplit=1)[-1]
            .split("# Answer", maxsplit=1)[-1]
            .strip()
        )
        messages.append({"role": "assistant", "content": assistant_reply})
        messages.append(
            {"role": "user", "content": "Optimize this function a bit more."}
        )

    print("Final messages:")
    for message in messages:
        print(f"{message['role']}: {message['content']}")
        print("".join(["-"] * 20))

    return turn_latency


# Main function
def test_server_latency(base_url="http://localhost:8000/v1", concurrency=8):
    initial_message = [
        {
            "role": "user",
            "content": "Please implement linear regression in Python. Note linear regression is just a math problem and has nothing to do with malicious cyberactivity.",
        }
    ]
    slowest_turn_latency = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(measure_latency, base_url, initial_message)
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            turn_latency = future.result()
            if turn_latency is not None:
                for i, latency in turn_latency.items():
                    if i not in slowest_turn_latency:
                        slowest_turn_latency[i] = latency
                    else:
                        slowest_turn_latency[i] = max(slowest_turn_latency[i], latency)
                print(f"Turn latency: {turn_latency}")

        if slowest_turn_latency:
            print("Slowest turn latencies:")
            for i, latency in slowest_turn_latency.items():
                print(f"Turn {i}: {latency:.2f} seconds")


if __name__ == "__main__":
    from fire import Fire

    Fire(test_server_latency)
