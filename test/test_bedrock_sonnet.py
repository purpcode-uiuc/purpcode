# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

response = completion(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=[{"content": "What are you doing?", "role": "user"}],
)

print(response.choices[0].message.content)
