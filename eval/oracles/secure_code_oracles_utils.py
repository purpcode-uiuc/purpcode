# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import zipfile
from typing import Sequence

SEVERITY_ORDER = {
    "info": 0,
    "informational": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def get_aws_region():
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-2"
    )


def check_min_severity(severity: str, min_sev_level: str = "MEDIUM") -> bool:

    sev_lower = severity.lower()
    min_sev_lower = min_sev_level.lower()

    if sev_lower not in SEVERITY_ORDER:
        raise ValueError(
            f"Invalid severity level: '{severity}'. Must be one of: {', '.join(SEVERITY_ORDER.keys())}"
        )

    if min_sev_lower not in SEVERITY_ORDER:
        raise ValueError(
            f"Invalid min severity level: '{min_sev_level}'. Must be one of: {', '.join(SEVERITY_ORDER.keys())}"
        )

    return SEVERITY_ORDER[sev_lower] >= SEVERITY_ORDER[min_sev_lower]


def encode_base64(original_text: str) -> str:
    if not isinstance(original_text, str):
        raise TypeError("Input must be a string.")
    message_bytes = original_text.encode("utf-8")
    return base64.b64encode(message_bytes).decode("ascii")


def decode_base64(encoded_text: str) -> str:
    if not isinstance(encoded_text, str):
        raise TypeError("Input must be a string.")
    base64_bytes = encoded_text.encode("ascii")
    original_message = base64.b64decode(base64_bytes).decode("utf-8")
    return original_message


def zip_files_flat(file_paths: Sequence[str], output_zip_path: str):
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))
