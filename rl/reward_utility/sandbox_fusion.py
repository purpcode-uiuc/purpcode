# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from sandbox_fusion import RunCodeRequest, RunCodeResponse, RunStatus, run_code

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 15


def skipline(line: str) -> bool:
    if not line.strip():
        return True
    for skipper in [
        "UserWarning:",
        "`nameko test`",
        "This warning can be suppressed",
        "warning.warn",
    ]:
        if skipper in line:
            return True

    return False


def render_error(response: RunCodeResponse):
    log = (
        _ERROR_MSG_PREFIX
        + "\n===== STDOUT =====\n"
        + response.run_result.stdout
        + "\n===== STDERR =====\n"
        + response.run_result.stderr
    )
    return "\n".join(l for l in log.split("\n") if not skipline(l))


def code_exec_sandbox_fusion(
    code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None
) -> Tuple[bool, str]:
    timeout = _DEFAULT_TIMEOUT_SECONDS

    if pytest:
        # remove line starting with "from solution import..."
        pytest_without_import = "\n".join(
            line
            for line in pytest.split("\n")
            if not line.startswith("from solution import")
        )
        response = run_code(
            RunCodeRequest(
                code=code + "\n" + pytest_without_import,
                timeout=timeout,
                language="pytest",
            )
        )
    else:
        response = run_code(
            RunCodeRequest(code=code, stdin=stdin, timeout=timeout, language="python")
        )

    if response.status != RunStatus.Success:
        return False, render_error(response)

    return True, response.run_result.stdout
