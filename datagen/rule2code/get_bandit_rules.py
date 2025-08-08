# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

"""
Scrape all flake8-bandit (Sxxx) rules from the Ruff docs.

Output: bandit_rules.json ― list[{code,name,short_msg,url,full_text}]
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urljoin

import fire
import requests
from bs4 import BeautifulSoup, Tag

SITE = "https://docs.astral.sh"
RULES_DIR = f"{SITE}/ruff/rules/"  # <-- NEW
LISTING = f"{RULES_DIR}#flake8-bandit-s"
HEADERS = {"User-Agent": "bandit-scraper/0.2 (+https://github.com/you)"}

SECTION_HEADINGS = {
    "what it does": "what_it_does",
    "why is this bad?": "why_bad",
    "example": "example_bad",
    "use instead:": "example_good",
}

TITLE_RE = re.compile(r"^(?P<title>.+?)\s+\((?P<code>S\d{3})\)$", re.I)

BANDIT_RE = re.compile(r"\b[bB](\d{3})\b")  # matches B605, b401, …


def load_ruff_rules(path: str | Path = "bandit_rules.json") -> Dict[str, dict]:
    """code → full rule dict (O(1) lookup)."""
    rules = json.loads(Path(path).read_text())
    return {r["code"]: r for r in rules}  # e.g. "S605": {...}


def bandit_id(text: str) -> Optional[str]:
    """Return 'B605' (str) or None."""
    m = BANDIT_RE.search(text)
    return f"B{m.group(1)}" if m else None


def ruff_code(bid: str) -> str:
    """'B605' → 'S605' (flake8-bandit / Ruff code)."""
    return "S" + bid[1:]


def enrich(recs: Iterable[dict], rules: Dict[str, Any]) -> Iterable[dict]:
    """Yield each rec + attached Ruff rule (or None)."""
    for rec in recs:
        bid = bandit_id(rec["recommendation_text"])
        rc = ruff_code(bid) if bid else None
        rec["bandit_id"] = bid
        rec["ruff_code"] = rc
        rec["ruff_rule"] = rules.get(rc)
        yield rec


def categorize_bandit_text(full_text: str) -> Dict[str, Optional[str]]:
    raw_lines = full_text.splitlines()
    lines = []

    for line in raw_lines:
        if line.strip():
            lines.append(line.rstrip())
        elif lines and lines[-1].strip():
            lines.append("")

    if not lines:
        raise ValueError("empty text")

    m = TITLE_RE.match(lines[0].strip())
    if not m:
        raise ValueError(f"unexpected title line {lines[0]!r}")

    out = {
        "code": m.group("code"),
        "title": m.group("title"),
        "what_it_does": None,
        "why_bad": None,
        "example_bad": None,
        "example_good": None,
        "remainder": None,
    }

    current_key = "remainder"
    buf = []

    def flush():
        if buf:
            text = "\n".join(buf).rstrip()
            if current_key in ["example_bad", "example_good"]:
                text = text.split("\nReferences")[0].rstrip()
                text = text.split("\nNote")[0].rstrip()
                text = text.split("\nOptions")[0].rstrip()
            elif current_key in ["what_it_does", "why_bad"]:
                text = " ".join(text.split())
            if out[current_key]:
                out[current_key] += "\n" + text
            else:
                out[current_key] = text
            buf.clear()

    for ln in lines[1:]:
        key = SECTION_HEADINGS.get(ln.strip().lower())
        if key:
            flush()
            current_key = key
            continue
        buf.append(ln)
    flush()
    return out


def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def bandit_table(doc: BeautifulSoup) -> Tag:
    h2 = doc.find(id="flake8-bandit-s")
    if not h2:
        raise RuntimeError("unable to find flake8-bandit section")
    return h2.find_next("table")


def row_to_meta(tr: Tag) -> dict[str, str]:
    tds = tr.find_all("td")
    code = tds[0].text.strip()
    a = tds[1].find("a")
    rel = a["href"]
    url = urljoin(RULES_DIR, rel.lstrip("/"))  # <-- FIX
    return {
        "code": code,
        "name": a.text.strip(),
        "short_msg": tds[2].get_text(" ", strip=True),
        "url": url,
    }


def page_markdown(url: str) -> str:
    soup = get_soup(url)
    body = soup.find("article") or soup
    for n in body.select("nav, aside, footer"):
        n.decompose()

    placeholders = []
    for pre in body.find_all("pre"):
        placeholders.append(pre.get_text(separator="", strip=False))
        pre.replace_with(f"__PRE_PLACEHOLDER_{len(placeholders)-1}__")

    text = body.get_text("\n", strip=False)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for i, content in enumerate(placeholders):
        text = text.replace(f"__PRE_PLACEHOLDER_{i}__", content)

    return text


def main(output_file: str = "bandit_rules.json") -> None:
    soup = get_soup(LISTING)
    rows = bandit_table(soup).tbody.find_all("tr")
    result = []
    for tr in rows:
        meta = row_to_meta(tr)
        try:
            meta["full_text"] = categorize_bandit_text(page_markdown(meta["url"]))
        except requests.HTTPError as e:
            print(f"[WARN] {meta['code']}: {e}")
            continue
        result.append(meta)
        time.sleep(0.3)
    Path(output_file).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✓ scraped {len(result)} rules → {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
