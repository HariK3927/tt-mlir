#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path
import subprocess


def convert_html_to_md(html_dir, md_dir):
    html_dir = Path(html_dir)
    md_dir = Path(md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    # Process HTML files
    for html_file in html_dir.rglob("*.html"):
        rel_path = html_file.relative_to(html_dir)
        md_file = md_dir / rel_path.with_suffix(".md")
        md_file.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "pandoc",
                "-f",
                "html",
                "-t",
                "markdown_strict",
                str(html_file),
                "-o",
                str(md_file),
            ],
            check=True,
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <html_dir> <md_dir>")
        sys.exit(1)

    convert_html_to_md(sys.argv[1], sys.argv[2])
