"""Detect appendix paragraphs longer than a word threshold.

Paragraph = blank-line separated block of LaTeX source, excluding:
  - comment-only lines (% ...)
  - lines inside display-math, table, figure, algorithm, longtable,
    tikzpicture, verbatim, lstlisting environments
  - section/subsection/subsubsection headers, \maketitle, \tableofcontents
  - lines that are pure LaTeX scaffolding (\\begin{...}, \\end{...}, \\item
    is kept because the surrounding text is the paragraph body)

Word count strips:
  - inline math $...$ and \\(...\\) (treated as single token)
  - \\cite[...]{...}, \\ref{...}, \\eqref{...}, \\label{...} (skipped)
  - other \\command{...} -> word count of the argument text
  - bare \\command without braces (skipped)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

APPENDIX = Path(__file__).resolve().parent.parent / "paper" / "appendix.tex"

# Environments whose body is NOT prose: treat as a single non-paragraph token.
SKIP_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "displaymath", "eqnarray", "eqnarray*",
    "table", "table*", "figure", "figure*", "longtable", "tabular",
    "algorithm", "algorithmic", "tikzpicture", "verbatim", "lstlisting",
    "thebibliography", "abstract", "IEEEbiography",
}

BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")

HEADER_RE = re.compile(
    r"^\s*\\("
    r"section|subsection|subsubsection|paragraph|subparagraph|"
    r"chapter|part|maketitle|tableofcontents|appendix|"
    r"input|include|bibliography|bibliographystyle|"
    r"clearpage|newpage|pagebreak|noindent|"
    r"caption|label|cite\w*"
    r")\b"
)

CITE_REF_RE = re.compile(r"\\(?:cite\w*|ref|eqref|pageref|label|nameref)(?:\[[^\]]*\])?\{[^}]*\}")
INLINE_MATH_RE = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)")
COMMAND_WITH_ARG_RE = re.compile(r"\\[a-zA-Z@]+\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
BARE_COMMAND_RE = re.compile(r"\\[a-zA-Z@]+\*?")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žçğıöşüÇĞİÖŞÜ][A-Za-zÀ-ÿĀ-žçğıöşüÇĞİÖŞÜ\-']*")


def strip_for_counting(text: str) -> str:
    text = INLINE_MATH_RE.sub(" math ", text)
    text = CITE_REF_RE.sub(" ", text)
    text = COMMAND_WITH_ARG_RE.sub(lambda m: " " + m.group(1) + " ", text)
    text = BARE_COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return text


def count_words(text: str) -> int:
    stripped = strip_for_counting(text)
    return len(WORD_RE.findall(stripped))


def parse_paragraphs(src: str):
    """Yield (start_line, end_line, raw_block) for prose-paragraph candidates."""
    lines = src.splitlines()
    env_stack: list[str] = []
    current: list[str] = []
    current_start = 1
    in_prose = True  # outside any SKIP env

    def flush(end_line: int):
        nonlocal current, current_start
        if current:
            block = "\n".join(current).strip()
            if block:
                yield_block.append((current_start, end_line, block))
        current = []

    yield_block: list[tuple[int, int, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        lineno = i + 1
        stripped_line = COMMENT_RE.sub("", line).rstrip()

        # Track environments
        for m in BEGIN_RE.finditer(stripped_line):
            env = m.group(1)
            env_stack.append(env)
            if env in SKIP_ENVS:
                # flush any in-progress paragraph BEFORE the env line
                # (the begin line itself is not prose)
                if current:
                    end_line = lineno - 1
                    block = "\n".join(current).strip()
                    if block:
                        yield_block.append((current_start, end_line, block))
                    current = []
                in_prose = False
        # If after parsing begins we are still in prose (env not in SKIP)
        # handle end tags
        for m in END_RE.finditer(stripped_line):
            env = m.group(1)
            if env_stack and env_stack[-1] == env:
                env_stack.pop()
            else:
                # mismatched; just try to pop any matching
                if env in env_stack:
                    env_stack.remove(env)
            if env in SKIP_ENVS and not any(e in SKIP_ENVS for e in env_stack):
                in_prose = True
                # blank-separate: do not absorb the end line into prose
                current_start = lineno + 1
                i += 1
                continue

        if not in_prose:
            i += 1
            continue

        # Header / scaffold lines flush and skip
        if HEADER_RE.match(stripped_line):
            if current:
                end_line = lineno - 1
                block = "\n".join(current).strip()
                if block:
                    yield_block.append((current_start, end_line, block))
                current = []
            current_start = lineno + 1
            i += 1
            continue

        if stripped_line == "":
            if current:
                end_line = lineno - 1
                block = "\n".join(current).strip()
                if block:
                    yield_block.append((current_start, end_line, block))
                current = []
            current_start = lineno + 1
        else:
            if not current:
                current_start = lineno
            current.append(line)
        i += 1

    if current:
        block = "\n".join(current).strip()
        if block:
            yield_block.append((current_start, len(lines), block))

    return yield_block


def main(threshold: int = 250):
    src = APPENDIX.read_text(encoding="utf-8")
    paras = parse_paragraphs(src)

    rows = []
    for start, end, block in paras:
        wc = count_words(block)
        rows.append((wc, start, end, block))

    rows.sort(key=lambda r: -r[0])

    over = [r for r in rows if r[0] > threshold]
    total = len(rows)

    # Histogram bins
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250),
            (250, 300), (300, 400), (400, 500), (500, 750), (750, 10**6)]
    hist = {f"{a}-{b if b < 10**5 else 'inf'}": 0 for a, b in bins}
    for wc, *_ in rows:
        for a, b in bins:
            if a <= wc < b:
                key = f"{a}-{b if b < 10**5 else 'inf'}"
                hist[key] += 1
                break

    print(f"Total prose paragraphs: {total}")
    print(f"Paragraphs > {threshold} words: {len(over)}")
    print()
    print("Histogram (word count buckets):")
    for k, v in hist.items():
        bar = "#" * v
        print(f"  {k:>10}  {v:>4}  {bar}")
    print()
    print(f"Top {min(20, len(over))} longest paragraphs:")
    print(f"{'wc':>5}  {'lines':>11}  first 80 chars")
    for wc, start, end, block in rows[:20]:
        snippet = block.replace("\n", " ")[:80]
        print(f"{wc:>5}  L{start:>4}-{end:<5}  {snippet}")


if __name__ == "__main__":
    th = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    main(th)
