"""Write src/unicode.c: the code point ranges of Unicode letters (L*), numbers (N*) and
marks (M*) from Python's unicodedata, the classes the Qwen pre-tokenizer regex uses as
\\p{L}, \\p{N} and \\p{M}. Rerun after a Python upgrade changes the Unicode version.

    python3 tools/gen_unicode.py
"""

import sys
import unicodedata
from pathlib import Path

OUT = Path("src/unicode.c")


def ranges(prefix: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)).startswith(prefix):
            if out and out[-1][1] == cp - 1:
                out[-1] = (out[-1][0], cp)
            else:
                out.append((cp, cp))
    return out


def table(name: str, rs: list[tuple[int, int]]) -> str:
    rows = "".join(f"    {{0x{a:x}, 0x{b:x}}},\n" for a, b in rs)
    return f"const uint32_t {name}[{len(rs)}][2] = {{\n{rows}}};\n"


def main() -> None:
    letters = ranges("L")
    numbers = ranges("N")
    marks = ranges("M")
    OUT.write_text(
        '#include "unicode.h"\n\n'
        + table("mda_unicode_letters", letters)
        + "\n"
        + table("mda_unicode_numbers", numbers)
        + "\n"
        + table("mda_unicode_marks", marks)
        + f"""
bool mda_unicode_in(const uint32_t (*rs)[2], uint32_t n, uint32_t cp) {{
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {{
        uint32_t mid = (lo + hi) / 2;
        if (rs[mid][1] < cp) {{
            lo = mid + 1;
        }} else {{
            hi = mid;
        }}
    }}
    return lo < n && rs[lo][0] <= cp;
}}

bool mda_unicode_letter(uint32_t cp) {{
    return mda_unicode_in(mda_unicode_letters, {len(letters)}, cp);
}}

bool mda_unicode_number(uint32_t cp) {{
    return mda_unicode_in(mda_unicode_numbers, {len(numbers)}, cp);
}}

bool mda_unicode_mark(uint32_t cp) {{
    return mda_unicode_in(mda_unicode_marks, {len(marks)}, cp);
}}
"""
    )
    print(f"{OUT}: Unicode {unicodedata.unidata_version}, {len(letters)} letter, {len(numbers)} number and {len(marks)} mark ranges")


if __name__ == "__main__":
    main()
