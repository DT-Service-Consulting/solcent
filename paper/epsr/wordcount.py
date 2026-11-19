"""Approximate word count of the EPSR manuscript (citations count as one word).

EPSR counts all text including abstract, tables, figure captions and appendices;
only references and acknowledgements are excluded. With no arguments this counts
the abstract, highlights, keywords, sections 01-07, tables, captions and the
declarations, for the EPSR wording only (the \\else branch of every \\ifepsr
switch is dropped). Run from this folder:

    python wordcount.py                 # whole manuscript, per file and total
    python wordcount.py sections/05_results.tex   # selected files only
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = ["sections/00_abstract.tex", "sections/00_highlights.tex",
           "sections/01_introduction.tex", "sections/02_related_work.tex",
           "sections/03_data.tex", "sections/04_methods.tex", "sections/05_results.tex",
           "sections/06_discussion.tex", "sections/07_conclusions.tex",
           "sections/08_declarations.tex"]


def count(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\else.*?\\fi", " ", s, flags=re.S)        # keep EPSR wording only
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", " CIT ", s)
    s = re.sub(r"\\(ref|label|eqref|includegraphics|begin|end|input|bibliography\w*)(\[[^]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", s)
    s = re.sub(r"[{}$\\~&]", " ", s)
    return len([w for w in s.split() if re.search(r"\w", w)])


files = sys.argv[1:] or DEFAULT
total = 0
for p in files:
    n = count((HERE / p).read_text(encoding="utf-8"))
    total += n
    print(f"{n:6d}  {p}")
if not sys.argv[1:]:
    main = (HERE / "main.tex").read_text(encoding="utf-8")
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main, flags=re.S)
    n = count(kw.group(1).replace(r"\sep", " ")) if kw else 0
    total += n
    print(f"{n:6d}  keywords (main.tex)")
print(f"{total:6d}  TOTAL")
