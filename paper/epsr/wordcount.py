"""Approximate word count for LaTeX section files (citations count as one word)."""
import re
import sys

total = 0
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", " CIT ", s)
    s = re.sub(r"\\(ref|label|eqref|includegraphics|begin|end|input|bibliography\w*)(\[[^]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", s)
    s = re.sub(r"[{}$\\~&]", " ", s)
    n = len([w for w in s.split() if re.search(r"\w", w)])
    total += n
    print(f"{n:6d}  {p}")
print(f"{total:6d}  TOTAL")
