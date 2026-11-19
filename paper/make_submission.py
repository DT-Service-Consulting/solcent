#!/usr/bin/env python3
"""
make_submission.py
==================
Builds a clean LaTeX package for upload to the journal from paper/epsr/.

The working sources carry internal comments (source-file pointers, TODO and
VERIFY notes) and journal switches (\\ifepsr ... \\else ... \\fi). Editorial
systems keep the uploaded source, so the package
  - flattens sections/*.tex into a single main.tex,
  - removes every LaTeX comment,
  - keeps the EPSR wording of each \\ifepsr switch and drops the other branch,
  - drops the \\ifwithsupp block (the supplementary material is uploaded separately),
  - copies only the figures main.tex uses, plus refs.bib, epsr.sty and the .bst.

Output: paper/epsr_submission/ and paper/epsr_submission.zip
Usage:  python paper/make_submission.py      (from the repository root)
"""

import re
import shutil
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent / "epsr"
OUT = SRC.parent / "epsr_submission"


def strip_comments(text):
    lines = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            lines.append(line)
        elif line[:m.start()].strip() == "":
            continue                                   # whole-line comment
        elif line[:m.start()][-1].isspace():
            lines.append(line[:m.start()].rstrip())    # inline note after a command
        else:
            lines.append(line[:m.start()] + "%")       # keep % so the line end stays suppressed
    return "\n".join(lines) + "\n"


def flatten(text):
    def inc(m):
        return (SRC / (m.group(1) + ".tex")).read_text(encoding="utf-8")
    return re.sub(r"\\input\{(sections/[^}]+)\}", inc, text)


def resolve_switches(text):
    text = re.sub(r"\\newif\\ifepsr\s*\\epsrtrue\s*", "", text)
    text = re.sub(r"\\newif\\ifwithsupp\s*\\withsupp(true|false)\s*", "", text)
    text = re.sub(r"\\ifwithsupp.*?\\fi\s*", "", text, flags=re.S)
    # \ifepsr A \else B \fi  ->  A ;  \ifepsr A \fi  ->  A   (no nesting in these files)
    text = re.sub(r"\\ifepsr(.*?)(?:\\else.*?)?\\fi(\\ )?",
                  lambda m: m.group(1).strip("\n") + (" " if m.group(2) else ""), text, flags=re.S)
    text = re.sub(r"(?m)^ (?=\S)", "", text)          # leading space left where \ifepsr opened a line
    text = re.sub(r"\.% *\n", ".\n", text)            # sentence-end % left by a removed note
    return text


def main():
    # empty the folder rather than delete it (Windows refuses to remove a folder
    # that a shell or editor has open)
    if OUT.exists():
        for p in sorted(OUT.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    tex = flatten((SRC / "main.tex").read_text(encoding="utf-8"))
    tex = strip_comments(tex)
    tex = resolve_switches(tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    for bad in (r"\ifepsr", r"\ifwithsupp", "TODO", "VERIFY", "supplementary_content"):
        assert bad not in tex, bad
    assert tex.count("{") == tex.count("}"), "unbalanced braces"
    (OUT / "main.tex").write_text(tex, encoding="utf-8")

    for f in ("refs.bib", "epsr.sty", "elsarticle-harv.bst"):
        shutil.copy2(SRC / f, OUT / f)
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)))
    for f in figs:
        shutil.copy2(SRC / f, OUT / f)

    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    print(f"main.tex flattened ({len(tex.split())} tokens), {len(figs)} figures")
    print(f"wrote {OUT} and {zpath}")


if __name__ == "__main__":
    main()
