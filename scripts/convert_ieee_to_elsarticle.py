# -*- coding: utf-8 -*-
"""Generate the elsarticle edition of the person-level manuscript.

paper/ieee/main.tex is the source of truth. This script produces
paper/elsevier_person/main.tex from it, so the two cannot drift in content:
only the format-bearing constructs are rewritten.

What changes, and why each is necessary:

  documentclass   IEEEtran -> elsarticle
  frontmatter     IEEEtran's \title/\author/\thanks/\maketitle -> elsarticle's
                  \begin{frontmatter} with \author[1]/\ead/\cortext/\address
  keywords        IEEEkeywords environment -> keyword environment, \sep-joined
  widths          \columnwidth is half a page under two-column IEEEtran but a
                  full page under single-column elsarticle; figures sized for
                  the former render oversized in the latter, so they are
                  remapped to \linewidth
  url package     elsarticle loads it; loading twice is an option clash

Run:  python scripts/convert_ieee_to_elsarticle.py
"""
import io
import os
import re
import sys

NL = chr(10)
B = chr(92)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "ieee", "main.tex")
DST = os.path.join(ROOT, "paper", "elsevier_person", "main.tex")

FRONTMATTER = (
    B + "begin{frontmatter}" + NL + NL +
    B + "title{Person-Level Kinship Verification: Leakage-Free Evaluation, "
        "Set-Based" + NL +
    "Aggregation, and a Negative Result for Quantum-Inspired Metrics}" + NL + NL +
    B + "author[1]{Sasi Vaibhav Sanka" + B + "corref{cor1}}" + NL +
    B + "ead{svaibhav@klu.ac.in}" + NL +
    B + "author[1]{Kesireddy Samith Reddy}" + NL +
    B + "author[1]{Konala Varshith}" + NL +
    B + "author[1]{Eelpanti Mythri}" + NL +
    B + "author[1]{Madhu Oruganti}" + NL +
    B + "cortext[cor1]{Corresponding author}" + NL + NL +
    B + "address[1]{Department of Artificial Intelligence and Data Science, "
        "Koneru Lakshmaiah Education Foundation, Hyderabad-500075, India}" +
    NL + NL
)

HEADER = (
    "% Person-Level Kinship Verification -- Elsevier (elsarticle) edition." + NL +
    "%" + NL +
    "% GENERATED from paper/ieee/main.tex by" + NL +
    "% scripts/convert_ieee_to_elsarticle.py. Edit the IEEE source, not this" + NL +
    "% file: content is identical by construction and every numeric claim" + NL +
    "% still traces to results/honest/*.json."
)


def convert(src_text):
    s = src_text

    s = s.replace(B + "documentclass[journal,12pt]{IEEEtran}",
                  B + "documentclass[preprint,12pt]{elsarticle}")

    # elsarticle already provides url; loading it again clashes.
    s = s.replace(B + "usepackage{url}" + NL, "")

    s = s.replace(B + "usepackage[hidelinks]{hyperref}",
                  B + "usepackage[hidelinks]{hyperref}" + NL +
                  B + "journal{Pattern Recognition}")

    s = s.replace("% Evaluation Protocol Flaws in Facial Kinship Verification",
                  HEADER)

    # Excise the IEEEtran title/author/maketitle block by span: it contains
    # nested braces that a regex handles badly.
    t0 = s.index(B + "title{")
    mk = s.index(B + "maketitle")
    s = s[:t0] + s[mk + len(B + "maketitle") + 1:]

    s = s.replace(B + "begin{document}" + NL,
                  B + "begin{document}" + NL + NL + FRONTMATTER, 1)

    # IEEEkeywords -> keyword, then close the frontmatter.
    open_kw = B + "begin{IEEEkeywords}"
    close_kw = B + "end{IEEEkeywords}"
    i, j = s.index(open_kw), s.index(close_kw)
    items = [k.strip().rstrip(".")
             for k in s[i + len(open_kw):j].strip().split(",") if k.strip()]
    body = (" " + B + "sep" + NL).join(items)
    s = (s[:i] + B + "begin{keyword}" + NL + body + NL + B + "end{keyword}" +
         NL + NL + B + "end{frontmatter}" + s[j + len(close_kw):])

    # Two-column half-page width -> single-column full width.
    s = s.replace(B + "columnwidth", B + "linewidth")

    # Reflowing to one column widens the text block, and a couple of tables
    # that fitted an IEEE column now overrun the margin. Size them to the
    # text width, matching what both other manuscripts do.
    s = _fit_wide_tables(s)

    # elsarticle sets a wider measure, so long typewriter paths and URLs that
    # broke acceptably in two columns no longer do. Same remedy as the
    # companion paper.
    s = s.replace(B + "begin{document}", B + "sloppy" + NL + B + "begin{document}", 1)

    return s


def _fit_wide_tables(s):
    """Wrap over-wide tabulars in a resizebox so they fit the text block."""
    lines = s.split(NL)
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(B + "begin{tabular}"):
            j = i
            while j < len(lines) and not lines[j].startswith(B + "end{tabular}"):
                j += 1
            block = lines[i:j + 1]
            widest = max((len(x) for x in block), default=0)
            already = out and out[-1].startswith(B + "resizebox")
            if widest > 78 and not already:
                out.append(B + "resizebox{" + B + "linewidth}{!}{%")
                out.extend(block[:-1])
                out.append(block[-1] + "}")
            else:
                out.extend(block)
            i = j + 1
            continue
        out.append(line)
        i += 1
    return NL.join(out)


def main():
    src = io.open(SRC, encoding="utf-8").read()
    out = convert(src)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    io.open(DST, "w", encoding="utf-8", newline=NL).write(out)

    # The conversion must not touch prose: a differing word count means a bug.
    def words(t):
        t = re.sub(r"%.*", "", t)
        return len(re.findall(r"[A-Za-z]{3,}", t))
    d = abs(words(src) - words(out))
    print("wrote %s" % os.path.relpath(DST, ROOT))
    print("  word-count delta vs IEEE source: %d (frontmatter only)" % d)
    if d > 120:
        print("  WARNING: larger than expected; inspect the diff", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
