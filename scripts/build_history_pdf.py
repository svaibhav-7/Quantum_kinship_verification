"""Convert the project-history markdown into a LaTeX report and compile it."""
import io, os, re

os.chdir(r"C:\Users\sasiv\Downloads\SasiVaibhav\SasiVaibhav\klu\3rd year\projects\Quantum_kinship")
SRC = os.path.join("docs", "project_history.md")
md = io.open(SRC, encoding="utf-8").read()


def esc(t):
    """Escape LaTeX specials in prose, preserving inline code and bold."""
    t = t.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        t = t.replace(ch, "\\" + ch)
    t = t.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    t = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + m.group(1).replace(" ", "~") + "}", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", t)
    t = t.replace("-->", r"$\rightarrow$").replace(" -> ", r" $\rightarrow$ ")
    t = t.replace("+/-", r"$\pm$")
    return t


def table(rows):
    hdr = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [r for r in rows[2:] if r.strip()]
    n = len(hdr)
    align = "l" + "r" * (n - 1)
    out = ["\\begin{center}", "\\small", "\\begin{tabular}{" + align + "}", "\\toprule",
           " & ".join(esc(h) for h in hdr) + " \\\\", "\\midrule"]
    for r in body:
        cells = [c.strip() for c in r.strip("|").split("|")]
        cells += [""] * (n - len(cells))
        out.append(" & ".join(esc(c) for c in cells[:n]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    return out


lines = md.split("\n")
body, i = [], 0
while i < len(lines):
    L = lines[i]

    if L.startswith("```"):
        i += 1
        code = []
        while i < len(lines) and not lines[i].startswith("```"):
            code.append(lines[i]); i += 1
        i += 1
        body += ["\\begin{quote}\\small\\begin{verbatim}"] + code + ["\\end{verbatim}\\end{quote}", ""]
        continue

    if L.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i]); i += 1
        body += table(rows)
        continue

    if L.startswith("### "):
        body.append("\\subsection*{" + esc(L[4:]) + "}")
    elif L.startswith("## "):
        body.append("\\section*{" + esc(L[3:]) + "}")
    elif L.strip() == "---":
        body.append("")
    elif L.startswith(("- ", "* ")):
        items = []
        while i < len(lines) and lines[i].startswith(("- ", "* ")):
            items.append("\\item " + esc(lines[i][2:])); i += 1
        body += ["\\begin{itemize}\\setlength{\\itemsep}{2pt}"] + items + ["\\end{itemize}", ""]
        continue
    elif re.match(r"^\d+\. ", L):
        items = []
        while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
            items.append("\\item " + esc(re.sub(r"^\d+\. ", "", lines[i]))); i += 1
        body += ["\\begin{enumerate}\\setlength{\\itemsep}{2pt}"] + items + ["\\end{enumerate}", ""]
        continue
    else:
        body.append(esc(L))
    i += 1

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{18pt}{6pt}

\title{\textbf{Facial Kinship Verification}\\[4pt]
\large Project History, Diagnostics, and Methodology}
\author{Sasi Vaibhav}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\noindent
This report records how a quantum-inspired kinship verification system became a
leakage-free evaluation protocol with a documented negative result for
quantum-inspired metrics. It covers the diagnostics that prompted the change,
why nine quantum formulations failed, the methodology that replaced them, and a
claim this project made and later had to retract. Every figure quoted here
derives from a JSON artefact in \texttt{results/honest/} and is reproducible
from the commands in the final section.
\end{abstract}

\tableofcontents
\vspace{12pt}

""" + "\n".join(body) + r"""

\end{document}
"""

os.makedirs("docs", exist_ok=True)
io.open("docs/project_history.tex", "w", encoding="utf-8").write(TEX)
print("wrote docs/project_history.tex")
