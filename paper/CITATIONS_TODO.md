# Citations to fill in

`paper/main.tex` has a structured Related Work section with placeholder keys.
Each entry below names the key, says what claim it must support, and gives the
search terms. **Only add an entry after reading the paper.** An earlier draft
carried a comparison table whose rows could not be traced to any source; that
table was removed rather than reproduced.

Priority order: P1 blocks submission, P2 strengthens, P3 optional.

---

## P1 — required

| Key | Must support | Search |
|---|---|---|
| `lu2014neighborhood` | NRML, the standard metric-learning baseline | "Neighborhood repressed metric learning kinship verification" Lu et al., IEEE TPAMI 2014 |
| `robinson2018fiw` | The FIW dataset: size, families, construction | "Families in the Wild large-scale kinship image database" Robinson et al., IEEE TPAMI 2018 |
| `xia2011tskinface` / `qin2015tri` | TSKinFace: the tri-subject protocol whose negative construction we show is exploitable | "TSKinFace tri-subject kinship verification" Qin, Tan, Chen |
| `lu2012kinfacew` | KinFaceW-I / KinFaceW-II and their official protocol | "KinFaceW kinship verification in the wild dataset" |
| `schroff2015facenet` | FaceNet, our frozen backbone | "FaceNet: A Unified Embedding for Face Recognition and Clustering" CVPR 2015 |
| `cao2018vggface2` | VGGFace2, the pretraining corpus | "VGGFace2: A dataset for recognising faces across pose and age" |
| `geirhos2020shortcut` | Shortcut learning — the frame for our TSKinFace finding | "Shortcut learning in deep neural networks" Nature Machine Intelligence 2020 |
| `whitelam2017ijbb` | IJB-B/C set-based (template) protocols, the precedent for our set-level formulation | "IARPA Janus Benchmark-B face dataset" |

## P2 — strengthens the argument

| Key | Must support | Search |
|---|---|---|
| `robinson2020rfiw` | RFIW challenge series; current reported accuracies | "Recognizing Families In the Wild data challenge" |
| *(recent SOTA, 2–4 papers)* | What current methods report and under which protocol | "kinship verification transformer 2023", "graph kinship verification", "contrastive kinship verification" |
| *(subject-disjoint evaluation)* | Prior warnings about identity leakage in face benchmarks | "subject-disjoint evaluation face recognition protocol bias" |
| *(dataset bias)* | Bias in vision benchmarks generally | "Unbiased look at dataset bias" Torralba & Efros, CVPR 2011 |
| `havlicek2019supervised` | Quantum-enhanced feature spaces, the strongest positive claim in QML | "Supervised learning with quantum-enhanced feature spaces" Nature 2019 |
| `schuld2019quantum` | Quantum kernels / feature maps | "Quantum machine learning in feature Hilbert spaces" Schuld & Killoran |
| *(quantum-inspired on classical data)* | Prior reports of quantum-inspired gains — the claims our nine formulations test | "quantum-inspired classical machine learning dequantization" |

## P3 — optional

| Key | Must support | Search |
|---|---|---|
| `deng2019arcface` | ArcFace, cited for scale comparison only | "ArcFace: Additive Angular Margin Loss" |
| *(negative results in ML)* | Precedent for publishing negative results | "negative results machine learning reproducibility" |

---

## What each Related Work subsection needs

**3.1 Kinship verification** — trajectory from metric learning (NRML/MNRML) through
deep pairwise models to transformer and graph approaches. State that reported
accuracies use each dataset's official protocol. Do **not** tabulate their
numbers against ours; the protocols differ, and we say so in §Results.

**3.2 Evaluation bias in vision benchmarks** — subject-disjoint splitting in face
recognition; shortcut learning; set/template protocols (IJB-B/C). This is the
section that positions our contribution: the practice exists in adjacent
fields and has not been applied to kinship verification.

**3.3 Quantum-inspired machine learning** — variational circuits, SWAP-test
similarity, density-matrix classifiers, quantum kernels. Include the strongest
positive claims, since our nine-formulation negative result is only meaningful
against them.

---

## Verifying novelty — do this before writing more

Two claims carry the paper. Both need checking, and neither can be checked
from inside this repository.

**1. The TSKinFace photo-session shortcut.** Search:

- `TSKinFace negative pair construction bias`
- `kinship verification same-family negative pairs`
- `photo session bias kinship dataset`
- `spurious correlation kinship verification benchmark`

*If already reported:* the finding becomes confirmation rather than discovery,
and the paper leans on leakage quantification plus set-level instead.
*If not:* this is the strongest single contribution, and it invalidates
published TSKinFace numbers.

**2. Set-level (person-level) kinship verification.** Search:

- `set-based kinship verification multiple images per subject`
- `template kinship verification`
- `multi-image kinship verification aggregation`
- `image set matching kinship`

*If already done:* our contribution is the measurement under a leakage-free
protocol, not the idea.
*If not:* +0.077 ROC-AUC on FIW is a novel, well-supported result.

**Also worth one search:** `identity leakage kinship verification splits` — to
confirm nobody has already quantified the 100% figure.

---

## How to add an entry

Replace the placeholder block at the end of `main.tex`:

```latex
\begin{thebibliography}{1}
\bibitem{lu2014neighborhood}
J.~Lu \emph{et al.}, "Neighborhood repulsed metric learning for kinship
verification," \emph{IEEE TPAMI}, vol.~36, no.~2, pp.~331--345, 2014.
\end{thebibliography}
```

Cite in text with `\cite{lu2014neighborhood}`.
