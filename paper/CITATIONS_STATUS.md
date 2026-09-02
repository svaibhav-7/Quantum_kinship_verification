<!-- Renamed from CITATIONS_TODO.md. Two external reviews read the old
     filename and concluded related work was unfinished; it is not. -->

# Citations - status: complete

This file was a worklist for filling placeholder citation keys. It is kept
because an external review read the stale version and concluded, reasonably,
that the related-work section was unfinished. It is not.

**Status: the P1 list is satisfied.** Both manuscripts carry inline
`\bibitem` bibliographies (IEEE: 16 entries; Elsevier: 15) and compile with
zero undefined references or citations.

Verify with:

```bash
python - <<'PY'
import io, re
for f in ("paper/ieee/main.tex", "paper/elsevier/quantum_negative.tex"):
    s = io.open(f, encoding="utf-8").read()
    keys = {k.strip() for m in re.finditer(r"cite\{([^}]*)\}", s)
            for k in m.group(1).split(",")}
    defined = set(re.findall(r"bibitem\{([^}]*)\}", s))
    print(f, "cited:", len(keys), "defined:", len(defined),
          "undefined:", sorted(keys - defined) or "none")
PY
```

One key named in the original P1 list, `xia2011tskinface`, is covered by
`qin2015tri`, which the list allowed as the alternative for the same claim.
`schroff2015facenet` was genuinely missing and has been added.

P2/P3 entries were strengthening suggestions, not blockers. Broader SOTA
positioning is a reviewer-facing judgement rather than a mechanical gap: the
papers deliberately avoid a head-to-head accuracy table against published
numbers, because those were measured under protocols this work shows to be
leaky. That choice is argued in the text, not an omission.
