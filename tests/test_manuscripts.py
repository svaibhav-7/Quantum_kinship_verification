"""Guards for defects that silently reach a submitted PDF.

Both defects tested here actually occurred: a table row terminated with a
single backslash made the Elsevier paper fail to compile while a stale PDF
sat in the repository looking fine, and a mangled escape left a literal tab
where a LaTeX command belonged, in the abstract.
"""
import io
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = [os.path.join(ROOT, "paper", "ieee", "main.tex"),
       os.path.join(ROOT, "paper", "elsevier", "quantum_negative.tex")]
BS = chr(92)


def _read(p):
    return io.open(p, encoding="utf-8").read()


class TestManuscriptSource(unittest.TestCase):
    def test_no_single_backslash_line_endings(self):
        """A row ending in one backslash makes LaTeX abort at the next rule."""
        for p in TEX:
            for n, line in enumerate(_read(p).split(chr(10)), 1):
                r = line.rstrip()
                if r.endswith(BS) and not r.endswith(BS + BS):
                    self.fail("%s:%d ends with a single backslash: %r"
                              % (os.path.basename(p), n, r[-40:]))

    def test_no_literal_control_characters(self):
        """A tab here means an escape was interpreted before LaTeX saw it."""
        for p in TEX:
            s = _read(p)
            self.assertNotIn(chr(9), s,
                             "%s contains a literal tab" % os.path.basename(p))

    def test_every_citation_is_defined(self):
        for p in TEX:
            s = _read(p)
            cited = {k.strip() for m in re.finditer(r"cite\{([^}]*)\}", s)
                     for k in m.group(1).split(",")}
            defined = set(re.findall(r"bibitem\{([^}]*)\}", s))
            self.assertEqual(cited - defined, set(),
                             "%s cites undefined keys" % os.path.basename(p))

    def test_backbone_claims_are_consistent(self):
        """Both papers report two backbones; neither may claim only one."""
        for p in TEX:
            s = _read(p)
            self.assertNotIn("single frozen backbone", s)
            self.assertNotIn("All results use frozen FaceNet embeddings", s)
            self.assertIn("ArcFace", s)


class TestManuscriptTypography(unittest.TestCase):
    """Overfull boxes print as text running into the margin.

    Both papers carried 15 each before submission preparation; a reviewer
    flagged them. Thresholds are set at the current state so a regression
    is caught, not so a clean build is demanded of prose TeX cannot break.
    """

    LIMITS = {"main.log": 1, "quantum_negative.log": 2}

    def test_overfull_boxes_within_budget(self):
        for name, limit in self.LIMITS.items():
            log = None
            for d in ("ieee", "elsevier"):
                c = os.path.join(ROOT, "paper", d, name)
                if os.path.exists(c):
                    log = c
                    break
            if log is None:
                # The build log is a local artefact and is gitignored, so a
                # clean checkout has none. Skipping is correct -- but note the
                # guard only protects a tree where the papers have been built.
                self.skipTest("%s not built; run pdflatex to enable this check"
                              % name)
            n = _read(log).count("Overfull")
            self.assertLessEqual(
                n, limit,
                "%s has %d overfull boxes (budget %d); rebuild and inspect"
                % (name, n, limit))


if __name__ == "__main__":
    unittest.main()
