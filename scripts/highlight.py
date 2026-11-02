"""Yellow-highlight every word of the revised PDF that is new or changed relative to the submitted version.
Word-level diff of the running text (difflib on normalised tokens); figures/images are not compared."""
import difflib
import re
import sys

import pymupdf

old_pdf, new_pdf, out_pdf = sys.argv[1:4]


def words(path):
    doc = pymupdf.open(path)
    W = []
    for pno, page in enumerate(doc):
        for w in page.get_text("words", sort=False, flags=pymupdf.TEXT_DEHYPHENATE):
            W.append((pno, pymupdf.Rect(w[:4]), w[4]))
    return doc, W


norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731
_, OW = words(old_pdf)
doc, NW = words(new_pdf)
o = [norm(w[2]) for w in OW]
n = [norm(w[2]) for w in NW]
sm = difflib.SequenceMatcher(None, o, n, autojunk=False)
mark = [False] * len(n)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag in ("replace", "insert"):
        for j in range(j1, j2):
            if n[j] and not re.fullmatch(r"[\[\(]?\d+[\]\)]?[,.;:\u2013-]*(\[?\d+\]?)?[,.;]*", NW[j][2]):
                mark[j] = True
cnt = 0
pages = [doc[i] for i in range(len(doc))]
for (pno, r, _), m in zip(NW, mark):
    if m:
        a = pages[pno].add_highlight_annot(r)
        a.set_colors(stroke=(1, 1, 0)); a.update(); cnt += 1
doc.save(out_pdf, garbage=3, deflate=True)
print(f"highlighted {cnt} of {len(n)} words ({cnt / len(n):.0%})")
