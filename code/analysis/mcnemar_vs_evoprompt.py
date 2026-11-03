"""
Paired comparison of two prompts on the same test items (McNemar's test)
=======================================================================
Two systems scored on identical items are not independent samples, so the test
conditions on the items where they disagree:

    b = A correct, B wrong
    c = A wrong,  B correct

Under the null these are equally likely, so b ~ Binomial(b + c, 1/2); the exact
binomial version is reported. Equivalence is tested at the same margin used for
the search comparison.

Reads two per-item files written by evaluate_full_test.py and aligns them on the
review text.

Run:  python mcnemar_vs_evoprompt.py <file_A.csv> <file_B.csv>
"""

import csv
import os
import sys
from math import comb


def load(path):
    """review text -> correct (0/1), keyed on text so row order need not match."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[r["text"]] = int(r["correct"])
    if len(out) != len(rows):
        print(f"  note: {path} has {len(rows) - len(out)} duplicate review texts; "
              f"keeping the last of each")
    return out


def exact_two_sided_p(b, n):
    """P(|X - n/2| >= |b - n/2|) for X ~ Binomial(n, 1/2)."""
    if n == 0:
        return 1.0
    target = abs(b - n / 2)
    total = sum(comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= target)
    return min(1.0, total / 2 ** n)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    pa, pb = sys.argv[1], sys.argv[2]
    A, B = load(pa), load(pb)

    shared = set(A) & set(B)
    if not shared:
        raise SystemExit("no overlapping review texts; are these the same test set?")
    for name, d in ((pa, A), (pb, B)):
        if len(d) != len(shared):
            print(f"  note: {os.path.basename(name)} has {len(d)} items, "
                  f"{len(shared)} shared")

    a = sum(1 for t in shared if A[t] and B[t])        # both right
    b = sum(1 for t in shared if A[t] and not B[t])    # only A right
    c = sum(1 for t in shared if not A[t] and B[t])    # only B right
    d = sum(1 for t in shared if not A[t] and not B[t])  # both wrong
    n = len(shared)

    accA, accB = (a + b) / n, (a + c) / n
    print(f"\nitems compared: {n}")
    print(f"  A  {os.path.basename(pa)}   accuracy {100 * accA:.2f}%")
    print(f"  B  {os.path.basename(pb)}   accuracy {100 * accB:.2f}%")
    print(f"  difference: {100 * (accA - accB):+.2f} points\n")

    print("contingency")
    print(f"  both correct          {a:5d}")
    print(f"  only A correct (b)    {b:5d}")
    print(f"  only B correct (c)    {c:5d}")
    print(f"  both wrong            {d:5d}")

    nd = b + c
    p = exact_two_sided_p(b, nd)
    print(f"\nMcNemar, exact binomial on the {nd} discordant pairs")
    print(f"  b = {b}, c = {c}   p = {p:.4f}")
    print(f"  -> {'a difference this large is unlikely under the null' if p < 0.05 else 'no significant difference'} at alpha = 0.05")

    if nd:
        # A non-significant difference is not evidence of parity, so equivalence
        # is tested directly.
        import math
        se = math.sqrt(nd) / n          # SE of (b - c)/n
        diff = (b - c) / n
        lo95, hi95 = diff - 1.96 * se, diff + 1.96 * se
        lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
        print(f"\n  paired difference {100 * diff:+.2f} points   SE {100 * se:.2f}")
        print(f"    95% CI [{100 * lo95:+.2f}, {100 * hi95:+.2f}]")
        print(f"    90% CI [{100 * lo90:+.2f}, {100 * hi90:+.2f}]   (the TOST interval)")

        delta = 0.01
        equiv = lo90 > -delta and hi90 < delta
        print(f"\n  equivalence at delta = {delta:.2f} (one accuracy point): "
              f"{'EQUIVALENT' if equiv else 'NOT established'}")
        print(f"    smallest margin that holds: {100 * max(abs(lo90), abs(hi90)):.2f} points")
        if not equiv:
            print("    (the test set cannot resolve a one-point parity claim)")


if __name__ == "__main__":
    main()
