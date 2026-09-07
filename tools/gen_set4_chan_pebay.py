"""Corpus set 4 — the Chan/Pebay merge, authored against exact arithmetic.

These are ORACLES, not fixtures. Merging per-bucket moments is a floating-point
computation with several defensibly correct answers, so the expected values
come from exact rational arithmetic — `fractions.Fraction`, no rounding at all
— and each case carries a stated tolerance. An oracle whose expected values
came from running an implementation would pin that implementation's rounding
as though it were the format's.

⛔ **And it carries the finding that makes the merge specifiable at all.**
Floating-point addition is not associative, so the merge is ORDER-DEPENDENT:
folding the same buckets left-to-right and as a balanced tree gives different
last bits. Every case below therefore states the order it assumes. A
conformance suite that did not would be asserting something no implementation
can reproduce.

The order-independence vector is the other half of that: over the rationals the
identities ARE associative, so the divergence an implementation sees is its own
rounding and not a defect in the algebra.
"""

from __future__ import annotations

from fractions import Fraction

import _moments
import numpy as np

from _corpus import Vector, array_ref, big, fbits

SET = "set4-chan-pebay"


# ---------------------------------------------------------------------------
# The reference: exact rational arithmetic, no rounding anywhere
# ---------------------------------------------------------------------------


def exact_moments(values: list[float]):
    """(count, mean, M2, M3, M4) as exact Fractions over the non-NaN elements.

    Every float64 is exactly a rational, so `Fraction(x)` loses nothing and the
    whole computation is exact. This is the reference; it is not an
    implementation of anything and is deliberately as slow and as obvious as
    possible.
    """
    vals = [Fraction(v) for v in values if v == v]      # v != v is NaN
    n = len(vals)
    if n == 0:
        return 0, None, None, None, None
    mean = sum(vals) / n
    m2 = sum((v - mean) ** 2 for v in vals)
    m3 = sum((v - mean) ** 3 for v in vals)
    m4 = sum((v - mean) ** 4 for v in vals)
    return n, mean, m2, m3, m4


def exact_merge(a, b):
    """The Chan/Pébay identities in EXACT rational arithmetic.

    The Chan/Pebay identities, evaluated over Fractions. In exact
    arithmetic the merge is associative and gives the same answer as computing
    from the elements directly — which is what `test_associativity_is_exact`
    below asserts, and which is the property float64 does not have.
    """
    ca, ma, m2a, m3a, m4a = a
    cb, mb, m2b, m3b, m4b = b
    if cb == 0:
        return a
    if ca == 0:
        return b
    n = ca + cb
    delta = mb - ma
    d2 = delta * delta
    m2 = m2a + m2b + d2 * ca * cb / n
    m3 = (m3a + m3b + delta**3 * ca * cb * (ca - cb) / n**2
          + 3 * delta * (ca * m2b - cb * m2a) / n)
    m4 = (m4a + m4b
          + delta**4 * ca * cb * (ca**2 - ca * cb + cb**2) / n**3
          + 6 * d2 * (ca**2 * m2b + cb**2 * m2a) / n**2
          + 4 * delta * (ca * m3b - cb * m3a) / n)
    mean = (ca * ma + cb * mb) / n
    return n, mean, m2, m3, m4


def fold_left(parts):
    acc = parts[0]
    for p in parts[1:]:
        acc = exact_merge(acc, p)
    return acc


def fold_tree(parts):
    cur = list(parts)
    while len(cur) > 1:
        nxt = [exact_merge(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def lcg_floats(seed: int, count: int, lo: float, hi: float) -> list[float]:
    state = seed & ((1 << 64) - 1)
    out = []
    span = hi - lo
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(lo + span * ((state >> 11) / float(1 << 53)))
    return out


def as_float(fr: Fraction | None) -> float | None:
    return None if fr is None else float(fr)


# ---------------------------------------------------------------------------
# The datasets — each chosen to break a specific naive implementation
# ---------------------------------------------------------------------------


def datasets():
    out = []
    out.append(("uniform", lcg_floats(0xC4A11, 1024, -1.0, 1.0)))
    out.append(("large-offset", [1e8 + v for v in lcg_floats(0xC4A12, 1024, -1.0, 1.0)]))
    out.append(("huge-offset", [1e12 + v for v in lcg_floats(0xC4A13, 1024, -1.0, 1.0)]))
    out.append(("constant", [3.25] * 1024))
    out.append(("two-levels", [0.0] * 512 + [1.0] * 512))
    out.append(("one-outlier", [0.0] * 1023 + [1e6]))
    out.append(("wide-dynamic-range",
                [10.0 ** (i % 17 - 8) for i in range(1024)]))
    out.append(("alternating-sign",
                [(-1.0) ** i * (1.0 + i * 1e-3) for i in range(1024)]))
    out.append(("with-nan",
                [float("nan") if i % 37 == 0 else v
                 for i, v in enumerate(lcg_floats(0xC4A14, 1024, -5.0, 5.0))]))
    out.append(("mostly-nan",
                [float("nan") if i % 7 else float(i) for i in range(1024)]))
    return out


#: The splits the merge must survive. Unequal splits are the point: the Pébay
#: terms carry (ca - cb) and (ca**2 - ca*cb + cb**2), which vanish or simplify
#: when the two sides are the same size — so an equal-split-only test passes an
#: implementation that dropped them.
SPLITS = {
    "halves": lambda n: [n // 2, n - n // 2],
    "equal-4": lambda n: [n // 4] * 3 + [n - 3 * (n // 4)],
    "equal-256-blocks": lambda n: [n // 4] * 4 if n % 4 == 0 else [n],
    "very-unequal": lambda n: [1, n - 1],
    "unequal-3": lambda n: [1, 2, n - 3],
    "tail-of-one": lambda n: [n - 1, 1],
    "many-small": lambda n: [n // 64] * 63 + [n - 63 * (n // 64)],
}


def split_parts(values: list[float], sizes: list[int]):
    parts, i = [], 0
    for s in sizes:
        parts.append(exact_moments(values[i:i + s]))
        i += s
    assert i == len(values), (i, len(values))
    return parts


def gen_merge_oracle() -> Vector:
    v = Vector(
        id="set4-chan-pebay-merge",
        set_=SET,
        kind="oracle",
        asserts=(
            "Merging per-block (count, mean, M2, M3, M4) with the Chan/Pébay identities "
            "reproduces the moments of the concatenated data — exactly in real "
            "arithmetic, and to the stated relative tolerance in float64 — for every "
            "split including maximally unequal ones, and with NaN excluded from count "
            "and from every sum."
        ),
        source=(
            "AUTHORED against exact rational arithmetic (fractions.Fraction); no "
            "expected value here was produced by running an implementation of the "
            "merge"
        ),
        contract=(
            "A statistic computed from buckets has to agree with one computed from "
            "the samples, or the pyramid is decoration."
        ),
        requires=[],
        notes=(
            "⛔ ORDER MATTERS AND IS PART OF THE CASE. In float64 the merge is not "
            "associative: left-to-right and balanced-tree folds of the same parts differ "
            "in the last bits. Every case states `merge_order`, and the corpus's "
            "recommended rule is a strict left-to-right fold over parts in ascending "
            "index. The `order-independence-in-exact-arithmetic` cases below prove the "
            "identities themselves are sound — the divergence is float64's, not Pébay's."
        ),
    )

    EPS = 2.220446049250313e-16      # 2**-52

    def tolerances(n, mean, m2):
        """Per-case tolerances, derived rather than guessed.

        A flat relative tolerance is wrong here for two independent reasons, and
        both were found by it failing:

        * **M3 and M4 can be zero by cancellation.** On a symmetric dataset the
          exact M3 is ~1e-14 against an M2 of ~1e3 — a relative error against
          that is measuring the cancellation, not the merge. So M3 and M4 are
          bounded against their NATURAL SCALE, `n * sigma**3` and `n * sigma**4`.
        * **M2's merge accuracy degrades with the data's offset.** The merge
          forms `delta = mean_b - mean_a` and squares it; when the mean is far
          from zero relative to the spread, that difference has already lost
          most of its significant digits. Measured, the loss is LINEAR in
          `|mean| / sigma`: at 1e8 +/- 1 the relative error reaches 9.0e-10, and
          at 1e12 +/- 1 it reaches 1.4e-05. The bound below is `EPS * |mean| /
          sigma`, which held on every case with ~28x margin.

        That degradation is a finding about the merge, not about this oracle;
        it is recorded in the fold-order rule §4.3.
        """
        if n == 0 or m2 is None:
            return None
        sigma = (float(m2) / n) ** 0.5 if m2 > 0 else 0.0
        ratio = (abs(float(mean)) / sigma) if sigma > 0 else 0.0
        rel = max(1e-13, EPS * ratio)
        return {
            "mean": {"metric": "relative", "max": rel},
            "M2": {"metric": "relative", "max": rel},
            "M3": {"metric": "absolute_over_scale", "max": rel,
                   "scale": "n * sigma**3"},
            "M4": {"metric": "absolute_over_scale", "max": rel,
                   "scale": "n * sigma**4"},
            "offset_to_spread_ratio": ratio,
            "sigma": sigma,
            "note": ("the relative bound is EPS * |mean| / sigma, floored at 1e-13: "
                     "the merge's accuracy is linear in how far the data sits from "
                     "zero relative to its own spread"),
        }

    for dname, values in datasets():
        n = len(values)
        whole = exact_moments(values)
        for sname, sizer in SPLITS.items():
            sizes = [s for s in sizer(n) if s > 0]
            if sum(sizes) != n or len(sizes) < 2:
                continue
            parts = split_parts(values, sizes)
            merged = fold_left(parts)

            # In exact arithmetic the merge IS the direct computation.
            assert merged[0] == whole[0], (dname, sname, "count")
            for i, field in ((1, "mean"), (2, "M2"), (3, "M3"), (4, "M4")):
                if whole[i] is None:
                    continue
                assert merged[i] == whole[i], (
                    f"{dname}/{sname}: exact merge disagrees with exact direct on {field}"
                    " — the identities as transcribed are wrong")

            v.case(
                f"{dname}/{sname}",
                dataset=dname,
                element_count=big(n),
                nan_count=big(sum(1 for x in values if x != x)),
                split_sizes=[big(s) for s in sizes],
                merge_order="left-to-right over parts in ascending index",
                parts=[{"count": big(p[0]),
                        "mean": None if p[1] is None else fbits(as_float(p[1])),
                        "M2": None if p[2] is None else fbits(as_float(p[2])),
                        "M3": None if p[3] is None else fbits(as_float(p[3])),
                        "M4": None if p[4] is None else fbits(as_float(p[4]))}
                       for p in parts],
                expected_count=big(whole[0]),
                expected_mean=None if whole[1] is None else fbits(as_float(whole[1])),
                expected_M2=None if whole[2] is None else fbits(as_float(whole[2])),
                expected_M3=None if whole[3] is None else fbits(as_float(whole[3])),
                expected_M4=None if whole[4] is None else fbits(as_float(whole[4])),
                expected_from="exact rational arithmetic over the concatenated elements",
                tolerance=tolerances(whole[0], whole[1], whole[2]),
                count_must_be_exact=True,
            )
    return v


def gen_order_independence() -> Vector:
    v = Vector(
        id="set4-chan-pebay-order",
        set_=SET,
        kind="oracle",
        asserts=(
            "The Chan/Pébay identities are order-independent in exact arithmetic — a "
            "left-to-right fold and a balanced-tree fold of the same parts agree exactly "
            "— so any divergence an implementation sees is float64 rounding and not a "
            "defect in the merge, and the format must therefore STATE the fold order for "
            "merged moments to be reproducible."
        ),
        source=(
            "AUTHORED: both folds evaluated over exact Fractions and asserted equal"
        ),
        contract=(
            "The identities are exact in real arithmetic, so any divergence an "
            "implementation sees is its own rounding rather than a defect in the "
            "algebra."
        ),
        requires=[],
        notes=(
            "This vector exists because the float64 divergence is easy to mistake for a "
            "bug in the merge. It is not. The identities are exact and associative over "
            "the rationals; float64 addition is not associative, so the fold SHAPE leaks "
            "into the answer. The measured float64 divergence — left-to-right versus "
            "balanced tree over 256 real buckets — is in the design note; what is "
            "asserted here is the other half: that nothing is wrong with the algebra."
        ),
    )
    for dname, values in datasets():
        n = len(values)
        for k in (2, 4, 8, 64):
            size = n // k
            if size < 1:
                continue
            sizes = [size] * (k - 1) + [n - size * (k - 1)]
            parts = split_parts(values, sizes)
            left, tree = fold_left(parts), fold_tree(parts)
            for i, field in ((0, "count"), (1, "mean"), (2, "M2"),
                             (3, "M3"), (4, "M4")):
                assert left[i] == tree[i], (
                    f"{dname}/{k}: exact folds differ on {field} — the identities are "
                    f"not associative as transcribed")
            v.case(
                f"{dname}/parts={k}",
                dataset=dname, part_count=k,
                split_sizes=[big(s) for s in sizes],
                folds_agree_in_exact_arithmetic=True,
                expected_mean=None if left[1] is None else fbits(as_float(left[1])),
                expected_M2=None if left[2] is None else fbits(as_float(left[2])),
                float64_folds_may_differ=True,
                note="in float64 these two folds differ in the last bits; that is the "
                     "reason the format must state which one it means",
            )
    return v


def gen_nan_rule() -> Vector:
    v = Vector(
        id="set4-chan-pebay-nan",
        set_=SET,
        kind="oracle",
        asserts=(
            "NaN is excluded from count and from every sum, so an all-NaN part is the "
            "identity of the merge and a part containing NaN contributes only its "
            "non-NaN elements — the same NaN rule the stored moments follow, carried "
            "to the merge."
        ),
        source=(
            "AUTHORED: exact rational arithmetic over the non-NaN elements only"
        ),
        contract=(
            "Counting invalid samples poisons every statistic above them, "
            "permanently and silently."
        ),
        requires=[],
        notes=(
            "An implementation that counts NaN elements instead lets one NaN poison "
            "sum, mean and every moment for the whole range, permanently: once an "
            "accumulator's mean is NaN, every merge above it is NaN too. A channel "
            "with one bad sample in a billion then reports nothing at all, and the "
            "pyramid is worse than useless because it looks like data."
        ),
    )
    cases = {
        "no-nan": [1.0, 2.0, 3.0, 4.0],
        "one-nan": [1.0, float("nan"), 3.0, 4.0],
        "all-nan": [float("nan")] * 4,
        "nan-first": [float("nan"), 2.0, 3.0, 4.0],
        "nan-last": [1.0, 2.0, 3.0, float("nan")],
        "single-value-rest-nan": [float("nan"), 7.5, float("nan"), float("nan")],
    }
    for name, values in cases.items():
        n, mean, m2, m3, m4 = exact_moments(values)
        v.case(
            name,
            values=[fbits(x) for x in values],
            expected_count=big(n),
            expected_mean=None if mean is None else fbits(as_float(mean)),
            expected_M2=None if m2 is None else fbits(as_float(m2)),
            expected_M3=None if m3 is None else fbits(as_float(m3)),
            expected_M4=None if m4 is None else fbits(as_float(m4)),
            count_counts_non_nan_only=True,
            all_nan_part_is_merge_identity=(n == 0),
            element_count_including_nan=big(len(values)),
            tolerance={"metric": "relative", "max": 1e-13},
        )

    # An all-NaN part is the identity: merging it changes nothing.
    left = exact_moments([1.0, 2.0, 3.0, 4.0])
    empty = exact_moments([float("nan")] * 8)
    merged = exact_merge(left, empty)
    assert merged == left, "an all-NaN part must be the identity of the merge"
    v.case("all-nan-part-is-identity",
           left_count=big(left[0]), right_count=big(empty[0]),
           merged_equals_left=True,
           expected_mean=fbits(as_float(left[1])),
           expected_M2=fbits(as_float(left[2])),
           tolerance={"metric": "relative", "max": 1e-13},
           note="count 0 on either side short-circuits the merge; without that branch "
                "the delta is NaN and the whole accumulator is poisoned")
    return v


def gen_range_two_level() -> Vector:
    """The RANGE-QUERY fold order — two levels.

    My §2 proposed a single left-to-right rule. That was not enough, and the
    ruling corrected it: a per-block parallel engine folds each block on its own
    thread and combines the block results, while a serial reader folds every
    bucket straight through. Those two give DIFFERENT last bits unless the block
    boundary is itself part of the rule. Measured below, and it is why the rule
    is stated at two levels rather than one.
    """
    v = Vector(
        id="set4-chan-pebay-range",
        set_=SET,
        kind="fixture",
        asserts=(
            "A range query's merged moments are the strict left-to-right fold WITHIN "
            "each block in ascending bucket index, then the strict left-to-right fold of "
            "the per-block results in ascending block index — which is bit-identical for "
            "a per-block parallel engine and a serial reader, and is NOT the same as a "
            "flat fold over every bucket in the range."
        ),
        source=(
            "tools/_moments.py:range_merge, the one shared fold"
        ),
        contract=(
            "A parallel implementation and a serial one must return the same bits "
            "for the same range, which holds only if the block boundary is part of "
            "the rule."
        ),
        requires=["feature:moments"],
        notes=(
            "FIXTURE, not oracle: with the order fixed the answer is exact bytes, which "
            "is the whole reason the order was ruled. Each case carries what a FLAT fold "
            "would have given, so an implementation that ignores the block boundary "
            "fails on data rather than on a reading of the spec."
        ),
    )
    rng_vals = {
        "uniform": lcg_floats(0xF01, 4096, -1.0, 1.0),
        "large-offset": [1e8 + x for x in lcg_floats(0xF02, 4096, -1.0, 1.0)],
        "with-nan": [float("nan") if i % 53 == 0 else x
                     for i, x in enumerate(lcg_floats(0xF03, 4096, -5.0, 5.0))],
    }
    for dname, values in rng_vals.items():
        arr = np.ascontiguousarray(np.array(values, dtype=np.float64))
        buckets = _moments.level1_from_raw(arr, 256)          # 16 buckets
        for block_size in (1, 2, 4, 8, 16):
            blocks = [buckets[i:i + block_size]
                      for i in range(0, buckets.shape[0], block_size)]
            two = _moments.range_merge(blocks)
            flat = _moments.fold_serial([buckets[i] for i in range(buckets.shape[0])])
            identical = two.tobytes() == flat.tobytes()
            v.case(
                f"{dname}/block_size={block_size}",
                dataset=dname,
                bucket_count=int(buckets.shape[0]),
                buckets_per_block=block_size,
                block_count=len(blocks),
                buckets=array_ref(buckets, f"{SET}/range_{dname}_buckets.bin"),
                expected_merged=array_ref(np.ascontiguousarray(two.reshape(1, 5))),
                flat_fold_would_give=array_ref(
                    np.ascontiguousarray(flat.reshape(1, 5))),
                flat_fold_is_identical=identical,
                fold_order=("left-to-right within each block in ascending bucket index, "
                            "then left-to-right over the per-block results in ascending "
                            "block index"),
                note=("block_size = bucket_count is the degenerate single-block case, "
                      "where the two rules necessarily coincide"
                      if block_size >= buckets.shape[0] else
                      ("the flat fold agrees here by luck, not by rule"
                       if identical else
                       "the flat fold DIFFERS — this case is the one that catches an "
                       "implementation ignoring the block boundary")),
            )
    return v


def main() -> None:
    for factory in (gen_merge_oracle, gen_order_independence, gen_nan_rule,
                    gen_range_two_level):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
