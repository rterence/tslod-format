"""Set 5 — the representative: the rule, and what one golden file stores.

Every expected value here is chosen by `_representative.py`, this repository's
expression of the specification's rule, and every time by the one time rule in
`_tslod_build.py`. Nothing is extracted from another implementation.

## Two vectors

* `set5-representative-rule` runs the rule over short channels built here —
  float64, float32 and int64, fixed and variable rate — one choice per case,
  and pins the one-bucket rule directly in three more. Every expected sample is
  stated per bucket, its value as the conventions require and its time beside
  it.
* `set5-representative-stored` runs it over the one golden channel whose level
  2 holds more than two buckets, `sig` in `v1_branching_factor_16.tslod`, and
  states the third column of the stored positions and values at levels 1 and 2.
  A level-2 bucket's representative is chosen from raw samples and not folded,
  and without this the only thing pinning that above level 1 is a file's hash.

## A case pins a choice only if the other choice fails it

So every case names the readings it exists to refuse, and before a vector is
written this generator recomputes the case under each of those readings and
requires a different answer. The readings are written here for that alone —
they are wrong, and nothing in the corpus is computed from them.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import _representative as R
import _tslod_build as B
from _corpus import Vector, array_ref, fbits, fbits_from_bits, i64
from gen_v1_format_vectors import FILES, ramp

SET = "set5-representative"
SOURCE = ("tools/_representative.py — the specification's rule in scalar Python; "
          "times by the exact rule in tools/_tslod_build.py; inputs constructed by "
          "tools/gen_set5_representative.py")
CONTRACT = ("A stored representative is a sample two writers must agree on, because a "
            "reader draws it, and two files that disagree about it draw different "
            "pictures from identical data.")

START = 1_700_000_000_000_000_000
RATE = 1000.0
NAN = float("nan")
_UINT = {"float32": np.uint32, "float64": np.uint64}


def _nan(payload: int, dtype: str = "float64") -> float:
    """A quiet NaN carrying `payload`, so a copied NaN is told from a made one."""
    bits = (0x7FF8000000000000 if dtype == "float64" else 0x7FC00000) | payload
    return np.array([bits], dtype=_UINT[dtype]).view(dtype)[0]


# --------------------------------------------------------------------------
# The wrong readings — each is the rule with one stated choice made otherwise
# --------------------------------------------------------------------------


def _half_up_time(i: int, rate: float) -> int:
    exact = Fraction(i) * B.NS / Fraction(rate)
    return START + math.floor(exact + Fraction(1, 2))


def _reading(values, span: int, ts=None, start: int = START, **swap) -> list[int]:
    """The level rule with the choices named in `swap` made the other way.

    tie="ge"            replace the maximum on an equal area too
    nan_area="argmax"   an argmax over every area, in which a NaN area wins
    fallback="first"    no winner takes the first sample, NaN or not
    anchor="exact"      anchors summed exactly, then rounded once
    anchor="span"       anchors divided by the bucket span, not the count
    anchor="with-nan"   NaN samples not skipped when summing the anchors
    x="index"           x is the sample index even at variable rate
    y="truncate"        y rounded toward zero rather than to nearest-even
    y="exact"           every step in exact rational arithmetic
    edges="skip-nan"    an edge bucket takes its first / last non-NaN sample
    one_bucket="last"   a level of one bucket takes the last sample
    """
    vals = np.ascontiguousarray(values).tolist()
    n = len(vals)
    exact = swap.get("y") == "exact"
    num = Fraction if exact else float

    def conv(v):
        if exact:
            return Fraction(v)
        if swap.get("y") == "truncate" and isinstance(v, int):
            f = float(v)
            return math.nextafter(f, 0.0) if abs(int(f)) > abs(v) else f
        return float(v)

    ys = [conv(v) for v in vals]
    if ts is None or swap.get("x") == "index":
        xs = [num(i) for i in range(n)]
    else:
        xs = [num(int(t) - start) for t in np.asarray(ts).tolist()]

    def isnan(v):
        return not exact and v != v

    n_buckets = -(-n // span)
    bounds = [(j * span, min((j + 1) * span, n)) for j in range(n_buckets)]
    anchors = []
    for a, b in bounds:
        keep = [k for k in range(a, b)
                if swap.get("anchor") == "with-nan" or not isnan(ys[k])]
        if not keep:
            anchors.append((NAN, NAN))
            continue
        div = span if swap.get("anchor") == "span" else len(keep)
        if swap.get("anchor") == "exact":
            sx = math.fsum(xs[k] for k in keep)
            sy = math.fsum(ys[k] for k in keep)
        else:
            sx = sy = num(0)
            for k in keep:
                sx += xs[k]
                sy += ys[k]
        anchors.append((sx / div, sy / div))

    out = []
    for j, (a, b) in enumerate(bounds):
        valid = [k for k in range(a, b) if not isnan(ys[k])]
        if j == 0:
            if n_buckets == 1 and swap.get("one_bucket") == "last":
                out.append(n - 1)
            elif swap.get("edges") == "skip-nan" and valid:
                out.append(valid[0])
            else:
                out.append(0)
            continue
        if j == n_buckets - 1:
            out.append(valid[-1] if swap.get("edges") == "skip-nan" and valid else n - 1)
            continue
        (pax, pay), (nax, nay) = anchors[j - 1], anchors[j + 1]
        areas = [abs((pax - nax) * (ys[k] - pay) - (pax - xs[k]) * (nay - pay))
                 for k in range(a, b)]
        if swap.get("nan_area") == "argmax":
            out.append(a + int(np.argmax(np.array(areas, dtype=np.float64))))
            continue
        best, chosen = num(-1), None
        for k in range(a, b):
            if isnan(ys[k]):
                continue
            area = areas[k - a]
            if area > best or (swap.get("tie") == "ge" and area == best):
                best, chosen = area, k
        if chosen is None:
            chosen = a if swap.get("fallback") == "first" or not valid else valid[0]
        out.append(chosen)
    return out


def _from_level_below(values, bf: int, level: int) -> list[int]:
    """Level `level` chosen among level `level - 1`'s representatives, not raw samples."""
    below = [int(i) for i in R.level_representatives(values, bf ** (level - 1))]
    sub = [values[i] for i in below]
    xs = [float(i) for i in below]
    ys = [float(v) for v in np.asarray(sub).tolist()]
    n_buckets = -(-len(below) // bf)
    bounds = [(j * bf, min((j + 1) * bf, len(below))) for j in range(n_buckets)]
    anchors = [R.bucket_anchor(xs[a:b], ys[a:b]) for a, b in bounds]
    out = []
    for j, (a, b) in enumerate(bounds):
        if j == 0:
            out.append(0)
        elif j == n_buckets - 1:
            out.append(len(values) - 1)
        else:
            out.append(below[a + R.bucket_representative(
                xs[a:b], ys[a:b], anchors[j - 1], anchors[j + 1])])
    return out


def _per_block(values, span: int, block_buckets: int) -> list[int]:
    """The level rule run block by block, so every block's edges are endpoints."""
    per = span * block_buckets
    out = []
    for lo in range(0, len(values), per):
        chunk = values[lo:lo + per]
        out += [lo + int(i) for i in R.level_representatives(chunk, span)]
    return out


def _bucket_reading(xs, ys, prev, nxt, **swap) -> int:
    """The one-bucket rule with one choice made the other way.

    area="shoelace"   the same area in the shoelace association
    area="exact"      the stated area in exact rational arithmetic
    tie="ge"          replace the maximum on an equal area too
    fallback="first"  no winner takes the first sample, NaN or not
    """
    (pax, pay), (nax, nay) = prev, nxt
    best, chosen = -1.0, None
    for k, (x, y) in enumerate(zip(xs, ys)):
        if y != y:
            continue
        if swap.get("area") == "shoelace":
            area = abs(pax * (y - nay) + x * (nay - pay) + nax * (pay - y))
        elif swap.get("area") == "exact":
            fp = [Fraction(v) for v in (pax, pay, nax, nay, x, y)]
            area = abs((fp[0] - fp[2]) * (fp[5] - fp[1]) - (fp[0] - fp[4]) * (fp[3] - fp[1]))
        else:
            area = abs((pax - nax) * (y - pay) - (pax - x) * (nay - pay))
        if area > best or (swap.get("tie") == "ge" and area == best):
            best, chosen = area, k
    if chosen is not None:
        return chosen
    if swap.get("fallback") == "first":
        return 0
    return next((k for k, y in enumerate(ys) if y == y), 0)


# --------------------------------------------------------------------------
# Case encoding
# --------------------------------------------------------------------------


def _rep_value(values: np.ndarray, i: int):
    """The stored sample as the conventions write it: bits for a float."""
    one = values[i:i + 1]
    if values.dtype.kind == "f":
        return fbits_from_bits(int(one.view(_UINT[values.dtype.name])[0]),
                               values.dtype.name)
    return i64(int(one[0]))


def _time(i: int, rate: float, ts) -> int:
    return B.sample_time_ns(START, rate, i) if ts is None else int(ts[i])


def _level_case(vec: Vector, name: str, values, bf: int, *, level: int = 1,
                pins: list[str], refuses: dict, note: str, ts=None,
                rate: float = RATE) -> list[int]:
    values = np.ascontiguousarray(values)
    span = bf ** level
    reps = [int(i) for i in R.level_representatives(values, span, ts, START)]
    for label, swap in refuses.items():
        if swap == "level-below":
            alt = _from_level_below(values, bf, level)
        elif swap in ("max", "min"):    # every middle bucket's first extreme
            arg = np.argmax if swap == "max" else np.argmin
            alt = [reps[0]] + [j * span + int(arg(values[j * span:(j + 1) * span]))
                               for j in range(1, len(reps) - 1)] + [reps[-1]]
        else:
            alt = _reading(values, span, ts, START, **swap)
        assert alt != reps, (
            f"{name}: reading it with {label} gives the same answer, "
            f"so the case does not pin that choice")
    fields = dict(
        applies_to="level", dtype=values.dtype.name,
        timing="fixed" if ts is None else "variable",
        branching_factor=bf, level=level, bucket_span=span,
        input=array_ref(values), start_timestamp=i64(START))
    if ts is None:
        fields["sample_rate"] = fbits(rate)
    else:
        fields["timestamps"] = array_ref(np.ascontiguousarray(ts, dtype=np.int64))
    fields.update(
        expected_bucket_count=len(reps),
        expected=[{"bucket": j, "raw_index": i64(i),
                   "bucket_relative_index": i - j * span,
                   "rep": _rep_value(values, i),
                   "rep_ts": i64(_time(i, rate, ts))}
                  for j, i in enumerate(reps)],
        pins=pins, refuses=sorted(refuses), note=note)
    vec.case(name, **fields)
    return reps


def _bucket_case(vec: Vector, name: str, xs, ys, prev, nxt, *, pins: list[str],
                 refuses: dict, note: str) -> None:
    k = R.bucket_representative(xs, ys, prev, nxt)
    for label, swap in refuses.items():
        assert _bucket_reading(xs, ys, prev, nxt, **swap) != k, (
            f"{name}: reading it with {label} gives the same answer")
    vec.case(name, applies_to="bucket",
             xs=array_ref(np.array(xs, dtype=np.float64)),
             ys=array_ref(np.array(ys, dtype=np.float64)),
             prev_anchor={"x": fbits(prev[0]), "y": fbits(prev[1])},
             next_anchor={"x": fbits(nxt[0]), "y": fbits(nxt[1])},
             expected_bucket_relative_index=k,
             pins=pins, refuses=sorted(refuses), note=note)


# --------------------------------------------------------------------------
# 1. the rule, one choice per case
# --------------------------------------------------------------------------


def gen_rule() -> Vector:
    vec = Vector(
        id="set5-representative-rule", set_=SET, kind="fixture",
        asserts=(
            "At every level L >= 1 each bucket of a numeric channel stores the "
            "representative the specification's rule names — the channel's first "
            "sample in the level's first bucket and its last in the last, and in every "
            "other bucket the non-NaN raw sample of greatest triangle area against its "
            "neighbours' anchors, the first of equal areas winning — with its value "
            "bit for bit and its time by the one time rule."),
        source=SOURCE, contract=CONTRACT,
        requires=["feature:representative", "dtype:float32", "dtype:float64",
                  "dtype:int64", "timing:fixed", "timing:variable"],
        notes=(
            "A level case gives a whole channel and the level's bucket span and states "
            "every bucket's representative: its raw index, its index within the bucket, "
            "its value, and its time. A bucket case gives one bucket's coordinates and "
            "its neighbours' anchors and states the index the triangle picks, so an "
            "implementation can be pinned at either grain. `pins` names the choice a "
            "case exists for and `refuses` the readings that fail it; the generator "
            "recomputed each case under each of those readings and got a different "
            "answer before writing it."),
    )

    _level_case(
        vec, "float64/triangle-not-an-extreme",
        np.array([-1.0, -5.0, -8.5, 8.5, -5.75, -0.75, 3.0, -4.25,
                  9.5, 10.0, 2.25, 3.25, 2.5, -4.5, 5.0, 10.0]), 4,
        pins=["the triangle"],
        refuses={"the bucket's max": "max", "the bucket's min": "min"},
        note=("the middle buckets take the sample of greatest triangle area against "
              "their neighbours' anchors — samples 7 and 8, neither of which is its "
              "bucket's max or its min; the first and last buckets take samples 0 "
              "and 15"))
    _level_case(
        vec, "float32/value-stored-in-its-own-dtype",
        np.array([0.1, -0.3, 1.7, 2.9, -4.1, 3.3, 0.7, -2.2,
                  5.5, -1.9, 0.3, 2.6, -0.8, 1.1, 0.4, -0.6], dtype=np.float32), 4,
        pins=["what is stored"], refuses={"the bucket's max": "max"},
        note=("y is each float32 widened to float64, which is exact; rep is the chosen "
              "sample's float32 bits, never the float64 it was compared as"))
    base = 2 ** 60
    _level_case(
        vec, "int64/above-2p53-y-rounds-to-nearest-even",
        np.array([base + d for d in (768, 1, 128, 128, 1408, 1664, 1023, 255,
                                     257, 640, 255, 384)], dtype=np.int64), 4,
        pins=["the coordinates"],
        refuses={"y rounded toward zero": {"y": "truncate"},
                 "exact integer arithmetic": {"y": "exact"}},
        note=("every value is near 2^60, where float64 steps by 256, so y rounds. The "
              "middle bucket's first sample, 2^60 + 1408, sits exactly halfway between "
              "two float64 values and rounds to the even one, 2^60 + 1536; rounding "
              "toward zero, or comparing the exact integers, picks the next sample "
              "instead. rep is stored as the exact int64, not as y"))
    _level_case(
        vec, "constant/first-of-equal-areas-wins",
        np.full(16, 5.0), 4,
        pins=["the strict comparison"], refuses={"a >= comparison": {"tie": "ge"}},
        note=("a constant integer series: every anchor is exactly 5.0 and every area "
              "exactly zero, so the strict comparison against a maximum seeded at -1 is "
              "all that decides, and each middle bucket takes its first sample"))
    _level_case(
        vec, "nan/in-a-bucket-before-its-winner",
        np.array([0.0, 1.0, 0.5, 1.5, 2.0, NAN, 9.0, 3.0,
                  -1.0, 6.0, -4.0, 2.0, 1.0, 0.0, 2.0, 1.0]), 4,
        pins=["the NaN rules"],
        refuses={"an argmax in which a NaN area wins": {"nan_area": "argmax"},
                 "NaN kept in the anchor sums": {"anchor": "with-nan"}},
        note=("sample 5 is NaN. It is skipped in bucket 1's anchor and is never a "
              "candidate, so bucket 1 takes the sample of greatest area after it and "
              "bucket 2, beside it, still runs the triangle against a real anchor"))
    _level_case(
        vec, "nan/invalid-bucket-and-its-two-neighbours",
        np.array([1.0, 2.0, 3.0, 4.0,
                  _nan(0x0A1), 6.0, -3.0, 8.0,
                  _nan(0xBEEF), _nan(0x0B2), _nan(0x0B3), _nan(0x0B4),
                  _nan(0x0C1), -5.0, 10.0, 2.0,
                  3.0, 1.0, 4.0, 1.5]), 4,
        pins=["the NaN rules"],
        refuses={"a fallback to the first sample, NaN or not": {"fallback": "first"}},
        note=("bucket 2 holds no valid sample, so it is invalid: its anchor is NaN and "
              "it stores its first sample as stored, the NaN with payload 0xBEEF copied "
              "bit for bit. Its neighbours' every area is then NaN and none wins, so "
              "each takes its first sample that is not NaN — the second of each, since "
              "both begin with a NaN of their own"))
    _level_case(
        vec, "edges/endpoints-even-when-nan",
        np.array([_nan(0x0E1), 2.0, 5.0, 1.0, 3.0, 7.0, -2.0, 4.0,
                  6.0, 0.5, 2.5, _nan(0x0E2)]), 4,
        pins=["the edges"],
        refuses={"edges that skip a NaN": {"edges": "skip-nan"}},
        note=("the channel's first and last samples are NaN with distinct payloads. The "
              "first and last buckets take them all the same, unconditionally, and "
              "store their bits; the NaN is skipped only in those buckets' anchors"))
    _level_case(
        vec, "one-bucket-level/stores-sample-0",
        np.array([7.5, 9.0, 8.0]), 4,
        pins=["the edges"],
        refuses={"a single bucket taking the last sample": {"one_bucket": "last"}},
        note=("three samples at branching factor 4 are one bucket, both the level's "
              "first and its last; it takes sample 0"))
    _level_case(
        vec, "ragged-last-bucket/anchor-over-the-samples-that-exist",
        np.array([-6.0, -7.0, 6.0, -3.0, -5.0, 3.0, 2.0, -2.0, 0.0, 1.0]), 4,
        pins=["the anchors"],
        refuses={"anchors divided by the bucket span": {"anchor": "span"}},
        note=("ten samples at branching factor 4: the last bucket holds two. Its anchor "
              "is the mean of those two, not a sum over four slots, and the middle "
              "bucket's choice depends on it; the last bucket itself takes sample 9"))
    _level_case(
        vec, "anchors/summed-left-to-right",
        np.array([1e16] + [1.0] * 14 + [-1e16]
                 + [-2.0, -4.0, -8.0, 6.0, -2.0, -4.0, -8.0, -5.0,
                    -6.0, 1.0, -4.0, 6.0, -3.0, 8.0, -8.0, 4.0]
                 + [5.0, 2.0, 3.0, -7.0, 9.0, -3.0, -2.0, 2.0,
                    -9.0, 2.0, 3.0, -1.0, 4.0, -6.0, 8.0, 2.0]
                 + [0.0]), 16,
        pins=["the anchors"],
        refuses={"anchors summed exactly": {"anchor": "exact"}},
        note=("bucket 0 is 1e16, fourteen 1.0s and -1e16. Summed left to right in "
              "float64 every 1.0 is absorbed by 1e16 and the sum is 0; summed exactly "
              "it is 14. Bucket 1 is judged against that anchor and takes a different "
              "sample under each, so the order of the sum is part of the rule and a "
              "pairwise or compensated sum is not"))
    level2 = [16.0, -13.0, 5.0, 19.0, -12.0, -20.0, 7.0, -15.0, 0.0, 18.0, 11.0, 11.0,
              2.0, 3.0, -17.0, -12.0, -2.0, -11.0, 16.0, 20.0, 12.0, -2.0, 15.0, 15.0,
              19.0, -6.0, -4.0, -16.0, 15.0, -5.0, -4.0, -2.0, 13.0, -12.0, -5.0, 3.0,
              9.0, 4.0, -9.0, -12.0, -19.0, 1.0, -15.0, 16.0, -18.0, -15.0, -13.0, 12.0,
              18.0, 9.0, -5.0, 4.0, 9.0, 10.0, 0.0, -14.0, 13.0, -19.0, 14.0, 4.0,
              -17.0, -11.0, 7.0, -6.0]
    _level_case(
        vec, "level-2/chosen-from-raw-samples", np.array(level2), 4, level=2,
        pins=["the raw samples at every level"],
        refuses={"a choice among level 1's representatives": "level-below"},
        note=("64 samples at branching factor 4: level 2 is four buckets of 16 raw "
              "samples each, and each middle bucket chooses among all 16 against its "
              "neighbours' anchors over their own 16. Choosing among the four level-1 "
              "representatives each covers instead stores different samples"))
    offsets = [0, 1, 2, 3, 4, 5, 55, 56, 106, 156, 157, 158, 159, 160, 161, 162]
    _level_case(
        vec, "variable-rate/x-is-the-timestamp-difference",
        np.array([5.0, -5.0, -4.0, -4.0, 4.0, -4.0, -7.0, -3.0,
                  -8.0, 8.0, -6.0, 3.0, -7.0, -1.0, -8.0, 9.0]), 4,
        ts=np.array([START + d * 1_000_000 for d in offsets], dtype=np.int64),
        pins=["the coordinates"],
        refuses={"x as the sample index": {"x": "index"}},
        note=("a variable-rate channel with three 50 ms gaps. x is each stored timestamp "
              "minus start_timestamp, so the gaps move the anchors and the areas; taking "
              "x as the sample index picks a different sample in bucket 1. rep_ts is "
              "the chosen sample's stored timestamp"))
    rate = 400_000_000.0
    values = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0,
                       0.0, -8.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    reps = _level_case(
        vec, "rep-ts/by-the-exact-time-rule", values, 4, rate=rate,
        pins=["what is stored"], refuses={},
        note=("at 400 MHz sample i is at i × 2.5 ns, so an odd sample falls on half a "
              "nanosecond. The middle buckets take samples 5 and 9, whose times 12.5 ns "
              "and 22.5 ns round half to even, to 12 and 22; rounding half up gives 13 "
              "and 23"))
    assert all(_half_up_time(i, rate) != B.sample_time_ns(START, rate, i)
               for i in reps[1:-1]), "rep-ts: half-up rounding does not differ here"

    _bucket_case(
        vec, "bucket/the-stated-association",
        [4.0, 5.0], [2.9297687251995264, 3.502507241000302],
        (1.762154486549691, -3.153396561451234),
        (9.535725917125417, 1.2988272021680194),
        pins=["the association"],
        refuses={"the shoelace association": {"area": "shoelace"},
                 "exact arithmetic": {"area": "exact"}},
        note=("two samples whose areas differ by less than the rounding of the "
              "expression that computes them. In exact arithmetic the second is the "
              "larger, and written as the shoelace sum it comes out larger too; in "
              "float64, in the stated association, the first does, and the first is "
              "what the bucket stores. The rule states the arithmetic and the "
              "association, so which sample wins is not left to either"))
    _bucket_case(
        vec, "bucket/first-of-equal-nonzero-areas",
        [1.0, 2.0, 3.0], [1.0, -1.0, 1.0], (0.0, 0.0), (4.0, 0.0),
        pins=["the strict comparison"], refuses={"a >= comparison": {"tie": "ge"}},
        note=("both anchors lie on y = 0, so each sample's area is 4|y|: three equal "
              "areas of 4, and the first wins"))
    _bucket_case(
        vec, "bucket/no-area-wins-beside-an-invalid-neighbour",
        [4.0, 5.0, 6.0, 7.0], [NAN, 3.0, -1.0, 5.0], (NAN, NAN), (8.0, 2.0),
        pins=["the NaN rules"],
        refuses={"a fallback to the first sample, NaN or not": {"fallback": "first"}},
        note=("the previous anchor is NaN, as an invalid bucket's is, so every area is "
              "NaN and none beats the seed; the bucket takes its first sample that is "
              "not NaN, index 1"))
    return vec


# --------------------------------------------------------------------------
# 2. what one golden channel stores
# --------------------------------------------------------------------------

GOLDEN = "v1_branching_factor_16.tslod"
GOLDEN_BF = 16


def gen_stored() -> Vector:
    raw = ramp("float32", 4096)
    vec = Vector(
        id="set5-representative-stored", set_=SET, kind="fixture",
        asserts=(
            f"Channel 0 of {GOLDEN} stores, at levels 1 and 2, the representative the "
            "rule names for every bucket — the third column of each level's values "
            "stream is the chosen sample's value and the third column of its positions "
            "stream that sample's time — and level 2's is chosen from the raw samples, "
            "not from level 1's representatives."),
        source=SOURCE, contract=CONTRACT,
        requires=["format:v1", "profile:0", "feature:representative",
                  "dtype:float32", "timing:fixed"],
        notes=(
            f"{GOLDEN} is the one golden file whose channel is long enough for a level "
            "2 of more than two buckets: 4,096 float32 samples at branching factor 16 "
            "give 256 buckets at level 1 and 16 at level 2, so fourteen level-2 buckets "
            "run the triangle over 256 raw samples each. `input` is the channel's "
            "samples exactly as its level-0 blocks hold them."),
    )
    for level in (1, 2):
        span = GOLDEN_BF ** level
        reps = R.level_representatives(raw, span)
        below = _from_level_below(raw, GOLDEN_BF, level) if level > 1 else None
        argmax = [j * span + int(np.argmax(raw[j * span:(j + 1) * span]))
                  for j in range(len(reps))]
        differ = {"the bucket's max": sum(a != b for a, b in zip(reps, argmax))}
        if len(reps) > GOLDEN_BF:       # more than one block at this level
            blocked = _per_block(raw, span, GOLDEN_BF)
            differ["a pass block by block"] = sum(a != b for a, b in zip(reps, blocked))
        if below is not None:
            differ["a choice among level 1's representatives"] = sum(
                a != b for a, b in zip(reps, below))
        for label, count in differ.items():
            assert count, f"level {level}: {label} stores the same samples"
        rel = f"{SET}/bf16_sig_level{level}"
        times = np.array([B.sample_time_ns(START, RATE, int(i)) for i in reps],
                         dtype=np.int64)
        vec.case(
            f"level-{level}",
            file=f"{FILES}/{GOLDEN}", channel=0, level=level,
            branching_factor=GOLDEN_BF, bucket_span=span,
            block_samples=GOLDEN_BF, dtype="float32", timing="fixed",
            start_timestamp=i64(START), sample_rate=fbits(RATE),
            input=array_ref(raw, f"{SET}/bf16_sig_input.bin"),
            expected_bucket_count=len(reps),
            expected_raw_index=array_ref(np.ascontiguousarray(reps, dtype=np.int64),
                                         f"{rel}_raw_index.bin"),
            expected_rep=array_ref(np.ascontiguousarray(raw[reps]), f"{rel}_rep.bin"),
            expected_rep_ts=array_ref(times, f"{rel}_rep_ts.bin"),
            stored_as=("values[:, 2] and positions[:, 2] of every block at this level, "
                       "in block order"),
            note=(f"level {level}: {len(reps)} buckets of {span} raw samples. Of them, "
                  + "; ".join(f"{count} would store a different sample under {label}"
                              for label, count in differ.items())))
    return vec


def main() -> None:
    for factory in (gen_rule, gen_stored):
        vector = factory()
        vector.write()
        print(f"  {vector.id}: {len(vector.cases)} cases")


if __name__ == "__main__":
    main()
