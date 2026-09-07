"""Set 3 — the bucket fold, and the moment tuple each bucket carries.

Every expected value here is folded by `_fold.py` and `_moments.py`, this
repository's re-expression of the fold from the specification. Nothing is
extracted from another implementation: `_corpus.py` states the rule this
generator exists to satisfy — "expected values come from this repository, and
another implementation only ever gets to CONFIRM them. A corpus that cannot be
rebuilt without the thing it checks is not independent of it."

## Where the inputs come from

Inputs are constructed here, deterministically, so the corpus rebuilds the same
bytes on any machine with no RNG library in the path:

  * a four-element **specials prefix** per dtype — the two zeros and the two
    unit values for floats, the type's extremes and 0/-1 for integers — so
    every case carries the values most likely to be mishandled, in the first
    bucket, before any filler;
  * **filler** from a small LCG written out below, seeded
    `0x5EED + n*31 + bf`. Floats land on a 3-decimal grid in [-1000, 1000];
    integers take the low bits of the draw, so a signed dtype sees negatives
    and an unsigned one sees its whole range.

The `mom_` inputs use a separate seed offset from the `raw_` ones, so the
moments are not folded over the same samples the min/max vectors use.

## Where the aggregated cases come from

A `from_raw=0` case's input is the `from_raw=1` case's OUTPUT — a real level,
folded by `_fold`, not a hand-written array that resembles one. That is what
makes the level-over-level fold testable: an implementation that gets level 1
right and level 2 wrong fails here rather than passing both.
"""

from __future__ import annotations

import struct

import numpy as np

import _fold
import _moments
from _corpus import Vector, array_ref, i64

SET = "set3-build-level"

FOLD_SOURCE = ("tools/_fold.py — the specification's fold in numpy; "
               "inputs constructed by tools/gen_set3_build_level.py")
MOMENT_SOURCE = ("tools/_moments.py, the one shared fold; inputs "
                 "constructed by tools/gen_set3_build_level.py")

FLOATS = ("float32", "float64")
SIGNED = ("int8", "int16", "int32", "int64")
UNSIGNED = ("uint8", "uint16", "uint32", "uint64")
DTYPES = FLOATS + SIGNED + UNSIGNED

#: (n, branching_factor). Two powers of two, a ragged tail (10/3), a full
#: block (512/256), a ragged block (700/256), and a single-bucket fold (64/64).
GRID = [(8, 2), (16, 4), (512, 256), (700, 256), (10, 3), (64, 64)]

COLUMN_ORDER = "min,max,first,last"
BUCKET_COUNT_RULE = "ceil(N / branching_factor)"

_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK64 = (1 << 64) - 1


def draws(seed: int, count: int) -> list[int]:
    """A tiny LCG, so regeneration is byte-stable on any interpreter."""
    state = seed & _MASK64
    out = []
    for _ in range(count):
        state = (state * _LCG_A + _LCG_C) & _MASK64
        out.append(state)
    return out


def specials(dtype: str) -> list:
    """The four values a fold is most likely to get wrong, per dtype."""
    if dtype in FLOATS:
        return [0.0, -0.0, 1.0, -1.0]
    info = np.iinfo(dtype)
    if dtype in SIGNED:
        return [info.min, info.max, 0, -1]
    return [0, info.max, 0, 1]


def samples(dtype: str, n: int, bf: int, salt: int = 0) -> np.ndarray:
    """`n` values of `dtype`: the specials prefix, then deterministic filler."""
    head = specials(dtype)[:n]
    raw = draws(0x5EED + n * 31 + bf + salt, max(n - len(head), 0))
    if dtype in FLOATS:
        tail = [(v % 2_000_001) / 1000.0 - 1000.0 for v in raw]
        return np.array(head + tail, dtype=dtype)
    width = np.dtype(dtype).itemsize * 8
    mask = (1 << width) - 1
    tail = np.array([v & mask for v in raw], dtype=f"uint{width}").view(dtype)
    return np.concatenate([np.array(head, dtype=dtype), tail]).astype(dtype, copy=False)


def _numeric_case(vec: Vector, name: str, arr, bf: int, from_raw: int,
                  rel_path: str | None = None, pattern: tuple | None = None,
                  extra: dict | None = None) -> np.ndarray:
    """One numeric-mode case. Returns the fold, so a level case can use it."""
    tuples, positions = _fold.build_level(arr, bf, from_raw, 0, True)
    fields = {
        "branching_factor": bf,
        "from_raw": from_raw,
        "aggregation_mode": 0,
        "positions_requested": True,
        "input": array_ref(arr, rel_path),
        "expected_tuples": array_ref(tuples),
        "expected_bucket_count": i64(tuples.shape[0]),
        "column_order": COLUMN_ORDER,
    }
    if pattern:
        fields[pattern[0]] = pattern[1]
    fields.update(extra or {})
    fields["expected_bucket_count_rule"] = BUCKET_COUNT_RULE
    tail = arr.shape[0] % bf
    if tail:
        # The tail is folded over exactly the elements it has, never padded and
        # never dropped, so its length is part of what the case asserts.
        fields["has_ragged_tail"] = True
        fields["tail_length"] = tail
    fields["expected_positions"] = array_ref(np.ascontiguousarray(positions, dtype=np.int64))
    fields["positions_are_bucket_relative"] = True
    vec.case(name, **fields)
    return tuples


# --------------------------------------------------------------------------
# 1. numeric mode, every dtype, raw and aggregated
# --------------------------------------------------------------------------


def gen_numeric() -> Vector:
    vec = Vector(
        id="set3-build-level-numeric", set_=SET, kind="fixture",
        asserts=(
            "build_level in numeric mode produces [min, max, first, last] per bucket "
            "in that column order, over all ten dtypes, for both from_raw=1 (raw "
            "samples) and from_raw=0 (tuples), with bucket-relative [min_idx, max_idx] "
            "positions, and folds the ragged tail rather than dropping it."),
        source=FOLD_SOURCE,
        contract=("All four columns share a type, so a swapped pair is invisible in the "
                  "data; the order is pinned because nothing else can recover it."),
        requires=["feature:positions"] + [f"dtype:{d}" for d in DTYPES],
        notes=("⟢ Column order is [min, max, first, last], NOT the first/last/min/max "
               "ordering the format is often described with. Getting this backwards is "
               "silent: all four are the same dtype and a swapped pair only shows on "
               "data where min != first."),
    )
    for dtype in DTYPES:
        for n, bf in GRID:
            arr = samples(dtype, n, bf)
            rel = f"{SET}/raw_{dtype}_bf{bf}_n{n}_input.bin"
            level = _numeric_case(vec, f"raw/{dtype}/bf{bf}/n{n}", arr, bf, 1, rel)
            # The level just produced, folded again — pairs of buckets this
            # time, so from_raw=0 is exercised on a level that really is one.
            if level.shape[0] >= 2:
                _numeric_case(vec, f"agg/{dtype}/bf2/n{level.shape[0]}",
                              np.ascontiguousarray(level), 2, 0)
    return vec


# --------------------------------------------------------------------------
# 2. the tie-break: strict < and >, so the FIRST occurrence wins
# --------------------------------------------------------------------------

TIE_PATTERNS = [
    ("all-equal", [5, 5, 5, 5, 5, 5, 5, 5]),
    ("min-repeated", [1, 7, 1, 3, 2, 8, 2, 9]),
    ("max-repeated", [9, 2, 9, 3, 8, 1, 8, 4]),
    ("both-repeated", [2, 6, 2, 6, 1, 5, 1, 5]),
    ("monotone-up", [1, 2, 3, 4, 5, 6, 7, 8]),
    ("monotone-down", [8, 7, 6, 5, 4, 3, 2, 1]),
]


def gen_tie_break() -> Vector:
    vec = Vector(
        id="set3-build-level-tie-break", set_=SET, kind="fixture",
        asserts=(
            "The comparison is strict < and >, so when a bucket's minimum or maximum "
            "occurs more than once the FIRST occurrence wins and its bucket-relative "
            "index is the one stored — which is what decides the stored min_ts/max_ts."),
        source=FOLD_SOURCE,
        contract=("The values are identical either way — only the positions separate a "
                  "strict comparison from a non-strict one, and positions are what "
                  "become stored timestamps."),
        requires=["feature:positions"],
        notes=("The value columns are identical either way; only the POSITIONS differ, "
               "and positions are what the writer turns into stored timestamps. A "
               "tie-break vector that checked only the tuples would pass a <= "
               "implementation."),
    )
    for dtype in DTYPES:
        for pattern, values in TIE_PATTERNS:
            arr = np.array(values, dtype=dtype)
            _numeric_case(vec, f"{dtype}/{pattern}/bf4", arr, 4, 1,
                          pattern=("tie_pattern", pattern))
    return vec


# --------------------------------------------------------------------------
# 3. the NaN matrix
# --------------------------------------------------------------------------

_N = float("nan")
_I = float("inf")

NAN_PATTERNS = [
    ("some-nan", [3.0, _N, 4.0, 1.0, _N, 9.0, 2.0, 6.0]),
    ("all-nan-bucket", [_N, _N, _N, _N, 5.0, 9.0, 2.0, 6.0]),
    ("every-bucket-all-nan", [_N] * 8),
    ("first-is-nan", [_N, 1.0, 4.0, 1.0, _N, 9.0, 2.0, 6.0]),
    ("last-is-nan", [3.0, 1.0, 4.0, _N, 5.0, 9.0, 2.0, _N]),
    ("first-and-last-nan", [_N, 1.0, 4.0, _N, _N, 9.0, 2.0, _N]),
    ("no-nan", [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]),
    ("nan-with-negative-zero", [-0.0, _N, 0.0, -0.0, _N, -0.0, 0.0, _N]),
    ("nan-with-infinities", [_I, _N, -_I, 1.0, _N, _I, -_I, 2.0]),
]


def gen_nan_matrix() -> Vector:
    vec = Vector(
        id="set3-build-level-nan-matrix", set_=SET, kind="fixture",
        asserts=(
            "NaN is skipped in min and max; an all-NaN bucket yields four NaNs and "
            "positions [0, 0]; first and last are the LITERAL first and last elements "
            "and may therefore be NaN even when min and max are finite — and every NaN "
            "that came from an input element keeps that element's bit pattern."),
        source=FOLD_SOURCE,
        contract=("NaN handling is the whole of a bucket's behaviour on invalid data, "
                  "and numeric equality cannot test it because NaN is equal to nothing."),
        requires=["feature:positions", "dtype:float32", "dtype:float64"],
        notes=("The spec's only NaN sentence is about bitfield mode, which has no NaN. "
               "This vector is the whole of the NaN contract. Bit-pattern equality "
               "matters here more than anywhere: a manufactured NaN is the canonical "
               "quiet NaN, a propagated one keeps its input's payload, and numeric "
               "equality cannot tell them apart because NaN != NaN."),
    )
    for dtype in FLOATS:
        for pattern, values in NAN_PATTERNS:
            arr = np.array(values, dtype=dtype)
            level = _numeric_case(vec, f"{dtype}/{pattern}/bf4", arr, 4, 1,
                                  pattern=("nan_pattern", pattern))
            _numeric_case(vec, f"{dtype}/{pattern}/agg-bf2",
                          np.ascontiguousarray(level), 2, 0,
                          pattern=("nan_pattern", pattern), extra={
                              "note": ("from_raw=0 over a level that already contains "
                                       "NaN — the class a hand-written fold and a "
                                       "vectorised one were measured to diverge on"),
                          })
    return vec


# --------------------------------------------------------------------------
# 4. bitfield mode
# --------------------------------------------------------------------------


def bitfield_patterns(dtype: str) -> list[tuple[str, list[int]]]:
    """The lane patterns, plus the sign-bit cases only a signed dtype has."""
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    out = [
        ("lanes", [1, 2, 4, 8, 3, 5, 9, 15]),
        ("all-zero", [0] * 8),
        ("all-ones", [hi] * 8),
    ]
    if dtype in SIGNED:
        out += [
            ("extremes", [lo, hi, 0, hi, lo, 0, hi, lo]),
            ("sign-bit", [lo, 1, -1, -2, lo, -1, 2, hi]),
            ("minus-one-and-minus-two", [-1, -2, -1, -2, -1, -2, -1, -2]),
        ]
    else:
        out += [("extremes", [0, hi, 0, hi, 0, 0, hi, 0])]
    return out


def _bitfield_case(vec: Vector, name: str, arr, bf: int, from_raw: int,
                   pattern: str) -> np.ndarray:
    tuples = _fold.build_level(arr, bf, from_raw, 1)
    vec.case(
        name,
        branching_factor=bf,
        from_raw=from_raw,
        aggregation_mode=1,
        positions_requested=False,
        input=array_ref(arr),
        expected_tuples=array_ref(tuples),
        expected_bucket_count=i64(tuples.shape[0]),
        column_order="OR,AND,first,last",
        bitfield_pattern=pattern,
        expected_bucket_count_rule=BUCKET_COUNT_RULE,
    )
    return tuples


def gen_bitfield() -> Vector:
    vec = Vector(
        id="set3-build-level-bitfield", set_=SET, kind="fixture",
        asserts=(
            "Bitfield mode produces [OR, AND, first, last] per bucket, is permitted on "
            "the eight integer dtypes and refused on the two float dtypes, operates on "
            "the two's-complement bit pattern including the sign bit for signed dtypes, "
            "and does not accept a request for positions."),
        source=FOLD_SOURCE,
        contract=("OR and AND over a signed integer act on the bit pattern, so a reader "
                  "that widens to a signed type and sign-extends produces different "
                  "flags from the same bytes."),
        requires=[f"dtype:{d}" for d in SIGNED + UNSIGNED],
        notes=("The sign bit is the case that separates a correct implementation from "
               "one that widened to a signed int and sign-extended: OR over int8 values "
               "-128 and 1 is -127, and AND over -1 and -2 is -2. Both are in the "
               "patterns below."),
    )
    for dtype in SIGNED + UNSIGNED:
        for pattern, values in bitfield_patterns(dtype):
            arr = np.array(values, dtype=dtype)
            level = _bitfield_case(vec, f"{dtype}/{pattern}/bf4", arr, 4, 1, pattern)
            _bitfield_case(vec, f"{dtype}/{pattern}/agg-bf2",
                           np.ascontiguousarray(level), 2, 0, pattern)

    for dtype in FLOATS:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=dtype)
        cls, message = _refusal(arr, 4, 1, 1, False, "bitfield-on-float-dtype")
        vec.case(
            f"{dtype}/bitfield-refused",
            branching_factor=4, from_raw=1, aggregation_mode=1,
            input=array_ref(arr),
            rejection_class=cls, message=message,
            note=("bitfield mode has no meaning on a float dtype: OR and AND over an "
                  "IEEE-754 bit pattern are not a summary of anything"),
        )

    arr = np.array([1, 2, 4, 8, 3, 5, 9, 15], dtype="int32")
    cls, message = _refusal(arr, 4, 1, 1, True, "positions-in-bitfield-mode")
    vec.case(
        "positions-with-bitfield-refused",
        branching_factor=4, from_raw=1, aggregation_mode=1,
        positions_requested=True,
        input=array_ref(arr),
        rejection_class=cls, message=message,
        note=("a bitfield bucket has no argmin or argmax to report, so asking for "
              "positions is refused rather than answered with zeros"),
    )
    return vec


def _refusal(arr, bf, from_raw, mode, positions, rejection_class) -> tuple[str, str]:
    """Run the fold, require a refusal, and record the class it belongs to.

    The class is stated in the format's terms and is what a port must
    reproduce. The message is whatever this generator's fold happened to say,
    kept because it is informative and compared by nothing: an implementation
    in another language refuses in its own words and with its own error type,
    and neither is part of the format.
    """
    try:
        _fold.build_level(arr, bf, from_raw, mode, positions)
    except Exception as exc:
        return rejection_class, str(exc)
    raise AssertionError(f"expected a refusal from build_level({bf}, {from_raw}, {mode})")


# --------------------------------------------------------------------------
# 5. edges: refusals, and the ragged single bucket
# --------------------------------------------------------------------------


def gen_edges() -> Vector:
    vec = Vector(
        id="set3-build-level-edges", set_=SET, kind="fixture",
        asserts=(
            "build_level refuses an empty input and a branching factor below 2 rather "
            "than returning an empty level, and a single-bucket fold of a bucket "
            "shorter than the branching factor is the identity on values."),
        source=FOLD_SOURCE,
        contract=("The fold refuses an empty input and a branching factor below two "
                  "rather than returning something plausible."),
        requires=[],
        notes=("`k == 1` is NOT handled here: the fold is defined for a branching "
               "factor of 2 or more, and a caller that wants one bucket per element "
               "is asking for the input back rather than for a fold."),
    )
    eight = np.arange(8, dtype="float64")
    for bf in (0, 1):
        cls, message = _refusal(eight, bf, 1, 0, False, "branching-factor-below-two")
        vec.case(f"branching-factor-{bf}-refused",
                 input=array_ref(eight), branching_factor=bf, from_raw=1,
                 aggregation_mode=0, rejection_class=cls, message=message)

    empty = np.array([], dtype="float64")
    cls, message = _refusal(empty, 4, 1, 0, False, "empty-input")
    vec.case("empty-input-refused",
             input=array_ref(empty), branching_factor=4, from_raw=1,
             aggregation_mode=0, rejection_class=cls, message=message)

    for n in (1, 2, 3):
        arr = np.arange(7.5, 7.5 + n, dtype="float64")
        _numeric_case(vec, f"n{n}-shorter-than-bf4", arr, 4, 1, extra={
            "note": ("one bucket, ragged: min/max/first/last all come from the same "
                     "short run"),
        })
    return vec


# --------------------------------------------------------------------------
# 6. the moment tuple each bucket carries
# --------------------------------------------------------------------------

MOMENT_GRID = [(16, 4), (512, 256), (700, 256), (10, 3)]
#: The moments are folded over their own samples, not the min/max vectors'.
MOMENT_SALT = 0x9E37

NAN_MOMENT_PATTERNS = [
    ("some-nan", [3.0, _N, 4.0, 1.0, _N, 9.0, 2.0, 6.0], None),
    ("all-nan-bucket", [_N, _N, _N, _N, 5.0, 9.0, 2.0, 6.0], "all-nan"),
    ("every-bucket-all-nan", [_N] * 8, "all-nan"),
    ("first-is-nan", [_N, 1.0, 4.0, 1.0, _N, 9.0, 2.0, 6.0], None),
    ("last-is-nan", [3.0, 1.0, 4.0, _N, 5.0, 9.0, 2.0, _N], None),
    ("single-valid-in-bucket", [_N, _N, _N, 7.5, 1.0, 2.0, 3.0, 4.0], None),
]


def gen_bucket_moments() -> Vector:
    vec = Vector(
        id="set3-bucket-moments", set_=SET, kind="fixture",
        asserts=(
            "Every numeric value bucket at level >= 1 carries (count, mean, M2, M3, M4) "
            "as float64, where count is the number of NON-NaN elements and the stored "
            "tuple is the strict left-to-right Chan/Pebay fold over the bucket's "
            "branching_factor children in ascending index — raw samples at level 1, the "
            "level below's stored tuples above it."),
        source=MOMENT_SOURCE,
        contract=("Moments computed by different implementations are only comparable if "
                  "the order they are folded in is part of the format."),
        requires=["format:v1", "feature:moments"],
        notes=("⛔ The fold ORDER is normative and stated at TWO levels, because "
               "floating-point addition is not associative and a single-level statement "
               "would leave a per-block parallel engine and a serial reader disagreeing "
               "in the last bits — measured: a flat fold over 256 level-1 buckets and a "
               "two-level fold over four blocks of 64 give different M2. The stored rule "
               "is here; the range-query rule is in set4-chan-pebay-range."),
    )
    fold_order = ("strict left-to-right over the bucket's children in ascending index, "
                  "float64, Chan/Pebay as written")
    level2_note = (
        "level 2 folds level 1's STORED tuples, not the raw samples again. The two "
        "differ in the last bits, and the hierarchical one is the rule — so a writer "
        "that recomputes each level from level 0 produces a file a conforming reader "
        "disagrees with")

    for dtype in DTYPES:
        for n, bf in MOMENT_GRID:
            arr = samples(dtype, n, bf, salt=MOMENT_SALT)
            level1 = _moments.level1_from_raw(arr, bf)
            level2 = _moments.next_level(level1, bf)
            vec.case(
                f"{dtype}/bf{bf}/n{n}",
                branching_factor=bf,
                level=1,
                dtype=dtype,
                input=array_ref(arr, f"{SET}/mom_{dtype}_bf{bf}_n{n}_input.bin"),
                expected_moments=array_ref(level1),
                moment_columns=list(_moments.COLUMNS),
                stream_shape=[int(s) for s in level1.shape],
                stream_dtype="float64",
                row_major_not_planar=True,
                fold_order=fold_order,
                expected_moments_level2=array_ref(level2),
                level2_folds_level1_not_raw=True,
                note=level2_note,
            )

    for dtype in FLOATS:
        for pattern, values, flag in NAN_MOMENT_PATTERNS:
            arr = np.array(values, dtype=dtype)
            level1 = _moments.level1_from_raw(arr, 4)
            counts = [int(c) for c in level1[:, 0]]
            fields = dict(
                branching_factor=4,
                level=1,
                dtype=dtype,
                nan_pattern=pattern,
                input=array_ref(arr),
                expected_moments=array_ref(level1),
                moment_columns=list(_moments.COLUMNS),
                expected_counts=counts,
                count_is_non_nan_count=True,
            )
            if flag == "all-nan":
                fields["all_nan_bucket_present"] = True
                fields["all_nan_bucket_moments"] = (
                    "canonical quiet NaN 0x7FF8000000000000")
            if pattern == "single-valid-in-bucket":
                fields["note"] = (
                    "a bucket with one valid element has count 1, mean = that element, "
                    "and M2 = M3 = M4 = 0 exactly — the variance of one sample is zero, "
                    "not undefined")
            vec.case(f"nan/{dtype}/{pattern}", **fields)

    for dtype in ("int8", "int16", "int32"):
        arr = samples(dtype, 16, 4, salt=MOMENT_SALT)
        level1 = _moments.level1_from_raw(arr, 4)
        vec.case(
            f"integers-have-no-nan/{dtype}",
            branching_factor=4,
            level=1,
            dtype=dtype,
            input=array_ref(arr),
            expected_moments=array_ref(level1),
            moment_columns=list(_moments.COLUMNS),
            all_counts_full=True,
            note=("an integer dtype has no NaN, so count is always the bucket's full "
                  "child count and the ragged tail is the only place it differs"),
        )
    return vec


def main() -> None:
    for factory in (gen_numeric, gen_tie_break, gen_nan_matrix, gen_bitfield,
                    gen_edges, gen_bucket_moments):
        vector = factory()
        path = vector.write()
        print(f"  {path.name}: {len(vector.cases)} cases")


if __name__ == "__main__":
    main()
