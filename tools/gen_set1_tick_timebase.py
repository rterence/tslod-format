"""Corpus set 1 — the tick timebase.

Pins `ticks_to_ns` / `ns_to_ticks` exactness — one normative rounding rule, shared with the time axis.

**Where the expected values come from.** These are FIXTURES — exact integer
arithmetic under one rounding rule has one right answer, not a family of them
— and every expected value is computed here by a round-half-even oracle over
`fractions.Fraction`, derived from the rule as this repository states it. A
corpus that took its expected values from an implementation of the conversion
would prove only that the implementation agrees with itself.

Stdlib only: no numpy, no build, nothing imported from any implementation of
the format.
"""

from __future__ import annotations

from fractions import Fraction

from _corpus import Vector, big, i64

NS_PER_SECOND = 10**9

# Three representative rates: a 24 MHz MCU counter, a
# 32.768 kHz RTC, and a rate that is not an integer number of hertz.
RATES = [
    ("mcu-24mhz", 24_000_000, 1),
    ("rtc-32768hz", 32_768, 1),
    ("rational-1e9-over-3", 10**9, 3),
]

START_TS = 1_700_000_000_000_000_000  # a realistic wall-clock ns anchor
ANCHOR_TICKS = -123_456                # deliberately negative


# ---------------------------------------------------------------------------
# The independent oracle — Fraction semantics, not the production divmod path
# ---------------------------------------------------------------------------


def rhe_fraction(fr: Fraction) -> int:
    """Round-half-even of an exact rational, derived from Fraction alone."""
    q = fr.numerator // fr.denominator  # floor; negatives handled
    r = fr - q
    half = Fraction(1, 2)
    if r > half or (r == half and q % 2 == 1):
        q += 1
    return q


def oracle_ticks_to_ns(ticks, start_timestamp, numer, denom, anchor):
    exact = Fraction((ticks - anchor) * NS_PER_SECOND * denom, numer)
    return start_timestamp + rhe_fraction(exact)


def oracle_ns_to_ticks(t_ns, start_timestamp, numer, denom, anchor):
    exact = Fraction((t_ns - start_timestamp) * numer, denom * NS_PER_SECOND)
    return anchor + rhe_fraction(exact)


# ---------------------------------------------------------------------------
# A self-contained deterministic sequence
# ---------------------------------------------------------------------------


def _straddle(threshold: int, denom: int, inclusive: bool) -> list[int]:
    """The two deltas either side of where the intermediate crosses `threshold`.

    The intermediate is `delta * 10**9 * denom`, so the crossing sits at
    `threshold / (10**9 * denom)`. `inclusive` selects which side the boundary
    itself falls on, matching how each case reports it: 2**53 is reported as
    exceeded when the intermediate is strictly greater, 2**63 when it is
    greater or equal.
    """
    step = NS_PER_SECOND * denom
    below = (threshold if inclusive else threshold - 1) // step
    return [below, below + 1]


def _ties(numer: int, denom: int, limit: int = 2_000_000) -> list[int]:
    """The smallest delta giving an exact tie at each quotient parity.

    A tie is where round-half-even actually decides something: the two
    parities round in opposite directions from the same remainder. Not every
    rate produces one — at 24 MHz and at 10^9/3 no delta does — so this
    returns what exists rather than a fixed count.
    """
    out: dict[int, int] = {}
    for delta in range(1, limit):
        quotient, remainder = divmod(delta * NS_PER_SECOND * denom, numer)
        if 2 * remainder == numer and quotient % 2 not in out:
            out[quotient % 2] = delta
            if len(out) == 2:
                break
    return [out[k] for k in sorted(out)]


def tick_deltas(numer: int, denom: int) -> list[int]:
    """Every delta this conversion behaves differently at, and no others.

    The rule is round-half-even over an exact rational, so it does not depend
    on the particular value of a delta. It depends on the sign, on how wide
    the intermediate `(ticks - anchor) * 10**9 * denom` has grown, and on
    whether the division lands on a tie. So:

      * the structural deltas — zero, the first few, and the day and 30-day
        magnitudes a real capture reaches;
      * the pairs that straddle 2**53 and 2**63, which is where a float64 and
        then an int64 intermediate stop being able to hold the value;
      * an exact tie at each quotient parity, where the rounding rule decides;
      * the same on the negative side, because flooring before the tie test is
        not the same as truncating toward zero.
    """
    day = numer * 86_400 // denom
    structural = [0, 1, 2, 3, 4, 5, 7, 8, day - 1, day, day + 1,
                  7 * day, 30 * day, 30 * day + 7]
    interesting = (_straddle(2**53, denom, inclusive=True)
                   + _straddle(2**63, denom, inclusive=False)
                   + _ties(numer, denom))
    negatives = [-1, -2, -3, -4, -day, -30 * day] + [-d for d in interesting]

    seen, ordered = set(), []
    for d in structural + interesting + negatives:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def gen_div_round_half_even() -> Vector:
    v = Vector(
        id="set1-div-round-half-even",
        set_="set1-tick-timebase",
        kind="fixture",
        asserts=(
            "Integer division rounds half to even over the exact rational: "
            "ties go to the even quotient, and negative numerators floor before "
            "the tie test rather than truncating toward zero."
        ),
        source=(
            "authored: round-half-even over fractions.Fraction, derived from the "
            "rule rather than from any implementation of it"
        ),
        contract=(
            "One rounding rule, stated: three implementations rounding three ways "
            "put the same instant in three different places."
        ),
        requires=["feature:tick-timebase"],
    )
    cases = [
        (5, 2), (7, 2), (-5, 2), (-7, 2),      # exact ties, both parities, both signs
        (1, 2), (3, 2), (-1, 2), (-3, 2),
        (7, 3), (8, 3), (-7, 3), (-8, 3),      # non-ties
        (10, 5), (0, 7), (-10, 5),             # exact
        (2**53 + 1, 2), (2**53 + 3, 2),        # past the double's integer range
        (-(2**53) - 1, 2),
        (2**63 - 1, 3), (-(2**63), 3),         # past int64 on the numerator
        (10**30 + 5, 10),                      # unbounded width
    ]
    for n, d in cases:
        expected = rhe_fraction(Fraction(n, d))
        v.case(f"rhe({n}/{d})", n=big(n), d=big(d), expected=big(expected))

    # The divisor domain is part of the contract.
    v.case("divisor-zero-rejected", n=big(1), d=big(0), expected=None,
           rejection_class="divisor-not-positive",
           message="the divisor is positive or the quotient is undefined; a "
                   "zero or negative divisor is refused, never silently floored")
    v.case("divisor-negative-rejected", n=big(1), d=big(-2), expected=None,
           rejection_class="divisor-not-positive",
           message="the divisor is positive or the quotient is undefined")
    return v


def gen_ticks_to_ns() -> Vector:
    v = Vector(
        id="set1-ticks-to-ns",
        set_="set1-tick-timebase",
        kind="fixture",
        asserts=(
            "ticks_to_ns(t) == start_timestamp + rhe((t - anchor) * 10^9 * denom / numer) "
            "evaluated as an EXACT rational at every magnitude: exact where the "
            "intermediate fits in a double, and still exact past 2^53 and past 2^63, "
            "where a float64 and then an int64 intermediate stop being able to hold it. "
            "The cases straddle both crossings from each side and at each sign."
        ),
        source=(
            "authored: exact-rational round-half-even oracle, so the expected "
            "values do not come from any implementation of the conversion. The delta "
            "set is chosen rather than swept: it was 160 pseudo-random magnitudes per "
            "rate, which sampled one behaviour many times, and is now the boundaries "
            "that behaviour actually turns on"
        ),
        contract=(
            "Evaluating this conversion in a double is correct on small inputs and "
            "wrong on real captures, so exactness is pinned at the magnitudes where "
            "it stops holding."
        ),
        requires=["feature:tick-timebase"],
        notes=(
            "The intermediate `(ticks - anchor) * 10^9 * denom` is stated in the case as "
            "`intermediate_exceeds_2p53` / `intermediate_exceeds_2p63` so an implementation "
            "can see which cases its arithmetic width is being tested by."
        ),
    )
    for rate_name, numer, denom in RATES:
        for delta in tick_deltas(numer, denom):
            ticks = ANCHOR_TICKS + delta
            expected = oracle_ticks_to_ns(ticks, START_TS, numer, denom, ANCHOR_TICKS)
            intermediate = abs((ticks - ANCHOR_TICKS) * NS_PER_SECOND * denom)
            v.case(
                f"{rate_name}/delta={delta}",
                ticks=i64(ticks),
                start_timestamp=i64(START_TS),
                tick_rate_numer=big(numer),
                tick_rate_denom=big(denom),
                anchor_ticks=i64(ANCHOR_TICKS),
                expected_ns=i64(expected),
                intermediate_exceeds_2p53=intermediate > 2**53,
                intermediate_exceeds_2p63=intermediate >= 2**63,
            )
    return v


def gen_ns_to_ticks_round_trip() -> Vector:
    v = Vector(
        id="set1-ns-to-ticks-round-trip",
        set_="set1-tick-timebase",
        kind="fixture",
        asserts=(
            "Below 1 GHz the round trip ticks -> ns -> ticks recovers the original tick "
            "bit-exactly, and ns_to_ticks is itself rhe((ns - start) * numer / (denom * 10^9)) "
            "— including at the magnitudes where the intermediate passes 2^53 and 2^63, "
            "where an implementation that narrows either direction loses the tick it "
            "started from."
        ),
        source=(
            "authored: exact-rational round-half-even oracle. The delta set is chosen "
            "rather than swept, on the same boundaries as set1-ticks-to-ns"
        ),
        contract=(
            "A timestamp that does not survive a round trip cannot be used to "
            "address the sample it came from."
        ),
        requires=["feature:tick-timebase"],
    )
    for rate_name, numer, denom in RATES:
        assert Fraction(numer, denom) < NS_PER_SECOND, rate_name
        for delta in tick_deltas(numer, denom):
            ticks = ANCHOR_TICKS + delta
            ns = oracle_ticks_to_ns(ticks, START_TS, numer, denom, ANCHOR_TICKS)
            back = oracle_ns_to_ticks(ns, START_TS, numer, denom, ANCHOR_TICKS)
            assert back == ticks, f"oracle round trip broke at {rate_name} delta={delta}"
            v.case(
                f"{rate_name}/delta={delta}",
                t_ns=i64(ns),
                start_timestamp=i64(START_TS),
                tick_rate_numer=big(numer),
                tick_rate_denom=big(denom),
                anchor_ticks=i64(ANCHOR_TICKS),
                expected_ticks=i64(back),
                round_trip_of=i64(ticks),
            )
    return v


def gen_ghz_boundary() -> Vector:
    """The >= 1 GHz bound, pinned as the measured bound and not as exactness.

    `ticks_to_ns`'s docstring states the round trip is bit-exact below 1 GHz
    and that at or above it the error is bounded by (1 + rate/10^9)/2 ticks.
    That is a real edge of the contract, so it gets vectors: the forward map
    stays exact (it is still RHE of the exact rational), and the ROUND TRIP is
    what degrades. A reader that assumes exactness at every rate fails here.
    """
    v = Vector(
        id="set1-ghz-boundary",
        set_="set1-tick-timebase",
        kind="fixture",
        asserts=(
            "At and above 1 GHz the forward map stays exact RHE of the exact rational, "
            "but the ticks -> ns -> ticks round trip is no longer bit-exact; the recovery "
            "error is bounded by (1 + rate/10^9)/2 ticks and each case states the observed error."
        ),
        source=(
            "authored: exact-rational round-half-even; the recovery error is "
            "measured per case, not assumed"
        ),
        contract=(
            "An implementation that assumes exactness at every rate is wrong above "
            "1 GHz, so the bound is part of the contract rather than a surprise "
            "found later."
        ),
        requires=["feature:tick-timebase"],
    )
    boundary_rates = [
        ("exactly-1ghz", 10**9, 1),
        ("2ghz", 2 * 10**9, 1),
        ("3ghz", 3 * 10**9, 1),
        ("just-below-1ghz", 10**9 - 1, 1),
        ("max-domain-2p32", 2**32, 1),  # the validated ceiling
    ]
    for rate_name, numer, denom in boundary_rates:
        rate = Fraction(numer, denom)
        bound = Fraction(1) if rate < NS_PER_SECOND else (1 + rate / NS_PER_SECOND) / 2
        worst = 0
        for delta in [0, 1, 2, 3, 7, 12345, 10**6, 10**9, 10**12]:
            ticks = ANCHOR_TICKS + delta
            ns = oracle_ticks_to_ns(ticks, START_TS, numer, denom, ANCHOR_TICKS)
            back = oracle_ns_to_ticks(ns, START_TS, numer, denom, ANCHOR_TICKS)
            err = abs(back - ticks)
            worst = max(worst, err)
            assert err <= bound, (
                f"{rate_name} delta={delta}: recovery error {err} exceeds the "
                f"documented bound {float(bound)}"
            )
            v.case(
                f"{rate_name}/delta={delta}",
                ticks=i64(ticks),
                start_timestamp=i64(START_TS),
                tick_rate_numer=big(numer),
                tick_rate_denom=big(denom),
                anchor_ticks=i64(ANCHOR_TICKS),
                expected_ns=i64(ns),
                expected_round_trip_ticks=i64(back),
                round_trip_error_ticks=big(err),
                round_trip_is_bit_exact=(err == 0),
                documented_error_bound_ticks=str(float(bound)),
            )
        v.case(
            f"{rate_name}/worst-observed",
            tick_rate_numer=big(numer),
            tick_rate_denom=big(denom),
            worst_observed_error_ticks=big(worst),
            documented_error_bound_ticks=str(float(bound)),
            note="the bound is an upper bound on recovery error, not an equality",
        )
    return v


def main() -> None:
    for factory in (gen_div_round_half_even, gen_ticks_to_ns,
                    gen_ns_to_ticks_round_trip, gen_ghz_boundary):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
