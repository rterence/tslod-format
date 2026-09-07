"""Corpus set 2 — the pyramid's geometry and its depth law.

Two rules a reader must agree with byte for byte, both derived here from the
format's own arithmetic:

  * where a bucket sits — bucket j at level K covers raw samples
    [j*BF^K, (j+1)*BF^K), anchored at global sample index 0;
  * how deep the pyramid is — `num_levels`, which is stored in the channel
    table entry and counts level 0.

Both are stated as arithmetic and computed here. Nothing in this module runs,
imports or transcribes an implementation of the format.
"""

from __future__ import annotations

from _corpus import Vector, u64


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def gen_compute_num_levels() -> Vector:
    v = Vector(
        id="set2-compute-num-levels",
        set_="set2-anchored-fold",
        kind="fixture",
        asserts=(
            "num_levels counts level 0, so it is ceil(log N / log BF) + 1; it is 1 for "
            "a zero- or one-sample channel, and branching_factor < 2 is refused rather "
            "than looping."
        ),
        source=(
            "derived here from the depth law: fold by branching_factor until one "
            "bucket remains, counting level 0"
        ),
        contract=(
            "The pyramid's depth is stored in the file, and a reader that derives "
            "it differently builds a level table of the wrong length."
        ),
        requires=[],
        notes=(
            "num_levels counts level 0, so it is one MORE than the number of "
            "aggregation levels. A reader that counts only the aggregation levels "
            "builds a level table one entry short and reads the block index of the "
            "level above the one it wanted."
        ),
    )
    import math

    for bf in (2, 4, 16, 256, 1000):
        for n in (0, 1, 2, 3, bf - 1, bf, bf + 1, bf * bf - 1, bf * bf,
                  bf * bf + 1, 1000, 65_536, 10**6, 2_916_533):
            if n < 0:
                continue
            # Derived here from the depth law, not read off an implementation.
            if n <= 0:
                produced = 1
            else:
                produced, samples = 1, n
                while samples > 1:
                    buckets = math.ceil(samples / bf)
                    produced += 1
                    if buckets <= 1:
                        break
                    samples = buckets
            v.case(
                f"N={n},BF={bf}",
                total_samples=u64(n), branching_factor=u64(bf),
                expected_num_levels=u64(produced),
            )

    for bad_bf in (0, 1):
        v.case(f"branching-factor-{bad_bf}-rejected",
               total_samples=u64(1000), branching_factor=u64(bad_bf),
               expected_num_levels=None,
               rejection_class="branching-factor-below-two",
               message="the pyramid has no depth for a branching factor below two: each level would hold at least as many buckets as the one below it")
    return v


def gen_bucket_geometry() -> Vector:
    v = Vector(
        id="set2-anchored-bucket-geometry",
        set_="set2-anchored-fold",
        kind="fixture",
        asserts=(
            "Bucket j at level K covers raw samples [j*BF^K, (j+1)*BF^K), anchored at "
            "global sample index 0 — never at a block boundary and never at a time — so "
            "block b at level K starts at raw sample b * BF^(K+1)."
        ),
        source=(
            "the anchored dyadic geometry, stated as arithmetic"
        ),
        contract=(
            "Bucket boundaries are anchored to the data rather than to a block or a "
            "time, which is what lets two implementations agree on them without "
            "negotiating."
        ),
        requires=[],
        notes=(
            "A block holds `block_samples` buckets "
            "rather than exactly BF, so the block-start column below is stated for "
            "block_samples == BF (today's geometry) and separately for block_samples == "
            "2 * BF, which is the first value the parameter makes legal."
        ),
    )
    for bf in (2, 16, 256):
        for level in range(0, 4):
            span = bf ** level
            for j in (0, 1, 2, 7, 1000):
                v.case(
                    f"bucket/BF={bf},K={level},j={j}",
                    branching_factor=u64(bf), level=u64(level), bucket_index=u64(j),
                    raw_start_inclusive=u64(j * span),
                    raw_end_exclusive=u64((j + 1) * span),
                    raw_span=u64(span),
                )
            for b in (0, 1, 5):
                for block_samples_mult in (1, 2):
                    block_samples = bf * block_samples_mult
                    # A level-K block holds `block_samples` buckets of span BF^K.
                    v.case(
                        f"block/BF={bf},K={level},b={b},block_samples={block_samples}",
                        branching_factor=u64(bf), level=u64(level), block_index=u64(b),
                        block_samples=u64(block_samples),
                        buckets_per_block=u64(block_samples),
                        raw_start_of_block=u64(b * block_samples * span),
                        raw_span_of_block=u64(block_samples * span),
                        note=("block_samples == branching_factor is today's geometry "
                              "(the default); the multiple is what block_samples parameterises"),
                    )
    return v


def main() -> None:
    for factory in (gen_compute_num_levels, gen_bucket_geometry):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
