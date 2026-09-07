# `.tslod` version 1

This document defines version 1 of the format. It is the source of truth: an implementation is
correct when it does what this says. The conformance data under `corpus/` illustrates these rules
and is named beside them, case by case, so a claim here can be checked against bytes rather than
taken on trust. Where this document and a vector disagree, the vector is wrong and is a bug.

Every multi-byte integer is little-endian, with one documented exception: `session_id` is the 16
RFC 4122 bytes in network order.

---

## Records

| record | size |
|---|---|
| header | 128 B |
| group table entry | 64 B |
| channel table entry | 160 B |
| level table entry | 24 B |
| block index entry | 48 B |

`record-layouts-v1` gives every field of all five — offset, width, wire type, name — together with a
*golden encoding*: a fully populated record with every field set to a distinguishable value, and
the exact bytes it must produce. A field table alone cannot catch a struct whose fields are all
correctly named and whose padding is wrong; the golden encoding can.

Header fields worth naming: `branching_factor` at 8, which is **two or more**; `compression_id` at 12, which is the file
**profile**; `file_state` at 13 (`0` sealed, `1` active — there is no per-block flag anywhere in the
format); `block_samples` at 20, the number of buckets a block holds, which must be non-zero and a
multiple of `branching_factor`; `features` at 92, whose low 32 bits are must-understand and whose
high 32 bits may be ignored.

Version 1 defines one feature bit and reserves the rest:

| bits | | meaning |
|---|---|---|
| 0 | must-understand | **The file carries the moment stream.** Every numeric value block at level ≥ 1 has three streams; when the bit is clear, none does. |
| 1–31 | must-understand | Reserved. A reader rejects a file with any of them set. |
| 32–63 | may-ignore | Reserved. A reader ignores them and reads the file. |

A writer sets bit 0 whenever it writes the moment stream, and never otherwise.

Group entry offsets 0, 8, 16 and 24 are **frozen**: a streaming writer patches `start_timestamp` and
`total_samples` in place after the entry is written, so a new field may only append at offset 25 or
later.

`prev_file_hash` at header offset 60 is the SHA-256 of the **entire** previous file in a chained
recording, or 32 zero bytes when there is no predecessor. `sequence_number` at 56 is that chain's
index.

Every field named `reserved` — 28 bytes at header offset 100, 6 and 8 in the group entry, 14 in the
channel entry, 4 in the block index entry — is **written zero and not checked on read**. A reader
must not reject a non-zero reserved field, because that is the space a later version writes into.
The one exception is the timing-flag byte, whose bits 2–7 are reserved and *are* rejected when set:
those bits change how the group is read, so a reader that ignores them reads it wrongly.

## The pyramid

Bucket *j* at level *k* covers raw samples `[j·BF^k, (j+1)·BF^k)`, **anchored at global sample index
0** — never at a block boundary and never at a time. Block *b* at level *k* therefore begins at raw
sample `b · block_samples · BF^k`. `num_levels` counts level 0, so it is one more than the count of
aggregation levels (`set2-anchored-bucket-geometry`, `set2-compute-num-levels`).

A numeric bucket is `[min, max, first, last]` **in that column order**. NaN is skipped in `min` and
`max`; an all-NaN bucket yields four NaNs; `first` and `last` are the literal first and last
elements and may themselves be NaN. The comparison is strict `<` and `>`, so on a tie the **first**
occurrence wins — and that choice is what decides the stored position timestamps, which is why the
tie-break vector checks positions and not only values. A bitfield bucket is `[OR, AND, first, last]`
on the two's-complement bit pattern including the sign bit, and is defined only for integer dtypes
(`set3-build-level-*`).

The fold is defined for a non-empty input and a `branching_factor` of **two or more** — a caller
asking for one bucket per element is asking for its input back, not for a fold — and positions are
defined in numeric mode only, which is why a bitfield block carries no position stream. A rejection
is named by its **class**, stated in these terms; the exception type an implementation raises is its
own language's business and no part of the format.

## Bucket statistics

Every numeric **value** bucket at level ≥ 1 also carries `(count, mean, M2, M3, M4)` as float64 —
none on timestamp, tick or position streams, none on bitfield buckets. `count` is the number of
**non-NaN** elements; the moments are over the non-NaN elements; an all-NaN bucket has `count = 0`
and the four moments as the canonical quiet NaN `0x7FF8000000000000`.

⛔ **The fold order is normative, at two levels.** Floating-point addition is not associative, so the
shape of the fold changes the answer's last bits.

- **Stored**: a level-*k* bucket's moments are the strict left-to-right Chan/Pébay fold over its
  `branching_factor` children in ascending index — raw samples at level 1, the level below's stored
  tuples above it.
- **A range query**: the strict left-to-right fold *within each block* in ascending bucket index,
  then the strict left-to-right fold of the per-block results in ascending block index.

⛔ **The arithmetic is normative too.** The order alone does not pin the bits. The merge is
evaluated using only `+`, `-`, `*` and `/` on float64 — the operations IEEE-754 requires to be
correctly rounded, and so the only ones that give the same answer on every conforming machine. `pow`
is not one of them and is not required to be correctly rounded, so an implementation must not reach
for `pow`, `powi`, `powf` or `cbrt` anywhere in the merge. Where an identity calls for a power it is
written as multiplication, in this association: with `d2 = delta * delta`, the cube of `delta` is
`d2 * delta` and its fourth power is `d2 * d2`; with `n2 = n * n`, the cubed count is `n2 * n`. The
association is stated because it is the association that fixes the last bits — two implementations
that agree on the order and differ here write different files.

A single-level rule is not enough, and this is measured rather than argued: a flat fold and a
per-block fold of the same 16 buckets differ in **8 of 15** cases (`set4-chan-pebay-range`). Six of
the seven that agree are the degenerate block sizes — one bucket per block, or every bucket in one
block — where the two folds are the same computation; of the nine cases where a block boundary
actually falls inside the range, eight differ and one agrees by luck. Two levels is what makes a
per-block parallel implementation and a serial one bit-identical.

Accuracy, so it is not assumed: the merge is exact in real arithmetic but not in float64, and its
error grows **linearly with `|mean|/σ`** — reaching `1.4 × 10⁻⁵` relative on `M2` for data offset to
`10¹²`. An offset channel is the normal case, not the pathological one (`set4-chan-pebay-merge`
carries the derived per-case tolerances).

## Blocks and streams

A block is one or more streams. **Every stream except the last is prefixed by its own byte length as
`u32` LE, counting its recipe byte; the last runs to `compressed_size`.** Stream order is fixed by
block kind: timestamps, then values, then moments.

| timing | level | aggregation | streams |
|---|---|---|---|
| fixed | 0 | either | values `(N,)` |
| fixed | ≥ 1 | numeric | positions `(N,2)` i64 `[min_ts, max_ts]`, values `(N,4)`, moments `(N,5)`† |
| fixed | ≥ 1 | bitfield | values `(N,4)` |
| variable | 0 | either | timestamps `(N,)` i64, values `(N,)` |
| variable | ≥ 1 | numeric | timestamps `(N,4)` `[first, last, min, max]`, values `(N,4)`, moments `(N,5)`† |
| variable | ≥ 1 | bitfield | timestamps `(N,)` `[first]`, values `(N,4)` |

† **Whether the moment stream is present is `features` bit 0, and nothing else.** It is a property
of the file, not of a block: either every numeric value block at level ≥ 1 carries the stream or
none does. `v1_no_moment_stream.tslod` is a file written without it and with the bit clear.

At profile 0 the framing is arithmetic, so a reader can check the bit rather than merely trust it.
`uncompressed_size` counts the **values** stream's decoded bytes and nothing else, so under the
identity recipe that stream occupies `1 + uncompressed_size` bytes: after the leading stream, either
the remaining bytes are exactly that — two streams — or the next length prefix is, and the moments
follow it. A block whose framing disagrees with the bit is rejected as
`moment-stream-flag-mismatch`, in either direction: a three-stream block in a file whose bit is
clear, or a two-stream block in a file where it is set.

⛔ **At profile 2 the bit is the only source, and a reader must not attempt to discover it.** The
values stream's encoded length is not known until it has been decoded, so the arithmetic above does
not exist; and the block's first byte does not answer it either. The three-stream block in
`v1_block_samples_256.tslod` begins `01 02 00 00`, where that `0x01` is the low byte of a length
prefix and not the zstd recipe byte it resembles. This is why the presence is a header bit and not
something a reader works out.

It follows that at profile 2 a file whose bit contradicts its blocks is malformed in a way **no rule
detects before the payload is decoded**. There is no arithmetic to catch it and no byte to read: the
reader takes the count the bit gives, splits the block accordingly, and hands the wrong bytes to a
decoder. This is stated rather than left to be discovered, and it is why there is no conformance
case for it — a case would pin one implementation's accidental failure mode as though it were the
format's.

⛔ **Every stream decodes to exactly the size its shape implies.** A block's `sample_count` is its
row count at its own level — raw samples at level 0, buckets above it — and the table above gives
each stream its columns. The size is therefore `sample_count × columns × the dtype's width`:
positions `N×2` i64, variable-rate timestamp tuples `N×4` i64, values `N×1` at level 0 and `N×4`
above it, moments `N×5` f64. A stream that decodes to any other length makes the file malformed
(`decoded-size-mismatch`).

For the values stream that size is also what the index entry's `uncompressed_size` states, so
`uncompressed_size` must agree with the shape as well as with the decoded bytes — three numbers that
have to be one number. At profile 0 the shape is knowable before anything is decoded and the whole
check is arithmetic; at profile 2 the decoded length is knowable only after decoding, and comparing
it is the only check a reader has on a block's size.

Every payload is the array's bytes, little-endian, **row-major** — the four values of bucket 0, then
the four of bucket 1, never planar (`v1-block-framing`).

### The order a reader checks in

Two readers must give the same broken file the same rejection class, so the order is part of the
format:

1. The header and the index entries, as above.
2. The **leading** stream's length prefix — the positions or timestamps stream — whose position
   feature bit 0 cannot move.
3. At profile 0 only, feature bit 0 against the length arithmetic
   (`moment-stream-flag-mismatch`).
4. The remaining length prefixes, and the last stream's recipe byte.
5. Decode each stream and compare its length with the size its shape implies, the values stream's
   being `uncompressed_size` (`decoded-size-mismatch`).
6. The moment stream, where bit 0 says there is one.

A reader that cross-checks `uncompressed_size` against the implied shape at step 1 refuses a
disagreeing index entry there; a reader that only compares after decoding refuses it at step 5. Both
give `decoded-size-mismatch`, because it is one rule and not two, and where `uncompressed_size` and
the decoded bytes disagree either reader catches it.

The shape comparison is nevertheless **required and not merely an early exit**. A `sample_count`
that disagrees with the block leaves `uncompressed_size` and the decoded bytes equal to each other —
both are wrong in the same way — so nothing but the shape notices, and every stream in that block is
then read at the wrong length.

Step 3 sits between the two prefix checks, and it has to. The cross-check needs only the leading
stream's prefix and the count of bytes remaining after it — pure length arithmetic that depends on
no later prefix being valid — while every check from step 4 onward is parameterised by the stream
count that bit 0 supplies. Run the full prefix walk first and a wrong bit derails the walk before
the cross-check is ever reached, so the file is refused for a damaged prefix or an unknown recipe
byte, whichever the values stream's first bytes happen to look like. The class would then depend on
the data rather than on the defect, and `moment-stream-flag-mismatch` would sit in this
specification while no file could produce it.

## Codecs

Each stream begins with a recipe byte. **No transform is implicit**: a timestamp payload under the
identity recipe is the absolute `i64` values, and if a delta ever earns its place it is a recipe a
reader can see, not a rule it must know.

`0x00` identity · `0x01` zstd (standard frame, content size on, checksum off, no dictionary) ·
`0x02` pco in its standalone format · `0x03` byte-transpose then zstd. `0xF0–0xFF` is experimental
and never appears in a released file; every other value is reserved and rejected with an error
naming the byte. There is no registry escape.

`compression_id` is the file profile: `0` = none, where **every** recipe byte must be `0x00`, and
`2` = recipe. Profile 0 is the interchange and conformance profile — it needs no codec at all, which
is what lets a standard-library reader be measured against it.

Conformance is `decode(vector) == expected payload`, **per recipe, never encoder byte-identity**:
two conforming encoders may emit different bytes for the same input and both be right
(`codec-decode-per-recipe`).

## Integrity

Every block carries a checksum — a CRC-32, the IEEE polynomial as zlib computes it (`0xEDB88320` reflected, init and
xor-out `0xFFFFFFFF`, check value `0xCBF43926` for `"123456789"`), over exactly
`[file_offset, file_offset + compressed_size)` — the index entry holding it lies outside that range
— **checked before decode**. An index entry whose `compressed_size` is zero is rejected, which is
what stops an all-zero entry passing on the CRC of an empty range (`v1-block-crc`).

## Time

For a fixed-rate group the rate is the exact rational value of the stored `float64 sample_rate`, and
the time of raw sample *i* is

```
start_timestamp + round_half_even(i × 10⁹ / rate)
```

evaluated as an exact rational — **one rule**. A writer stores every block's `start_timestamp` and
every position timestamp by that rule; a reader treats stored values as data and never recomputes
them (`v1-time-axis`).

A group may carry a tick timebase, converting with the same round-half-even convention over exact
integers:

```
ticks_to_ns(t) = start_timestamp + round_half_even((t − anchor_ticks) × 10⁹ × denom / numer)
ns_to_ticks(n) = anchor_ticks    + round_half_even((n − start_timestamp) × numer / (denom × 10⁹))
```

`tick_rate_numer` and `tick_rate_denom` are each **at least 1 and at most 2³²**, and a file outside
that range is rejected. The bound is what lets an implementation evaluate both conversions in
128-bit integers rather than arbitrary precision: the widest intermediate is
`|t − anchor_ticks| × 10⁹ × denom`, which for `|t − anchor_ticks| < 2⁶³` and `denom ≤ 2³²` stays
below `2¹²⁵`. Both ends are pinned, the accepting end included — `2³²` is the largest legal value,
not the first illegal one (`v1-negative-vectors`).

The round trip is bit-exact below 1 GHz; at and above it the recovery error is bounded by
`(1 + rate/10⁹)/2` ticks, and the vectors state the observed error per case rather than claiming
exactness (`set1-*`).

## Rejection

A version-1 reader reads **exactly** version 1 and refuses every other value. Every validity
constraint a writer enforces is enforced again at read time, because a reader's input is the file
and not the writer. The negative vectors are the enumeration: a well-formed file, one patched field,
and the rejection class it must trigger (`v1-negative-vectors`).
