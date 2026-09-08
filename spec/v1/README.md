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

Bit 0 states the file's **policy** rather than what has been written so far, and it does not change
during the file's life. A writer sets it at the moment the header is written when every numeric
value block at level ≥ 1 the file will carry has the moment stream, and leaves it clear when none
will; on a file that has not yet produced such a block the bit says what its blocks will carry, and
is vacuously true until the first one exists. A reader that re-reads an active file must find it
unchanged. Tying the bit to the act of writing a stream instead would leave it clear to a follower
that opened a growing file before its first level-≥-1 block and set to that same follower on the
next read, and a must-understand header bit whose meaning changes during a file's life is worse than
one set slightly before the first stream it describes exists.

`file_state` also says how far the rest of the file may be trusted. In a **sealed** file every
stored count is final, and `num_levels` in particular is the depth the channel's own samples imply
(**The pyramid**). In an **active** file it is not: a writer mid-append may have flushed level-0
blocks whose level table has not been extended yet, so a reader derives an active channel's depth by
walking the level table and takes the stored field as authoritative only once the file is sealed.
That is why the depth rejection binds sealed files alone — an active file is where the two
legitimately differ, in the same way `block_count` and `allocated` do.

Group entry offsets 0, 8, 16 and 24 are **frozen**: a streaming writer patches `start_timestamp` and
`total_samples` in place after the entry is written, so a new field may only append at offset 25 or
later.

`total_samples` is the **group's** count and no channel's. The channels of one group may hold
different numbers of samples — a channel that stopped recording early is still a channel of that
group — and the field holds the largest of their counts, which is how far the group's time base
runs. A channel's own count is the sum of its level-0 block index entries' `sample_count`, as the
depth rule under **The pyramid** states, and it is that count, never `total_samples`, that says what
the channel holds and how deep its pyramid is. A reader serving two channels of one group therefore
reads each channel's buckets from its own level table and block index — never assuming the two hold
the same number of buckets at any level — and pairs them over the span both cover
(`v1_group_unequal_lengths.tslod`). Which level a reader chooses to serve is its own policy and is
not stated here. The zero-sample channel already in the conformance set is the extreme of this rule
and not an exception to it.

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

⛔ **The last bucket at a level may be partial, and it summarises only the samples that exist.**
Where a channel's level-0 sample count is not a multiple of `BF^k`, the final bucket's range as
stated above runs past the last sample: the bucket folds the samples inside it and no others, and
the time it covers ends at the channel's last sample rather than at raw index `(j+1)·BF^k − 1`. A
reader deriving a bucket's time span from the geometry must clip it there. Nothing in the block says
so where it matters most — a fixed-rate bitfield block at level ≥ 1 carries no timestamp stream at
all, so the geometry is the only thing a reader has, and an unclipped derivation reports a bucket
that ends after the recording did (`v1_bitfield_ragged_tail.tslod`, whose last level-1 bucket folds
76 of a 1,100-sample channel).

⛔ **A channel's depth is not a free field: it is the depth its own samples imply.** Let *N* be the
channel's level-0 sample count — the sum of its own level-0 block index entries' `sample_count`.
The depth is 1 when *N* is 0 or 1; otherwise it is the number of levels produced by replacing *N*
with the ceiling of *N* / `branching_factor` until a single bucket remains, counting level 0. Every
channel in a sealed file stores that number and no other, and a reader refuses one that does not
(`num-levels-mismatch`, under **Rejection**).

A numeric bucket is `[min, max, first, last]` **in that column order**. `first` and `last` are the
literal first and last elements and may themselves be NaN.

⛔ **`min` and `max` are resolved independently, one column each.** The bucket's `min` is the
minimum over its children's `min` column with NaN skipped; if that whole column is NaN, it is the
**first child's `min` as stored**, at position 0. The bucket's `max` is the maximum over the
children's `max` column, by the same rule and separately. **Neither column gates the other**: a
`min` column that is entirely NaN says nothing about the `max` column, and a fold that lets one
decide the other returns the first child's max where it should return the largest. At level 1 both
columns are the raw samples, so the two rules coincide there and diverge only above it.

**A NaN carried through the fold keeps the payload it was written with**, and is never replaced by a
canonical one. A wholly-NaN bucket's `min` and `max` are the first child's own bits. This is a
different rule from the moments', where an all-NaN bucket's four moments are the canonical quiet NaN
because nothing was carried through — there, the value is manufactured; here, it is copied.

The comparison is strict `<` and `>`, so on a tie the **first**
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

Those tolerances are relative to **the larger of `n × σᵏ` and the expected value's own magnitude**,
never to `n × σᵏ` alone. Both ends of that need it: a moment that is zero by cancellation has no
magnitude to be relative to, and an outlier-dominated moment runs the other way — `n × σᵏ` is
`M2²/n` while `M4` is carried by the single outlier, so the moment exceeds its natural scale by
about a factor of `n`, and bounding it against the smaller of the two would demand accuracy float64
does not have.

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
and the rejection class it must trigger (`v1-negative-vectors`). A whole-file rule cannot be a
patched field, so those cases state a length to truncate to instead.

The rules stated with the fields they belong to are rejections wherever they are stated. Beyond
them:

**The file as a whole.** Fewer than 128 bytes, so the header is incomplete
(`file-shorter-than-header`) — refused for its length and not for a magic it is too short to hold.
The first six bytes not `TSLOD\0` (`bad-magic`); the sixth is part of the marker. A block index
entry whose `file_offset + compressed_size` reaches past the end of the file
(`block-extent-past-eof`), checked **before** the CRC, because a checksum over a range that does not
exist is the read the rule prevents.

**The header.** `num_groups` or `num_channels` of zero (`num-groups-zero`, `num-channels-zero`): a
file with no group has no time base, and a file with no channel holds no data. A group or channel
table whose entries end past the end of the file (`group-table-out-of-bounds`,
`channel-table-out-of-bounds`) — both offsets are `u64` and nothing bounds them but the file's own
length. A `file_state` outside `{0, 1}` (`file-state-unknown`): a reader that treats any non-zero
value as active reads an unknown state as one it happens to know. A `branching_factor` below two
(`header-branching-factor-below-two`), which would make a level its own parent so the pyramid never
terminates — the class names the surface, because a *caller* passing the same value to the fold is a
separate rule with its own class.

**The group entry.** A `timing_mode` outside `{0, 1}` (`timing-mode-unknown`), which decides both
which streams a block carries and how its first stream is read. For a **fixed-rate** group, a
`sample_rate` that is not positive and finite (`sample-rate-not-positive-finite`) — zero divides,
negative runs time backwards, a NaN makes every derived timestamp compare equal to nothing including
itself, and an infinity collapses the group onto its start. The constraint is on fixed-rate groups
because that is where this field defines the time of a sample; a variable-rate group carries its
timestamps instead.

**The channel entry.** A `dtype` outside 0–9 (`dtype-unknown`), which leaves a reader without the
width of anything. An `aggregation_mode` outside `{0, 1}` (`aggregation-mode-unknown`), which
decides what a bucket's four columns mean. Bitfield mode on a float dtype
(`bitfield-on-float-dtype`), which is the file-level form of a rule set 3 already pins at the
kernel. A `group_id` past the group table (`group-id-out-of-range`). `num_levels` of zero
(`num-levels-zero`): it counts level 0, so 1 is the smallest a channel can have. In a **sealed**
file, a `num_levels` that is not the depth the law gives for that channel's own level-0 sample count
and the header's `branching_factor` (`num-levels-mismatch`, stated under **The pyramid**); an active
file's stored depth is not authoritative and this rule does not bind it. The input is the channel's
own count — the sum of its level-0 block index entries' `sample_count` — and never the group's
`total_samples`, which sits one record away under a plausible name and gives the wrong answer for a
channel that is empty in a group that is not: such a channel's depth is 1 whatever its group holds,
and the conformance set carries one, so a reader taking the group total refuses a valid file. The
check compares the channel entry against its own level-0 block index and reads nothing else, so it
falls in **step 1** of the reader's evaluation order — with the header and the index entries, before
any stream is read. A level table whose entries end past the end of the file
(`level-table-out-of-bounds`). `name`, `unit` or `calibration_id` that is not valid UTF-8
(`text-field-not-utf8`) — one rule for the three, so one class. A `scaling_type` outside `{0, 1}`
(`scaling-type-unknown`), and a `scaling_gain` or `scaling_offset` that is not finite
(`scaling-parameter-not-finite`), either of which turns every sample in the channel into a NaN or a
saturation.

**The level entry and the block index entry.** A `block_count` above `allocated`
(`block-count-exceeds-allocated`), which points at entries never filled in — the rule is an upper
bound and not equality, because an active file is exactly where the two differ legitimately. A block
index whose entries end past the end of the file (`block-index-out-of-bounds`), the last of the four
tables under the same bound as the other three. An `uncompressed_size` or `sample_count` of zero
(`zero-size-index-entry`): one rule, that an index entry describes a block that exists, so one
class. A `compressed_size` of zero is refused before the CRC is computed, which is what stops an
all-zero entry passing on the checksum of an empty range (`v1-block-crc`).

`total_samples` of zero is **not** a rejection. A zero-sample channel is legal and in the
conformance set, and `num_levels` is 1 for one, so a group whose channels are all empty is a
recording that captured nothing rather than a malformed file.
