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
high 32 bits may be ignored; and, only while feature bit 32 is set, `metadata_offset` at 100 and
`metadata_length` at 108, each `u64` (**The metadata block**).

Version 1 defines two feature bits and reserves the rest:

| bits | | meaning |
|---|---|---|
| 0 | must-understand | **The file carries the moment stream.** Every numeric value block at level ≥ 1 has three streams; when the bit is clear, none does. |
| 1–31 | must-understand | Reserved. A reader rejects a file with any of them set. |
| 32 | may-ignore | **The file carries a metadata block.** The first 16 of the header's 28 reserved bytes hold its offset and length. A reader that does not implement the bit ignores it, those bytes and the block, and reads the file. |
| 33–63 | may-ignore | Reserved. A reader ignores them and reads the file. |

Bit 0 states the file's **policy** rather than what has been written so far, and it does not change
during the file's life. A writer sets it at the moment the header is written when every numeric
value block at level ≥ 1 the file will carry has the moment stream, and leaves it clear when none
will; on a file that has not yet produced such a block the bit says what its blocks will carry, and
is vacuously true until the first one exists. A reader that re-reads an active file must find it
unchanged. Tying the bit to the act of writing a stream instead would leave it clear to a follower
that opened a growing file before its first level-≥-1 block and set to that same follower on the
next read, and a must-understand header bit whose meaning changes during a file's life is worse than
one set slightly before the first stream it describes exists.

**A may-ignore bit may license reserved bytes.** Within version 1, a may-ignore bit may give meaning
to bytes this document otherwise reserves, and those bytes mean something only while the bit is
set: while it is clear they are reserved bytes like any other, written zero and never read. **No
writer sets a may-ignore bit it does not implement, or carries one over from a file it read, and no
writer writes the bytes such a bit licenses.** A writer that rewrites a file therefore clears every
may-ignore bit it does not implement and writes every reserved byte as zero, except those a bit it
does set licenses. A bit carried over by a writer that does not know it would vouch for bytes that
writer never wrote, and a reader that implements the bit would trust them.

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
channel entry, 4 in the block index entry — is **written zero and not checked on read**, except
where a may-ignore feature bit the writer sets licenses it. A reader must not reject a non-zero
reserved field, because a may-ignore bit may license reserved bytes within version 1, and a reader
that does not implement that bit must still read the file. Bit 32 licenses the first 16 of the
header's 28 (**The metadata block**); the other 12, and every other reserved field, stay reserved.
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

A numeric bucket is `[min, max, first, mid, last]` **in that column order**. `min` and `max` are the
bucket's extremes, and they serve every range statistic and search. `first` and `last` are the
bucket's first and last samples, which give continuity at the joins between drawn buckets. `mid` is a
sample at the bucket's centre, and because its position is fixed by the bucket's geometry rather than
by the data, every channel of a group names the same sample there — which is what lets two channels
be paired against each other. Each of the five is a sample as stored: it may itself be NaN, and it
keeps the bits it was written with.

⛔ **Every column of a stored row folds from the row below it.** A level can therefore be rebuilt,
extended or repaired from the level below without reading a raw sample: `min` and `max` fold from
their children's `min` and `max` columns by the rules that follow, `first` is the first child's
`first`, `last` is the last child's `last`, the moments merge under the order **Bucket statistics**
fixes, and `mid` is the first field of child ⌊k/2⌋ of the bucket's k existing children. A column with
no such rule could be recovered only by reading the raw samples again, which is the one thing the
pyramid exists to avoid, so no such column is stored.

⛔ **`mid` is the first field of child ⌊k/2⌋ of the bucket's k existing children**, and never "the
sample at the bucket's middle index". At level 1 a child is one raw sample, so `mid` is raw sample
⌊n/2⌋ of the n samples the bucket holds. Where the bucket is full and `branching_factor` is even it
is exactly the bucket's middle raw sample, at index `BF^L / 2` from the bucket's start. The two part
company on a ragged bucket above level 1 — three children holding 256, 256 and 100 samples hold 612
samples, whose middle index is 306, while the rule gives child 1's `first`, index 256 — and an odd
branching factor above level 1 parts from it the same way, naming the child just below the centre.
Where they differ the child rule is the rule, because it is what makes the column fold from the row
below. Two channels of one group whose lengths differ may hold a different k in their last buckets
and so name different samples there; a reader pairs them over the span both cover, as above.

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
occurrence wins — and that choice is what decides the stored `min_ts` and `max_ts`, which is why the
tie-break vector checks positions and not only values. `first`, `mid` and `last` are named by the
geometry instead, so no comparison decides them. A bitfield bucket is `[OR, AND]` on the
two's-complement bit pattern including the sign bit, and is defined only for integer dtypes
(`set3-build-level-*`). It carries no `first`, `mid` or `last`: a mask has no shape to draw and no
join to make continuous, so its two folded columns are the whole of it.

The fold is defined for a non-empty input and a `branching_factor` of **two or more** — a caller
asking for one bucket per element is asking for its input back, not for a fold — and positions are
defined in numeric mode only, which is why a bitfield block carries no position stream. A rejection
is named by its **class**, stated in these terms; the exception type an implementation raises is its
own language's business and no part of the format.

How a reader turns stored points into drawn ones — which of a bucket's points it emits, whether it
drops one that coincides with another, what it does with a series no longer than its bucket count or
with no valid sample, and how it combines stored buckets when it draws fewer than it read — is the
reader's own, because a read-side rule is the format's only when it computes a quantity the format
names, and choosing among stored values names none. The moment fold's range order, under **Bucket
statistics**, is one of the two read-side rules stated here — the other is the window rule under
**Time** — because a merged moment is such a quantity.

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
| fixed | ≥ 1 | numeric | positions `(N,2)` i64 `[min_ts, max_ts]`, values `(N,5)` `[min, max, first, mid, last]`, moments `(N,5)`† |
| fixed | ≥ 1 | bitfield | values `(N,2)` `[OR, AND]` |
| variable | 0 | either | timestamps `(N,)` i64, values `(N,)` |
| variable | ≥ 1 | numeric | timestamps `(N,5)` i64 `[min_ts, max_ts, first_ts, mid_ts, last_ts]`, values `(N,5)` `[min, max, first, mid, last]`, moments `(N,5)`† |
| variable | ≥ 1 | bitfield | timestamps `(N,)` i64 `[first]`, values `(N,2)` `[OR, AND]` |

† **Whether the moment stream is present is `features` bit 0, and nothing else.** It is a property
of the file, not of a block: either every numeric value block at level ≥ 1 carries the stream or
none does. `v1_no_moment_stream.tslod` is a file written without it and with the bit clear.

A numeric bucket stores two timestamps at fixed rate and five at variable rate, and a column that
appears at both rates sits in the same place in both. At fixed rate `first`, `mid` and `last` need no
stamp: the geometry gives their raw indices and **Time** gives the time of an index. At variable rate
every stamp is stored, each the stamp of the sample its own column names, looked up and never
computed. The one timestamp a variable-rate bitfield bucket carries is the time of its first sample,
which places the bucket when it is drawn; at fixed rate the geometry places it.

⚠ **Two streams of the same decoded size are told apart by their order alone.** On an 8-byte dtype a
five-column value row decodes to exactly as many bytes as the five-column moment stream, and at
variable rate as the five-column timestamp stream too, so a block whose streams were written in the
wrong order is not caught by the size rule below. No reader rule changes: stream order is fixed by
block kind, as above, and whether a moment stream is there is `features` bit 0 and nothing else. What
pins the order is the conformance set, which records every stream's kind, shape and the SHA-256 of
its decoded bytes in block order — `v1_all_ten_dtypes.tslod`, whose `ch_float64` channel has its
values and moments at one size, and `v1_both_timing_modes.tslod`, whose variable-rate `var_num`
channel has all three at that size. A swapped pair fails both.

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
positions `N×2` i64, variable-rate timestamp tuples `N×5` i64, values `N×1` at level 0 and above it
`N×5` numeric and `N×2` bitfield, moments `N×5` f64. A stream that declares any other size makes the
file malformed (`decoded-size-mismatch`), and one that does not decode to the size it declares is not
a stream of its recipe (`stream-payload-undecodable`); **Codecs** says what each recipe declares, and
step 6 of the order says which is taken first.

For the values stream that size is also what the index entry's `uncompressed_size` states, so
`uncompressed_size` must agree with the shape as well as with the decoded bytes — three numbers that
have to be one number. At profile 0 the shape is knowable before anything is decoded and the whole
check is arithmetic; at profile 2 each stream declares its decoded size in its codec's own header,
and comparing that declaration — before a byte of the stream is decoded — is the only check a reader
has on a block's size.

Every payload is the array's bytes, little-endian, **row-major** — the five values of bucket 0, then
the five of bucket 1, never planar (`v1-block-framing`).

### The order a reader checks in

Two readers must give the same broken file the same rejection class, so the order is part of the
format:

1. **The header and the index entries — the whole file's, before any block is read.** The header's
   own fields; then every group entry; then every channel entry — its `dtype`, `aggregation_mode`,
   `group_id` and `num_levels`, then that its level table lies within the file
   (`level-table-out-of-bounds`), then its three text fields (`text-field-not-utf8`), then its `name`
   against every name already seen (`channel-name-duplicate`), then its scaling fields, then in a
   sealed file its depth (`num-levels-mismatch`); then every level entry and every block index
   entry, under the rules stated in **Rejection** — among them `uncompressed_size` against the shape
   the entry implies (`decoded-size-mismatch`), an entry whose block reaches past the end of the
   file (`block-extent-past-eof`), and the zero-valued fields. Nothing here reads a block. The
   channel table is walked **per channel and not per kind**: for each channel entry in turn — the
   entry's own fields, then its level entries and their block index entries — before the next
   channel's. In a sealed file the depth is read from level 0's entry and its block index, so those
   two entries' own rules (`block-count-exceeds-allocated`, `block-index-out-of-bounds`) are checked
   for level 0 before the depth is compared; the remaining levels follow. For a
   variable-rate channel's level 0 the index entries' `start_timestamp`s must also be
   **non-decreasing in block order**: each one is a stored stamp by the rule under **Time**, so a
   decrease among them is `timestamp-stream-decreasing`, and a reader raises it here — it locates
   blocks by searching those starts before anything is decoded, and an unsorted index answers it
   wrongly with no stream ever examined. Last, where feature bit 32 is set and the reader implements
   it, the metadata block, as one unit and in the order **The metadata block** fixes. A defect there
   refuses the block and not the file, and the read goes on to step 2.
2. **Each block's CRC**, over `[file_offset, file_offset + compressed_size)`, before any of that
   block's streams is parsed (`block-crc-mismatch`) — which is what **Integrity**'s *checked before
   decode* means for this order. Every step below runs on a block whose checksum has already
   verified, which is why a defect built into a block's data is one this list can reach at all.
3. The **leading** stream's length prefix — the positions or timestamps stream — whose position
   feature bit 0 cannot move.
4. At profile 0 only, feature bit 0 against the length arithmetic
   (`moment-stream-flag-mismatch`).
5. The remaining length prefixes, and the last stream's recipe byte.
6. Each stream in wire order, one finished before the next is begun: first the size it **declares**
   under **Codecs** against the size its shape implies, the values stream's being
   `uncompressed_size` (`decoded-size-mismatch`); then that declaration against what its payload
   could deliver under **Codecs** (`stream-payload-undecodable`); then its payload decoded, which
   must be exactly one payload of its recipe delivering exactly the size it declared
   (`stream-payload-undecodable`). A payload whose declaration cannot be read is
   `stream-payload-undecodable` at the first of the three.
7. For a level-0 block of a **variable-rate** channel, once its timestamps stream has passed step 6:
   first that the block's first stored stamp **equals** its index entry's `start_timestamp`
   (`block-start-timestamp-mismatch`), and then that its stamps are non-decreasing within the block
   and that its first is not below the last stamp of the block before it
   (`timestamp-stream-decreasing`). In that order, because an entry that lies about a stream is
   refused before the stream it lies about is judged.
8. The moment stream, where bit 0 says there is one.

Step 1 is the whole file's and precedes every block. Steps 2 to 8 are one block's, taken in block
order within a level, level order within a channel and channel order within the file, so a file
whose defects lie in different blocks is refused for the earlier block's. Steps 7 and 8 never both
apply to the same block: a level-0 block carries no moment stream, and a block that carries one is
above level 0.

`uncompressed_size` against the implied shape is a **step-1** check for every conforming reader. It
is arithmetic on the index entry — `sample_count` × columns × width — and needs no byte decoded, so
no reader has a reason to defer it: a disagreeing entry is refused at step 1, and an entry that also
reaches past the end of the file is refused for `decoded-size-mismatch` and not for
`block-extent-past-eof`. Step 6's comparison remains and raises the same class, where the size a
stream declares disagrees with an `uncompressed_size` the shape check passed. One rule, one class,
two sites, and the order says which of them speaks.

The comparison is **required and not merely an early exit**, which is why it cannot be left to a
reader's choice. A `sample_count` that disagrees with the block leaves `uncompressed_size` and the
decoded bytes equal to each other — both are wrong in the same way — so nothing but the shape
notices, and every stream in that block is then read at the wrong length.

The declaration is compared **before** the payload is decoded, and that is the bound: no conforming
reader decodes a stream past the size its block's shape implies, and none needs to in order to know
the class. A payload that declares the wrong size is `decoded-size-mismatch` whether or not its bytes
would decode; a payload that declares the right size and then fails — a damaged block, a count
delivered short or long, a byte after its frame or its termination — is `stream-payload-undecodable`
whether the reader stopped at the declared size or decoded to the end. A reader that decodes first
and compares after gives a frame that is both corrupt and mis-declared the other class, and does not
conform. A declaration is a number in the file, so it is **compared and never reserved on**: a
reader allocates against the bytes it has read — a zstd block yields at most 128 KiB and a pco chunk
at most 2²⁴ numbers — so no file makes an allocation larger than the bytes that arrived to justify
it.

Step 4 sits between the two prefix checks, and it has to. The cross-check needs only the leading
stream's prefix and the count of bytes remaining after it — pure length arithmetic that depends on
no later prefix being valid — while every check from step 5 onward is parameterised by the stream
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

A stream's payload is **exactly one** payload of its recipe, with nothing after it: one zstd frame,
or one pco standalone file ending at its termination byte. Each **declares its decoded size in its
own header**, before any of it is decoded — identity by its length; zstd by the frame's content
size, which for byte-transpose is the planar length and so the same number; pco by its header's size
hint times the dtype's width. pco treats that hint as advisory; **binding it is a writer rule this
format adds over pco's own**: a writer sets it to the stream's element count. The declaration is
what step 6 of the order compares.

**A declaration is bounded by the payload that carries it.** A zstd frame's blocks cost at least
four bytes each — a three-byte header and a byte of content — and each yields at most 128 KiB, so a
frame declares at most **32,768 bytes for every byte of its payload**, byte-transpose included; the
identity recipe declares its own length, so its bound is that length. A stream declaring more is
refused before it is decoded (`stream-payload-undecodable`), and that refusal cannot disagree with
decoding the payload to the end: 32,768 is the format's own maximum, so a payload that thin can
never deliver what it declared, and the reader that refuses early and the reader that decodes to the
end meet at one class. pco carries no ratio and needs none: its chunks state their own counts, at
most 2²⁴ numbers each, and a reader walks one standalone file chunk by chunk to its termination,
holding what the payload has delivered and stopping at the declared count — so a pco declaration is
not reserved on either.

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
— **checked before decode**. An index entry whose `compressed_size` is zero is rejected
(`zero-size-index-entry`), which is what stops an all-zero entry passing on the CRC of an empty
range (`v1-compressed-size-zero` in `v1-negative-vectors`).

## Time

For a fixed-rate group the rate is the exact rational value of the stored `float64 sample_rate`, and
the time of raw sample *i* is

```
start_timestamp + round_half_even(i × 10⁹ / rate)
```

evaluated as an exact rational — **one rule**. A writer stores every block's `start_timestamp` and
every position timestamp — `min_ts` and `max_ts` — by that rule; a reader treats stored values as
data and never recomputes them (`v1-time-axis`). It is also the rule a reader evaluates for the
`first`, `mid` and `last` of a fixed-rate bucket, whose times are not stored because the geometry
already names their raw indices.

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

For a variable-rate group every sample's stamp is stored rather than computed, and a channel's
stored stamps are **non-decreasing** — within a block and across every block boundary. **Equal
consecutive stamps are legal**: two samples may carry the same nanosecond, and no reader may assume
a strict increase anywhere. A decreasing pair makes the file malformed
(`timestamp-stream-decreasing`). The rule is stated as non-decreasing rather than increasing because
a recording whose source stamps two samples alike is a recording and not a fault, and a reader that
treats equality as impossible drops one of the two.

For a variable-rate channel, a level-0 block's index `start_timestamp` **is its first stored stamp**
— the entry says what the stream says, so a reader may locate blocks from the index alone and the
window rule below still tests the stamp itself. A block whose entry and stream disagree makes the
file malformed (`block-start-timestamp-mismatch`, `v1-negative-vectors`). It follows that a
channel's level-0 starts are non-decreasing in block order, each one being a stored stamp, and a
reader raises a decrease among them from the index before it reads a block. The rule binds **level 0
only**: a fixed-rate block's `start_timestamp` is given by the time rule above, and an upper level's
is not constrained here — its leading stream holds a stamp per column rather than one per sample.

⛔ **A request for the half-open window [t₀, t₁) selects every sample whose stored stamp `s`
satisfies `t₀ ≤ s < t₁`, wherever a block boundary falls.** The test is on the stamp and never on the
block that holds it: where a block's first stamp equals `t₀`, an equal stamp ending the block before
it is selected too, and both copies of a duplicated `t₀` are in the window. This is stated because
the shape that breaks it is the natural implementation — a reader that finds its first block by
comparing the block's `start_timestamp` against `t₀` and then scans forward drops every earlier copy
of `t₀`, and answers one sample short with nothing in the file to say so
(`v1-variable-window-straddle`).

## The metadata block

A file may carry what its writer knew when it wrote it, beyond the channel entry's own fields: per
channel a **description**, and per file the writer's **provenance** — who wrote the file and from
what. Both travel in one **metadata block**, behind may-ignore feature bit 32. The block holds
nothing concluded about a recording after it was written: no member relates one channel to
another, and none states a limit or a level.

### Where it is

While bit 32 is set, `metadata_offset` (`u64` at header offset 100) is the block's absolute offset
and `metadata_length` (`u64` at 108) its length in bytes. While the bit is clear, those 16 bytes are
reserved bytes: written zero, never read and never checked, whatever they hold — a reader that
finds them non-zero with the bit clear serves no block (`v1_metadata_bit_clear_pointer_set.tslod`).
The other 12 reserved bytes at 116 stay reserved either way.

⛔ **The block lies directly after the channel table and before the first data block**: its offset
is `channel_table_offset + 160 × num_channels`, and no data block begins before it ends. Every
writer places it there, so a file has one layout for one content, and a file written without a
block is laid out exactly as it was before bit 32 existed.

⛔ **The block is written once and never changed.** A writer that sets bit 32 writes the block's
bytes before the header that points at them, with the header's first write, and every later write
of the header — a flush, the seal — carries the bit, the offset and the length unchanged. A reader
that re-reads an active file therefore finds the block it found before, and never a pointer to bytes
not yet written. A block is never added to an existing file and never edited in one: a file gains
or changes a block only by being written again. A writer that rewrites a file copies the block byte
for byte when the file's set of channel names is unchanged; otherwise it drops the block, clears the
bit and zeroes the pointer. Rewriting one file of a chained recording changes its SHA-256, so its
successors, whose `prev_file_hash` names it, are written again in order.

A writer takes the block's content as values and serialises it itself, in the canonical form below.

### The canonical form

The block is UTF-8 JSON (RFC 8259) within I-JSON (RFC 7493), restricted to **one byte string per
content**, so that two writers given the same content write the same bytes and a file's SHA-256
does not depend on which of them ran. The form is a grammar over code points:

```
block   = value                          ; nothing before it and nothing after it
value   = object / array / string
object  = "{" [ member *( "," member ) ] "}"
member  = string ":" value
array   = "[" [ value *( "," value ) ] "]"
string  = %x22 *char %x22
char    = %x5C %x22                      ; \"
        / %x5C %x5C                      ; \\
        / any code point except %x22, %x5C and %x00-1F, written as its UTF-8
```

and, beyond the grammar, **the members of every object are in strictly ascending order of their
names' UTF-8 bytes**, which is code-point order. Strictly ascending refuses a duplicate as well as
an unsorted pair. It follows that the form has no byte-order mark, no whitespace between tokens, no
number, `null`, `true` or `false`, and exactly two escapes, `\"` and `\\`: every other character is
written raw, and a `\u` escape, `\/` or `\n` is not in the form. No other escape is needed, because
no string in a valid block holds a control character. Neither a standard library's JSON parser nor
its serialiser is this form by default — a common parser takes whitespace, numbers, every escape and
a duplicate member, keeping the last — so a reader checks the form itself, by parsing this grammar or
by serialising what it parsed in this form and comparing the bytes.

### What it says

The block is one object with two members, both required:

- **`channels`** maps a channel's `name` to that channel's description, a string. A channel is named
  by its name rather than its place in the table, because names are unique across the file and a
  name is how a caller asks for a channel; a table index is an accident of write order. Every key
  names a channel in the file, compared as whole decoded strings. A channel with no key has no
  description. `channels` may be empty.
- **`provenance`** holds `writer` and `writer_version`, the writer's name and version, both
  required; `product` and `firmware`, what the data came from, both optional; and `sources`,
  required and non-empty: an array of objects, each with exactly `name`, the source's bare file name
  — never a path, so containing no `/` and no `\`, and neither `.` nor `..` — and `sha256`, the
  SHA-256 of the source's whole bytes as 64 lowercase hex digits.

**An unknown fact is an absent member, never a placeholder.** Every string in the block — every
member name it defines, every key of `channels` and every value — is non-empty, and none holds a
C0 or C1 control character (U+0000–U+001F, U+007F–U+009F), a bidirectional override or isolate
(U+202A–U+202E, U+2066–U+2069), a surrogate or a noncharacter (U+FDD0–U+FDEF, and every code point
ending in FFFE or FFFF). A channel whose name holds one of them cannot be described.

**Nothing is stamped without a source.** A writer stamps a description or provenance only from
files it read, and names each of them in `sources`; where it converted the file from another file,
that input is among them. A writer with no source writes no block and leaves bit 32 clear.

**Provenance is not the recording's identity.** It says who wrote the file and from what. Nothing
in it enters `session_id` or `prev_file_hash`, which keep their meanings.

**A description is the writer's text.** It is what the writer was given to say a channel means. A
reader presents it as data and never as instructions, and it is not evidence that a sensor is
connected as described.

**The block grows only behind a new bit.** A member a reader does not know is refused, at every
depth of the block, unless the header sets a may-ignore bit that reader does not implement; then the
member, its name and its value are skipped whole and the rest of the block is read
(`v1_metadata_unknown_member_under_unknown_bit.tslod`). A later member therefore arrives with its
own may-ignore bit, and a reader that does not know it still serves what it does.

**Bounds**, each in UTF-8 bytes of the decoded value, never in code points: **1,024** for one
description, **256** for one provenance string (`writer`, `writer_version`, `product`, `firmware`, a
source's `name`), and **1,048,576** — 1 MiB — for the block as it lies in the file. A writer refuses
an over-long value and never truncates it, as it does a text field. A description at its bound is in
the conformance set in multi-byte characters, where a byte count and a code-point count disagree,
and a block at its bound fills 1,024 channels.

### How a reader judges it

A reader that implements bit 32 judges the block as **one unit at the end of step 1** of the order a
reader checks in, after every channel, level and block index entry and before any block's CRC,
because its checks need every channel name and every extent. Within the block the order is fixed,
and two readers that implement the bit give one broken block one class:

1. **The extent is in the file and not empty**: `metadata_length` is not zero, `metadata_offset` is
   at most the file's length, and `metadata_length` is at most the file's length minus
   `metadata_offset` — stated so that no sum is formed, because `offset + length` wraps in 64-bit
   arithmetic (`metadata-extent-out-of-bounds`).
2. **It overlaps nothing**: no byte of it lies in the header, the group or channel table, a level
   table, a block index or a data block (`metadata-extent-overlaps`).
3. **It is placed**: it begins where the channel table ends and no data block begins before it ends
   (`metadata-misplaced`).
4. **Its length is within the block bound** (`metadata-too-long`).
5. **Its bytes are strict UTF-8** — no byte outside a valid sequence, no overlong form, no encoded
   surrogate (`metadata-not-utf8`).
6. **It is the canonical form** (`metadata-not-canonical`).
7. **It is the block's grammar**: the two members, every required member present, every value of
   its type, no string empty or holding a forbidden code point, and no unknown member except under a
   may-ignore bit the reader does not implement (`metadata-grammar-violation`).
8. **Every key of `channels` names a channel in the file** (`metadata-key-names-no-channel`).
9. **Every source's `sha256` is 64 lowercase hex digits and its `name` a bare file name**
   (`metadata-provenance-invalid`).
10. **Every description and provenance string is within its bound** (`metadata-value-too-long`).

⛔ **A defect in the block refuses the block, not the file.** The reader reports the block's class,
serves the file whole with no description and no provenance, never serves part of a block, and goes
on to step 2: a file whose block is refused and whose blocks fail their CRC is still refused for the
CRC. A defect elsewhere in step 1 refuses the file before the block is judged, so the file's class
is reported and the block's is not. A reader that does not implement bit 32 ignores the bit, the
pointer and the block and reads the file, which is what may-ignore means — so no file it could open
before bit 32 was assigned stops opening. `v1-metadata-block-conformance-set` holds the files a
reader serves a block from, or none; `v1-metadata-block-negatives` holds a block per class, two cases
with two defects each that pin the order above, and two that pin the block's place in step 1. The
negatives bind readers that implement bit 32.

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
decides how many columns a bucket has and what they mean. Bitfield mode on a float dtype
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

Each of the three text fields holds up to the field's **full width** of UTF-8 — 64 bytes for `name`,
16 for `unit`, 32 for `calibration_id` — NUL-padded where the value is shorter. A value that fills
its field carries **no terminator**, and a reader takes the bytes before the first NUL or the field's
end, whichever comes first. A writer refuses a value longer than the field and **never truncates one
to fit**: the field's width is a byte count and a value's length is not, so a cut lands inside a
multi-byte sequence whenever the two disagree, and what it produces is exactly the invalid UTF-8
`text-field-not-utf8` rejects — a writer that accommodates an over-long name silently writes a file
no conforming reader will open.

Two channel entries carrying the same `name` — compared as the bytes each decodes to, before the
first NUL or the field's end — make the file malformed (`channel-name-duplicate`), across the whole
file and not merely within a group. A name is how a caller asks for a channel, so a file holding two
of one name has no answer to the question; a reader that resolves a name to the first entry matching
it returns a channel the caller did not ask for and reports nothing, which is the one failure worse
than a refusal.

**The level entry and the block index entry.** A `block_count` above `allocated`
(`block-count-exceeds-allocated`), which points at entries never filled in — the rule is an upper
bound and not equality, because an active file is exactly where the two differ legitimately. A block
index whose entries end past the end of the file (`block-index-out-of-bounds`), the last of the four
tables under the same bound as the other three. An `uncompressed_size`, `sample_count` or
`compressed_size` of zero (`zero-size-index-entry`): one rule, that an index entry describes a block
that exists, so one class. The `compressed_size` case is refused before the CRC is computed, which
is what stops an all-zero entry passing on the checksum of an empty range (`v1-compressed-size-zero`
in `v1-negative-vectors`).

Within step 1 these are taken in one order — for each level: its entry's `block_count` against
`allocated`; then the whole index's bound; then each index entry in order — the zero-valued fields,
`uncompressed_size` against the shape the entry implies, the block's extent — before that block's
CRC.

**The stream.** A payload that is not a payload of the recipe its byte names
(`stream-payload-undecodable`): a zstd payload that is not exactly one standard frame declaring its
content size, a pco payload that is not exactly one standalone file of the stream's dtype declaring
its size hint, either one with a byte after it, either one declaring more than its payload could
deliver (the bound under **Codecs**), or either one decoding to anything but the size it declares. One rule — the bytes are not a stream of this recipe — so one class, whichever decoder
refuses and however. It is taken after the declared size is compared with the shape (step 6), so a
payload that both declares the wrong size and would not decode is `decoded-size-mismatch`.

**The metadata block.** Where feature bit 32 is set, the ten classes under **The metadata block**,
in the order stated there. They refuse the block, never the file, and they bind readers that
implement the bit.

`total_samples` of zero is **not** a rejection. A zero-sample channel is legal and in the
conformance set, and `num_levels` is 1 for one, so a group whose channels are all empty is a
recording that captured nothing rather than a malformed file.
