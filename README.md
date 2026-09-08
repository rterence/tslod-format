# tslod-format

`.tslod` is a file format for long time-series — sensor and telemetry recordings with millions to
billions of samples per channel. This repository is the format's specification and the conformance
data an implementation checks itself against.

## What the format does

A `.tslod` file stores its samples and a pyramid of pre-computed summaries above them.

Level 0 is the raw samples. Each bucket at level 1 summarises `branching_factor` consecutive
samples; each bucket at level 2 summarises `branching_factor` level-1 buckets, and so on. With
`branching_factor` 64, one level-3 bucket covers 262,144 samples, so a range spanning a billion
samples is about 3,800 buckets at that level instead of a billion values. A reader that needs a
coarser or finer answer moves up or down a level rather than reading more raw samples.

Each numeric bucket stores nine numbers about the samples underneath it:

| field | meaning |
|---|---|
| `min`, `max` | the smallest and largest sample in the bucket — the true extremes of its span, not an approximation of them |
| `first`, `last` | the first and last sample in the bucket, in time order — the values at each end of its span |
| `count` | how many samples were valid; NaN is not counted |
| `mean`, `M2`, `M3`, `M4` | running moments, giving mean, variance, σ, RMS and kurtosis for any range at any level without reading the raw samples |

A file also stores a checksum for every block, so a reader detects damaged data before using it; one
exact rule for the timestamp of every sample; optional per-stream compression; and support for both
fixed-rate and irregularly-timestamped channels.

**[`spec/v1/`](spec/v1/README.md) defines the format.** It is the source of truth: an implementation
is correct when it does what the specification says.

---

## The conformance suite

`corpus/` holds 1,321 test cases across 25 vectors. Each case gives an input, the exact output
expected from it, and what that output demonstrates. They are stored as JSON and raw binary rather
than in any programming language, so implementations in Rust, C or Python are checked against
identical expectations.

A **vector** is one JSON file plus any bulk arrays it needs as `.bin` sidecars. Its name is its
assertion, and it carries one sentence saying what must be true and one saying why that is a
contract between implementations rather than one implementation's private detail.

| set | what it pins |
|---|---|
| `v1-format` | 18 golden `.tslod` files, the profile-0 and profile-2 conformance sets, block framing, the CRC, the time axis, and the read-time rejections |
| `record-layouts` | every field of all five records, with a golden encoding, and the wire enums |
| `set1-tick-timebase` | the tick conversions and the one rounding rule they share |
| `set2-anchored-fold` | where a bucket sits, and how deep the pyramid is |
| `set3-build-level` | what a bucket contains: column order, NaN, the tie-break, the bitfield sign bit |
| `set4-chan-pebay` | the moment merge, its fold order, and its accuracy |
| `codec-vectors` | what each recipe byte means, per recipe, decode only |

### An example

`set3-build-level-tie-break` checks how a bucket is summarised when its lowest value occurs twice.

**Input** — eight `int16` samples, `branching_factor` 4, so two buckets:

```
[1, 7, 1, 3]   [2, 8, 2, 9]
```

**Expected output** — one summary per bucket, as `[min, max, first, last]`, plus the index within
the bucket where the min and max were found:

```
bucket 0:  [1, 7, 1, 3]   min at index 0, max at index 1
bucket 1:  [2, 9, 2, 9]   min at index 0, max at index 3
```

The value `1` appears at index 0 and index 2 of bucket 0, and `2` appears twice in bucket 1. The
expected index is the **first** occurrence in both. Two implementations comparing with `<` and `<=`
produce identical summary values and different indices — and the index is what identifies the sample
the minimum came from, and so its timestamp. The case pins the index, not just the values.

It is `corpus/vectors/set3-build-level/set3-build-level-tie-break.json`.

### Two kinds of case

Each case says which kind it is:

- A **fixture** has exactly one right answer — a byte layout, a set of indices, integer arithmetic.
  Reproduce it exactly.
- An **oracle** is a floating-point result where several answers are defensibly correct. Its
  expected value is computed independently at high precision, never taken from an existing
  implementation, and it carries a stated tolerance.

A **negative** case is a well-formed file with one field patched, and the rejection it must produce.

### Encoding

Floating-point values are compared as bit patterns rather than numerically, so results are identical
across CPU architectures and a NaN keeps its exact payload. 64-bit integers are written as decimal
strings so none is rounded through a double. Full details in [`corpus/CONVENTIONS.md`](corpus/CONVENTIONS.md).

Each vector lists the capabilities it needs in `requires` — a dtype, a codec recipe, a profile, a
feature such as `feature:crc`. It is there for you to read when deciding which vectors your
implementation is ready for; nothing in this repository interprets it.

---

## Running the checker

```bash
python3 check_corpus.py
```

`check_corpus.py` verifies that every file matches its SHA-256 in `corpus/MANIFEST.json`, then
re-derives every case's expected values from the rules the specification states and compares them
against what is committed. It reports `passed`, `failed` and `skipped`.

It checks **this repository against itself** — that no vector has drifted from the rule it claims to
pin. It is not a conformance runner for your implementation: it does not know how to call your code.
Checking your reader means reading the vectors and running your own code against them, which is what
the language-neutral encoding is for. When a reference reader exists in `reader/`, this will run it
against the golden files too.

A case is **skipped** for one reason only: an optional third-party codec is not installed. Every skip
names the package.

```
corpus: 25 vectors, 1321 cases

skipped, because an optional codec is not installed:
    44 cases need pcodec — pip install pcodec
    106 cases need zstandard — pip install zstandard

passed 1171  failed 0  skipped 150
RESULT: PASS, with codec cases skipped
```

Every vector must have a check. A vector without one fails the run rather than being counted
separately, because a suite that quietly ignores part of itself reports a number that means nothing.

### Rebuilding the data

```bash
python3 tools/generate_all.py
```

Rebuilds every vector from the generators in `tools/`, then rebuilds `corpus/MANIFEST.json`. You do
not need this to use the suite: the data is committed, so an implementation in Rust or C runs the
suite with no Python toolchain, and a change to a generator arrives in review as the change it makes
to the expected values. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the generators
on every push and pull request and fails if the rebuilt tree differs from the commit by a single
byte, so the data cannot drift from the code that defines it. The same job then runs
`check_corpus.py` and requires that nothing failed and nothing was skipped.

Requires numpy, plus `zstandard` and `pcodec` to rebuild the compression cases. The exact
versions the committed data was generated with are pinned in
[`tools/requirements.txt`](tools/requirements.txt):

```bash
pip install -r tools/requirements.txt
```

---

## Layout

```
spec/v1/           what version 1 of the format is — the source of truth
corpus/
  CONVENTIONS.md   how cases are encoded
  MANIFEST.json    every vector and a SHA-256 of every file
  vectors/         the conformance data
check_corpus.py    checks the data against the specification's rules
tools/             the generators that produce corpus/vectors
reader/            a reference reader — not yet written; see its README
```

## Contributing

A change to the format starts with the specification: state the rule, then add or amend the case
that shows it, and regenerate. If an expected value cannot be computed independently of an
implementation, it is an oracle and must carry a tolerance.

## Status

Version 1 is not released and carries no stability guarantee. From this commit on, history on
`main` is never rewritten, so a consumer may pin any commit.

## Licence

This repository is licensed under the Apache License 2.0; see [`LICENSE`](LICENSE).
Copyright the tslod-format authors.
