# Corpus conventions

The corpus is **language-neutral**. Nothing in it may be readable only by Python, and nothing in it
may lose a value that a 64-bit implementation can hold. These rules are binding on every generator
in `tools/` and on anything that reads the data.

---

## 1. What a vector is

A vector is one JSON file under `vectors/<set>/`, plus zero or more sidecar `.bin` files for bulk
arrays. Every vector carries these keys:

| key | meaning |
|---|---|
| `id` | stable slug, unique across the corpus; the filename stem |
| `set` | the corpus set it belongs to |
| `kind` | `fixture`, `oracle`, or `negative` |
| `asserts` | **one sentence: what must be true.** A vector's name is its assertion |
| `contract` | **one sentence: why this is a contract between implementations**, rather than one implementation's private detail |
| `source` | in plain language, where the expected values came from |
| `requires` | capabilities an implementation needs to run it (see §5) |
| `cases` | the list of cases; each has `name`, inputs, and expected outputs |

`kind` is not decoration. It is the FIXTURE/ORACLE rule :

- **`fixture`** — the current output **is** the contract. Byte layouts, index sets, exact integer
  arithmetic, enum values. A port must reproduce it exactly.
- **`oracle`** — the current output is *one correct answer among many*. The expected value comes
  from an **independent** reference, never from the implementation under test. Numerical results
  with a tolerance.
- **`negative`** — a well-formed input with one field patched; the assertion is that the reader
  **rejects** it, and the rejection class is named.

**A rejection is a class, never a language's exception type.** Any case whose expected outcome is a
refusal carries `rejection_class`: a kebab-case name stated in the format's terms, corresponding to
a rule in `spec/v1`. It may also carry `message`, which is informative and compared by nothing — an
implementation refuses in its own words and with whatever error type its language has, and neither
is part of the format. A vector that named a Python exception would fail every conforming
implementation that is not Python, and would fail a Python one that chose a different subclass.
Flattening `oracle` into `fixture` is how a rewrite inherits its predecessor's bugs with full
coverage. It is the one thing this corpus exists to prevent.

There is no fourth kind. A rule that carries no data is a rule, and it belongs in `spec/v1`, not in
a vector with zero cases.

`asserts` and `contract` answer different questions and both are required. `asserts` says *what
must be true*; `contract` says *why that is a contract between implementations* rather than an
accident of one of them. A vector that cannot answer the second question is pinning an accident,
and the generator refuses to write it.

---

## 2. Integers

**Every integer that can exceed 2⁵³ is a JSON string in decimal.** That is: every `i64`, every
`u64`, every nanosecond timestamp, every tick count, every file offset, every sample count.

```json
{ "start_timestamp": "1700000000000000000", "sample_count": "9007199254740993" }
```

Small, bounded integers (a dtype code, a level index, a branching factor, an array length) are
plain JSON numbers. When in doubt, stringify: a reader that parses `"256"` and a reader that parses
`256` both work, but a value silently rounded through a double is unrecoverable.

Negative values keep their sign inside the string: `"-123456"`.

---

## 3. Floats

**Equality of floating-point fixtures is equality of the stored bit pattern, never numeric
equality**. A float is therefore written as its IEEE-754 bit pattern, big-endian hex,
with the width implied by the dtype:

```json
{ "dtype": "float64", "bits": "0x3FF0000000000000", "repr": "1.0" }
```

- `bits` is the assertion. `repr` is a courtesy for a human reading the file and is **never
  compared**.
- A NaN that comes from an input element keeps **that element's** bits. A NaN a kernel manufactures
  is the one canonical quiet NaN: `0x7FC00000` (float32), `0x7FF8000000000000` (float64). A fixture
  extracted on arm64 must pass on x86_64, which is exactly what this rule buys.
- `oracle` vectors are the exception: they carry `expected` plus an explicit `tolerance` object
  naming the metric (`abs`, `rel`, or `ulp`). An oracle with no stated tolerance is malformed.

---

## 4. Bulk arrays

Anything longer than a few dozen elements is a sidecar file, never inline JSON:

```json
{
  "values": {
    "file": "set3-build-level/bl_float32_bf256_values.bin",
    "dtype": "float32", "shape": [4096], "order": "C", "byte_order": "little"
  }
}
```

- Raw little-endian, C-contiguous, row-major, no header, no padding.
- `shape` is row-major. A `(N, 4)` tuple array stores the four values of bucket 0, then the four of
  bucket 1 — **never planar**.
- The `.bin` is hashed in the manifest, so a corrupted sidecar is a CI failure and not a silent
  pass.

---

## 5. `requires`

Each vector's `requires` is a flat list of capability tokens, saying what an implementation needs in
order to run it. It is **data for you to read**, not a protocol: nothing in this repository
negotiates over it, and `check_corpus.py` checks every vector regardless. Use it to decide which
vectors your implementation is ready for.

| token | meaning |
|---|---|
| `format:v1` | it parses version 1's records |
| `dtype:<name>` | the implementation handles that wire dtype |
| `recipe:<hex>` | it decodes that codec recipe (`0x00`, `0x01`, `0x02`, `0x03`) |
| `profile:<n>` | it reads v1 files with that header `compression_id` (`0` none, `2` recipe) |
| `timing:<mode>` | `fixed` or `variable` |
| `feature:positions` | its fold produces bucket-relative argmin/argmax positions |
| `feature:tick-timebase` | it implements the tick conversions |
| `feature:streaming` | it opens `file_state = active` files |
| `feature:crc` | it checks the per-block CRC |
| `feature:moments` | it reads or writes the `(count, mean, M2, M3, M4)` moment stream |
| `feature:read-rejection` | it parses a whole file and refuses a malformed one |

`check_corpus.py` skips for one reason only — an optional third-party codec is not installed — and
names the package. It does not skip on `requires`.

---

## 6. Drift

`MANIFEST.json` carries a SHA-256 for every vector file and every sidecar. The generators are in
`tools/`, and CI regenerates the corpus and byte-compares it against the committed tree. A generator
and its corpus cannot drift, because a drift is a failing build.

Regenerate with:

```
python3 tools/generate_all.py
```

Nothing outside this repository writes here, and nothing here needs another implementation of the
format installed. A corpus that could not be rebuilt without the thing it checks would not be
independent of it.
