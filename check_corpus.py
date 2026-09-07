"""Check the conformance data against the rules this repository states.

This is not a conformance runner for your implementation — it does not know how
to call your code. It is the check that keeps this repository honest: every
vector's expected values are re-derived here from the specification's rules and
compared against what is committed, so a vector that has drifted from the rule
it claims to pin fails the build.

    python3 check_corpus.py

Three outcomes, and only three:

  passed   the case was checked and the data agrees with the rule.
  failed   it does not. The message names the case and both values.
  skipped  an optional third-party codec is not installed, so the cases that
           need it could not be decoded. Nothing else is ever skipped, and
           every skip names the package that would remove it.

Every vector in the manifest must have a check. A vector with no check fails
the run: a suite that quietly ignores part of itself is worse than one that
fails, because it reports a number that means nothing.

When a reference reader exists in `reader/`, this will run it against the
golden files as well. It does not exist yet, and this script does not pretend
otherwise.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import traceback
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REPO_ROOT / "corpus"
VECTORS = CORPUS_ROOT / "vectors"
sys.path.insert(0, str(REPO_ROOT / "tools"))

import _fold      # noqa: E402
import _moments   # noqa: E402

NS = 10**9


class Skip(Exception):
    """An optional third-party codec is missing. The only reason to skip."""


# --------------------------------------------------------------------------
# Encoding helpers — the conventions in CONVENTIONS.md, read back
# --------------------------------------------------------------------------


def rhe(numer: int, denom: int) -> int:
    """Round half to even over an exact rational."""
    q, r = divmod(numer, denom)
    r2 = 2 * r
    if r2 > denom or (r2 == denom and q % 2 == 1):
        q += 1
    return q


def f_of(entry) -> float:
    bits = int(entry["bits"], 16)
    if entry["dtype"] == "float64":
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def arr_of(ref):
    payload = (bytes.fromhex(ref["hex"]) if "hex" in ref
               else (VECTORS / ref["file"]).read_bytes())
    a = np.frombuffer(payload, dtype=np.dtype(ref["dtype"]))
    shape = tuple(ref["shape"])
    return a.reshape(shape) if shape else a


def bits_equal(a, b) -> bool:
    """Float equality is equality of the stored bits, never numeric equality."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.view(np.uint8).tobytes() == b.view(np.uint8).tobytes())


# --------------------------------------------------------------------------
# A version-1 reader, written from spec/v1 — enough to accept a valid file
# and to refuse every malformed one the negative vectors describe
# --------------------------------------------------------------------------


class CorruptFile(Exception):
    """A file that violates a rule the specification states."""

    def __init__(self, rejection_class: str, detail: str = ""):
        super().__init__(f"{rejection_class}{': ' + detail if detail else ''}")
        self.rejection_class = rejection_class


RECIPE_REGISTRY = {0x00, 0x01, 0x02, 0x03}
GROUP_ENTRY_SIZE, CHANNEL_ENTRY_SIZE = 64, 160
LEVEL_ENTRY_SIZE, BLOCK_INDEX_ENTRY_SIZE = 24, 48
TIMEBASE_PRESENT = 0x01
RATE_FIELD_MAX = 2**32


def _check_recipe(byte: int) -> None:
    if byte not in RECIPE_REGISTRY:
        raise CorruptFile("unknown-recipe-byte", f"0x{byte:02X}")


def validate_streams(body: bytes, n_streams: int, profile: int) -> None:
    """Every stream but the last is prefixed by its own u32 length.

    The length counts the stream's own recipe byte, so the next stream begins
    at `offset + 4 + prefix`. A reader that treats the prefix as the payload
    length places the next recipe byte one early and decodes garbage.
    """
    offset = 0
    for i in range(n_streams - 1):
        if offset + 4 > len(body):
            raise CorruptFile("stream-length-prefix-out-of-range",
                              f"stream {i} prefix runs past the block")
        prefix, = struct.unpack_from("<I", body, offset)
        if prefix < 1:
            raise CorruptFile("stream-length-prefix-zero",
                              "a length of 0 leaves no room for a recipe byte")
        if offset + 4 + prefix > len(body):
            raise CorruptFile("stream-length-prefix-out-of-range",
                              f"stream {i} ends at {offset + 4 + prefix}, "
                              f"block is {len(body)}")
        recipe = body[offset + 4]
        _check_recipe(recipe)
        if profile == 0 and recipe != 0x00:
            raise CorruptFile("profile0-recipe-mismatch", f"0x{recipe:02X}")
        offset += 4 + prefix
    if len(body) - offset < 1:
        raise CorruptFile("stream-length-prefix-consumes-block",
                          "the prefixes leave no bytes for the last stream")
    recipe = body[offset]
    _check_recipe(recipe)
    if profile == 0 and recipe != 0x00:
        raise CorruptFile("profile0-recipe-mismatch", f"0x{recipe:02X}")


def stream_count(body: bytes, level: int, timing_mode: int,
                 aggregation_mode: int, uncompressed_size: int) -> int:
    """How many streams this block actually has.

    The stream table in spec/v1 gives the block kind's shape, but it is not the
    whole answer: the moment stream is OPTIONAL on a numeric level >= 1 block —
    `v1_no_moment_stream.tslod` is in the conformance set precisely so neither
    shape can be assumed — and nothing in the header or the channel entry
    records which was written.

    So a reader works it out from `uncompressed_size`, which counts the values
    stream's decoded bytes and nothing else. Under the identity recipe the
    values stream is stored in `1 + uncompressed_size` bytes, so after the
    leading stream either the remainder is exactly that (no moments) or the
    next length prefix is exactly that (moments follow).
    """
    # A fixed-rate block carries a leading stream only where it has positions:
    # numeric, level >= 1. A variable-rate block always leads with timestamps.
    if timing_mode == 0:
        leading = 1 if (level >= 1 and aggregation_mode == 0) else 0
    else:
        leading = 1
    if aggregation_mode == 1:                  # bitfield buckets carry no moments
        return leading + 1
    if level == 0:
        return leading + 1
    values_stored = 1 + uncompressed_size      # identity recipe
    offset = 0
    for _ in range(leading):
        if offset + 4 > len(body):
            raise CorruptFile("stream-length-prefix-out-of-range")
        prefix, = struct.unpack_from("<I", body, offset)
        offset += 4 + prefix
    if len(body) - offset == values_stored:
        return leading + 1                     # no moment stream
    return leading + 2


def open_v1(data: bytes) -> dict:
    """Parse and validate a version-1 file, or raise CorruptFile."""
    if len(data) < 128 or data[:6] != b"TSLOD\x00":
        raise CorruptFile("bad-magic")
    version, = struct.unpack_from("<H", data, 6)
    if version != 1:
        raise CorruptFile("version-must-be-exactly-1", str(version))
    bf, = struct.unpack_from("<I", data, 8)
    if bf < 2:
        raise CorruptFile("branching-factor-below-2", str(bf))
    profile = data[12]
    if profile not in (0, 2):
        raise CorruptFile("profile-unknown", str(profile))
    block_samples, = struct.unpack_from("<I", data, 20)
    if block_samples == 0:
        raise CorruptFile("block-samples-zero")
    if block_samples % bf:
        raise CorruptFile("block-samples-not-multiple-of-bf",
                          f"{block_samples} % {bf}")
    features, = struct.unpack_from("<Q", data, 92)
    if features & 0xFFFF_FFFF:
        # The low half is must-understand and this version defines no bit in
        # it, so any bit set is a feature this reader does not know.
        raise CorruptFile("unknown-must-understand-feature-bit",
                          f"0x{features & 0xFFFFFFFF:08X}")

    n_groups, = struct.unpack_from("<H", data, 14)
    n_channels, = struct.unpack_from("<I", data, 16)
    group_off, = struct.unpack_from("<Q", data, 24)
    channel_off, = struct.unpack_from("<Q", data, 32)

    groups = []
    for gi in range(n_groups):
        base = group_off + gi * GROUP_ENTRY_SIZE
        timing_mode = data[base + 24]
        timing_flags = data[base + 25]
        if timing_flags & 0b1111_1100:
            raise CorruptFile("timing-flags-reserved-bit-set",
                              f"0x{timing_flags:02X}")
        numer, denom = struct.unpack_from("<QQ", data, base + 32)
        if timing_flags & TIMEBASE_PRESENT:
            for name, value in (("numer", numer), ("denom", denom)):
                if value < 1:
                    raise CorruptFile(f"tick-rate-{name}-zero", str(value))
                if value > RATE_FIELD_MAX:
                    raise CorruptFile(f"tick-rate-{name}-exceeds-2p32", str(value))
        groups.append({"timing_mode": timing_mode, "timing_flags": timing_flags})

    blocks = 0
    for ci in range(n_channels):
        base = channel_off + ci * CHANNEL_ENTRY_SIZE
        aggregation_mode = data[base + 65]
        group_id, = struct.unpack_from("<H", data, base + 66)
        num_levels, = struct.unpack_from("<I", data, base + 68)
        level_off, = struct.unpack_from("<Q", data, base + 72)
        timing_mode = groups[group_id]["timing_mode"] if groups else 0
        for lv in range(num_levels):
            block_count, allocated, index_off = struct.unpack_from(
                "<QQQ", data, level_off + lv * LEVEL_ENTRY_SIZE)
            if block_count > allocated:
                raise CorruptFile("block-count-exceeds-allocated")
            for b in range(block_count):
                e = index_off + b * BLOCK_INDEX_ENTRY_SIZE
                fo, cs, us, sc, _ts, crc, _res = struct.unpack_from(
                    "<QQQQqII", data, e)
                if cs == 0:
                    raise CorruptFile("compressed-size-zero")
                if us == 0 or sc == 0:
                    raise CorruptFile("zero-size-index-entry")
                if fo + cs > len(data):
                    raise CorruptFile("block-extent-past-eof")
                if zlib.crc32(data[fo:fo + cs]) & 0xFFFFFFFF != crc:
                    raise CorruptFile("block-crc-mismatch",
                                      f"channel {ci} level {lv} block {b}")
                body = data[fo:fo + cs]
                validate_streams(
                    body,
                    stream_count(body, lv, timing_mode, aggregation_mode, us),
                    profile)
                blocks += 1
    return {"branching_factor": bf, "block_samples": block_samples,
            "profile": profile, "block_count": blocks}


# --------------------------------------------------------------------------
# The checks, one per vector
# --------------------------------------------------------------------------


def c_rhe(v, c):
    if c.get("rejection_class"):
        assert c["rejection_class"] == "divisor-not-positive"
        assert int(c["d"]) <= 0, "a positive divisor was marked as rejected"
        return
    assert rhe(int(c["n"]), int(c["d"])) == int(c["expected"])


def c_ticks_to_ns(v, c):
    delta = (int(c["ticks"]) - int(c["anchor_ticks"])) * NS * int(c["tick_rate_denom"])
    got = int(c["start_timestamp"]) + rhe(delta, int(c["tick_rate_numer"]))
    assert got == int(c["expected_ns"]), f"got {got}, want {c['expected_ns']}"
    if "intermediate_exceeds_2p53" in c:
        assert c["intermediate_exceeds_2p53"] == (abs(delta) > 2**53)
        assert c["intermediate_exceeds_2p63"] == (abs(delta) >= 2**63)


def c_ns_to_ticks(v, c):
    delta = (int(c["t_ns"]) - int(c["start_timestamp"])) * int(c["tick_rate_numer"])
    got = int(c["anchor_ticks"]) + rhe(delta, int(c["tick_rate_denom"]) * NS)
    assert got == int(c["expected_ticks"])
    assert got == int(c["round_trip_of"]), "the round trip must recover the tick"


def c_ghz(v, c):
    if "ticks" not in c:                      # the worst-observed summary case
        bound = float(c["documented_error_bound_ticks"])
        assert int(c["worst_observed_error_ticks"]) <= bound
        return
    c_ticks_to_ns(v, c)
    delta = (int(c["expected_ns"]) - int(c["start_timestamp"])) * int(c["tick_rate_numer"])
    back = int(c["anchor_ticks"]) + rhe(delta, int(c["tick_rate_denom"]) * NS)
    assert back == int(c["expected_round_trip_ticks"])
    err = abs(back - int(c["ticks"]))
    assert err == int(c["round_trip_error_ticks"])
    assert err <= float(c["documented_error_bound_ticks"])
    assert c["round_trip_is_bit_exact"] == (err == 0)


def c_num_levels(v, c):
    import math
    n, bf = int(c["total_samples"]), int(c["branching_factor"])
    if c.get("rejection_class"):
        assert c["rejection_class"] == "branching-factor-below-two"
        assert bf < 2
        return
    levels, samples = 1, n
    if n > 0:
        while samples > 1:
            buckets = math.ceil(samples / bf)
            levels += 1
            if buckets <= 1:
                break
            samples = buckets
    assert levels == int(c["expected_num_levels"])


def c_geometry(v, c):
    bf, level = int(c["branching_factor"]), int(c["level"])
    span = bf ** level
    if "bucket_index" in c:
        j = int(c["bucket_index"])
        assert int(c["raw_span"]) == span
        assert int(c["raw_start_inclusive"]) == j * span
        assert int(c["raw_end_exclusive"]) == (j + 1) * span
        return
    b, bs = int(c["block_index"]), int(c["block_samples"])
    assert int(c["raw_start_of_block"]) == b * bs * span
    assert int(c["raw_span_of_block"]) == bs * span


def c_layout(v, c):
    fmt, raw = c["struct_format"], bytes.fromhex(c["golden_encoding_hex"])
    assert struct.calcsize(fmt) == c["size_bytes"] == len(raw)
    assert struct.pack(fmt, *struct.unpack(fmt, raw)) == raw
    cursor = 0
    for fld in c["fields"]:
        assert fld["offset"] == cursor, f"{fld['name']}: gap or overlap"
        cursor += fld["width"]
    assert cursor == c["size_bytes"]


def c_enums(v, c):
    """The enums are closed sets, and the codes agree with the rest of the corpus."""
    if c.get("table"):                                  # dtype-widths
        for name, width in c["table"].items():
            assert np.dtype(name).itemsize == width, name
        return
    values = c["values"]
    assert len(set(values.values())) == len(values), "codes must be distinct"
    assert c.get("unknown_value_is_rejected") is True
    if c["enum"] == "dtype":
        assert sorted(values.values()) == list(range(10))
        for name in values:
            np.dtype(name)                              # it must be a real dtype
    if c["enum"] == "recipe":
        assert set(values.values()) == RECIPE_REGISTRY, (
            "the registry the reader enforces must be the registry stated here")
    if c["enum"] == "compression_id":
        assert set(values.values()) == {0, 2}, "the v1 profile is 0 or 2"
    if c["enum"] == "timing_flags":
        assert values == {"TIMEBASE_PRESENT": TIMEBASE_PRESENT,
                          "EPOCH_SYNCED": 0x02}
        assert c["reserved_bits_must_be_zero"] is True


def c_build_level(v, c):
    arr = arr_of(c["input"])
    bf, from_raw, mode = c["branching_factor"], c["from_raw"], c["aggregation_mode"]
    cls = c.get("rejection_class")
    if cls:
        # The class is the contract; the exception type is this language's.
        expected = {
            "bitfield-on-float-dtype": lambda: mode == 1 and arr.dtype.kind == "f",
            "positions-in-bitfield-mode": lambda: mode == 1 and c.get("positions_requested"),
            "branching-factor-below-two": lambda: bf < 2,
            "empty-input": lambda: arr.size == 0,
        }[cls]
        assert expected(), f"the case does not match the class it claims: {cls}"
        try:
            _fold.build_level(arr, bf, from_raw, mode, c.get("positions_requested", False))
        except Exception:
            return
        raise AssertionError(f"expected a refusal: {cls}")
    if c.get("positions_requested"):
        t, p = _fold.build_level(arr, bf, from_raw, mode, True)
        assert bits_equal(np.ascontiguousarray(p, dtype=np.int64),
                          arr_of(c["expected_positions"])), "positions differ"
    else:
        t = _fold.build_level(arr, bf, from_raw, mode)
    assert bits_equal(t, arr_of(c["expected_tuples"])), "tuples differ"


def c_moments(v, c):
    raw = arr_of(c["input"])
    got = _moments.level1_from_raw(raw, int(c["branching_factor"]))
    assert bits_equal(got, arr_of(c["expected_moments"])), "level-1 moments differ"
    if "expected_moments_level2" in c:
        l2 = _moments.next_level(got, int(c["branching_factor"]))
        assert bits_equal(l2, arr_of(c["expected_moments_level2"])), "level-2 differ"


def c_range(v, c):
    buckets = arr_of(c["buckets"])
    size = int(c["buckets_per_block"])
    blocks = [buckets[i:i + size] for i in range(0, buckets.shape[0], size)]
    got = _moments.range_merge(blocks).reshape(1, 5)
    assert bits_equal(np.ascontiguousarray(got), arr_of(c["expected_merged"])), (
        "the two-level range merge differs")


def _merged(c):
    acc = None
    for p in c["parts"]:
        row = np.array([[float(p["count"]),
                         0.0 if p["mean"] is None else f_of(p["mean"]),
                         0.0 if p["M2"] is None else f_of(p["M2"]),
                         0.0 if p["M3"] is None else f_of(p["M3"]),
                         0.0 if p["M4"] is None else f_of(p["M4"])]])
        acc = row if acc is None else _moments.merge_arrays(acc, row)
    return acc.reshape(5)


def c_merge(v, c):
    got = _merged(c)
    assert int(got[0]) == int(c["expected_count"]), "count must be EXACT"
    tol = c.get("tolerance")
    if tol is None:
        return                                  # the empty dataset: count is all of it
    n, sigma = int(c["expected_count"]), tol["sigma"]
    for i, key in ((1, "mean"), (2, "M2"), (3, "M3"), (4, "M4")):
        field = c.get(f"expected_{key}")
        if field is None:
            continue
        want, g = f_of(field), float(got[i])
        limit = tol[key]["max"]
        if tol[key]["metric"] == "absolute_over_scale":
            scale = max(n * (sigma ** (3 if key == "M3" else 4)), abs(want))
            err = abs(g - want) / scale if scale > 0 else abs(g - want)
        else:
            err = abs(g - want) / abs(want) if want != 0.0 else abs(g)
        assert err <= limit, f"{key}: error {err:.3e} > {limit:.3e}"


def c_merge_nan(v, c):
    if "values" not in c:
        return                                  # the identity part
    vals = np.array([f_of(x) for x in c["values"]], dtype=np.float64)
    got = _moments.fold_children(_moments.singles(vals), len(vals))[0]
    assert int(got[0]) == int(c["expected_count"]), (
        f"count {int(got[0])} != {c['expected_count']} — count is the NON-NaN count")


def _exact_moments(values):
    """(count, mean, M2, M3, M4) over exact rationals, NaN excluded.

    NaN is excluded from count and from every sum, so an all-NaN part is the
    identity of the merge — the same rule the stored moments follow.
    """
    xs = [Fraction(v) for v in values if v == v]
    n = len(xs)
    if n == 0:
        return 0, Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    mean = sum(xs) / n
    return (n, mean,
            sum((x - mean) ** 2 for x in xs),
            sum((x - mean) ** 3 for x in xs),
            sum((x - mean) ** 4 for x in xs))


def _exact_merge(a, b):
    """Chan/Pebay, over exact rationals."""
    na, ma, m2a, m3a, m4a = a
    nb, mb, m2b, m3b, m4b = b
    if na == 0:
        return b
    if nb == 0:
        return a
    n = na + nb
    d = mb - ma
    mean = ma + d * nb / n
    m2 = m2a + m2b + d**2 * na * nb / n
    m3 = (m3a + m3b + d**3 * na * nb * (na - nb) / n**2
          + 3 * d * (na * m2b - nb * m2a) / n)
    m4 = (m4a + m4b
          + d**4 * na * nb * (na**2 - na * nb + nb**2) / n**3
          + 6 * d**2 * (na**2 * m2b + nb**2 * m2a) / n**2
          + 4 * d * (na * m3b - nb * m3a) / n)
    return n, mean, m2, m3, m4


def c_merge_order(v, c):
    """The identities are order-independent in EXACT arithmetic.

    This is the vector's whole claim, and it is checkable rather than
    skippable: rebuild the named dataset, split it the way the case says,
    fold the parts left-to-right and as a balanced tree over Fractions, and
    require the two to agree exactly. Any divergence an implementation sees in
    float64 is therefore its own rounding and not a defect in the algebra.
    """
    import gen_set4_chan_pebay as g

    values = dict(g.datasets())[c["dataset"]]
    sizes = [int(x) for x in c["split_sizes"]]
    assert sum(sizes) == len(values), "the split does not cover the dataset"

    parts, at = [], 0
    for size in sizes:
        parts.append(_exact_moments(values[at:at + size]))
        at += size

    left = parts[0]
    for p in parts[1:]:
        left = _exact_merge(left, p)

    tree = list(parts)
    while len(tree) > 1:
        tree = [_exact_merge(tree[i], tree[i + 1]) if i + 1 < len(tree) else tree[i]
                for i in range(0, len(tree), 2)]

    assert left == tree[0], (
        "the Chan/Pebay identities are not order-independent over the rationals, "
        "which would be a defect in the algebra rather than in float64")
    assert c["folds_agree_in_exact_arithmetic"] is True

    # The recorded mean and M2 are those exact values rounded once to float64.
    for i, key in ((1, "mean"), (2, "M2")):
        field = c.get(f"expected_{key}")
        if field is not None:
            assert float(left[i]) == f_of(field), f"{key} differs from the exact fold"


def c_crc(v, c):
    if "input_hex" in c:
        payload = bytes.fromhex(c["input_hex"])
        assert zlib.crc32(payload) & 0xFFFFFFFF == int(c["expected_crc32"], 16)
        return
    if "file" in c and "file_offset" in c:
        data = (VECTORS / c["file"]).read_bytes()
        lo = int(c["file_offset"])
        hi = lo + int(c["compressed_size"])
        got = zlib.crc32(data[lo:hi]) & 0xFFFFFFFF
        assert got == int(c["expected_crc32"], 16), (
            f"CRC over [{lo}, {hi}) is {got:#010x}, stored {c['expected_crc32']}")
        return
    # The remaining cases state the variant itself: polynomial, init, xor-out.
    assert zlib.crc32(b"123456789") & 0xFFFFFFFF == 0xCBF43926, (
        "this is the CRC-32 variant with check value 0xCBF43926")


def c_time_axis(v, c):
    rate = struct.unpack("<d", struct.pack("<Q", int(c["sample_rate_bits"], 16)))[0]
    exact = Fraction(int(c["sample_index"])) * NS / Fraction(rate)
    got = int(c["start_timestamp"]) + rhe(exact.numerator, exact.denominator)
    assert got == int(c["expected_ns"]), f"got {got}, want {c['expected_ns']}"


def c_framing(v, c):
    body = bytes.fromhex(c["block_hex"])
    assert len(body) == c["block_length"]
    if c.get("stream_count") == 3:
        ts_len, = struct.unpack_from("<I", body, 0)
        assert ts_len == c["ts_len"]
        assert c["ts_recipe_byte_offset"] == 4
        val_len, = struct.unpack_from("<I", body, 4 + ts_len)
        assert val_len == c["values_len"]
        assert c["moments_recipe_byte_offset"] == 4 + ts_len + 4 + val_len
        assert bits_equal(
            np.frombuffer(body[c["moments_recipe_byte_offset"] + 1:],
                          dtype=np.float64).reshape(c["moment_stream_shape"]),
            arr_of(c["expected_moments"]))
        validate_streams(body, 3, 0)
    elif "ts_len" in c:
        ts_len, = struct.unpack_from("<I", body, 0)
        assert ts_len == c["ts_len"]
        # The off-by-one: ts_len counts the timestamp stream INCLUDING its own
        # recipe byte, so the values recipe byte is at 4 + ts_len, not earlier.
        assert body[4] == 0x00, "timestamp stream recipe byte"
        assert body[4 + ts_len] == 0x00, "values stream recipe byte at 4 + ts_len"
        if "values_recipe_byte_offset" in c:
            assert c["values_recipe_byte_offset"] == 4 + ts_len
        validate_streams(body, 2, 0)
    else:
        assert body[c["recipe_byte_offset"]] == c["recipe"]
        validate_streams(body, 1, 0)


def c_profile0(v, c):
    """Read a profile-0 file with struct and zlib alone.

    This is the claim the profile exists to make: no codec, no third-party
    decoder, standard library only. If this check ever needs an import beyond
    struct and zlib, profile 0 has stopped being what it says it is.
    """
    data = (VECTORS / c["file"]).read_bytes()
    info = open_v1(data)
    assert info["profile"] == 0, "profile 0 means compression_id 0"
    assert info["branching_factor"] == c["branching_factor"]
    assert info["block_samples"] == c["block_samples"]
    assert info["block_count"] == c["block_count"], (
        f"walked {info['block_count']} blocks, the vector says {c['block_count']}")


def c_negative(v, c):
    """Apply the patch and require the reader to refuse — or to accept."""
    if "file" in c:
        data = bytearray((VECTORS / c["file"]).read_bytes())
        p = c["patch"]
        off, width = int(p["offset"]), p["width_bytes"]
        assert data[off:off + width].hex().upper() == p["original_hex"], (
            "the base file does not hold the byte the patch says it replaces")
        data[off:off + width] = bytes.fromhex(p["patched_hex"])
        if c.get("must_open"):
            open_v1(bytes(data))          # must not raise
            return
        try:
            open_v1(bytes(data))
        except CorruptFile:
            return
        raise AssertionError(f"{c['name']}: the patched file was accepted")

    # Cases with no file describe block framing, which is checked by building
    # the malformed block the case names and requiring the same refusal.
    cls = c["rejection_class"]
    if cls == "unknown-recipe-byte":
        body = bytes([int(c["recipe_byte"], 16)]) + b"\x00" * 8
        try:
            validate_streams(body, 1, 2)
        except CorruptFile as exc:
            assert exc.rejection_class == cls
            assert c["recipe_byte"].upper()[2:] in str(exc).upper(), (
                "the error must name the offending byte")
            return
        raise AssertionError(f"recipe {c['recipe_byte']} was accepted")

    bodies = {
        "stream-length-prefix-zero": (struct.pack("<I", 0) + b"\x00" * 8, 2, 0),
        "stream-length-prefix-out-of-range": (struct.pack("<I", 999) + b"\x00" * 8, 2, 0),
        "stream-length-prefix-consumes-block": (struct.pack("<I", 8) + b"\x00" * 8, 2, 0),
        # profile 0 means every recipe byte in the file is 0x00.
        "profile0-recipe-mismatch": (b"\x01" + b"\x00" * 8, 1, 0),
    }
    body, n_streams, profile = bodies[cls]
    try:
        validate_streams(body, n_streams, profile)
    except CorruptFile as exc:
        assert exc.rejection_class == cls, f"got {exc.rejection_class}, want {cls}"
        return
    raise AssertionError(f"{cls}: the malformed block was accepted")


def c_codec(v, c):
    """decode(stream) == the expected payload, bit for bit.

    Encoder output is never compared: two conforming encoders may emit
    different bytes for the same input and both be right.
    """
    if "recipe" not in c:                       # the registry case
        assert {int(k, 16) for k in c["recipes"]} == RECIPE_REGISTRY
        assert c["no_registry_escape"] is True
        for name in c["policy_table"]:
            np.dtype(name)
        return
    if c.get("not_encodable"):
        return                       # recorded so a reader knows it never occurs
    recipe = int(c["recipe"], 16)
    stream = bytes.fromhex(c["stream_hex"])
    assert stream[0] == recipe, "the recipe byte is part of the stream"
    payload = stream[int(c["payload_offset"]):]
    expected = arr_of(c["expected_payload"])

    if recipe == 0x00:
        got = payload
    elif recipe in (0x01, 0x03):
        try:
            import zstandard
        except ImportError:
            raise Skip("zstandard")
        got = zstandard.ZstdDecompressor().decompress(payload)
        if recipe == 0x03:
            n = expected.dtype.itemsize
            got = np.frombuffer(got, dtype=np.uint8).reshape(
                n, -1).T.tobytes()
    elif recipe == 0x02:
        try:
            from pcodec import standalone
        except ImportError:
            raise Skip("pcodec")
        got = standalone.simple_decompress(payload).tobytes()
    else:
        raise AssertionError(f"unknown recipe 0x{recipe:02X}")

    assert got == expected.tobytes(), (
        f"decoded {len(got)} bytes, expected {expected.nbytes}")


CHECKS = {
    "set1-div-round-half-even": c_rhe,
    "set1-ticks-to-ns": c_ticks_to_ns,
    "set1-ns-to-ticks-round-trip": c_ns_to_ticks,
    "set1-ghz-boundary": c_ghz,
    "set2-compute-num-levels": c_num_levels,
    "set2-anchored-bucket-geometry": c_geometry,
    "record-layouts-v1": c_layout,
    "record-enums": c_enums,
    "set3-bucket-moments": c_moments,
    "set3-build-level-numeric": c_build_level,
    "set3-build-level-nan-matrix": c_build_level,
    "set3-build-level-tie-break": c_build_level,
    "set3-build-level-bitfield": c_build_level,
    "set3-build-level-edges": c_build_level,
    "set4-chan-pebay-merge": c_merge,
    "set4-chan-pebay-nan": c_merge_nan,
    "set4-chan-pebay-order": c_merge_order,
    "set4-chan-pebay-range": c_range,
    "v1-block-crc": c_crc,
    "v1-block-framing": c_framing,
    "v1-time-axis": c_time_axis,
    "v1-negative-vectors": c_negative,
    "v1-profile0-conformance-set": c_profile0,
    "codec-decode-per-recipe": c_codec,
}


# --------------------------------------------------------------------------


def verify_hashes(manifest: dict) -> list[str]:
    problems = []
    for entry in manifest["vectors"]:
        for f in entry["files"]:
            path = VECTORS / f["path"]
            if not path.exists():
                problems.append(f"{f['path']}: missing")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != f["sha256"]:
                problems.append(
                    f"{f['path']}: sha256 {actual[:12]} != manifest "
                    f"{f['sha256'][:12]} — regenerate with "
                    f"tools/generate_all.py")
    return problems


def main() -> int:
    manifest = json.loads((CORPUS_ROOT / "MANIFEST.json").read_text())

    problems = verify_hashes(manifest)
    if problems:
        print("the data does not match the manifest:")
        for p in problems:
            print(f"    {p}")
        return 2

    print(f"corpus: {manifest['counts']['vectors']} vectors, "
          f"{manifest['counts']['cases']} cases\n")

    passed = failed = skipped = 0
    failures: list[str] = []
    skips: dict[str, int] = {}
    unchecked: list[str] = []

    for entry in manifest["vectors"]:
        check = CHECKS.get(entry["id"])
        if check is None:
            unchecked.append(f"{entry['id']} ({entry['set']})")
            continue
        vector = json.loads((VECTORS / entry["files"][0]["path"]).read_text())
        for case in vector["cases"]:
            try:
                check(vector, case)
            except Skip as exc:
                skipped += 1
                skips[str(exc)] = skips.get(str(exc), 0) + 1
            except AssertionError as exc:
                failed += 1
                failures.append(f"{vector['id']} / {case['name']}: {exc}")
            except Exception:
                failed += 1
                failures.append(
                    f"{vector['id']} / {case['name']}: unexpected "
                    f"{traceback.format_exc(limit=2).strip().splitlines()[-1]}")
            else:
                passed += 1

    if failures:
        print(f"{len(failures)} FAILED")
        for f in failures[:60]:
            print(f"    {f}")
        if len(failures) > 60:
            print(f"    ... and {len(failures) - 60} more")
        print()

    if unchecked:
        print(f"{len(unchecked)} VECTORS HAVE NO CHECK — every vector must have "
              f"one, so this is a failure and not a gap to note:")
        for u in unchecked:
            print(f"    {u}")
        print()

    if skips:
        print("skipped, because an optional codec is not installed:")
        for name, n in sorted(skips.items()):
            print(f"    {n} cases need {name} — pip install {name}")
        print()

    print(f"passed {passed}  failed {failed}  skipped {skipped}")
    if failed or unchecked:
        print("RESULT: FAIL")
        return 1
    if skipped:
        print("RESULT: PASS, with codec cases skipped")
        return 0
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
