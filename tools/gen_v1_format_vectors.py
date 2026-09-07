"""The version-1 format vectors, at `compression_id = 0` (profile 0).

Profile 0 is the interchange and conformance profile: every recipe byte must be
`0x00` and every payload is the array's little-endian bytes, so a reader needs
only struct unpacking and a CRC-32 to read one. That is what lets an
implementation be measured before it has a codec.

What is here:

  * the profile-0 conformance files — every enum value, both timing modes,
    both aggregation modes, all ten dtypes, both file states, two units,
    `scaling_type = linear`, a zero-sample channel, and `block_samples` both
    equal to and a multiple of the branching factor;
  * the block framing vectors — one-stream and two-stream, with `ts_len`
    counting the timestamp stream INCLUDING its recipe byte, so the values
    recipe byte sits at `4 + ts_len`;
  * the CRC vectors — IEEE CRC-32 as zlib computes it over exactly
    `[file_offset, file_offset + compressed_size)`, checked before decode;
  * the time-axis vectors — the exact-rational rule, never a float period
    accumulated per sample;
  * the v1 negative vectors — `block_samples`, the feature word, and the
    profile byte.

Every numeric bucket at level >= 1 may carry `(count, mean, M2, M3, M4)` as a
third stream. It is optional, and nothing in the file records whether it was
written, so both shapes are in the conformance set and neither can be assumed.
"""

from __future__ import annotations

import copy
import hashlib
import struct
import zlib
from fractions import Fraction

import numpy as np

import _tslod_build as B
from _corpus import VECTORS, Vector, array_ref, i64, u64

SET = "v1-format"
FILES = f"{SET}/files"

ALL_DTYPES = ["float32", "float64", "int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64"]


def ramp(dtype: str, n: int, seed: int = 0) -> np.ndarray:
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return np.ascontiguousarray((((np.arange(n) + seed) % 97) * 1.5 - 40).astype(dtype))
    info = np.iinfo(dt)
    arr = np.ascontiguousarray(((np.arange(n) + seed) % 97).astype(np.int64).astype(dtype))
    arr[0] = info.min
    arr[1] = info.max
    return arr


def _write_file(name: str, result: B.BuildResult) -> str:
    rel = f"{FILES}/{name}"
    path = VECTORS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.data)
    return rel


# ---------------------------------------------------------------------------
# The profile-0 conformance set
# ---------------------------------------------------------------------------


def gen_profile0_set() -> Vector:
    v = Vector(
        id="v1-profile0-conformance-set",
        set_=SET,
        kind="fixture",
        asserts=(
            "The profile-0 conformance files are readable with no codec at all: "
            "compression_id = 0 means every recipe byte in the file is 0x00 and every "
            "payload is the array's little-endian row-major bytes, so a reader needs "
            "only struct unpacking and zlib.crc32 to pass this set."
        ),
        source=(
            "written by tools/_tslod_build.py"
        ),
        contract=(
            "A profile that needs no codec at all is what lets an implementation be "
            "checked before it has one."
        ),
        requires=["format:v1", "profile:0", "recipe:0x00", "feature:crc"],
        notes=(
            "Profile 0 exists so the reference reader can be written against the SPEC "
            "with the standard library alone (the one exemption is "
            "pco and zstd for profile-2 files, and profile 0 needs neither). Every file "
            "below is decodable by hand."
        ),
    )
    ts = 1_700_000_000_000_000_000

    def emit(name, spec, checks, note, roll_up=False):
        """`roll_up` swaps per-stream detail for one hash over the whole file.

        A file carries per-block, per-stream records because its purpose is to
        vary SHAPE. Where a file exists to vary block COUNT instead, those
        records are the same few kinds repeated thousands of times, so it
        carries the rolled-up form the profile-2 set uses and nothing is lost:
        every kind, dtype and shape it contains appears in the other files, and
        its bytes stay pinned by the hash and by every block's CRC.
        """
        result = B.build(spec)
        rel = _write_file(name, result)
        # Every block's CRC must verify over exactly its stored extent.
        for (ci, level, bi, blk) in result.blocks:
            lo = blk["file_offset"]
            hi = lo + blk["compressed_size"]
            assert zlib.crc32(result.data[lo:hi]) & 0xFFFFFFFF == blk["crc32"], (
                f"{name}: CRC does not verify for channel {ci} level {level} block {bi}")
            # Check the recipe byte at its REAL offset, per stream, rather than
            # the block's first byte. The weaker check passed for months by
            # coincidence: a float32 payload of -40.0 begins 00 00 20 C2, so
            # its first byte is 0x00 whether or not a recipe byte precedes it,
            # and a whole file written with the wrong framing slipped through.
            off = 0
            if blk.get("ts_columns"):
                ts_len = struct.unpack_from("<I", result.data, lo)[0]
                assert result.data[lo + 4] == 0x00, (
                    f"{name}: timestamp stream's recipe byte is not identity")
                off = 4 + ts_len
            has_moments = (level >= 1
                           and spec.channels[ci].aggregation_mode == 0
                           and spec.moments)
            if has_moments:
                val_len = struct.unpack_from("<I", result.data, lo + off)[0]
                assert result.data[lo + off + 4] == 0x00, (
                    f"{name}: values stream's recipe byte is not identity")
                assert result.data[lo + off + 4 + val_len] == 0x00, (
                    f"{name}: moment stream's recipe byte is not identity")
            else:
                assert result.data[lo + off] == 0x00, (
                    f"{name}: values stream's recipe byte is not identity")
        features = struct.unpack_from("<Q", result.data, 92)[0]
        assert bool(features & B.FEATURE_MOMENT_STREAM) == bool(spec.moments), (
            f"{name}: feature bit 0 does not match what the builder wrote")
        body = dict(file=rel, file_size_bytes=u64(len(result.data)),
                    block_count=len(result.blocks),
                    format_version=1, compression_id=0,
                    branching_factor=spec.branching_factor,
                    block_samples=spec.block_samples or spec.branching_factor,
                    features=u64(features),
                    moment_stream_present=bool(features & B.FEATURE_MOMENT_STREAM),
                    every_crc_verifies=True, note=note)
        if roll_up:
            payloads = _walk_payloads(spec, result)
            rolled = hashlib.sha256()
            for _n, _d, payload in payloads:
                rolled.update(payload)
            body.update(decoded_stream_count=len(payloads),
                        decoded_bytes=u64(sum(len(p) for _n, _d, p in payloads)),
                        decoded_sha256=rolled.hexdigest())
        else:
            body["blocks"] = _block_records(spec, result)
        body.update(checks)      # a check may restate a field; the check wins
        v.case(name, **body)

    emit("v1_all_ten_dtypes.tslod",
         B.FileSpec(compression_id=0, branching_factor=256,
                    groups=[B.GroupSpec(1000.0, ts)],
                    channels=[B.ChannelSpec(f"ch_{d}", ramp(d, 1024, i))
                              for i, d in enumerate(ALL_DTYPES)]),
         {"dtypes": ALL_DTYPES},
         "all ten wire dtypes in one file; six of them appear in no .tslod that exists")

    emit("v1_both_timing_modes.tslod",
         B.FileSpec(compression_id=0, branching_factor=256,
                    groups=[
                        B.GroupSpec(1000.0, ts, timing_mode=0),
                        B.GroupSpec(1000.0, ts, timing_mode=1,
                                    timestamps=np.ascontiguousarray(
                                        ts + np.arange(1024, dtype=np.int64) * 999_983)),
                    ],
                    channels=[
                        B.ChannelSpec("fixed_num", ramp("float32", 1024), group_id=0),
                        B.ChannelSpec("fixed_bits", ramp("uint16", 1024), group_id=0,
                                      aggregation_mode=1),
                        B.ChannelSpec("var_num", ramp("float64", 1024), group_id=1),
                        B.ChannelSpec("var_bits", ramp("uint8", 1024), group_id=1,
                                      aggregation_mode=1),
                    ]),
         {"timing_modes": [0, 1], "aggregation_modes": [0, 1],
          "stream_shapes": {
              "fixed/L0": "one stream: values (N,)",
              "fixed/L>=1 numeric": "two: positions (N,2) i64 [min_ts,max_ts], then tuples (N,4)",
              "fixed/L>=1 bitfield": "one: tuples (N,4)",
              "variable/L0": "two: timestamps (N,) i64, then values (N,)",
              "variable/L>=1 numeric": "two: timestamps (N,4) [first,last,min,max], then tuples (N,4)",
              "variable/L>=1 bitfield": "two: timestamps (N,) [first], then tuples (N,4)"}},
         "all six rows of the stream table in one file — which streams a block has "
         "is fixed by its KIND, and this is the file that proves each shape exists")

    emit("v1_enums_and_units.tslod",
         B.FileSpec(compression_id=0, branching_factor=256,
                    groups=[B.GroupSpec(1000.0, ts, timing_flags=0x03,
                                        tick_rate_numer=24_000_000, tick_rate_denom=1,
                                        anchor_ticks=-123_456)],
                    channels=[
                        B.ChannelSpec("accel", ramp("float32", 1024), unit="m/s^2"),
                        B.ChannelSpec("volts", ramp("float32", 1024, 5), unit="V"),
                        B.ChannelSpec("temp", ramp("int16", 1024), unit="degC",
                                      scaling_type=1, scaling_gain=0.0625,
                                      scaling_offset=-273.15,
                                      calibration_id="cal-2026-09-06"),
                        B.ChannelSpec("empty", np.ascontiguousarray(
                            np.array([], dtype="float32")), unit="raw"),
                    ]),
         {"units": ["m/s^2", "V", "degC", "raw"], "scaling_types": [0, 1],
          "timing_flags": 3, "zero_sample_channel": "empty"},
         "two units, both scaling types, both timing flags, and a zero-sample channel — "
         "the list of what the conformance files must carry")

    emit("v1_active_file_state.tslod",
         B.FileSpec(compression_id=0, branching_factor=256, file_state=1,
                    groups=[B.GroupSpec(1000.0, ts)],
                    channels=[B.ChannelSpec("streaming", ramp("float32", 1024))]),
         {"file_state": 1, "file_state_name": "active"},
         "readable up to the last flushed block; num_levels derived by walking the "
         "level table; the stored field authoritative once sealed. The spec's "
         "'potentially corrupt' advisory is dropped")

    emit("v1_no_moment_stream.tslod",
         B.FileSpec(compression_id=0, branching_factor=256,
                    moments=False,
                    groups=[B.GroupSpec(1000.0, ts)],
                    channels=[B.ChannelSpec("sig", ramp("float32", 4096))]),
         {"has_moment_stream": False},
         "the same data with NO moment stream. A reader must handle both, because "
         "bitfield buckets never carry moments and a writer may omit them; the stream "
         "count is a function of the block KIND plus this presence, and both shapes are "
         "in the corpus so neither can be assumed")

    for bs_mult in (1, 2, 4):
        bs = 256 * bs_mult
        emit(f"v1_block_samples_{bs}.tslod",
             B.FileSpec(compression_id=0, branching_factor=256,
                        block_samples=bs,
                        groups=[B.GroupSpec(1000.0, ts)],
                        channels=[B.ChannelSpec("sig", ramp("float32", 8192))]),
             {"block_samples_multiple_of_bf": bs_mult},
             f"block_samples = {bs} = {bs_mult}x branching_factor. The bucket "
             f"geometry is UNCHANGED — only how many buckets share a block moves")

    for bf in (2, 16, 1024):
        # BF=2 is the deepest pyramid in the set: 4,096 blocks, whose per-stream
        # records would be the same three kinds repeated and nine tenths of the
        # vector's bytes. It varies block count, not shape, so it rolls up.
        emit(f"v1_branching_factor_{bf}.tslod",
             B.FileSpec(compression_id=0, branching_factor=bf,
                        groups=[B.GroupSpec(1000.0, ts)],
                        channels=[B.ChannelSpec("sig", ramp("float32", 4096))]),
             {"branching_factor": bf,
              "num_levels": B.compute_num_levels(4096, bf)},
             f"BF={bf}; no file that exists uses anything but 256",
             roll_up=(bf == 2))
    return v


# ---------------------------------------------------------------------------
# Block framing
# ---------------------------------------------------------------------------


def _split_streams(body: bytes, n: int) -> list:
    """The framing, read back: every stream but the last carries a u32 prefix."""
    out, off = [], 0
    for _ in range(n - 1):
        prefix, = struct.unpack_from("<I", body, off)
        out.append(body[off + 4:off + 4 + prefix])
        off += 4 + prefix
    out.append(body[off:])
    return out


def _decode_stream(stream: bytes, dtype: str) -> bytes:
    """`[recipe][payload]` -> the decoded payload bytes, per the recipe registry."""
    recipe, payload = stream[0], stream[1:]
    if recipe == 0x00:
        return payload
    if recipe == 0x01:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(payload)
    if recipe == 0x03:
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(payload)
        return B.byte_untranspose(raw, np.dtype(dtype).itemsize)
    if recipe == 0x02:
        from pcodec import standalone
        return np.ascontiguousarray(
            standalone.simple_decompress(payload)).tobytes()
    raise ValueError(f"recipe {recipe:#04x}")


def _stream_kinds(spec, ci: int, level: int, blk: dict) -> list:
    """(name, dtype, columns) per stream of this block, in wire order.

    The leading stream is POSITIONS on a fixed-rate numeric level >= 1 block —
    `[min_ts, max_ts]` per bucket — and TIMESTAMPS everywhere else it appears.
    `ts_columns` is what tells them apart, so the two names are derived from
    the data rather than asserted.
    """
    ch = spec.channels[ci]
    kinds = []
    tsc = blk.get("ts_columns") or 0
    if tsc:
        kinds.append(("positions" if tsc == 2 else "timestamps", "int64", tsc))
    kinds.append(("values", np.dtype(ch.data.dtype).name, 4 if level >= 1 else 1))
    if level >= 1 and ch.aggregation_mode == 0 and spec.moments:
        kinds.append(("moments", "float64", 5))
    return kinds


def _block_records(spec, result) -> list:
    """Per block, per stream: kind, dtype, shape, recipe and decoded SHA-256.

    This is what a reader test wants and what neither conformance set carried:
    the shape says how to interpret the bytes, and the hash says whether the
    bytes are right, per stream rather than per file.
    """
    out = []
    for (ci, level, bi, blk) in result.blocks:
        lo = blk["file_offset"]
        body = result.data[lo:lo + blk["compressed_size"]]
        kinds = _stream_kinds(spec, ci, level, blk)
        streams = []
        for stream, (kind, dtype, cols) in zip(_split_streams(body, len(kinds)),
                                               kinds):
            payload = _decode_stream(stream, dtype)
            item = np.dtype(dtype).itemsize
            assert len(payload) % (item * cols) == 0, (
                f"{kind} payload is not a whole number of {cols}-column rows")
            n = len(payload) // (item * cols)
            streams.append({
                "kind": kind, "dtype": dtype,
                "shape": [n] if cols == 1 else [n, cols],
                "recipe": f"0x{stream[0]:02X}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        out.append({"channel": ci, "level": level, "block": bi,
                    "streams": streams})
    return out


def _walk_payloads(spec, result) -> list:
    """Every block's decoded stream payloads, in file order."""
    out = []
    for (ci, level, bi, blk) in result.blocks:
        lo = blk["file_offset"]
        body = result.data[lo:lo + blk["compressed_size"]]
        kinds = _stream_kinds(spec, ci, level, blk)
        for stream, (name, dtype, _cols) in zip(_split_streams(body, len(kinds)),
                                                kinds):
            out.append((name, dtype, _decode_stream(stream, dtype)))
    return out


# ---------------------------------------------------------------------------
# The profile-2 conformance set
# ---------------------------------------------------------------------------


#: The profile-2 files, kept so a negative case can patch one. A rule that
#: exists FOR profile 2 has to have a case at profile 2.
_PROFILE2_BUILDS: dict = {}


def gen_profile2_set() -> Vector:
    v = Vector(
        id="v1-profile2-conformance-set",
        set_=SET,
        kind="fixture",
        asserts=(
            "A profile-2 file mixes recipes freely — each stream carries its own "
            "recipe byte and none is implicit — and decodes to exactly the payloads "
            "the identity-encoded file of the same data holds, byte for byte. The "
            "number of streams in each block is header feature bit 0 and nothing a "
            "reader computes."
        ),
        source="written by tools/_tslod_build.py",
        contract=(
            "At profile 2 the stream count is known only from feature bit 0, and a "
            "reader that discovers it any other way is wrong."
        ),
        requires=["format:v1", "profile:2", "recipe:0x01", "recipe:0x02",
                  "recipe:0x03", "feature:crc"],
        notes=(
            "Profile 0 cannot exercise any of this: its every recipe byte is 0x00, so "
            "nothing in that set decodes, and the length arithmetic that settles the "
            "stream count there does not exist here. Each file below is written with a "
            "DIFFERENT recipe on each of its streams, so a reader that applies one "
            "recipe to a whole block fails on the first file. `decoded_sha256` is over "
            "every block's decoded streams concatenated in file order; it is verified "
            "here against the identity-encoded file of the same data before the vector "
            "is written."
        ),
    )
    ts = 1_700_000_000_000_000_000

    def emit2(name, spec, recipes, note):
        spec.compression_id = 2
        spec.recipes = recipes
        result = B.build(spec)
        rel = _write_file(name, result)
        _PROFILE2_BUILDS[name] = (rel, result)

        for (ci, level, bi, blk) in result.blocks:
            lo = blk["file_offset"]
            hi = lo + blk["compressed_size"]
            assert zlib.crc32(result.data[lo:hi]) & 0xFFFFFFFF == blk["crc32"], (
                f"{name}: CRC does not verify for channel {ci} level {level} block {bi}")

        # The claim, checked before it is written: decoding this file gives
        # exactly what the identity-encoded file of the same data holds.
        twin_spec = copy.deepcopy(spec)
        twin_spec.compression_id = 0
        twin_spec.recipes = None
        twin_spec.recipe = B.RECIPE_IDENTITY
        twin = B.build(twin_spec)
        got = _walk_payloads(spec, result)
        want = _walk_payloads(twin_spec, twin)
        assert len(got) == len(want), f"{name}: stream count differs from the twin"
        for (gn, gd, gp), (wn, wd, wp) in zip(got, want):
            assert (gn, gd) == (wn, wd), f"{name}: stream kinds differ"
            assert gp == wp, f"{name}: {gn} stream does not decode to the twin's bytes"

        digest = hashlib.sha256()
        for _, _, payload in got:
            digest.update(payload)
        features = struct.unpack_from("<Q", result.data, 92)[0]
        assert bool(features & B.FEATURE_MOMENT_STREAM) == bool(spec.moments)

        v.case(name, file=rel, file_size_bytes=u64(len(result.data)),
               block_count=len(result.blocks),
               format_version=1, compression_id=2,
               branching_factor=spec.branching_factor,
               block_samples=spec.block_samples or spec.branching_factor,
               features=u64(features),
               moment_stream_present=bool(features & B.FEATURE_MOMENT_STREAM),
               recipes={k: f"0x{r:02X}" for k, r in recipes.items()},
               stream_count_per_numeric_level_block=(
                   3 if spec.moments else 2),
               decoded_stream_count=len(got),
               decoded_bytes=u64(sum(len(p) for _, _, p in got)),
               decoded_sha256=digest.hexdigest(),
               every_crc_verifies=True,
               blocks=_block_records(spec, result), note=note)

    emit2("v1_p2_with_moments.tslod",
          B.FileSpec(branching_factor=256,
                     groups=[B.GroupSpec(1000.0, ts)],
                     channels=[B.ChannelSpec("sig", ramp("float32", 4096)),
                               B.ChannelSpec("count", ramp("int32", 4096))]),
          {"timestamps": B.RECIPE_PCO,
           "values": B.RECIPE_TRANSPOSE_ZSTD,
           "moments": B.RECIPE_ZSTD},
          "three recipes in one file, one per stream kind: positions under pco, values "
          "under transpose+zstd, moments under zstd. Feature bit 0 is set, so every "
          "numeric level-1 block has three streams and a reader knows it before it "
          "decodes anything")

    emit2("v1_p2_no_moments.tslod",
          B.FileSpec(branching_factor=256, moments=False,
                     groups=[B.GroupSpec(1000.0, ts)],
                     channels=[B.ChannelSpec("sig", ramp("float32", 4096)),
                               B.ChannelSpec("count", ramp("int32", 4096))]),
          {"timestamps": B.RECIPE_ZSTD,
           "values": B.RECIPE_PCO},
          "the same shape with NO moment stream and bit 0 clear. This is the pair that "
          "matters at profile 2: the two files differ in stream count, and only the "
          "header bit says so — there is no length arithmetic here to fall back on")

    emit2("v1_p2_active_with_moments.tslod",
          B.FileSpec(branching_factor=256, file_state=1,
                     groups=[B.GroupSpec(1000.0, ts)],
                     channels=[B.ChannelSpec("streaming", ramp("int64", 2048))]),
          {"timestamps": B.RECIPE_TRANSPOSE_ZSTD,
           "values": B.RECIPE_PCO,
           "moments": B.RECIPE_TRANSPOSE_ZSTD},
          "an ACTIVE profile-2 file: file_state = 1 and moments present. A writer that "
          "is still appending compresses what it has already flushed, so the active "
          "case and the codec case meet here rather than in separate files")

    return v


def gen_block_framing() -> Vector:
    v = Vector(
        id="v1-block-framing",
        set_=SET,
        kind="fixture",
        asserts=(
            "A one-stream v1 block is [recipe u8][payload]; a two-stream block is "
            "[ts_len u32 LE][recipe u8][ts payload][recipe u8][values payload] where "
            "ts_len counts the timestamp stream INCLUDING its recipe byte, so the values "
            "recipe byte sits at offset 4 + ts_len."
        ),
        source=(
            "the bytes are produced by tools/_tslod_build.py"
        ),
        contract=(
            "A length that counts its own recipe byte and one that does not differ "
            "by one, and a reader that gets it wrong decodes garbage from the "
            "second stream onward."
        ),
        requires=["format:v1", "recipe:0x00"],
        notes=(
            "The off-by-one this vector exists to catch: a reader that treats ts_len as "
            "the payload length EXCLUDING the recipe byte places the second recipe byte "
            "one byte early and decodes garbage. the framing rule names it — 'two readers placing "
            "the second recipe byte one byte apart' — and this is the vector that fails "
            "for one of them."
        ),
    )
    ts = 1_700_000_000_000_000_000

    # one-stream: fixed-rate level 0
    values = np.ascontiguousarray(np.arange(8, dtype=np.float32) * 1.5)
    block = bytes([0x00]) + values.tobytes()
    v.case("one-stream/fixed-L0-values",
           kind="fixed-rate level 0",
           block_hex=block.hex().upper(),
           block_length=len(block),
           recipe_byte_offset=0, recipe=0,
           values_payload_offset=1,
           values_payload_length=len(values.tobytes()),
           expected_values=array_ref(values),
           uncompressed_size=u64(len(values.tobytes())),
           note="uncompressed_size in the index entry counts the VALUES stream's decoded "
                "bytes only, and the recipe byte is not part of it")

    # two-stream: variable-rate level 0
    stamps = np.ascontiguousarray(ts + np.arange(8, dtype=np.int64) * 1_000_003)
    ts_stream = bytes([0x00]) + stamps.tobytes()
    val_stream = bytes([0x00]) + values.tobytes()
    block2 = struct.pack("<I", len(ts_stream)) + ts_stream + val_stream
    v.case("two-stream/variable-L0",
           kind="variable-rate level 0",
           block_hex=block2.hex().upper(),
           block_length=len(block2),
           ts_len=len(ts_stream),
           ts_len_includes_its_recipe_byte=True,
           ts_recipe_byte_offset=4,
           values_recipe_byte_offset=4 + len(ts_stream),
           expected_timestamps=array_ref(stamps),
           expected_values=array_ref(values),
           timestamps_are_absolute_i64=True,
           note="⟢ ABSOLUTE i64, not deltas. No transform is implicit: if a delta "
                "ever earns its place it is a RECIPE the reader can see, not a rule it "
                "must know")

    # two-stream: fixed-rate level >= 1 numeric — positions then tuples
    positions = np.ascontiguousarray(
        np.column_stack([ts + np.arange(4, dtype=np.int64) * 256_000_000,
                         ts + np.arange(4, dtype=np.int64) * 256_000_000 + 1_000]))
    tuples = np.ascontiguousarray(
        np.arange(16, dtype=np.float32).reshape(4, 4))
    ts_stream = bytes([0x00]) + positions.tobytes()
    block3 = struct.pack("<I", len(ts_stream)) + ts_stream + bytes([0x00]) + tuples.tobytes()
    # three-stream: fixed-rate level >= 1 numeric, WITH the moment stream
    import _moments
    mom = _moments.level1_from_raw(
        np.ascontiguousarray(np.arange(16, dtype=np.float64)), 4)
    ts_stream3 = bytes([0x00]) + positions.tobytes()
    val_stream3 = bytes([0x00]) + tuples.tobytes()
    mom_stream3 = bytes([0x00]) + np.ascontiguousarray(mom).tobytes()
    block4 = (struct.pack("<I", len(ts_stream3)) + ts_stream3
              + struct.pack("<I", len(val_stream3)) + val_stream3
              + mom_stream3)
    v.case("three-stream/fixed-L1-numeric-with-moments",
           kind="fixed-rate level >= 1 numeric, with the moment stream",
           block_hex=block4.hex().upper(),
           block_length=len(block4),
           stream_count=3,
           stream_order="timestamps, then values, then moments",
           ts_len=len(ts_stream3),
           values_len=len(val_stream3),
           last_stream_has_no_length_prefix=True,
           ts_recipe_byte_offset=4,
           values_recipe_byte_offset=4 + len(ts_stream3) + 4,
           moments_recipe_byte_offset=4 + len(ts_stream3) + 4 + len(val_stream3),
           expected_positions=array_ref(positions),
           expected_tuples=array_ref(tuples),
           expected_moments=array_ref(np.ascontiguousarray(mom)),
           moment_columns=list(_moments.COLUMNS),
           moment_stream_dtype="float64",
           moment_stream_shape=[int(mom.shape[0]), 5],
           note="Every stream but the LAST carries its own u32 LE byte "
                "length including its recipe byte; the last runs to compressed_size. The "
                "two-stream form is the same rule with one fewer stream, which is why "
                "ts_len's meaning did not change. A reader that wants only the plot "
                "stops after the values stream and never touches the moments")

    v.case("two-stream/fixed-L1-numeric",
           kind="fixed-rate level >= 1 numeric",
           block_hex=block3.hex().upper(),
           block_length=len(block3),
           ts_len=len(ts_stream),
           ts_columns=2, ts_column_meaning="[min_ts, max_ts]",
           expected_positions=array_ref(positions),
           expected_tuples=array_ref(tuples),
           row_major_not_planar=True,
           note="the four values of bucket 0, then the four of bucket 1 — NEVER planar "
                ". A planar reader gets four plausible-looking arrays of the "
                "wrong thing")
    return v


# ---------------------------------------------------------------------------
# CRC
# ---------------------------------------------------------------------------


def gen_crc() -> Vector:
    v = Vector(
        id="v1-block-crc",
        set_=SET,
        kind="fixture",
        asserts=(
            "The block CRC is IEEE CRC-32 exactly as zlib computes it, over exactly the "
            "bytes [file_offset, file_offset + compressed_size) — the index entry that "
            "holds it lies outside that range — and it is checked BEFORE decode, so a "
            "flipped bit is refused rather than mis-decoded."
        ),
        source=(
            "verified against zlib.crc32 and against the polynomial's own published "
            "check value"
        ),
        contract=(
            "A corrupt block that decodes to a plausible value draws a wrong plot "
            "silently; checking before decode is what makes the failure loud."
        ),
        requires=["format:v1", "feature:crc"],
        notes=(
            "The check value pins the variant unambiguously: reflected polynomial "
            "0xEDB88320, init and xor-out 0xFFFFFFFF, giving 0xCBF43926 for the ASCII "
            "string '123456789'. There are several CRC-32s and they disagree; this one "
            "is the one crc32fast and zlib.crc32 both compute."
        ),
    )
    v.case("check-value",
           input_ascii="123456789",
           input_hex=b"123456789".hex().upper(),
           expected_crc32=f"0x{zlib.crc32(b'123456789') & 0xFFFFFFFF:08X}",
           polynomial="0xEDB88320 (reflected)",
           init="0xFFFFFFFF", xor_out="0xFFFFFFFF",
           note="the standard check value for this CRC-32 variant")
    assert zlib.crc32(b"123456789") & 0xFFFFFFFF == 0xCBF43926

    for label, payload in [
        ("empty", b""),
        ("single-zero-byte", b"\x00"),
        ("all-zero-64", b"\x00" * 64),
        ("all-ff-64", b"\xff" * 64),
        ("ascending-256", bytes(range(256))),
    ]:
        v.case(f"payload/{label}",
               input_hex=payload.hex().upper(),
               input_length=len(payload),
               expected_crc32=f"0x{zlib.crc32(payload) & 0xFFFFFFFF:08X}")

    # On a real block, and with a flipped bit.
    ts = 1_700_000_000_000_000_000
    result = B.build(B.FileSpec(
        compression_id=0, branching_factor=256,
        groups=[B.GroupSpec(1000.0, ts)],
        channels=[B.ChannelSpec("sig", ramp("float32", 1024))]))
    rel = _write_file("v1_crc_reference.tslod", result)
    ci, level, bi, blk = result.blocks[0]
    lo, hi = blk["file_offset"], blk["file_offset"] + blk["compressed_size"]
    v.case("on-a-real-block",
           file=rel,
           channel_index=ci, level=level, block_index=bi,
           file_offset=u64(lo), compressed_size=u64(blk["compressed_size"]),
           coverage=f"[{lo}, {hi}) — the whole block as stored, framing and recipe bytes included",
           expected_crc32=f"0x{blk['crc32']:08X}",
           index_entry_lies_outside_coverage=True)

    flipped = bytearray(result.data)
    flipped[lo] ^= 0x01
    v.case("flipped-bit-must-be-rejected",
           file=rel,
           byte_offset_to_flip=u64(lo),
           bit_mask="0x01",
           original_byte=f"0x{result.data[lo]:02X}",
           flipped_byte=f"0x{flipped[lo]:02X}",
           stored_crc32=f"0x{blk['crc32']:08X}",
           crc32_after_flip=f"0x{zlib.crc32(bytes(flipped[lo:hi])) & 0xFFFFFFFF:08X}",
           must_be_rejected_before_decode=True,
           note="one bit, in the first byte of the payload. The CRC differs, so the "
                "reader refuses the block; without the check it would decode to a "
                "plausible wrong value and draw a wrong plot")
    assert zlib.crc32(bytes(flipped[lo:hi])) & 0xFFFFFFFF != blk["crc32"]

    v.case("compressed-size-zero-rejected-before-crc",
           compressed_size=u64(0),
           crc32_of_empty_range=f"0x{zlib.crc32(b'') & 0xFFFFFFFF:08X}",
           must_be_rejected=True,
           note="an index entry whose compressed_size is zero is rejected as today "
                ", which is what stops an all-zero entry passing: the CRC of an "
                "empty range is 0x00000000, and an all-zero entry stores exactly that")
    return v


# ---------------------------------------------------------------------------
# The time axis
# ---------------------------------------------------------------------------


def gen_time_axis() -> Vector:
    v = Vector(
        id="v1-time-axis",
        set_=SET,
        kind="fixture",
        asserts=(
            "The time of raw sample i in a fixed-rate group is "
            "start_timestamp + rhe(i * 10^9 / rate), where rate is the EXACT rational "
            "value of the stored float64 and rhe is round-half-even — evaluated as a "
            "rational, never as a float period accumulated per sample."
        ),
        source=(
            "computed with fractions.Fraction, which is exact because every finite "
            "double is a rational"
        ),
        contract=(
            "One rule for the time of a sample, so a file's timestamps mean the "
            "same thing to everyone who reads it."
        ),
        requires=["format:v1", "timing:fixed"],
        notes=(
            "One rule, and the reason it has to be exactly one: 10^9 / rate is not "
            "representable for most rates, so a writer that computes a float period "
            "and multiplies, one that truncates the period to an integer, and one that "
            "rounds the product all produce different times for the same sample. At "
            "24 MHz those three span 666,666 ns by sample 10^9 — two thirds of a "
            "millisecond of disagreement about when a sample was taken. Evaluating the "
            "rational exactly is what removes the choice."
        ),
    )
    ts = 1_700_000_000_000_000_000
    rates = [
        ("1khz-exact", 1000.0),
        ("24mhz", 24_000_000.0),
        ("44100hz", 44_100.0),
        ("7hz", 7.0),
        ("3076.923076923077hz", 3076.923076923077),   # a rate with no exact period
        ("99.9900009999hz", 99.9900009999),           # and another
    ]
    for name, rate in rates:
        exact_rate = Fraction(rate)
        for i in [0, 1, 2, 3, 255, 256, 257, 1000, 65_536, 10**6, 10**9]:
            t = B.sample_time_ns(ts, rate, i)
            v.case(
                f"{name}/i={i}",
                sample_rate_bits="0x%016X" % struct.unpack(
                    "<Q", struct.pack("<d", rate))[0],
                sample_rate_repr=repr(rate),
                sample_rate_exact_numerator=str(exact_rate.numerator),
                sample_rate_exact_denominator=str(exact_rate.denominator),
                start_timestamp=i64(ts),
                sample_index=u64(i),
                expected_ns=i64(t),
                rule="start + round_half_even(i * 10^9 / rate), rate exact",
            )
    return v


# ---------------------------------------------------------------------------
# v1-specific rejections
# ---------------------------------------------------------------------------


def gen_v1_negatives() -> Vector:
    v = Vector(
        id="v1-negative-vectors",
        set_=SET,
        kind="negative",
        asserts=(
            "A v1 reader reads EXACTLY version 1 and rejects every other value, "
            "and rejects block_samples that is zero or not a multiple of "
            "branching_factor, any must-understand feature bit it does not know, a "
            "profile byte outside {0, 2}, any recipe byte outside the registry, and a "
            "moment-stream feature bit that disagrees with how the blocks are framed — "
            "while skipping may-ignore feature bits it does not know."
        ),
        source=(
            "the version-1 header and framing rules"
        ),
        contract=(
            "A reader that accepts a malformed file produces wrong answers where it "
            "should have produced an error."
        ),
        requires=["format:v1", "feature:read-rejection"],
        notes=(
            "Each case is a well-formed file with one field patched and the rejection "
            "class it must trigger. Two of them are ACCEPTING cases — a may-ignore "
            "feature bit, and a tick rate at exactly the largest legal value — because "
            "a bound pinned only where it fails is half pinned: a reader that refuses "
            "the boundary passes every rejection here and still refuses valid files."
        ),
    )
    ts = 1_700_000_000_000_000_000
    good = B.build(B.FileSpec(
        compression_id=0, branching_factor=256, block_samples=256,
        groups=[B.GroupSpec(1000.0, ts)],
        channels=[B.ChannelSpec("sig", ramp("float32", 1024))]))
    rel = _write_file("v1_negative_base.tslod", good)
    H = B.header_offset

    def case(slug, offset, code, value, field, why, rejection_class=None):
        """One patched field and the class it must be refused as.

        `rejection_class` defaults to the slug because most cases are the only
        member of their class. Where several patches are refused by the SAME
        rule the class must be passed explicitly: the slug names the case, and
        a class no reader can produce is not a check.
        """
        raw = struct.pack("<" + code, value)
        v.case(slug, file=rel, rejection_class=rejection_class or slug,
               patch={"offset": u64(offset), "width_bytes": len(raw),
                      "original_hex": good.data[offset:offset + len(raw)].hex().upper(),
                      "patched_hex": raw.hex().upper()},
               field=field, expected_error="CorruptFileError",
               verified_against_an_implementation=False, reason=why)

    case("v1-version-must-be-exactly-1", H("version"), "H", 4,
         "header.version",
         "the wire version of this format is 1. A version-1 reader reads EXACTLY 1 "
         "and rejects every other value. The error should name the version it found, "
         "because a user holding a file this reader cannot read needs to know what "
         "they are holding")
    case("v1-version-zero", H("version"), "H", 0, "header.version",
         "zero is not a version; it is what an uninitialised or truncated header reads as",
         rejection_class="v1-version-must-be-exactly-1")
    case("v1-version-unassigned", H("version"), "H", 3, "header.version",
         "no version other than 1 is assigned, so a file claiming any of them is "
         "refused rather than read on a guess",
         rejection_class="v1-version-must-be-exactly-1")

    case("v1-block-samples-zero", H("block_samples"), "I", 0,
         "header.block_samples",
         "block_samples that is zero is rejected")
    case("v1-block-samples-not-multiple-of-bf", H("block_samples"), "I", 300,
         "header.block_samples",
         "300 is not a multiple of branching_factor 256; the block-length parameter requires a multiple so "
         "that a level k+1 block is built from exactly branching_factor complete level k "
         "blocks and position chaining is unchanged")
    case("v1-profile-unassigned-1", H("compression_id"), "B", 1,
         "header.compression_id",
         "the profile is 0 (none) or 2 (recipe); 1 is not assigned and a file carrying "
         "it is rejected rather than guessed at",
         rejection_class="v1-profile-unknown")
    case("v1-profile-unknown", H("compression_id"), "B", 3,
         "header.compression_id",
         "the v1 profile is 0 (none) or 2 (recipe); nothing else")

    def raw_case(slug, offset, raw, field, why, cls=None):
        """A patch of arbitrary bytes, where a struct code will not do."""
        v.case(slug, file=rel, rejection_class=cls or slug,
               patch={"offset": u64(offset), "width_bytes": len(raw),
                      "original_hex": good.data[offset:offset + len(raw)].hex().upper(),
                      "patched_hex": raw.hex().upper()},
               field=field, expected_error="CorruptFileError",
               verified_against_an_implementation=False, reason=why)

    def trunc_case(slug, keep, field, why, cls):
        """A whole-file rule: the defect is the file's LENGTH, not a field."""
        v.case(slug, file=rel, rejection_class=cls, truncate_to=u64(keep),
               field=field, expected_error="CorruptFileError",
               original_size_bytes=u64(len(good.data)),
               verified_against_an_implementation=False, reason=why)

    # ---- the file as a whole. A truncated file is not a patched field: the
    # length IS the defect, so these cases cut rather than overwrite.
    trunc_case("v1-empty-file", 0, "file",
               "zero bytes is the commonest damaged file there is — a create "
               "that never wrote, a copy that never ran. It is refused for "
               "being too short to hold a header, not for a magic that was "
               "never there to mismatch",
               "file-shorter-than-header")
    trunc_case("v1-file-shorter-than-header", 64, "file",
               "half a header. A reader that unpacks fields before checking "
               "the length reads whatever follows the buffer",
               "file-shorter-than-header")
    trunc_case("v1-file-one-byte-short-of-header", 127, "file",
               "the boundary, and the one an off-by-one gets wrong: 127 bytes "
               "is not a header and 128 is. A reader comparing with > rather "
               "than >= accepts this and reads one byte past the end",
               "file-shorter-than-header")

    raw_case("v1-magic-mismatch", H("magic"), b"TSLOD\x01", "header.magic",
             "the sixth byte of the magic is part of it. A reader that matches "
             "only the five ASCII bytes accepts a file whose format marker "
             "says something else, and then reads it as version 1",
             cls="bad-magic")

    # ---- header fields whose rule the reader already knew and no case pinned
    case("v1-branching-factor-below-2", H("branching_factor"), "I", 1,
         "header.branching_factor",
         "a branching factor of 1 gives a level that is its own parent, so the "
         "pyramid never terminates. The specification says two or more and the "
         "reader has always refused it; nothing pinned it until now",
         rejection_class="branching-factor-below-2")
    case("v1-file-state-unknown", H("file_state"), "B", 2, "header.file_state",
         "file_state is 0 sealed or 1 active and there is no third value. A "
         "reader that treats anything non-zero as active reads an unknown "
         "state as one it happens to know",
         rejection_class="file-state-unknown")
    case("v1-num-groups-zero", H("num_groups"), "H", 0, "header.num_groups",
         "every channel belongs to a group and a group carries the time base, "
         "so a file with no groups has no way to say when any sample was "
         "taken",
         rejection_class="num-groups-zero")
    case("v1-num-channels-zero", H("num_channels"), "I", 0, "header.num_channels",
         "a file with no channels holds no data. It is refused rather than "
         "read as an empty success, because the likelier cause is a header "
         "written before the channels were",
         rejection_class="num-channels-zero")
    case("v1-group-table-offset-out-of-bounds", H("group_table_offset"), "Q",
         1 << 40, "header.group_table_offset",
         "the group table is refused for ending past the end of the file, "
         "which is a stated rule, rather than for whatever a reader's language "
         "does when it indexes past a buffer",
         rejection_class="group-table-out-of-bounds")
    case("v1-channel-table-offset-out-of-bounds", H("channel_table_offset"), "Q",
         1 << 40, "header.channel_table_offset",
         "the same rule for the channel table. Both offsets are u64 and "
         "neither is bounded by anything but the file's own length",
         rejection_class="channel-table-out-of-bounds")

    # ---- the group entry
    G = lambda name: B.group_offset(good.layout, 0, name)
    for slug, value, why in (
        ("v1-sample-rate-zero", 0.0,
         "a rate of zero makes the time of sample i a division by zero, so "
         "every timestamp in the group is undefined rather than merely wrong"),
        ("v1-sample-rate-negative", -1000.0,
         "a negative rate runs the group's time backwards, so sample i+1 "
         "precedes sample i and any search over the axis is unsound"),
        ("v1-sample-rate-nan", float("nan"),
         "a NaN rate makes every derived timestamp compare equal to nothing, "
         "including itself, which is worse than an error because a reader "
         "finds no sample rather than failing to look"),
        ("v1-sample-rate-infinite", float("inf"),
         "an infinite rate collapses every sample onto the group's start "
         "timestamp. It is refused with the other three because the rule is "
         "one rule: the rate is positive and finite"),
    ):
        case(slug, G("sample_rate"), "d", value, "group_entry.sample_rate", why,
             rejection_class="sample-rate-not-positive-finite")

    case("v1-timing-mode-unknown", G("timing_mode"), "B", 2,
         "group_entry.timing_mode",
         "timing_mode is 0 fixed or 1 variable. It decides which streams a "
         "block carries and how its first stream is read, so a reader that "
         "defaults an unknown value parses every block in the group wrongly",
         rejection_class="timing-mode-unknown")
    case("v1-timing-flags-reserved-bit-set", G("timing_flags"), "B", 0x04,
         "group_entry.timing_flags",
         "bits 2 to 7 of the timing-flag byte are reserved AND rejected, which "
         "is the one exception to reserved fields being ignored: these bits "
         "change how the group is read, so a reader that skips one it does not "
         "know reads the group wrongly rather than incompletely. Stated since "
         "version 1 and enforced all along; this is the case that pins it",
         rejection_class="timing-flags-reserved-bit-set")

    # ---- the channel entry
    C = lambda name: B.channel_offset(good.layout, 0, name)
    case("v1-dtype-unknown", C("dtype"), "B", 10, "channel_entry.dtype",
         "the dtype enum is 0 to 9 and closed. It fixes the width of every "
         "value in the channel, so an unknown code leaves a reader with no way "
         "to know how long anything is",
         rejection_class="dtype-unknown")
    case("v1-aggregation-mode-unknown", C("aggregation_mode"), "B", 2,
         "channel_entry.aggregation_mode",
         "aggregation_mode is 0 numeric or 1 bitfield, and it decides what a "
         "bucket's four columns MEAN — min/max/first/last against OR/AND/"
         "first/last. A reader that defaults an unknown value reports one as "
         "the other with no sign that it did",
         rejection_class="aggregation-mode-unknown")
    case("v1-bitfield-aggregation-on-float-dtype", C("aggregation_mode"), "B", 1,
         "channel_entry.aggregation_mode",
         "bitfield mode is defined only for the eight integer dtypes: OR and "
         "AND over a float's bit pattern are not statistics of anything. The "
         "base channel is float32, so setting bitfield mode on it is the "
         "combination the rule forbids. The class is the one set 3 already "
         "uses for the same rule at the kernel",
         rejection_class="bitfield-on-float-dtype")
    case("v1-group-id-out-of-range", C("group_id"), "H", 7,
         "channel_entry.group_id",
         "a channel names the group whose time base it uses. An index past the "
         "table is refused as an out-of-range group, not as whatever a "
         "reader's language does when it indexes a list past its end",
         rejection_class="group-id-out-of-range")
    case("v1-num-levels-zero", C("num_levels"), "I", 0,
         "channel_entry.num_levels",
         "num_levels counts level 0, so the smallest legal value is 1 — a "
         "channel with raw samples and no pyramid above them. Zero says the "
         "channel has not even a level 0, which no channel can be, and a "
         "reader looping over levels would silently read nothing",
         rejection_class="num-levels-zero")
    case("v1-level-table-offset-out-of-bounds", C("level_table_offset"), "Q",
         1 << 40, "channel_entry.level_table_offset",
         "the same bound as the group and channel tables, one level down: the "
         "level table's entries must end inside the file",
         rejection_class="level-table-out-of-bounds")
    for slug, fname, why in (
        ("v1-channel-name-not-utf8", "name",
         "the channel name is UTF-8, NUL-padded. 0xFF 0xFE begins no valid "
         "sequence, and a reader that decodes it leniently shows a user a name "
         "the writer never wrote"),
        ("v1-channel-unit-not-utf8", "unit",
         "the unit string is the same rule in a 16-byte field, and a mangled "
         "unit is worse than a mangled name: it is the thing a plot's axis is "
         "labelled with"),
        ("v1-channel-calibration-id-not-utf8", "calibration_id",
         "and the calibration identifier, which is what ties the channel to "
         "the record of how it was calibrated. One rule for all three text "
         "fields, so one class"),
    ):
        raw_case(slug, C(fname), b"\xff\xfe", f"channel_entry.{fname}", why,
                 cls="text-field-not-utf8")
    case("v1-scaling-type-unknown", C("scaling_type"), "B", 3,
         "channel_entry.scaling_type",
         "scaling_type is 0 identity or 1 linear. It decides whether the "
         "stored values are the real ones or have to be transformed first, so "
         "an unknown value leaves a reader unable to say what a sample means",
         rejection_class="scaling-type-unknown")
    for slug, delta, value, pname, why in (
        ("v1-scaling-gain-nan", 0, float("nan"), "gain",
         "a NaN gain turns every scaled sample into a NaN, so a channel that "
         "recorded perfectly reads as entirely invalid"),
        ("v1-scaling-offset-infinite", 8, float("inf"), "offset",
         "an infinite offset does the same by saturation. Gain and offset are "
         "one rule — a scaling parameter is finite — so they share a class"),
    ):
        case(slug, C("scaling_params") + delta, "d", value,
             f"channel_entry.scaling_{pname}", why,
             rejection_class="scaling-parameter-not-finite")

    # ---- a block whose extent leaves the file
    _blk = B.block_offset(good.layout, 0, 1, 0, "file_offset")
    _csz = B.block_offset(good.layout, 0, 1, 0, "compressed_size")
    case("v1-block-extent-past-eof", _csz, "Q", 1 << 40,
         "block_index_entry.compressed_size",
         "file_offset + compressed_size reaches past the end of the file. This "
         "is checked before the CRC, because computing a CRC over a range that "
         "does not exist is the read the rule is there to prevent",
         rejection_class="block-extent-past-eof")
    case("v1-block-file-offset-past-eof", _blk, "Q", 1 << 40,
         "block_index_entry.file_offset",
         "the same rule reached by moving the start rather than the length. "
         "One rule, one class: a reader needs to know the block is not in the "
         "file, not which of the two numbers was wrong",
         rejection_class="block-extent-past-eof")
    # This case used to patch bit 0. Bit 0 is now ASSIGNED — it is what says
    # whether the moment stream is present — so patching it no longer describes
    # an unknown feature; it describes a flag that disagrees with the framing,
    # which is a different rejection and has its own two cases below. The case
    # moves to bit 31, the top of the must-understand half, which version 1
    # still assigns nothing. The patched value KEEPS bit 0 set, because the
    # base file carries moments: this case must fail for the unknown bit alone.
    case("v1-unknown-must-understand-feature-bit", H("features"), "Q",
         0x0000_0000_8000_0001, "header.features",
         "bit 31 is in the must-understand half and version 1 assigns it nothing, so a "
         "reader refuses the file rather than reading it as though the feature were "
         "absent. Taken with its two neighbours this is the whole rule in three cases: "
         "bit 0 is must-understand and KNOWN, so the file is accepted and the bit is "
         "acted on — it is what says whether the moment stream is there; bit 31 is "
         "must-understand and unknown, so the file is refused; bit 32 is may-ignore and "
         "unknown, so the file is accepted and the bit skipped. The patched value here "
         "keeps bit 0 set because the base file carries moments, so the only thing "
         "wrong with it is the bit a reader does not know")

    v.case("v1-unknown-may-ignore-feature-bit-is-ACCEPTED",
           file=rel, rejection_class=None,
           patch={"offset": u64(H("features")), "width_bytes": 8,
                  "original_hex": good.data[H("features"):H("features") + 8].hex().upper(),
                  # bit 32 set AND bit 0 kept: the base file carries moments,
                  # so clearing bit 0 would make this file rejectable for a
                  # reason that has nothing to do with the half being tested.
                  "patched_hex": struct.pack(
                      "<Q", (1 << 32) | B.FEATURE_MOMENT_STREAM).hex().upper()},
           field="header.features",
           expected_error=None, must_open=True,
           verified_against_an_implementation=False,
           reason="the HIGH 32 bits are may-ignore. This is the case that proves the "
                  "split is real: the same field, a different half, and the opposite "
                  "outcome. A reader that refuses both has no forward compatibility at "
                  "all; a reader that accepts both has no safety")

    # ---- feature bit 0 must agree with the framing, in both directions.
    # Only statable at profile 0, where the framing is arithmetic: at profile 2
    # the bit is the only source and there is nothing to disagree with it.
    MOMENT_FLAG = B.FEATURE_MOMENT_STREAM
    assert struct.unpack_from("<Q", good.data, H("features"))[0] & MOMENT_FLAG, (
        "the negative base file is supposed to carry moments")

    v.case("v1-moment-stream-flag-clear-but-block-has-three-streams",
           file=rel, rejection_class="moment-stream-flag-mismatch",
           patch={"offset": u64(H("features")), "width_bytes": 8,
                  "original_hex": good.data[H("features"):H("features") + 8].hex().upper(),
                  "patched_hex": struct.pack("<Q", 0).hex().upper()},
           field="header.features",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="the file's numeric level-1 blocks are framed with three streams, and "
                  "clearing bit 0 says there are two. A reader must not decide the bit "
                  "is stale and read the block anyway, nor decide the block is stale and "
                  "trust the bit: it cannot tell which of the two is the damage, and "
                  "either guess silently hands back a moment stream read as values, or "
                  "values read as a truncated block")

    nomom_rel = f"{FILES}/v1_no_moment_stream.tslod"
    nomom = (VECTORS / nomom_rel).read_bytes()
    assert not struct.unpack_from("<Q", nomom, H("features"))[0] & MOMENT_FLAG, (
        "v1_no_moment_stream.tslod is supposed to have bit 0 clear")

    v.case("v1-moment-stream-flag-set-but-block-has-two-streams",
           file=nomom_rel, rejection_class="moment-stream-flag-mismatch",
           patch={"offset": u64(H("features")), "width_bytes": 8,
                  "original_hex": nomom[H("features"):H("features") + 8].hex().upper(),
                  "patched_hex": struct.pack("<Q", MOMENT_FLAG).hex().upper()},
           field="header.features",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="the same rule from the other side: the bit promises a moment stream "
                  "the blocks do not carry. This is the direction that corrupts a "
                  "reader quietly rather than loudly — it reads the values stream's "
                  "length prefix as the values stream, and then reads whatever follows "
                  "as moments. The file that proves the rule is the one file in the set "
                  "written without moments, which is also why it is kept")

    # ---- the values stream must decode to exactly uncompressed_size.
    # The index entry lies outside the block's CRC range, so these patches
    # leave every CRC verifying: the file is well formed until the length it
    # promises is compared with the length it delivers.
    us_off = B.block_offset(good.layout, 0, 1, 0, "uncompressed_size")
    good_us, = struct.unpack_from("<Q", good.data, us_off)

    for slug, value, why in (
        ("v1-decoded-size-smaller-than-index-entry", good_us - 1,
         "the index entry promises one byte fewer than the values stream "
         "decodes to. A reader that trusts the entry allocates short and "
         "either truncates the block or walks off the end of it; a reader "
         "that trusts the stream silently disagrees with the index it will "
         "use to seek. Neither is a reading of this file, so it is refused"),
        ("v1-decoded-size-larger-than-index-entry", good_us + 1,
         "the same disagreement from the other side, and the one a writer is "
         "likelier to produce: the entry was written before the stream was "
         "encoded and never corrected. Feature bit 0 still agrees with the "
         "framing here, so this fails at the decoded length and not before it"),
    ):
        raw = struct.pack("<Q", value)
        v.case(slug, file=rel, rejection_class="decoded-size-mismatch",
               patch={"offset": u64(us_off), "width_bytes": 8,
                      "original_hex": good.data[us_off:us_off + 8].hex().upper(),
                      "patched_hex": raw.hex().upper()},
               field="block_index_entry.uncompressed_size",
               expected_error="CorruptFileError",
               verified_against_an_implementation=False, reason=why)

    sc_off = B.block_offset(good.layout, 0, 1, 0, "sample_count")
    good_sc, = struct.unpack_from("<Q", good.data, sc_off)
    v.case("v1-sample-count-disagrees-with-block-shape",
           file=rel, rejection_class="decoded-size-mismatch",
           patch={"offset": u64(sc_off), "width_bytes": 8,
                  "original_hex": good.data[sc_off:sc_off + 8].hex().upper(),
                  "patched_hex": struct.pack("<Q", good_sc + 1).hex().upper()},
           field="block_index_entry.sample_count",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="the same rule from the shape's side rather than the byte "
                  "count's. sample_count is the block's row count at its own "
                  "level, so it fixes the decoded size of EVERY stream in the "
                  "block; one more row than the block holds and no stream is "
                  "the length its shape implies. A reader can refuse this "
                  "before it decodes anything, by comparing uncompressed_size "
                  "with the shape, or after decoding, by comparing the bytes. "
                  "It is one rule, so it is one class either way")

    p2_rel, p2 = _PROFILE2_BUILDS["v1_p2_with_moments.tslod"]
    p2_off = B.block_offset(p2.layout, 0, 1, 0, "uncompressed_size")
    p2_us, = struct.unpack_from("<Q", p2.data, p2_off)
    v.case("v1-decoded-size-mismatch-at-profile-2",
           file=p2_rel, rejection_class="decoded-size-mismatch",
           patch={"offset": u64(p2_off), "width_bytes": 8,
                  "original_hex": p2.data[p2_off:p2_off + 8].hex().upper(),
                  "patched_hex": struct.pack("<Q", p2_us - 1).hex().upper()},
           field="block_index_entry.uncompressed_size",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="the rule exists for this profile, so this is the case that "
                  "proves it. Here there is no length arithmetic to catch the "
                  "disagreement earlier — the values stream is compressed, so "
                  "its decoded length is not knowable until it has been "
                  "decoded, and comparing that length with the index entry is "
                  "the only check a profile-2 reader has on the block's size")

    v.case("v1-stream-length-prefix-exceeds-block",
           rejection_class="stream-length-prefix-out-of-range",
           field="block.stream[0].length_prefix",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="a stream's u32 length prefix that runs past the block's "
                  "compressed_size must be refused before any decode. With k streams "
                  "there are k-1 such prefixes and each is a rejection site")

    v.case("v1-stream-length-prefix-zero",
           rejection_class="stream-length-prefix-zero",
           field="block.stream[0].length_prefix",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="a length of 0 leaves no room for the stream's own recipe byte, so it "
                  "cannot describe a well-formed stream; the minimum is 1")

    v.case("v1-stream-length-prefixes-leave-no-last-stream",
           rejection_class="stream-length-prefix-consumes-block",
           field="block stream framing",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="the prefixes of the first k-1 streams must leave at least one byte "
                  "for the last stream, which has no prefix and runs to compressed_size. "
                  "Prefixes summing to the whole block describe a block with no values")

    # The stored tick-rate domain, as file rejections. Both fields are u64 on
    # the wire, so the constraint a reader enforces is 1 <= x <= 2**32: the
    # floor because a zero denominator has no value and a zero numerator makes
    # every tick the same instant, the ceiling because it is what keeps every
    # intermediate in the two tick conversions inside 128 bits.
    ticks_base = B.build(B.FileSpec(
        compression_id=0, branching_factor=256, block_samples=256,
        groups=[B.GroupSpec(1000.0, ts, timing_flags=0x01,
                            tick_rate_numer=24_000_000, tick_rate_denom=1,
                            anchor_ticks=-123_456)],
        channels=[B.ChannelSpec("sig", ramp("float32", 1024))]))
    ticks_rel = _write_file("v1_negative_tick_rate_base.tslod", ticks_base)

    def rate_case(slug, fieldname, value, why):
        offset = B.group_offset(ticks_base.layout, 0, fieldname)
        raw = struct.pack("<Q", value)
        v.case(slug, file=ticks_rel, rejection_class=slug,
               patch={"offset": u64(offset), "width_bytes": len(raw),
                      "original_hex": ticks_base.data[offset:offset + len(raw)].hex().upper(),
                      "patched_hex": raw.hex().upper()},
               field=f"group_entry.{fieldname}", expected_error="CorruptFileError",
               verified_against_an_implementation=False, reason=why)

    rate_case("v1-tick-rate-numer-zero", "tick_rate_numer", 0,
              "with TIMEBASE_PRESENT set, a numerator of zero makes every tick the same "
              "instant; the field is >= 1 or the group is malformed")
    rate_case("v1-tick-rate-denom-zero", "tick_rate_denom", 0,
              "a denominator of zero is a division by zero in both tick conversions; "
              "the field is >= 1 or the group is malformed")
    rate_case("v1-tick-rate-numer-exceeds-2p32", "tick_rate_numer", 2**32 + 1,
              "both rate fields are at most 2^32. That bound is what lets an "
              "implementation evaluate the tick conversions in 128-bit integers rather "
              "than arbitrary precision: the widest intermediate is "
              "|ticks - anchor| * 10^9 * denom, below 2^125 for |ticks - anchor| < 2^63 "
              "and denom <= 2^32")
    # The accepting side of the same bound. Without it "at most 2^32" is pinned
    # only where it fails, and a reader that refuses the boundary value itself
    # passes every rejection case above.
    _numer_off = B.group_offset(ticks_base.layout, 0, "tick_rate_numer")
    v.case("v1-tick-rate-at-2p32-is-ACCEPTED",
           file=ticks_rel, rejection_class=None,
           patch={"offset": u64(_numer_off), "width_bytes": 8,
                  "original_hex": ticks_base.data[_numer_off:_numer_off + 8].hex().upper(),
                  "patched_hex": struct.pack("<Q", 2**32).hex().upper()},
           field="group_entry.tick_rate_numer",
           expected_error=None, must_open=True,
           verified_against_an_implementation=False,
           reason="2^32 is the largest legal value, not the first illegal one. This is "
                  "the case that makes the bound inclusive: a reader that treats it as "
                  "exclusive refuses a valid file and passes every rejection case above")

    rate_case("v1-tick-rate-denom-exceeds-2p32", "tick_rate_denom", 2**32 + 1,
              "the same bound on the denominator, which is the field the widest "
              "intermediate is actually multiplied by")

    for recipe in (0x04, 0x7F, 0xEF, 0xFF):
        v.case(f"v1-unknown-recipe-{recipe:#04x}",
               rejection_class="unknown-recipe-byte",
               recipe_byte=f"0x{recipe:02X}",
               expected_error="CorruptFileError",
               error_must_name_the_byte=True,
               verified_against_an_implementation=False,
               reason="every value outside {0x00, 0x01, 0x02, 0x03} and the 0xF0-0xFF "
                      "experimental range is reserved and rejected with an actionable "
                      "error naming the byte; there is no registry escape")

    v.case("v1-profile0-with-non-identity-recipe",
           rejection_class="profile0-recipe-mismatch",
           compression_id=0, recipe_byte="0x01",
           expected_error="CorruptFileError",
           verified_against_an_implementation=False,
           reason="under profile 0 EVERY recipe byte must be 0x00. A profile-0 "
                  "file carrying a zstd recipe is the file that would make the "
                  "stdlib-only reference reader need a codec, which is the whole point "
                  "of the profile")
    return v


def main() -> None:
    for factory in (gen_profile0_set, gen_profile2_set, gen_block_framing, gen_crc,
                    gen_time_axis, gen_v1_negatives):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
