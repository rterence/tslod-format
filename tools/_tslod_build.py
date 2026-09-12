"""Build `.tslod` files from a spec.

The record layouts plus `struct` are the generator for exactly this: given a
file spec, emit the bytes version 1 defines. It writes shapes no recording
happens to contain — two units, `scaling_type = linear`, a bitfield channel
beside a numeric one in the same group, a zero-sample channel — because a
conformance set has to cover the format rather than the recordings that exist.

Its only claim is that the bytes it emits satisfy the layout the specification
states. It is not a writer: it chooses no codec by policy, it does not stream,
and it patches nothing in place.

Two rules it follows without exception:

  * **timestamps come from the exact-rational rule**, evaluated over
    `fractions.Fraction`, never from a float period accumulated per sample;
  * **no transform is implicit.** A timestamp payload under the identity
    recipe is the absolute `i64` values. If a delta ever earns its place it is
    a recipe a reader can see, not a rule it must know.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

import _fold
import _moments

# ---------------------------------------------------------------------------
# Layout constants — the record layouts, as the specification states them
# ---------------------------------------------------------------------------

MAGIC = b"TSLOD\x00"

HEADER_FMT_V1 = "<6sHIBBHIIQQ16sI32sQ28s"
GROUP_ENTRY_FMT = "<dqQBB6sQQq8s"
CHANNEL_ENTRY_FMT = "<64sBBHIQ16sB1s16s32s14s"
LEVEL_ENTRY_FMT = "<QQQ"
BLOCK_INDEX_FMT_V1 = "<QQQQqII"

HEADER_SIZE = 128
GROUP_ENTRY_SIZE = 64
CHANNEL_ENTRY_SIZE = 160
LEVEL_ENTRY_SIZE = 24
BLOCK_INDEX_ENTRY_SIZE_V1 = 48

DTYPE_ENUM = {"float32": 0, "float64": 1, "int8": 2, "int16": 3, "int32": 4,
              "int64": 5, "uint8": 6, "uint16": 7, "uint32": 8, "uint64": 9}

#: Must-understand feature bit 0: the file carries the moment stream, so every
#: numeric value block at level >= 1 has three streams. Clear means none does.
#: The builder derives it from what it actually wrote; nothing else may set it.
FEATURE_MOMENT_STREAM = 1 << 0

RECIPE_IDENTITY = 0x00
RECIPE_ZSTD = 0x01
RECIPE_PCO = 0x02
RECIPE_TRANSPOSE_ZSTD = 0x03

NS = 10**9


# ---------------------------------------------------------------------------
# the time-axis rule's time rule
# ---------------------------------------------------------------------------


def rhe(numer: int, denom: int) -> int:
    """Round-half-even of the exact rational numer/denom, denom > 0."""
    q, r = divmod(numer, denom)
    r2 = 2 * r
    if r2 > denom or (r2 == denom and q % 2 == 1):
        q += 1
    return q


def sample_time_ns(start_timestamp: int, sample_rate: float, i: int) -> int:
    """the time-axis rule: t(i) = start + rhe(i * 10^9 / rate), rate the EXACT rational
    value of the stored float64 (every finite double is one)."""
    rate = Fraction(sample_rate)
    exact = Fraction(i) * NS / rate
    return start_timestamp + rhe(exact.numerator, exact.denominator)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class GroupSpec:
    sample_rate: float
    start_timestamp: int
    timing_mode: int = 0            # 0 fixed, 1 variable
    timing_flags: int = 0
    tick_rate_numer: int = 0
    tick_rate_denom: int = 0
    anchor_ticks: int = 0
    #: variable-rate only: the int64 stamp of every level-0 sample
    timestamps: np.ndarray | None = None
    #: written into the entry; None means "derive from the channels"
    total_samples: int | None = None


@dataclass
class ChannelSpec:
    name: str
    data: np.ndarray
    group_id: int = 0
    aggregation_mode: int = 0       # 0 numeric, 1 bitfield
    unit: str = "raw"
    scaling_type: int = 0           # 0 identity, 1 linear
    scaling_gain: float = 1.0
    scaling_offset: float = 0.0
    calibration_id: str = ""
    #: override num_levels in the channel entry (for negative vectors)
    num_levels_override: int | None = None


@dataclass
class FileSpec:
    groups: list[GroupSpec]
    channels: list[ChannelSpec]
    branching_factor: int = 256
    block_samples: int | None = None      # v1 only; defaults to branching_factor
    compression_id: int = 0               # the file profile: 0 none, 2 recipe
    file_state: int = 0                   # 0 sealed, 1 active
    session_id: bytes = b"\x00" * 16
    sequence_number: int = 0
    prev_file_hash: bytes = b"\x00" * 32
    features: int = 0                     # v1 only
    recipe: int = RECIPE_IDENTITY         # v1 only; the default for every stream
    #: v1 only. Per-stream override of `recipe`, keyed "timestamps", "values"
    #: and "moments". Profile 2 exists so a file can mix recipes — each stream
    #: carries its own byte and nothing is implicit — so the writer has to be
    #: able to write a different one per stream or the profile is untested.
    recipes: dict | None = None
    #: v1 only. The (count, mean, M2, M3, M4) third stream on numeric levels
    #: >= 1, False builds a file without it, which is what
    #: the "reader that does not want moments skips it" vector needs.
    moments: bool = True


# ---------------------------------------------------------------------------
# The pyramid
# ---------------------------------------------------------------------------


def compute_num_levels(total_samples: int, bf: int) -> int:
    if bf < 2:
        raise ValueError(f"branching_factor must be >= 2, got {bf}")
    if total_samples <= 0:
        return 1
    levels, samples = 1, total_samples
    while samples > 1:
        buckets = -(-samples // bf)
        levels += 1
        if buckets <= 1:
            break
        samples = buckets
    return levels


def build_pyramid(raw: np.ndarray, bf: int, agg_mode: int):
    """Return (levels, raw_idx, moments).

    `levels[0]` is the raw array; `levels[k]` for k >= 1 is the fold's
    `(N, 5)` array, `[min, max, first, mid, last]`, or `(N, 2)` `[OR, AND]` in
    bitfield mode.

    `raw_idx[k]` maps each of the five column names to the absolute LEVEL-0
    index of the sample that column holds, per bucket at level k. `min` and
    `max` are chained through the parent level's positions from level 2 up,
    which is what makes their stored timestamp the true raw-sample instant
    rather than a bucket-boundary approximation. `first`, `mid` and `last` are
    arithmetic in the bucket's geometry: `first` is the bucket's first sample,
    `last` its last existing one, and `mid` the first sample of child
    `k // 2` of its `k` existing children — the child rule, evaluated on
    indices. The two derivations are required to agree below: the fold takes
    the values from the children and this takes the indices from the geometry,
    and a disagreement between them would store a value under the wrong time.
    """
    levels = [raw]
    n_raw = len(raw)
    idx_by_level: dict[int, dict[str, np.ndarray]] = {}
    #: level -> (N, 5) float64 stored moments. : level 1 folds
    #: raw samples left to right; level k folds level k-1's stored tuples. It is
    #: hierarchical by rule, not a re-read of level 0 at each level.
    moments_by_level: dict[int, np.ndarray] = {}

    current, from_raw = raw, 1
    level = 0
    while True:
        size = len(current) if from_raw else current.shape[0]
        if size <= 1:
            break
        if agg_mode == 0:
            tuples, pos = _fold.build_level(
                np.ascontiguousarray(current), bf, from_raw, agg_mode, True)
            n_out = tuples.shape[0]
            offsets = np.arange(n_out, dtype=np.int64) * bf
            child_min = offsets + pos[:, 0].astype(np.int64)
            child_max = offsets + pos[:, 1].astype(np.int64)
            # The children a bucket actually has, which is `bf` for every
            # bucket but a ragged last one — and k is what names mid's child.
            counts = np.minimum(offsets + bf, size) - offsets
            span_below = bf ** level          # raw samples under one child
            here = {
                "first": offsets * span_below,
                "mid": (offsets + counts // 2) * span_below,
                "last": np.minimum((offsets + counts) * span_below, n_raw) - 1,
            }
            if level == 0:
                here["min"], here["max"] = child_min, child_max
                moments_by_level[1] = _moments.level1_from_raw(current, bf)
            else:
                moments_by_level[level + 1] = _moments.next_level(
                    moments_by_level[level], bf)
                # Chain: the child index is a bucket at `level`, whose own
                # stored raw index is already resolved.
                here["min"] = idx_by_level[level]["min"][child_min]
                here["max"] = idx_by_level[level]["max"][child_max]
            idx_by_level[level + 1] = here
            # The fold took these five values from the children; the indices
            # above came from the geometry. They must name the same samples, or
            # a stored value would be paired with another sample's time.
            for name, col in (("min", 0), ("max", 1), ("first", 2),
                              ("mid", 3), ("last", 4)):
                named = np.ascontiguousarray(raw[here[name]])
                folded = np.ascontiguousarray(tuples[:, col])
                assert named.tobytes() == folded.tobytes(), (
                    f"level {level + 1}: the folded {name} column and the raw "
                    f"sample its index names disagree")
        else:
            tuples = _fold.build_level(
                np.ascontiguousarray(current), bf, from_raw, agg_mode)
        levels.append(tuples)
        current, from_raw = tuples, 0
        level += 1

    return levels, idx_by_level, moments_by_level


# ---------------------------------------------------------------------------
# Stream encoding
# ---------------------------------------------------------------------------


def encode_stream_v1(payload: bytes, recipe: int) -> bytes:
    """`[recipe u8][payload]` — the framing rules. No transform is implicit."""
    if recipe == RECIPE_IDENTITY:
        return bytes([recipe]) + payload
    if recipe == RECIPE_ZSTD:
        import zstandard
        c = zstandard.ZstdCompressor(write_content_size=True, write_checksum=False)
        return bytes([recipe]) + c.compress(payload)
    if recipe == RECIPE_TRANSPOSE_ZSTD:
        raise ValueError("byte-transpose needs the element width; use encode_block_v1")
    raise ValueError(f"recipe {recipe:#04x} is not encodable here")


def byte_transpose(payload: bytes, width: int) -> bytes:
    """Byte plane i of every element before plane i+1."""
    if len(payload) % width:
        raise ValueError("payload is not a whole number of elements")
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(-1, width)
    return arr.T.tobytes()


def byte_untranspose(payload: bytes, width: int) -> bytes:
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(width, -1)
    return arr.T.tobytes()


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    data: bytes
    #: [(ch_idx, level, block_idx, file_offset, compressed_size, uncompressed_size,
    #:   sample_count, start_timestamp, crc32)]
    blocks: list[tuple] = field(default_factory=list)
    layout: dict = field(default_factory=dict)


def build(spec: FileSpec) -> BuildResult:
    if spec.features & FEATURE_MOMENT_STREAM:
        raise ValueError(
            "feature bit 0 is derived from whether a moment stream is written, "
            "so a caller must not set it in FileSpec.features")
    #: Set by the block loop below the moment it writes a moment stream. The
    #: header's feature bit 0 is this and nothing else, so the bit cannot
    #: disagree with the framing.
    wrote_moments = False
    bf = spec.branching_factor
    block_samples = spec.block_samples or bf
    if block_samples <= 0 or block_samples % bf:
        raise ValueError("block_samples must be non-zero and a multiple of "
                         f"branching_factor, got {block_samples} with bf={bf}")
    idx_size = BLOCK_INDEX_ENTRY_SIZE_V1

    n_groups, n_channels = len(spec.groups), len(spec.channels)
    group_table_offset = HEADER_SIZE
    channel_table_offset = group_table_offset + n_groups * GROUP_ENTRY_SIZE
    data_start = channel_table_offset + n_channels * CHANNEL_ENTRY_SIZE

    # -- per-channel pyramids and blocks ----------------------------------
    blob = bytearray()
    per_channel = []
    for ch in spec.channels:
        grp = spec.groups[ch.group_id]
        raw = np.ascontiguousarray(ch.data)
        n_raw = len(raw)
        num_levels = (ch.num_levels_override if ch.num_levels_override is not None
                      else compute_num_levels(n_raw, bf))

        if n_raw == 0:
            per_channel.append({"levels": [], "num_levels": max(num_levels, 1)})
            continue

        levels, raw_idx, moments = build_pyramid(raw, bf, ch.aggregation_mode)

        def t_of_raw(i):
            if grp.timing_mode == 0:
                return sample_time_ns(grp.start_timestamp, grp.sample_rate, int(i))
            return int(grp.timestamps[int(i)])

        level_blocks = []
        for level in range(min(len(levels), num_levels)):
            arr = levels[level]
            n_units = len(arr) if level == 0 else arr.shape[0]
            span = bf ** level          # raw samples per bucket at this level
            blocks = []
            for b0 in range(0, n_units, block_samples):
                b1 = min(b0 + block_samples, n_units)
                chunk = np.ascontiguousarray(arr[b0:b1])
                count = b1 - b0
                first_raw = b0 * span
                start_ts = t_of_raw(min(first_raw, n_raw - 1))

                ts_payload = None
                ts_columns = 0
                if level == 0:
                    if grp.timing_mode == 1:
                        stamps = np.ascontiguousarray(
                            grp.timestamps[b0:b1].astype(np.int64))
                        ts_payload, ts_columns = stamps.tobytes(), 1
                    values_payload = chunk.tobytes()
                elif ch.aggregation_mode == 0:
                    # Two stamps at fixed rate and five at variable, in the
                    # values' column order. first, mid and last need none at
                    # fixed rate: the geometry names their raw indices and the
                    # time rule gives the time of an index. At variable rate
                    # every stamp is the stored stamp of the sample its column
                    # names, looked up and never computed.
                    named = (("min", "max") if grp.timing_mode == 0 else
                             ("min", "max", "first", "mid", "last"))
                    cols = np.array(
                        [[t_of_raw(raw_idx[level][name][j]) for name in named]
                         for j in range(b0, b1)], dtype=np.int64)
                    ts_payload = np.ascontiguousarray(cols).tobytes()
                    ts_columns = len(named)
                    values_payload = chunk.tobytes()
                else:
                    if grp.timing_mode == 1:
                        first = np.array([t_of_raw(min(j * span, n_raw - 1))
                                          for j in range(b0, b1)], dtype=np.int64)
                        ts_payload = np.ascontiguousarray(first).tobytes()
                        ts_columns = 1
                    values_payload = chunk.tobytes()

                uncompressed_size = len(values_payload)
                moments_payload = None
                if (level >= 1
                        and ch.aggregation_mode == 0 and spec.moments):
                    moments_payload = np.ascontiguousarray(
                        moments[level][b0:b1]).tobytes()
                    wrote_moments = True

                def _r(kind: str) -> int:
                    return (spec.recipes or {}).get(kind, spec.recipe)

                block_streams = []
                if ts_payload is not None:
                    block_streams.append((ts_payload, _r("timestamps"), "int64"))
                block_streams.append(
                    (values_payload, _r("values"), chunk.dtype.name))
                if moments_payload is not None:
                    block_streams.append(
                        (moments_payload, _r("moments"), "float64"))
                body = _frame_v1(block_streams)

                offset = data_start + len(blob)
                blob += body
                crc = zlib.crc32(body) & 0xFFFFFFFF
                blocks.append({
                    "file_offset": offset,
                    "compressed_size": len(body),
                    "uncompressed_size": uncompressed_size,
                    "sample_count": count,
                    "start_timestamp": start_ts,
                    "crc32": crc,
                    "ts_columns": ts_columns,
                })
            level_blocks.append(blocks)

        while len(level_blocks) < num_levels:
            level_blocks.append([])
        per_channel.append({"levels": level_blocks, "num_levels": num_levels})

    # -- block index arrays, then level tables ----------------------------
    index_region = bytearray()
    index_start = data_start + len(blob)
    for ch_state in per_channel:
        for blocks in ch_state["levels"]:
            ch_state.setdefault("index_offsets", []).append(
                index_start + len(index_region))
            for b in blocks:
                index_region += struct.pack(
                    BLOCK_INDEX_FMT_V1, b["file_offset"], b["compressed_size"],
                    b["uncompressed_size"], b["sample_count"],
                    b["start_timestamp"], b["crc32"], 0)

    level_region = bytearray()
    level_start = index_start + len(index_region)
    for ch_state in per_channel:
        ch_state["level_table_offset"] = level_start + len(level_region)
        offsets = ch_state.get("index_offsets", [])
        for li in range(ch_state["num_levels"]):
            blocks = ch_state["levels"][li] if li < len(ch_state["levels"]) else []
            count = len(blocks)
            off = offsets[li] if li < len(offsets) else index_start
            level_region += struct.pack(LEVEL_ENTRY_FMT, count, count, off)

    # -- tables ------------------------------------------------------------
    group_table = bytearray()
    for gi, g in enumerate(spec.groups):
        total = g.total_samples
        if total is None:
            total = max((len(c.data) for c in spec.channels if c.group_id == gi),
                        default=0)
        group_table += struct.pack(
            GROUP_ENTRY_FMT, g.sample_rate, g.start_timestamp, total,
            g.timing_mode, g.timing_flags, b"\x00" * 6,
            g.tick_rate_numer, g.tick_rate_denom, g.anchor_ticks, b"\x00" * 8)

    channel_table = bytearray()
    for ch, ch_state in zip(spec.channels, per_channel):
        channel_table += struct.pack(
            CHANNEL_ENTRY_FMT,
            ch.name.encode("utf-8").ljust(64, b"\x00")[:64],
            DTYPE_ENUM[str(np.dtype(ch.data.dtype).name)],
            ch.aggregation_mode, ch.group_id, ch_state["num_levels"],
            ch_state["level_table_offset"],
            ch.unit.encode("utf-8").ljust(16, b"\x00")[:16],
            ch.scaling_type, b"\x00",
            struct.pack("<dd", ch.scaling_gain, ch.scaling_offset),
            ch.calibration_id.encode("utf-8").ljust(32, b"\x00")[:32],
            b"\x00" * 14)

    header = struct.pack(
        HEADER_FMT_V1, MAGIC, 1, bf, spec.compression_id, spec.file_state,
        n_groups, n_channels, block_samples,
        group_table_offset, channel_table_offset,
        spec.session_id, spec.sequence_number, spec.prev_file_hash,
        spec.features | (FEATURE_MOMENT_STREAM if wrote_moments else 0),
        b"\x00" * 28)

    data = bytes(header) + bytes(group_table) + bytes(channel_table) \
        + bytes(blob) + bytes(index_region) + bytes(level_region)

    flat = []
    for ci, ch_state in enumerate(per_channel):
        for li, blocks in enumerate(ch_state["levels"]):
            for bi, b in enumerate(blocks):
                flat.append((ci, li, bi, b))

    return BuildResult(
        data=data,
        blocks=flat,
        layout={
            "header_size": HEADER_SIZE,
            "group_table_offset": group_table_offset,
            "channel_table_offset": channel_table_offset,
            "data_start": data_start,
            "index_start": index_start,
            "level_start": level_start,
            "block_index_entry_size": idx_size,
            "total_size": len(data),
            "version": 1,
            "branching_factor": bf,
            "block_samples": block_samples,
            # Per-channel, so a negative vector can address a level table entry
            # or a block index entry by (channel, level, block) rather than by
            # a magic number computed at the call site.
            "channels": [
                {
                    "num_levels": st["num_levels"],
                    "level_table_offset": st["level_table_offset"],
                    "index_offsets": st.get("index_offsets", []),
                    "block_counts": [len(b) for b in st["levels"]],
                }
                for st in per_channel
            ],
        },
    )


# ---------------------------------------------------------------------------
# Field addressing — where a negative vector patches
# ---------------------------------------------------------------------------

#: field -> (offset within the record, struct code)
HEADER_FIELDS = {
    "magic": (0, "6s"), "version": (6, "H"), "branching_factor": (8, "I"),
    "compression_id": (12, "B"), "file_state": (13, "B"), "num_groups": (14, "H"),
    "num_channels": (16, "I"), "block_samples": (20, "I"),
    "channel_dict_version": (20, "I"),
    "group_table_offset": (24, "Q"), "channel_table_offset": (32, "Q"),
    "session_id": (40, "16s"), "sequence_number": (56, "I"),
    "prev_file_hash": (60, "32s"), "features": (92, "Q"),
}

GROUP_FIELDS = {
    "sample_rate": (0, "d"), "start_timestamp": (8, "q"), "total_samples": (16, "Q"),
    "timing_mode": (24, "B"), "timing_flags": (25, "B"),
    "tick_rate_numer": (32, "Q"), "tick_rate_denom": (40, "Q"),
    "anchor_ticks": (48, "q"),
}

CHANNEL_FIELDS = {
    "name": (0, "64s"), "dtype": (64, "B"), "aggregation_mode": (65, "B"),
    "group_id": (66, "H"), "num_levels": (68, "I"), "level_table_offset": (72, "Q"),
    "unit": (80, "16s"), "scaling_type": (96, "B"), "scaling_params": (98, "16s"),
    "calibration_id": (114, "32s"),
}

LEVEL_FIELDS = {
    "block_count": (0, "Q"), "allocated_blocks": (8, "Q"),
    "block_index_offset": (16, "Q"),
}

BLOCK_FIELDS = {
    "file_offset": (0, "Q"), "compressed_size": (8, "Q"),
    "uncompressed_size": (16, "Q"), "sample_count": (24, "Q"),
    "start_timestamp": (32, "q"), "crc32": (40, "I"),
}


def patch(data: bytes, offset: int, code: str, value) -> bytes:
    """Return `data` with one field overwritten. The rest is untouched."""
    raw = struct.pack("<" + code, value)
    return data[:offset] + raw + data[offset + len(raw):]


def header_offset(fieldname: str) -> int:
    return HEADER_FIELDS[fieldname][0]


def group_offset(layout: dict, group_index: int, fieldname: str) -> int:
    off, _ = GROUP_FIELDS[fieldname]
    return layout["group_table_offset"] + group_index * GROUP_ENTRY_SIZE + off


def channel_offset(layout: dict, channel_index: int, fieldname: str) -> int:
    off, _ = CHANNEL_FIELDS[fieldname]
    return layout["channel_table_offset"] + channel_index * CHANNEL_ENTRY_SIZE + off


def level_offset(layout: dict, channel_index: int, level: int, fieldname: str) -> int:
    off, _ = LEVEL_FIELDS[fieldname]
    ch = layout["channels"][channel_index]
    return ch["level_table_offset"] + level * LEVEL_ENTRY_SIZE + off


def block_offset(layout: dict, channel_index: int, level: int, block: int,
                 fieldname: str) -> int:
    off, _ = BLOCK_FIELDS[fieldname]
    ch = layout["channels"][channel_index]
    entry = BLOCK_INDEX_ENTRY_SIZE_V1
    return ch["index_offsets"][level] + block * entry + off


def encode_stream_typed(payload: bytes, recipe: int, dtype: str) -> bytes:
    """`[recipe][payload]` for any registered recipe, given the stream's dtype.

    The dtype is what the two width-aware recipes need: byte-transpose has to
    know the element width, and pco takes typed values rather than bytes.
    """
    if recipe in (RECIPE_IDENTITY, RECIPE_ZSTD):
        return encode_stream_v1(payload, recipe)
    width = np.dtype(dtype).itemsize
    if recipe == RECIPE_TRANSPOSE_ZSTD:
        import zstandard
        c = zstandard.ZstdCompressor(write_content_size=True, write_checksum=False)
        return bytes([recipe]) + c.compress(byte_transpose(payload, width))
    if recipe == RECIPE_PCO:
        from pcodec import ChunkConfig, standalone
        arr = np.frombuffer(payload, dtype=np.dtype(dtype))
        if arr.size == 0 or width == 1:
            raise ValueError("pco takes neither an empty array nor an 8-bit dtype")
        return bytes([recipe]) + standalone.simple_compress(
            np.ascontiguousarray(arr), ChunkConfig())
    raise ValueError(f"recipe {recipe:#04x} is not encodable")


def _frame_v1(streams_in: list) -> bytes:
    """v1 framing: `(payload, recipe, dtype)` per stream, in wire order.

    **Every stream except the last is prefixed by its own byte length as u32 LE,
    counting its recipe byte** — the same rule `ts_len` already followed. The
    last stream runs to `compressed_size` and needs no prefix. Stream order is
    fixed by block kind: timestamps, then values, then moments.

    So a three-stream block is
    `[len_ts u32][recipe][ts][len_vals u32][recipe][vals][recipe][moments]`,
    and the two-stream form is the same rule with one fewer stream — which is
    why a two-stream block's `ts_len` means what it means.

    Each stream carries its own recipe byte, so a file may mix them; that is
    the whole of profile 2, and nothing about the framing changes with it.
    """
    streams = [encode_stream_typed(payload, recipe, dtype)
               for payload, recipe, dtype in streams_in]
    out = b""
    for stream in streams[:-1]:
        out += struct.pack("<I", len(stream)) + stream
    return out + streams[-1]


def sha256_of(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()
