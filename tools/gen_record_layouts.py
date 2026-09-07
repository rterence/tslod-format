"""The five record layouts, byte for byte, and the wire enums.

This generator emits two things per record:

  1. the field table — offset, width, wire type, name — as data an
     implementation can check its own struct against;
  2. a golden encoding — a fully populated record with every field set to a
     distinguishable value, and the exact bytes it must produce, hex-encoded.

The golden encoding is the part that catches what a field table cannot: a
struct whose fields are all correctly named and whose padding is wrong.
"""

from __future__ import annotations

import struct
import uuid

from _corpus import Vector


class WIRE:
    """The wire enums, as the format's own constants."""
    DTYPE_ENUM = {"float32": 0, "float64": 1, "int8": 2, "int16": 3, "int32": 4,
                  "int64": 5, "uint8": 6, "uint16": 7, "uint32": 8, "uint64": 9}
    DTYPE_SIZES = {"float32": 4, "float64": 8, "int8": 1, "int16": 2, "int32": 4,
                   "int64": 8, "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8}
    AGGREGATION_MODE_ENUM = {"numeric": 0, "bitfield": 1}
    TIMING_MODE_ENUM = {"fixed": 0, "variable": 1}
    FILE_STATE_ENUM = {"sealed": 0, "active": 1}
    SCALING_TYPE_ENUM = {"identity": 0, "linear": 1}
    TIMING_FLAG_TIMEBASE_PRESENT = 0x01
    TIMING_FLAG_EPOCH_SYNCED = 0x02


def hexs(b: bytes) -> str:
    return b.hex().upper()


# ---------------------------------------------------------------------------
# v1 — the version rules, stated here as the layout, with the delta named
# ---------------------------------------------------------------------------

V1_HEADER_FMT = "<6sHIBBHIIQQ16sI32sQ28s"
V1_GROUP_ENTRY_FMT = "<dqQBB6sQQq8s"
V1_CHANNEL_ENTRY_FMT = "<64sBBHIQ16sB1s16s32s14s"
V1_LEVEL_ENTRY_FMT = "<QQQ"
V1_BLOCK_INDEX_ENTRY_FMT = "<QQQQqII"

V1_RECORDS = {
    "header": {
        "fmt": V1_HEADER_FMT,
        "size": struct.calcsize(V1_HEADER_FMT),
        "fields": [
            (0,   6, "bytes",  "magic", "b'TSLOD\\x00'"),
            (6,   2, "uint16", "version", "1"),
            (8,   4, "uint32", "branching_factor", ">= 2"),
            (12,  1, "uint8",  "compression_id",
             "v1: the file PROFILE. 0 = none (every recipe byte must be 0x00), "
             "2 = recipe. Every other value is reserved and rejected"),
            (13,  1, "uint8",  "file_state", "0 sealed, 1 active"),
            (14,  2, "uint16", "num_groups", ""),
            (16,  4, "uint32", "num_channels", ""),
            (20,  4, "uint32", "block_samples",
             "buckets per block; must be non-zero and a multiple of "
             "branching_factor, and defaults to it"),
            (24,  8, "uint64", "group_table_offset", ""),
            (32,  8, "uint64", "channel_table_offset", ""),
            (40, 16, "bytes",  "session_id", "the 16 RFC 4122 bytes in network order — the one exception to little-endian"),
            (56,  4, "uint32", "sequence_number", ""),
            (60, 32, "bytes",  "prev_file_hash", "SHA-256 over the ENTIRE previous file"),
            (92,  8, "uint64", "features",
             "low 32 bits must-understand (an unknown one is fatal), high 32 "
             "may-ignore (an unknown one is skipped)"),
            (100, 28, "bytes", "reserved", "written zero, ignored on read"),
        ],
    },
    "group_entry": {
        "fmt": V1_GROUP_ENTRY_FMT,
        "size": struct.calcsize(V1_GROUP_ENTRY_FMT),
        "fields": [
            (0,   8, "float64", "sample_rate", "FROZEN offset"),
            (8,   8, "int64",   "start_timestamp", "FROZEN offset; writer patches in place"),
            (16,  8, "uint64",  "total_samples", "FROZEN offset; writer patches in place"),
            (24,  1, "uint8",   "timing_mode", "FROZEN offset; 0 fixed, 1 variable"),
            (25,  1, "uint8",   "timing_flags", "bit 0 TIMEBASE_PRESENT, bit 1 EPOCH_SYNCED"),
            (26,  6, "bytes",   "reserved", "written zero"),
            (32,  8, "uint64",  "tick_rate_numer", ">= 1 when TIMEBASE_PRESENT"),
            (40,  8, "uint64",  "tick_rate_denom", ">= 1 when TIMEBASE_PRESENT"),
            (48,  8, "int64",   "anchor_ticks", ""),
            (56,  8, "bytes",   "reserved2", "written zero"),
        ],
    },
    "channel_entry": {
        "fmt": V1_CHANNEL_ENTRY_FMT,
        "size": struct.calcsize(V1_CHANNEL_ENTRY_FMT),
        "fields": [
            (0,   64, "bytes",  "name", "UTF-8, NUL-padded; invalid UTF-8 rejected at open"),
            (64,   1, "uint8",  "dtype", "wire codes 0..9"),
            (65,   1, "uint8",  "aggregation_mode", "0 numeric, 1 bitfield"),
            (66,   2, "uint16", "group_id", ""),
            (68,   4, "uint32", "num_levels", "counts level 0"),
            (72,   8, "uint64", "level_table_offset", ""),
            (80,  16, "bytes",  "unit", "UTF-8, NUL-padded"),
            (96,   1, "uint8",  "scaling_type", "0 identity, 1 linear"),
            (97,   1, "bytes",  "_pad_scaling", "explicit pad, never checked"),
            (98,  16, "bytes",  "scaling_params", "float64 gain, float64 offset when linear"),
            (114, 32, "bytes",  "calibration_id", "UTF-8, NUL-padded"),
            (146, 14, "bytes",  "reserved", "written zero"),
        ],
    },
    "level_entry": {
        "fmt": V1_LEVEL_ENTRY_FMT,
        "size": struct.calcsize(V1_LEVEL_ENTRY_FMT),
        "fields": [
            (0,  8, "uint64", "block_count", ""),
            (8,  8, "uint64", "allocated_blocks", "block_count > allocated_blocks is rejected"),
            (16, 8, "uint64", "block_index_offset", ""),
        ],
    },
    "block_index_entry": {
        "fmt": V1_BLOCK_INDEX_ENTRY_FMT,
        "size": struct.calcsize(V1_BLOCK_INDEX_ENTRY_FMT),
        "fields": [
            (0,  8, "uint64", "file_offset", ""),
            (8,  8, "uint64", "compressed_size",
             "the WHOLE block as stored — framing, recipe bytes, payloads"),
            (16, 8, "uint64", "uncompressed_size",
             "the VALUES stream's decoded bytes only; a timestamp stream's "
             "decoded size is sample_count * columns * 8"),
            (24, 8, "uint64", "sample_count", "zero is rejected"),
            (32, 8, "int64",  "start_timestamp", ""),
            (40, 4, "uint32", "crc32",
             "IEEE CRC-32 as zlib computes it, over exactly "
             "[file_offset, file_offset + compressed_size), checked BEFORE decode"),
            (44, 4, "uint32", "reserved", "written zero, ignored on read"),
        ],
    },
}

V1_GOLDEN = {
    "header": (
        b"TSLOD\x00", 1, 256, 0, 0, 3, 82, 256,
        128, 128 + 3 * 64,
        bytes(range(0x10, 0x20)), 7,
        bytes(range(0x20, 0x40)),
        0, b"\x00" * 28,
    ),
    "group_entry": (
        24_000_000.0, 1_700_000_000_000_000_000, 2_916_533, 1, 0x03,
        b"\x00" * 6, 24_000_000, 1, -123_456, b"\x00" * 8,
    ),
    "channel_entry": (
        b"motor_vib_x".ljust(64, b"\x00"), 0, 0, 2, 3, 4096,
        b"m/s^2".ljust(16, b"\x00"), 1, b"\x00",
        struct.pack("<dd", 0.001953125, -12.5), b"cal-2026-09".ljust(32, b"\x00"),
        b"\x00" * 14,
    ),
    "level_entry": (11_401, 16_384, 1_048_576),
    "block_index_entry": (
        4_294_967_296, 1_040, 2_048, 256, 1_700_000_000_000_000_000,
        0xCBF43926, 0,
    ),
}


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------


def _layout_cases(v: Vector, records: dict, golden: dict, version: int) -> None:
    for name, rec in records.items():
        packed = struct.pack(rec["fmt"], *golden[name]) if name in golden else None
        if packed is not None:
            assert len(packed) == rec["size"], (name, len(packed), rec["size"])
        # The field table must actually tile the record with no gap and no overlap.
        cursor = 0
        for off, width, _t, fname, _n in rec["fields"]:
            assert off == cursor, (
                f"{name}.{fname}: field table has a gap or overlap at offset {off} "
                f"(expected {cursor})"
            )
            cursor += width
        assert cursor == rec["size"], (
            f"{name}: field table covers {cursor} bytes, record is {rec['size']}"
        )
        case = dict(
            record=name,
            format_version=version,
            struct_format=rec["fmt"],
            size_bytes=rec["size"],
            byte_order="little",
            fields=[
                {"offset": off, "width": width, "type": t, "name": fname, "note": note}
                for off, width, t, fname, note in rec["fields"]
            ],
        )
        if packed is not None:
            case["golden_encoding_hex"] = hexs(packed)
            case["golden_encoding_len"] = len(packed)
            # session_id is the one field that is NOT little-endian: it is the
            # 16 RFC 4122 bytes in network order. Stating the same value as a
            # text UUID makes that checkable — the text parses to network-order
            # bytes by definition, so a reader can compare two derivations
            # instead of taking the byte order on trust.
            for off, width, _t, fname, _n in rec["fields"]:
                if fname == "session_id":
                    assert width == 16, "session_id is 16 bytes"
                    case["session_id_text"] = str(
                        uuid.UUID(bytes=packed[off:off + width]))
                    break
        v.case(f"v{version}/{name}", **case)


def gen_v1_layouts() -> Vector:
    v = Vector(
        id="record-layouts-v1",
        set_="record-layouts",
        kind="fixture",
        asserts=(
            "The five records are exactly 128, 64, 160, 24 and 48 bytes, little-endian "
            "throughout except session_id which is the 16 RFC 4122 bytes in network "
            "order, and each field sits at the offset given — proved by a golden "
            "encoding and not only by a field table, so a struct with the right names "
            "and the wrong padding fails."
        ),
        source=(
            "the version-1 record definitions, with a golden encoding produced from "
            "them"
        ),
        contract=(
            "A struct with the right field names and the wrong padding reads "
            "plausible nonsense, so the layout is pinned as bytes and not only as a "
            "table."
        ),
        requires=["format:v1"],
        notes=(
            "channel_dict_version is DROPPED, not reserved: it was written 0 and read by "
            "nothing (it was written zero and read by nothing), so offset 20 is reused rather than left dead. A v1 "
            "reader rejects block_samples that is zero or not a multiple of "
            "branching_factor, and rejects any must-understand feature bit it does not know."
        ),
    )
    _layout_cases(v, V1_RECORDS, V1_GOLDEN, 1)
    return v


def gen_enums() -> Vector:
    v = Vector(
        id="record-enums",
        set_="record-layouts",
        kind="fixture",
        asserts=(
            "The wire enums are closed sets with these exact codes: ten dtypes 0..9, two "
            "aggregation modes, two timing modes, two timing-flag bits, two file states, "
            "two scaling types, and the compression/profile codes; an unrecognised value "
            "in any of them is a read-time rejection, never a default."
        ),
        source=(
            "the enum definitions in spec/v1, cross-checked against the dtype widths "
            "and the recipe registry the rest of the corpus uses"
        ),
        contract=(
            "An enum is a closed set: a reader that defaults an unknown value reads "
            "a file its writer never wrote."
        ),
        requires=[],
    )
    for name, mapping, note in [
        ("dtype", WIRE.DTYPE_ENUM,
         "the wire dtype enum: an implementation's dtype type carries these exact codes"),
        ("aggregation_mode", WIRE.AGGREGATION_MODE_ENUM,
         "bitfield mode is rejected at read time on a float dtype"),
        ("timing_mode", WIRE.TIMING_MODE_ENUM,
         "timing is a property of the GROUP, not the block"),
        ("file_state", WIRE.FILE_STATE_ENUM,
         "one byte at header offset 13; there is NO per-block open/closed flag"),
        ("scaling_type", WIRE.SCALING_TYPE_ENUM,
         "a linear channel stores float64 gain then float64 offset in scaling_params"),
    ]:
        v.case(f"enum/{name}", enum=name,
               values={k: int(val) for k, val in mapping.items()},
               closed=True, unknown_value_is_rejected=True, note=note)

    v.case("enum/timing_flags", enum="timing_flags",
           values={"TIMEBASE_PRESENT": WIRE.TIMING_FLAG_TIMEBASE_PRESENT,
                   "EPOCH_SYNCED": WIRE.TIMING_FLAG_EPOCH_SYNCED},
           reserved_bits="2-7", reserved_bits_must_be_zero=True,
           unknown_value_is_rejected=True,
           note="a set reserved bit is rejected at open")

    v.case("enum/compression_id-v1-profile", enum="compression_id", format_version=1,
           values={"none": 0, "recipe": 2},
           closed=True, unknown_value_is_rejected=True,
           note="v1 reinterprets the field as the file PROFILE: 0 = none, where every "
                "recipe byte must be 0x00 (the interchange and conformance profile); "
                "2 = recipe. Every other value, 1 included, is reserved and rejected")

    v.case("enum/recipe-registry-v1", enum="recipe", format_version=1,
           values={"identity": 0x00, "zstd": 0x01, "pco": 0x02, "transpose_zstd": 0x03},
           experimental_range="0xF0-0xFF",
           experimental_never_in_a_released_file=True,
           closed=True, unknown_value_is_rejected=True,
           note="one recipe byte per stream. Every other value is reserved and rejected "
                "with an actionable error naming the byte; there is no registry escape")

    v.case("dtype-widths", table={k: int(val) for k, val in WIRE.DTYPE_SIZES.items()},
           note="byte width per dtype; a bucket tuple is 4 * width, a position pair is 16")
    return v


def main() -> None:
    for factory in (gen_v1_layouts, gen_enums):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
