"""Codec conformance vectors — one per registered recipe.

**The assertion is `decode(vector) == expected payload`, and never encoder
byte-identity.** An encoder is round-tripped through the decoder; two
conforming encoders may emit different bytes for the same input and both be
right. So every vector here carries *compressed bytes as its input* and the
*decoded payload as its expected output*, and an implementation passes by
decoding — which is the only half of a codec a format can require.

The block classes are the shapes real data actually takes, plus the degenerate
ones a writer emits at the edges of a channel:

  * `constant` — a held signal; every codec's best case
  * `ramp` — a monotone counter, which is what a timestamp stream looks like
    once v1 stops delta-encoding it
  * `noise` — high entropy, every codec's worst case
  * `all-nan` — a fully invalid bucket run
  * `single-sample` — a one-element tail block

Expected payloads cross as bit patterns, so a NaN keeps the
payload a float round-trip would not preserve.
"""

from __future__ import annotations

import numpy as np
import zstandard
from pcodec import ChunkConfig, standalone

from _corpus import Vector, array_ref, u64

SET = "codec-vectors"

RECIPES = {0x00: "identity", 0x01: "zstd", 0x02: "pco", 0x03: "transpose_zstd"}

#: the framing rules's writer policy table at v1. The reader is total over the registry
#: regardless of policy, which is why every class below is encoded under EVERY
#: recipe that accepts it and not only under its policy choice.
POLICY = {
    "int8": 0x01, "uint8": 0x01, "uint16": 0x01,
    "int16": 0x02, "int32": 0x02, "int64": 0x02,
    "uint32": 0x02, "uint64": 0x02, "float32": 0x02, "float64": 0x02,
}

PCO_DTYPES = {"int16", "int32", "int64", "uint16", "uint32", "uint64",
              "float32", "float64"}


def zstd_compress(payload: bytes) -> bytes:
    """the framing rules: a standard zstd frame, content size on, checksum off, no dictionary."""
    c = zstandard.ZstdCompressor(level=3, write_content_size=True,
                                 write_checksum=False)
    return c.compress(payload)


def byte_transpose(payload: bytes, width: int) -> bytes:
    """Byte plane i of every element before plane i+1."""
    return np.frombuffer(payload, dtype=np.uint8).reshape(-1, width).T.tobytes()


def byte_untranspose(payload: bytes, width: int) -> bytes:
    return np.frombuffer(payload, dtype=np.uint8).reshape(width, -1).T.tobytes()


def encode(arr: np.ndarray, recipe: int) -> bytes | None:
    """`[recipe][payload]`, or None when the recipe cannot take this array."""
    payload = np.ascontiguousarray(arr).tobytes()
    if recipe == 0x00:
        return bytes([recipe]) + payload
    if recipe == 0x01:
        return bytes([recipe]) + zstd_compress(payload)
    if recipe == 0x03:
        return bytes([recipe]) + zstd_compress(
            byte_transpose(payload, arr.dtype.itemsize))
    if recipe == 0x02:
        if arr.dtype.name not in PCO_DTYPES or arr.size == 0:
            return None
        return bytes([recipe]) + standalone.simple_compress(
            np.ascontiguousarray(arr), ChunkConfig())
    raise ValueError(recipe)


def decode(stream: bytes, dtype: str, count: int) -> np.ndarray:
    """The reference decode, used here to VERIFY every vector before writing it."""
    recipe, payload = stream[0], stream[1:]
    width = np.dtype(dtype).itemsize
    if recipe == 0x00:
        raw = payload
    elif recipe == 0x01:
        raw = zstandard.ZstdDecompressor().decompress(payload)
    elif recipe == 0x03:
        raw = byte_untranspose(
            zstandard.ZstdDecompressor().decompress(payload), width)
    elif recipe == 0x02:
        return standalone.simple_decompress(payload)
    else:
        raise ValueError(f"unknown recipe {recipe:#04x}")
    return np.frombuffer(raw, dtype=np.dtype(dtype), count=count)


def block_classes():
    """(name, array) over the classes and dtypes v1 actually writes."""
    out = []
    n = 256
    for dtype in POLICY:
        dt = np.dtype(dtype)
        if dt.kind == "f":
            const = np.full(n, 3.25, dtype=dtype)
            ramp = (np.arange(n, dtype=np.float64) * 0.125).astype(dtype)
            state = 0x9E3779B97F4A7C15
            vals = []
            for _ in range(n):
                state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
                vals.append((state >> 11) % 2_000_001 / 1000.0 - 1000.0)
            noise = np.array(vals, dtype=dtype)
            allnan = np.full(n, np.nan, dtype=dtype)
            single = np.array([np.nan], dtype=dtype)
            out += [(f"{dtype}/constant", np.ascontiguousarray(const)),
                    (f"{dtype}/ramp", np.ascontiguousarray(ramp)),
                    (f"{dtype}/noise", np.ascontiguousarray(noise)),
                    (f"{dtype}/all-nan", np.ascontiguousarray(allnan)),
                    (f"{dtype}/single-sample-nan", np.ascontiguousarray(single))]
        else:
            info = np.iinfo(dt)
            const = np.full(n, 7 if info.min == 0 else -7, dtype=dtype)
            # `info.max + 1` is 2**63 for int64 and 2**64 for uint64, neither of
            # which fits the C long numpy would need to take the modulus in.
            span = min(int(info.max), n * 8 - 1) + 1
            ramp = np.ascontiguousarray(
                (np.arange(n, dtype=np.int64) % span).astype(dtype))
            state = 0x9E3779B97F4A7C15
            vals = []
            width_bits = dt.itemsize * 8
            for _ in range(n):
                state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
                vals.append(state & ((1 << width_bits) - 1))
            noise = np.array(vals, dtype=f"uint{width_bits}").view(dtype).copy()
            extremes = np.ascontiguousarray(
                np.array([info.min, info.max] * (n // 2), dtype=dtype))
            out += [(f"{dtype}/constant", np.ascontiguousarray(const)),
                    (f"{dtype}/ramp", ramp),
                    (f"{dtype}/noise", np.ascontiguousarray(noise)),
                    (f"{dtype}/extremes", extremes),
                    (f"{dtype}/single-sample", np.ascontiguousarray(
                        np.array([info.max], dtype=dtype)))]

    # ⛔ NO real capture data here, deliberately. An earlier revision drew three
    # classes from production files, which put capture payloads and capture
    # NAMES into a vector set intended to be publishable. The synthetic classes
    # above bracket the codecs' behaviour; real-data codec vectors live beside
    # a real recording, which this repository does not ship.

    # A timestamp stream, which version 1 stores ABSOLUTE.
    base = 1_700_000_000_000_000_000
    out.append(("int64/timestamps-absolute-fixed-rate", np.ascontiguousarray(
        base + np.arange(n, dtype=np.int64) * 1_000_000)))
    out.append(("int64/timestamps-absolute-jittered", np.ascontiguousarray(
        base + np.arange(n, dtype=np.int64) * 1_000_000
        + (np.arange(n, dtype=np.int64) % 13) * 971)))
    return out


def gen() -> Vector:
    v = Vector(
        id="codec-decode-per-recipe",
        set_=SET,
        kind="fixture",
        asserts=(
            "For every registered v1 recipe — 0x00 identity, 0x01 zstd, 0x02 pco, "
            "0x03 byte-transpose+zstd — decoding the stored bytes yields exactly the "
            "expected payload, bit for bit. Encoder output is NEVER compared; two "
            "conforming encoders may differ and both be right."
        ),
        source=(
            "encoded with pcodec 1.0.3 (standalone format) and zstandard (level 3, "
            "content size on, checksum off); every vector is decoded back and "
            "checked bit-exact before it is written"
        ),
        contract=(
            "A file written by one implementation has to decode in another, so what "
            "a recipe byte means is part of the format while how an encoder chooses "
            "to use it is not."
        ),
        requires=["format:v1", "recipe:0x00", "recipe:0x01",
                  "recipe:0x02", "recipe:0x03"],
        notes=(
            "The recipe byte is part of the stream, so `stream_hex` includes it and the "
            "payload begins at offset 1. `pco` is the STANDALONE format — the "
            "self-describing one — never the wrapped format, on the 1.x compatibility "
            "line. pco declines empty arrays and the two 8-bit dtypes, and those "
            "cases are recorded as `not_encodable` rather than omitted, so a reader is "
            "told which combinations it will never see."
        ),
    )

    for name, arr in block_classes():
        dtype = arr.dtype.name
        payload = arr.tobytes()
        for recipe, rname in RECIPES.items():
            stream = encode(arr, recipe)
            if stream is None:
                v.case(f"{name}/{rname}",
                       block_class=name, dtype=dtype, recipe=f"0x{recipe:02X}",
                       recipe_name=rname, element_count=u64(arr.size),
                       not_encodable=True,
                       reason=("pco takes neither an empty array nor the 8-bit dtypes; "
                               "the writer's policy table sends int8/uint8/uint16 to zstd"))
                continue
            got = decode(stream, dtype, arr.size)
            assert np.ascontiguousarray(got).view(np.uint8).tobytes() == payload, (
                f"{name}/{rname}: the reference decode does not reproduce the payload")
            v.case(
                f"{name}/{rname}",
                block_class=name, dtype=dtype,
                recipe=f"0x{recipe:02X}", recipe_name=rname,
                element_count=u64(arr.size),
                stream_hex=stream.hex().upper(),
                stream_length=len(stream),
                payload_offset=1,
                expected_payload=array_ref(arr, f"{SET}/{name.replace('/', '_')}_payload.bin"),
                expected_payload_bytes=u64(len(payload)),
                compression_ratio=round(len(payload) / max(1, len(stream) - 1), 4),
                is_policy_choice_for_this_dtype=(POLICY.get(dtype) == recipe),
            )

    v.case("registry",
           recipes={f"0x{k:02X}": val for k, val in RECIPES.items()},
           experimental_range="0xF0-0xFF",
           experimental_never_in_a_released_file=True,
           every_other_value="reserved and rejected with an error naming the byte",
           no_registry_escape=True,
           zstd_frame="standard frame, content size on, checksum off, no dictionary",
           pco_format="standalone (self-describing), 1.x compatibility line",
           byte_transpose="byte plane i of every element before plane i+1",
           policy_table=POLICY,
           note="the writer's policy table decides which recipe a stream GETS; the "
                "reader is total over the registry regardless of policy, which is why "
                "every class above is encoded under every recipe that accepts it")
    return v


def main() -> None:
    vector = gen()
    vector.write()
    print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
