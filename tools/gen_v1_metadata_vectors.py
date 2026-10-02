"""The metadata block vectors: may-ignore feature bit 32.

A version-1 file may carry one metadata block — a description per channel and
the writer's provenance — behind may-ignore feature bit 32, with its offset
and length in the first 16 of the header's 28 reserved bytes. These vectors
pin it from both sides:

  * the conformance files: a block, a block with every optional member absent,
    a block exactly at the block bound, the bit clear with the pointer bytes
    non-zero, an unknown may-ignore bit beside the block, and an unknown member
    under an unknown may-ignore bit;
  * the negatives: one case or more per rejection class, cases with two
    defects that pin the order the block is judged in, and cases where a block
    defect meets a file defect, which pin where the block sits in the reader's
    order.

**A defect in the block refuses the block, not the file.** Every block-refused
case here is a file a reader opens and serves with no description and no
provenance. The negatives bind readers that implement bit 32; a reader that
does not implement it ignores the bit, the pointer and the block, and reads
every one of these files.

Every file is profile 0, so a reader needs only struct, zlib and a UTF-8
decoder to run the whole set.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

import _tslod_build as B
from _corpus import VECTORS, Vector, u64

SET = "v1-format"
FILES = f"{SET}/files"
TS = 1_700_000_000_000_000_000
MEBIBYTE = 1 << 20


def ramp(dtype: str, n: int, seed: int = 0) -> np.ndarray:
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return np.ascontiguousarray(
            (((np.arange(n) + seed) % 97) * 1.5 - 40).astype(dtype))
    return np.ascontiguousarray(((np.arange(n) + seed) % 97).astype(dtype))


def _write(name: str, data: bytes) -> str:
    rel = f"{FILES}/{name}"
    path = VECTORS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: The sources' hashes are of fixed stand-in bytes. A source is named by its
#: bare file name and the SHA-256 of its whole bytes; what the bytes were is
#: the writer's business, and nothing here needs them.
SOURCE_LIST = {"name": "variables.csv", "sha256": _sha("variables.csv, stand-in")}
SOURCE_INPUT = {"name": "capture-0001.bin",
                "sha256": _sha("capture-0001.bin, stand-in")}

#: Exactly 1,024 UTF-8 bytes in far fewer code points: the description
#: bound's accepting side, measured in bytes, where a byte count and a
#: code-point count part company.
_HEAD = "Phase current, Ω-weighted \U0001d714 estimate: "
DESCRIPTION_AT_BOUND = _HEAD + "µ" * ((1024 - len(_HEAD.encode("utf-8"))) // 2)
assert len(DESCRIPTION_AT_BOUND.encode("utf-8")) == 1024
assert len(DESCRIPTION_AT_BOUND) < 1024
AT_BOUND_CODE_POINTS = len(DESCRIPTION_AT_BOUND)


def block_at_bound(names: list[str]) -> dict:
    """A block of exactly 1 MiB describing every channel in `names`, no
    description over 1,024 bytes: each starts at the bound, and descriptions
    from the last backwards are shortened until the block fits exactly."""
    descs = {n: "d" * 1024 for n in names}
    block = {"channels": descs, "provenance": MINIMAL_BLOCK["provenance"]}
    over = len(canonical(block)) - MEBIBYTE
    for n in reversed(names):
        if over <= 0:
            break
        cut = min(over, 1023)
        descs[n] = "d" * (1024 - cut)
        over -= cut
    assert len(canonical(block)) == MEBIBYTE, "the block is not exactly 1 MiB"
    return block

FULL_BLOCK = {
    "channels": {
        # A quote and a backslash, the two characters the form escapes, and
        # text outside ASCII, which it writes raw.
        "sig_a": ('Température du stator, "brute" avant étalonnage '
                  '\\ capteur 2'),
        "sig_b": DESCRIPTION_AT_BOUND,
        # sig_c has no key, so it has no description.
    },
    "provenance": {
        "writer": "example-writer",
        "writer_version": "1.4.0",
        "product": "Example drive",
        "firmware": "fw 2.3.1",
        "sources": [SOURCE_LIST, SOURCE_INPUT],
    },
}

MINIMAL_BLOCK = {
    "channels": {},
    "provenance": {"writer": "example-writer", "writer_version": "1.4.0",
                   "sources": [SOURCE_INPUT]},
}


def canonical(value) -> bytes:
    return B.canonical_json(value)


def base_spec(**kw) -> B.FileSpec:
    """Three channels in one fixed-rate group; sig_c is never described."""
    return B.FileSpec(
        compression_id=0, branching_factor=256, block_samples=256,
        groups=[B.GroupSpec(1000.0, TS)],
        channels=[B.ChannelSpec("sig_a", ramp("float32", 1024)),
                  B.ChannelSpec("sig_b", ramp("float32", 1024, 1)),
                  B.ChannelSpec("sig_c", ramp("int16", 512))],
        **kw)


def _with(block: dict, path: tuple, value) -> dict:
    """A deep copy of `block` with one member replaced (or removed, for
    value None)."""
    import copy
    out = copy.deepcopy(block)
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is None:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


def _pointer(data: bytes) -> tuple[int, int]:
    return struct.unpack_from("<QQ", data, B.METADATA_OFFSET_AT)


# ---------------------------------------------------------------------------
# The conformance files
# ---------------------------------------------------------------------------


def gen_conformance() -> Vector:
    v = Vector(
        id="v1-metadata-block-conformance-set",
        set_=SET,
        kind="fixture",
        asserts=(
            "A reader that implements feature bit 32 serves each of these files "
            "whole, and serves exactly the metadata block the case states: the "
            "descriptions and provenance where bit 32 is set and the block is "
            "valid, and none where bit 32 is clear, whatever the pointer bytes "
            "hold."
        ),
        source="written by tools/_tslod_build.py, which serialises the block itself",
        contract=(
            "The block has one canonical byte form, so one content has one byte "
            "string, and two readers that implement the bit must serve one file "
            "the same descriptions."
        ),
        requires=["format:v1", "profile:0", "recipe:0x00", "feature:crc",
                  "feature:metadata-block"],
        notes=(
            "metadata_hex is the block exactly as it lies in the file, so a "
            "writer's serialiser can be compared with these bytes. The members a "
            "reader serves are stated as decoded JSON values; a member the reader "
            "skips under an unknown may-ignore bit is not among them."
        ),
    )

    def emit(name, spec, served, note, **extra):
        result = B.build(spec)
        rel = _write(name, result.data)
        features = struct.unpack_from("<Q", result.data, 92)[0]
        offset, length = _pointer(result.data)
        body = dict(file=rel, file_size_bytes=u64(len(result.data)),
                    block_count=len(result.blocks), features=u64(features),
                    metadata_bit_set=bool(features & B.FEATURE_METADATA_BLOCK),
                    metadata_offset=u64(offset), metadata_length=u64(length))
        if served is None:
            body["metadata"] = "absent"
        else:
            raw = result.data[offset:offset + length]
            body.update(metadata="served",
                        metadata_sha256=hashlib.sha256(raw).hexdigest())
            if length <= 4096:
                # Small enough to carry whole: the bytes and what they say.
                body.update(metadata_hex=raw.hex().upper(), served=served)
        body.update(extra)
        body["note"] = note
        v.case(name, **body)
        return result

    full = emit(
        "v1_metadata_block.tslod", base_spec(metadata=FULL_BLOCK),
        FULL_BLOCK,
        "a block with every member: two of three channels described, sig_c "
        "not, so a channel with no key has no description. sig_a's text "
        "carries the quote and the backslash, the only two characters the "
        "form escapes, and text outside ASCII, which it writes raw. sig_b's is "
        f"exactly 1,024 UTF-8 bytes in {AT_BOUND_CODE_POINTS} code points, the "
        "accepting side of the description bound, which is counted in bytes. "
        "The block lies "
        "directly after the channel table and before the first data block",
        undescribed_channels=["sig_c"],
        description_utf8_bytes={"sig_a": len(FULL_BLOCK["channels"]["sig_a"]
                                             .encode("utf-8")),
                                "sig_b": 1024})
    lay = full.layout
    assert lay["metadata_extent"][0] == lay["channel_table_end"]

    emit("v1_metadata_block_minimal.tslod", base_spec(metadata=MINIMAL_BLOCK),
         MINIMAL_BLOCK,
         "every optional member absent: no channel described, no product, no "
         "firmware, one source. An unknown fact is an absent member and never "
         "a placeholder, so a writer that knows only its own name, its version "
         "and what it read writes exactly this")

    # The block bound's accepting side: a block of exactly 1 MiB. It takes
    # 1,024 channels to fill one without breaking the description bound, so
    # they are zero-sample channels, which cost a channel entry and a level
    # entry each and no data.
    names = [f"c{i:04d}" for i in range(1024)]
    channels = [B.ChannelSpec(n, np.zeros(0, dtype="float32")) for n in names]

    at_bound = block_at_bound(names)
    emit("v1_metadata_block_at_bound.tslod",
         B.FileSpec(compression_id=0, branching_factor=256,
                    groups=[B.GroupSpec(1000.0, TS)], channels=channels,
                    metadata=at_bound),
         at_bound,
         "a block of exactly 1,048,576 bytes, the block bound's accepting side. "
         "1,024 zero-sample channels, each described, so that no description "
         "passes its own bound of 1,024 bytes",
         block_utf8_bytes=u64(MEBIBYTE))

    # The bit is the only switch. Here the pointer bytes hold the extent of a
    # well-formed block that lies where a block belongs, and the bit is clear:
    # a reader that reads the pointer whenever it is non-zero serves a block
    # this file does not carry.
    emit("v1_metadata_bit_clear_pointer_set.tslod",
         base_spec(metadata=FULL_BLOCK, metadata_bit=False), None,
         "bit 32 clear, the pointer bytes non-zero, and a well-formed block at "
         "the offset they name. The bytes are reserved bytes while the bit is "
         "clear, so a reader neither reads nor checks them, and serves no "
         "block. The file opens")

    emit("v1_metadata_unknown_may_ignore_bit.tslod",
         base_spec(metadata=FULL_BLOCK, features=1 << 33), FULL_BLOCK,
         "bit 32 and bit 33 set. Bit 33 is may-ignore and assigned nothing, so "
         "a reader skips it, reads the file and serves the block")

    later = _with(_with(FULL_BLOCK, ("annotations",), {"x": ["later"]}),
                  ("provenance", "site"), "bench 4")
    emit("v1_metadata_unknown_member_under_unknown_bit.tslod",
         base_spec(metadata_raw=B.canonical_json(later), features=1 << 63),
         FULL_BLOCK,
         "bit 63 set, which this reader does not implement, and a block with "
         "two members it does not know: annotations at the top and site in "
         "provenance. A member a reader does not know is refused unless the "
         "header sets a may-ignore bit the reader does not implement; this "
         "header does, so both are skipped whole and the rest of the block is "
         "served. The same block without bit 63 is refused "
         "(v1-metadata-unknown-member, in the negatives)")

    emit("v1_metadata_unknown_may_ignore_bit_no_block.tslod",
         base_spec(features=1 << 40), None,
         "bit 40 set and bit 32 clear: an unknown may-ignore bit with no block. "
         "The file opens and no block is served")

    # A file written without a block. Every conformance file before bit 32 is
    # one; this case names one so the absent shape is pinned by this vector too.
    v.case("v1_all_ten_dtypes.tslod",
           file=f"{FILES}/v1_all_ten_dtypes.tslod", metadata="absent",
           metadata_bit_set=False, metadata_offset=u64(0), metadata_length=u64(0),
           note="a file with no block: bit 32 clear and the pointer bytes zero, "
                "as every file written before bit 32 had them")
    return v


# ---------------------------------------------------------------------------
# The negatives
# ---------------------------------------------------------------------------


def gen_negatives() -> Vector:
    v = Vector(
        id="v1-metadata-block-negatives",
        set_=SET,
        kind="negative",
        asserts=(
            "A reader that implements feature bit 32 refuses each of these "
            "metadata blocks with the rejection class the case names, taking the "
            "block's checks in the order spec/v1 fixes, and refuses the block "
            "and not the file: where the case's outcome is block-refused, the "
            "file opens and is served with no description and no provenance."
        ),
        source="the version-1 metadata block rules",
        contract=(
            "Two readers that implement the bit must give one broken block one "
            "class, and neither may lose a file's samples to its metadata."
        ),
        requires=["format:v1", "profile:0", "feature:read-rejection",
                  "feature:metadata-block"],
        notes=(
            "These bind readers that implement bit 32. A reader that does not "
            "implement it ignores the bit, the pointer and the block, and opens "
            "every block-refused file here. Each file is built with its defect; "
            "the cross-order cases patch one field of such a file, outside every "
            "block's CRC."
        ),
    )
    built: dict[str, bytes] = {}

    def block_case(slug, spec, cls, why, **extra):
        name = f"v1_metadata_neg_{slug.removeprefix('v1-metadata-').replace('-', '_')}.tslod"
        data = B.build(spec).data
        built[slug] = data
        rel = _write(name, data)
        v.case(slug, file=rel, outcome="block-refused", rejection_class=cls,
               file_opens=True, verified_against_an_implementation=False,
               reason=why, **extra)

    def raw_case(slug, raw: bytes, cls, why, **extra):
        block_case(slug, base_spec(metadata_raw=raw), cls, why,
                   metadata_hex=raw.hex().upper(), **extra)

    good = B.build(base_spec(metadata=FULL_BLOCK))
    n = len(good.data)
    lay = good.layout
    canon = B.canonical_json

    # ---- 1. the extent lies within the file and is not empty
    block_case("v1-metadata-pointer-zero", base_spec(metadata_bit=True),
               "metadata-extent-out-of-bounds",
               "bit 32 set and the pointer bytes zero: an empty extent at offset "
               "0. This is the header a writer that sets the bit without "
               "implementing it writes, and it is the shape of the may-ignore "
               "case in v1-negative-vectors, which a reader that implements the "
               "bit opens and refuses the block of (v1-metadata-may-ignore-case-"
               "under-bit-32, below)")
    block_case("v1-metadata-offset-past-eof",
               base_spec(metadata=FULL_BLOCK, metadata_pointer=(n + 8, 16)),
               "metadata-extent-out-of-bounds",
               "the offset lies past the end of the file. The block is refused "
               "for that rule, never for whatever a reader's language does when "
               "it slices past a buffer")
    block_case("v1-metadata-length-past-eof",
               base_spec(metadata=FULL_BLOCK,
                         metadata_pointer=(lay["metadata_extent"][0], n)),
               "metadata-extent-out-of-bounds",
               "the offset is the block's own and the length runs past the end "
               "of the file")
    block_case("v1-metadata-extent-wraps",
               base_spec(metadata=FULL_BLOCK,
                         metadata_pointer=(2**64 - 8, 16)),
               "metadata-extent-out-of-bounds",
               "offset 2^64 - 8 and length 16, whose sum wraps to 8 in 64-bit "
               "arithmetic and passes a check written as offset + length <= "
               "file length. The rule is offset <= file length and length <= "
               "file length - offset, which no pair can wrap")

    # ---- 2. no overlap
    block_case("v1-metadata-overlaps-header",
               base_spec(metadata=FULL_BLOCK, metadata_pointer=(64, 128)),
               "metadata-extent-overlaps",
               "an extent inside the file that covers the header's second half "
               "and the group table. A block shares no byte with the header, a "
               "table, a block index or a block")
    first_block = lay["data_start"]
    block_case("v1-metadata-overlaps-a-block",
               base_spec(metadata=FULL_BLOCK,
                         metadata_pointer=(first_block, 16)),
               "metadata-extent-overlaps",
               "an extent over the first data block's first 16 bytes. Read as a "
               "block it would be bytes the CRC covers for another purpose")

    # ---- 3. placement
    block_case("v1-metadata-misplaced",
               base_spec(metadata=FULL_BLOCK, metadata_at="end"),
               "metadata-misplaced",
               "a well-formed block after the level tables, overlapping nothing. "
               "A block lies directly after the channel table and before the "
               "first data block, so a file has one layout for one content")

    # ---- 4. the block bound
    names = [f"c{i:04d}" for i in range(1024)]
    channels = [B.ChannelSpec(nm, np.zeros(0, dtype="float32")) for nm in names]
    big = block_at_bound(names)
    assert len(big["channels"][names[0]]) == 1024
    big["channels"][names[0]] += "d"
    raw_big = canon(big)
    assert len(raw_big) == MEBIBYTE + 1
    block_case("v1-metadata-block-over-bound",
               B.FileSpec(compression_id=0, branching_factor=256,
                          groups=[B.GroupSpec(1000.0, TS)], channels=channels,
                          metadata_raw=raw_big),
               "metadata-too-long",
               "1,048,577 bytes, one more than the block bound, made by "
               "lengthening one description of the at-bound conformance file "
               "by one byte. That description is now 1,025 bytes, over its own "
               "bound too, and the block bound is checked first, so the class "
               "is the block's",
               block_utf8_bytes=u64(MEBIBYTE + 1))

    # ---- 5. UTF-8
    valid = canon(FULL_BLOCK)
    at = valid.index(b"stator")
    raw_case("v1-metadata-invalid-utf8-byte",
             valid[:at] + b"\xff" + valid[at + 1:],
             "metadata-not-utf8",
             "one byte 0xFF inside a description. No UTF-8 sequence contains it")
    raw_case("v1-metadata-encoded-surrogate",
             valid[:at] + b"\xed\xa0\x80" + valid[at + 3:],
             "metadata-not-utf8",
             "ED A0 80, the three-byte encoding of the surrogate U+D800. Strict "
             "UTF-8 does not encode surrogates, so the span is not UTF-8; a "
             "decoder that passes surrogates through is not a strict one")

    # ---- 6. the canonical form
    prov = canon(MINIMAL_BLOCK["provenance"])
    for slug, raw, why in (
        ("v1-metadata-whitespace",
         b'{"channels": {},"provenance":' + prov + b"}",
         "one space after a colon. The form has no insignificant whitespace"),
        ("v1-metadata-leading-bom",
         b"\xef\xbb\xbf" + canon(MINIMAL_BLOCK),
         "a byte-order mark before the object. It is valid UTF-8 and it is "
         "not part of the form"),
        ("v1-metadata-members-unsorted",
         b'{"provenance":' + prov + b',"channels":{}}',
         "provenance before channels. Members are in ascending order of "
         "their names' UTF-8 bytes"),
        ("v1-metadata-duplicate-member",
         b'{"channels":{"sig_a":"one","sig_a":"two"},"provenance":' + prov + b"}",
         "sig_a twice. A parser that keeps the last of two members serves "
         "\"two\"; one that keeps the first serves \"one\"; the form refuses "
         "both, because each name follows its predecessor strictly"),
        ("v1-metadata-escaped-duplicate",
         b'{"channels":{"sig_a":"one","sig_\\u0061":"two"},"provenance":'
         + prov + b"}",
         "sig_a, then sig_a again spelled with a \\u escape. Compared before "
         "unescaping the two differ; the form has no \\u escape, so the block "
         "is refused without comparing them at all"),
        ("v1-metadata-unicode-escape",
         b'{"channels":{"sig_a":"\\u00e9"},"provenance":' + prov + b"}",
         "an e-acute written as \\u00e9. The form writes every character raw "
         "except the quote and the backslash"),
        ("v1-metadata-lone-surrogate-escape",
         b'{"channels":{"sig_a":"\\ud800"},"provenance":' + prov + b"}",
         "\\ud800, a surrogate escape with no pair. The bytes are valid UTF-8, "
         "so this is not the UTF-8 class; the form has no \\u escape, so it is "
         "this one"),
        ("v1-metadata-newline-escape",
         b'{"channels":{"sig_a":"a\\nb"},"provenance":' + prov + b"}",
         "\\n inside a description. A line break is a control character, which "
         "no string in a block holds, so the form needs no escape for it"),
        ("v1-metadata-raw-tab",
         b'{"channels":{"sig_a":"a\tb"},"provenance":' + prov + b"}",
         "a raw U+0009 inside a string, which no JSON string may hold"),
        ("v1-metadata-number",
         b'{"channels":{"sig_a":"x"},"provenance":' + prov[:-1]
         + b',"zz_build":7}}',
         "a number as a member's value. The form holds strings, objects and "
         "arrays and nothing else: no number, no null, no boolean"),
        ("v1-metadata-null",
         b'{"channels":{"sig_a":null},"provenance":' + prov + b"}",
         "null where a description belongs. An unknown fact is an absent "
         "member, never a placeholder"),
        ("v1-metadata-trailing-bytes",
         canon(MINIMAL_BLOCK) + b"{}",
         "a second value after the block's object"),
    ):
        raw_case(slug, raw, "metadata-not-canonical", why)

    # ---- 7. the grammar
    for slug, block, why in (
        ("v1-metadata-top-level-array", [FULL_BLOCK["provenance"]],
         "an array where the block's object belongs"),
        ("v1-metadata-provenance-missing", _with(FULL_BLOCK, ("provenance",), None),
         "no provenance. A block says who wrote it and from what, or it is not "
         "written"),
        ("v1-metadata-sources-empty",
         _with(FULL_BLOCK, ("provenance", "sources"), []),
         "an empty sources array. A writer stamps nothing without a named, "
         "hashed source, so a writer with none writes no block"),
        ("v1-metadata-writer-version-missing",
         _with(FULL_BLOCK, ("provenance", "writer_version"), None),
         "writer_version absent; it is required beside writer"),
        ("v1-metadata-description-empty",
         _with(FULL_BLOCK, ("channels", "sig_a"), ""),
         "an empty description. No string is empty: a channel with nothing to "
         "say has no key"),
        ("v1-metadata-description-not-a-string",
         _with(FULL_BLOCK, ("channels", "sig_a"), ["one", "two"]),
         "an array where a description belongs. A description is a string"),
        ("v1-metadata-unknown-member",
         _with(FULL_BLOCK, ("annotations",), {"x": ["later"]}),
         "a member this reader does not know, with no may-ignore bit set that "
         "it does not implement. The block grows only behind a new bit, so an "
         "unknown member is refused at every depth; under such a bit it is "
         "skipped (v1_metadata_unknown_member_under_unknown_bit.tslod)"),
        ("v1-metadata-unknown-source-member",
         _with(FULL_BLOCK, ("provenance", "sources"),
               [dict(SOURCE_INPUT, size="12")]),
         "an unknown member inside a source: the rule holds at every depth"),
        ("v1-metadata-bidi-override",
         _with(FULL_BLOCK, ("channels", "sig_a"), "phase ‮cba"),
         "U+202E, a right-to-left override, inside a description. It makes "
         "text display as other text, so no string in a block carries a "
         "bidirectional override or isolate"),
        ("v1-metadata-c1-control",
         _with(FULL_BLOCK, ("provenance", "product"), "drive\u009b31m"),
         "U+009B, a C1 control (the single-byte control sequence introducer), "
         "inside product. The form writes it raw, so it is canonical and "
         "refused here, with the other forbidden code points"),
        ("v1-metadata-noncharacter",
         _with(FULL_BLOCK, ("channels", "sig_b"), "value ￾"),
         "U+FFFE, a noncharacter, inside a description. Valid UTF-8, and "
         "refused, as I-JSON refuses it"),
    ):
        raw_case(slug, canon(block), "metadata-grammar-violation", why)

    # ---- 8. keys
    raw_case("v1-metadata-key-names-no-channel",
             canon(_with(FULL_BLOCK, ("channels", "sig_d"), "a fourth")),
             "metadata-key-names-no-channel",
             "sig_d, a name no channel in the file carries")
    raw_case("v1-metadata-key-is-a-prefix",
             canon(_with(FULL_BLOCK, ("channels", "sig"), "a prefix")),
             "metadata-key-names-no-channel",
             "sig, a prefix of three channel names and the name of none. Keys "
             "compare with names as whole strings")

    # ---- 9. provenance
    for slug, block, why in (
        ("v1-metadata-sha256-uppercase",
         _with(FULL_BLOCK, ("provenance", "sources"),
               [dict(SOURCE_INPUT, sha256=SOURCE_INPUT["sha256"].upper())]),
         "the hash in uppercase hex. It is 64 lowercase hex digits, one "
         "spelling, so two writers stamp one source alike"),
        ("v1-metadata-sha256-short",
         _with(FULL_BLOCK, ("provenance", "sources"),
               [dict(SOURCE_INPUT, sha256=SOURCE_INPUT["sha256"][:63])]),
         "63 hex digits, one short of a SHA-256"),
        ("v1-metadata-source-name-is-a-path",
         _with(FULL_BLOCK, ("provenance", "sources"),
               [dict(SOURCE_INPUT, name="exports/capture-0001.bin")]),
         "a source named by a relative path. A name is a bare file name, so "
         "a file carries no directory of the machine that wrote it"),
        ("v1-metadata-source-name-backslash",
         _with(FULL_BLOCK, ("provenance", "sources"),
               [dict(SOURCE_INPUT, name="exports\\capture-0001.bin")]),
         "the same path with a backslash separator"),
    ):
        raw_case(slug, canon(block), "metadata-provenance-invalid", why)

    # ---- 10. the value bounds
    over = "µ" * 513
    assert len(over.encode("utf-8")) == 1026 and len(over) == 513
    raw_case("v1-metadata-description-over-bound",
             canon(_with(FULL_BLOCK, ("channels", "sig_a"), over)),
             "metadata-value-too-long",
             "a description of 513 code points and 1,026 UTF-8 bytes. The bound "
             "is 1,024 bytes of the decoded value, so a reader that counts code "
             "points accepts it and is wrong")
    raw_case("v1-metadata-provenance-string-over-bound",
             canon(_with(FULL_BLOCK, ("provenance", "firmware"), "f" * 257)),
             "metadata-value-too-long",
             "a firmware string of 257 bytes; every provenance string is at "
             "most 256")

    # ---- the order within the block: two defects, one class
    two = _with(_with(FULL_BLOCK, ("channels", "sig_a"), over),
                ("channels", "zz_none"), "names no channel")
    raw_case("v1-metadata-order-key-before-bound",
             canon(two), "metadata-key-names-no-channel",
             "two defects: sig_a's description is over its bound, and zz_none, "
             "which sorts after it, names no channel. The keys are judged before "
             "the bounds, so the class is the key's. A reader that walks the "
             "members once, judging each fully, meets sig_a first and gives the "
             "other class",
             defects=["metadata-value-too-long", "metadata-key-names-no-channel"])
    raw_case("v1-metadata-order-grammar-before-provenance",
             canon(_with(_with(FULL_BLOCK, ("provenance", "sources"),
                               [dict(SOURCE_INPUT,
                                     sha256=SOURCE_INPUT["sha256"].upper())]),
                         ("provenance", "zz_later"), "unknown")),
             "metadata-grammar-violation",
             "two defects: an uppercase hash and an unknown provenance member. "
             "The grammar is judged before provenance's forms",
             defects=["metadata-provenance-invalid", "metadata-grammar-violation"])

    # ---- where the block sits in the reader's order: patches on a file whose
    # block is already refused, each outside every block's CRC.
    oob = "v1-metadata-offset-past-eof"
    oob_rel = f"{FILES}/v1_metadata_neg_offset_past_eof.tslod"
    oob_layout = B.build(base_spec(metadata=FULL_BLOCK,
                                   metadata_pointer=(n + 8, 16))).layout

    def cross(slug, offset, raw: bytes, field, cls, why):
        data = built[oob]
        v.case(slug, file=oob_rel, outcome="file-refused", rejection_class=cls,
               file_opens=False,
               patch={"offset": u64(offset), "width_bytes": len(raw),
                      "original_hex": data[offset:offset + len(raw)].hex().upper(),
                      "patched_hex": raw.hex().upper()},
               field=field, defects=["metadata-extent-out-of-bounds", cls],
               verified_against_an_implementation=False, reason=why)

    cross("v1-metadata-order-file-defect-first",
          B.channel_offset(oob_layout, 2, "dtype"), bytes([10]),
          "channel_entry.dtype", "dtype-unknown",
          "the block's pointer is past the end of the file, and sig_c's dtype is "
          "10, outside the registry. The block is judged at the END of step 1, "
          "after every channel, level and block index entry, so the file is "
          "refused for the dtype and no block class is reported")
    crc_at = B.block_offset(oob_layout, 0, 0, 0, "crc32")
    stored = struct.unpack_from("<I", built[oob], crc_at)[0]
    cross("v1-metadata-file-read-on-past-a-refused-block", crc_at,
          struct.pack("<I", stored ^ 1), "block_index_entry.crc32",
          "block-crc-mismatch",
          "the block is refused, and channel sig_a's first data block's stored "
          "CRC has one bit flipped. A refused block does not end the read: the "
          "reader goes on to step 2 and refuses the file for the CRC, which is "
          "what reading on means")

    # The may-ignore case of v1-negative-vectors, judged by a reader that
    # implements bit 32. That case sets bit 32 on a file whose pointer bytes
    # are zero; it was written before the bit was assigned and stays as it is,
    # because its outcome — the file opens — holds for every reader.
    base_rel = f"{FILES}/v1_negative_base.tslod"
    base = (VECTORS / base_rel).read_bytes()
    feat = B.header_offset("features")
    v.case("v1-metadata-may-ignore-case-under-bit-32", file=base_rel,
           outcome="block-refused", rejection_class="metadata-extent-out-of-bounds",
           file_opens=True,
           patch={"offset": u64(feat), "width_bytes": 8,
                  "original_hex": base[feat:feat + 8].hex().upper(),
                  "patched_hex": struct.pack(
                      "<Q", (1 << 32) | B.FEATURE_MOMENT_STREAM).hex().upper()},
           field="header.features", verified_against_an_implementation=False,
           reason="the patch of v1-unknown-may-ignore-feature-bit-is-ACCEPTED in "
                  "v1-negative-vectors: bit 32 set, bit 0 kept, the pointer bytes "
                  "zero. A reader that does not implement bit 32 skips it, as that "
                  "case says. One that does finds an empty extent at offset 0, "
                  "refuses the block and opens the file — so that case's outcome "
                  "holds for both")
    return v


def main() -> None:
    for factory in (gen_conformance, gen_negatives):
        vector = factory()
        vector.write()
        print(f"  {vector.kind:9s} {vector.id:34s} {len(vector.cases):5d} cases")


if __name__ == "__main__":
    main()
