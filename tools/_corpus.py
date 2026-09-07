"""Shared machinery for every corpus generator.

Rules this module enforces so no generator has to remember them (see
`corpus/CONVENTIONS.md`):

  * every 64-bit integer leaves as a decimal STRING;
  * every float leaves as its IEEE-754 bit pattern in big-endian hex, with a
    `repr` beside it that no runner compares;
  * bulk arrays leave as raw little-endian sidecar files, never inline;
  * every vector carries the sentence it asserts, and the manifest is built
    from the vectors rather than maintained beside them.

Stdlib only, and nothing here imports an implementation of the format. The
corpus must be rebuildable by someone who has only this repository.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus"
#: Where vectors are written. Overridable so a generator that lives in another
#: repository — one whose vectors prove an implementation rather than the
#: format — can use this module's encoding rules while writing somewhere else.
VECTORS = Path(os.environ.get("TSLOD_CORPUS_VECTORS", CORPUS_ROOT / "vectors"))

# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def i64(value: int) -> str:
    """A 64-bit integer as a decimal string. Sign stays inside the string."""
    value = int(value)
    if not (-(2**63) <= value < 2**63):
        raise ValueError(f"not representable as i64: {value}")
    return str(value)


def u64(value: int) -> str:
    value = int(value)
    if not (0 <= value < 2**64):
        raise ValueError(f"not representable as u64: {value}")
    return str(value)


def big(value: int) -> str:
    """An integer of unbounded width as a decimal string.

    Used for the tick vectors, where an intermediate deliberately exceeds
    2**53 and the whole point is that an implementation must not evaluate it
    in a double.
    """
    return str(int(value))


#: dtype -> (float struct code, same-width unsigned struct code, byte width)
_FLOAT_PACK = {"float32": ("f", "I", 4), "float64": ("d", "Q", 8)}


def fbits(value: float, dtype: str = "float64") -> dict:
    """A float as `{bits, repr, dtype}`. `bits` is the assertion.

    Equality of floating-point fixtures is equality of the bit pattern, never
    numeric equality. `repr` is for a human reading the file; no runner
    compares it, and it does not round-trip every NaN payload.
    """
    fcode, icode, width = _FLOAT_PACK[dtype]
    bits = struct.unpack(f"<{icode}", struct.pack(f"<{fcode}", value))[0]
    return {"dtype": dtype, "bits": f"0x{bits:0{width * 2}X}", "repr": repr(value)}


def fbits_from_bits(bits: int, dtype: str = "float64") -> dict:
    """Same shape as `fbits`, built from a bit pattern that already exists.

    The path a NaN must take: carried through as bits it keeps the payload a
    float round-trip would not preserve.
    """
    fcode, icode, width = _FLOAT_PACK[dtype]
    value = struct.unpack(f"<{fcode}", struct.pack(f"<{icode}", bits))[0]
    return {"dtype": dtype, "bits": f"0x{bits:0{width * 2}X}", "repr": repr(value)}


CANONICAL_QNAN = {"float32": 0x7FC00000, "float64": 0x7FF8000000000000}


#: Arrays at or below this many bytes are carried inline as hex; larger ones
#: become sidecar files. One decode path either way: bytes -> array.
INLINE_ARRAY_LIMIT = 512


def array_ref(arr, rel_path: str | None = None) -> dict:
    """Encode a numpy array as a corpus array reference.

    The payload is always the array's raw little-endian C-contiguous bytes —
    inline as `hex` when small, as a sidecar `file` when not. A runner decodes
    one way in both cases, and a float's bit pattern survives by construction
    because no float is ever written as a decimal.
    """
    import numpy as np  # local: the stdlib generators must not need numpy

    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder not in ("<", "=", "|"):
        raise ValueError(f"array must be little-endian or native, got {arr.dtype}")
    payload = arr.tobytes(order="C")
    ref = {
        "dtype": arr.dtype.name,
        "shape": [int(s) for s in arr.shape],
        "order": "C",
        "byte_order": "little",
    }
    if len(payload) <= INLINE_ARRAY_LIMIT:
        ref["hex"] = payload.hex().upper()
    else:
        if rel_path is None:
            raise ValueError("array exceeds the inline limit and no sidecar path was given")
        write_bin(rel_path, payload)
        ref["file"] = rel_path
    return ref


def bin_ref(rel_path: str, dtype: str, shape: list[int]) -> dict:
    """Describe a sidecar array. The file itself is written by `write_bin`."""
    return {
        "file": rel_path,
        "dtype": dtype,
        "shape": [int(s) for s in shape],
        "order": "C",
        "byte_order": "little",
    }


def write_bin(rel_path: str, payload: bytes) -> None:
    path = VECTORS / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------

_KINDS = {"fixture", "oracle", "negative", "assertion"}


class Vector:
    """One corpus vector. Its `asserts` sentence is its reason to exist."""

    def __init__(self, *, id: str, set_: str, kind: str, asserts: str,
                 source: str, contract: str, requires: list[str] | None = None,
                 notes: str | None = None):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
        if not asserts.strip():
            raise ValueError(f"vector {id} has no assertion; that is the one required field")
        if not contract.strip():
            raise ValueError(
                f"vector {id} has no contract sentence. `asserts` says WHAT must be "
                f"true; `contract` says why that is a contract between implementations "
                f"rather than one implementation's private detail. A vector that cannot "
                f"answer the second question is pinning an accident.")
        self.id = id
        self.set = set_
        self.kind = kind
        self.asserts = asserts.strip()
        self.source = source
        self.contract = contract.strip()
        self.requires = list(requires or [])
        self.notes = notes
        self.cases: list[dict] = []

    def case(self, name: str, **fields) -> None:
        self.cases.append({"name": name, **fields})

    def as_dict(self) -> dict:
        body = {
            "id": self.id,
            "set": self.set,
            "kind": self.kind,
            "asserts": self.asserts,
            "source": self.source,
            "contract": self.contract,
            "requires": self.requires,
        }
        if self.notes:
            body["notes"] = self.notes
        body["case_count"] = len(self.cases)
        body["cases"] = self.cases
        return body

    def write(self) -> Path:
        if not self.cases and self.kind != "assertion":
            raise ValueError(
                f"vector {self.id} has no cases — an empty set is a finding, not a pass"
            )
        path = VECTORS / self.set / f"{self.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict:
    """Walk `vectors/` and build the manifest from what is actually there.

    Built from the tree rather than maintained beside it, so a vector that was
    generated but never listed cannot exist, and a listed vector that was
    never generated is a missing file at build time.
    """
    entries = []
    for vector_path in sorted((CORPUS_ROOT / "vectors").rglob("*.json")):
        body = json.loads(vector_path.read_text())
        rel = vector_path.relative_to(CORPUS_ROOT / "vectors").as_posix()
        files = [{"path": rel, "sha256": sha256(vector_path)}]
        for referenced in _referenced_files(body):
            side_path = CORPUS_ROOT / "vectors" / referenced
            if not side_path.exists():
                raise FileNotFoundError(
                    f"{rel} references {referenced} which does not exist"
                )
            files.append({"path": referenced, "sha256": sha256(side_path)})
        entries.append({
            "id": body["id"],
            "set": body["set"],
            "kind": body["kind"],
            "asserts": body["asserts"],
            "source": body["source"],
            "contract": body["contract"],
            "requires": body.get("requires", []),
            "case_count": body.get("case_count", len(body.get("cases", []))),
            "files": files,
        })

    ids = [e["id"] for e in entries]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate vector ids: {dupes}")

    by_set: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for e in entries:
        by_set[e["set"]] = by_set.get(e["set"], 0) + 1
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1

    return {
        "manifest_version": 1,
        "format_target": "tslod v1",
        "conventions": "corpus/CONVENTIONS.md",
        "generated_from": "tools/generate_all.py",
        "counts": {
            "vectors": len(entries),
            "cases": sum(e["case_count"] for e in entries),
            "by_set": dict(sorted(by_set.items())),
            "by_kind": dict(sorted(by_kind.items())),
        },
        "vectors": entries,
    }


def _referenced_files(body: dict) -> list[str]:
    """Every file a vector points at, so the manifest hashes all of them.

    Two shapes: a bulk array reference, which carries dtype and shape beside
    the path, and a plain `file` key naming a golden `.tslod`. Both are data
    the vector's expected values depend on, so both are hashed — a golden file
    that is edited or corrupted has to be a failing build and not a silent
    pass.
    """
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("file"), str):
                found.append(node["file"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(body)
    return found


def write_manifest() -> Path:
    manifest = build_manifest()
    path = CORPUS_ROOT / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
