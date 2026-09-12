"""Regenerate the whole corpus, then rebuild the manifest from the tree.

CI runs this and byte-compares the result against the committed tree, so a
generator and its corpus cannot drift.

Every generator here runs on numpy, zstandard and pcodec. Nothing in this
repository needs another implementation of the format to rebuild itself, and
nothing outside it writes here.

    python3 tools/generate_all.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import write_manifest  # noqa: E402

#: (module, what it needs). Order is the order they are listed in.
GENERATORS = [
    ("gen_set1_tick_timebase", "stdlib"),
    ("gen_set2_anchored_fold", "stdlib"),
    ("gen_record_layouts", "stdlib"),
    ("gen_set3_build_level", "numpy"),
    ("gen_set4_chan_pebay", "stdlib"),
    ("gen_v1_format_vectors", "numpy"),
    ("gen_codec_vectors", "numpy + zstandard + pcodec"),
]


def main() -> int:
    failures = []
    for name, needs in GENERATORS:
        print(f"[generate] {name}  ({needs})")
        try:
            module = importlib.import_module(name)
            module.main()
        except Exception:
            traceback.print_exc()
            print(f"  FAILED — {name} raised; nothing from it is trustworthy")
            return 2

    path = write_manifest()
    import json
    manifest = json.loads(path.read_text())
    counts = manifest["counts"]
    print(f"\nmanifest: {path}")
    print(f"  vectors {counts['vectors']}, cases {counts['cases']}")
    print(f"  by kind: {counts['by_kind']}")
    print(f"  by set:  {counts['by_set']}")

    if failures:
        print("\nFAILURES. Absence is a finding, not a pass:")
        for f in failures:
            print(f"    {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
