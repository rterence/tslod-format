# `v1-format` — the version-1 file vectors

`files/` holds every `.tslod` this set uses, and **they are not all readable**. Some are goldens that
a conforming reader must open and read exactly; others are built with a defect and a valid CRC, and
opening one is supposed to fail with the rejection class its case names.

**A file's role is stated only by the vectors that name it.** Nothing about the directory, and
nothing about a file's name, says which a file is:

- a **golden** is named by a conformance set (`v1-profile0-conformance-set`,
  `v1-profile2-conformance-set`) or by a fixture vector (`v1-block-crc`,
  `v1-variable-window-straddle`);
- a **base for patched negatives** is named by `v1-negative-vectors` cases that carry a `patch` or a
  `truncate_to`. The base itself is well formed and opens;
- a file **built with its defect** is named by a `v1-negative-vectors` case carrying `file` with no
  `patch` and no `truncate_to`. It does not open. That predicate is the whole rule — the engine's
  corpus crate implements it as `is_built_with_its_defect`.

At this commit the directory holds 29 files, of which 8 are built with their defect.

A reader that enumerates this directory and assumes every file is a golden will fail on them, and
the failure will look like a bug in the reader rather than a wrong assumption about the corpus.
Enumerate the vectors instead: `corpus/MANIFEST.json` lists them, each vector names its own files,
and `corpus/CONVENTIONS.md` states the four shapes a negative case can take.
