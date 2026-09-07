# `reader/` — a reference reader

**Empty, deliberately.**

The reader is written against the specification once the specification is normative, sharing no code
with any engine — because a reference reader derived from an existing implementation proves that
implementation rather than the format, which is the one thing it exists not to do.

Profile 0 is what it will be measured against first: every recipe byte `0x00`, every payload the
array's little-endian bytes, so a reader needs only the standard library. Profile 2 now has a
conformance set too, and reading it needs `pcodec` and zstd — published third-party formats the
specification names by version, not another project's code, and the only exemption there is.
