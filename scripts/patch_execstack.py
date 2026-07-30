"""Flip PT_GNU_STACK X-bit off in onnxruntime's pybind .so.

onnxruntime <= 1.17 ships onnxruntime_pybind11_state.*.so with
PT_GNU_STACK = RWE. Modern Debian kernels reject the execstack mmap,
raising ImportError 'cannot enable executable stack'. This script
clears the X bit, leaving RW which the loader accepts without trying
to enable execstack.

Idempotent; safe to run on already-fixed .so files.
"""
from __future__ import annotations

import glob
import struct
import sys

PT_GNU_STACK = 0x6474E551


def patch(path: str) -> bool:
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF":
        return False
    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize = struct.unpack_from("<H", data, 54)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]
    patched = 0
    for i in range(e_phnum):
        ph = e_phoff + i * e_phentsize
        if struct.unpack_from("<I", data, ph)[0] == PT_GNU_STACK:
            off = ph + 4
            flags = struct.unpack_from("<I", data, off)[0]
            if flags & 0x1:
                new = flags & ~0x1
                struct.pack_into("<I", data, off, new)
                patched += 1
                print(f"  patched {path}: PT_GNU_STACK {flags:#x} -> {new:#x}")
    if patched:
        with open(path, "wb") as f:
            f.write(data)
    return bool(patched)


def main() -> int:
    import os

    root = "/usr/local/lib/python3.11/site-packages/onnxruntime/capi"
    candidates = glob.glob(os.path.join(root, "onnxruntime_pybind11_state.*.so"))
    if not candidates:
        print("no onnxruntime pybind .so found — nothing to patch")
        return 0
    for c in candidates:
        patch(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
