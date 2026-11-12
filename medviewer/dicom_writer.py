from __future__ import annotations
import struct
import numpy as np
from .dicomlite import LONG_VRS, TS_EXPLICIT_LE, TS_IMPLICIT_LE


def _pad(b: bytes, vr: str) -> bytes:
    if len(b) % 2:
        b += b"\x00" if vr in ("UI", "OB") else b" "
    return b


def _elem(tag, vr: str, value: bytes, explicit: bool) -> bytes:
    value = _pad(value, vr)
    g, e = tag
    if not explicit:
        return struct.pack("<HHI", g, e, len(value)) + value
    if vr.encode() in LONG_VRS:
        return struct.pack("<HH2sHI", g, e, vr.encode(), 0, len(value)) + value
    return struct.pack("<HH2sH", g, e, vr.encode(), len(value)) + value


def _val(vr: str, v) -> bytes:
    if vr == "US":
        return struct.pack("<%dH" % len(np.atleast_1d(v)), *np.atleast_1d(v))
    if isinstance(v, (list, tuple, np.ndarray)):
        v = "\\".join(str(x) for x in v)
    return str(v).encode("latin-1")


def write_dicom(path, fields, pixels, explicit=True, bits=16, signed=False):
    rows, cols = pixels.shape
    dt = {8: "u1", 16: "u2", 32: "u4"}[bits]
    if signed:
        dt = dt.replace("u", "i")
    base = [
        ((0x0028, 0x0002), "US", 1),
        ((0x0028, 0x0004), "CS", "MONOCHROME2"),
        ((0x0028, 0x0010), "US", rows),
        ((0x0028, 0x0011), "US", cols),
        ((0x0028, 0x0100), "US", bits),
        ((0x0028, 0x0101), "US", bits),
        ((0x0028, 0x0102), "US", bits - 1),
        ((0x0028, 0x0103), "US", 1 if signed else 0),
    ]
    allf = sorted(list(fields) + base, key=lambda t: t[0])
    body = b"".join(_elem(t, vr, _val(vr, v), explicit) for t, vr, v in allf)
    px = np.ascontiguousarray(pixels.astype("<" + dt)).tobytes()
    body += _elem((0x7FE0, 0x0010), "OW", px, explicit)
    ts = TS_EXPLICIT_LE if explicit else TS_IMPLICIT_LE
    meta = _elem((0x0002, 0x0010), "UI", ts.encode(), True)
    meta = _elem((0x0002, 0x0000), "UL", struct.pack("<I", len(meta)), True) + meta
    with open(path, "wb") as f:
        f.write(b"\x00" * 128 + b"DICM" + meta + body)
