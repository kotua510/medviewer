from __future__ import annotations
import struct
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

UNDEF = 0xFFFFFFFF
LONG_VRS = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"OD", b"OL", b"OV", b"UC", b"UR", b"SV", b"UV"}

TS_IMPLICIT_LE = "1.2.840.10008.1.2"
TS_EXPLICIT_LE = "1.2.840.10008.1.2.1"

# 読み取るタグ -> デコード用VR
WANTED = {
    (0x0002, 0x0010): "UI",
    (0x0008, 0x0060): "CS",  # Modality
    (0x0008, 0x103E): "LO",  # SeriesDescription
    (0x0010, 0x0010): "PN",  # PatientName
    (0x0018, 0x0050): "DS",  # SliceThickness
    (0x0018, 0x0088): "DS",  # SpacingBetweenSlices
    (0x0020, 0x000E): "UI",  # SeriesInstanceUID
    (0x0020, 0x0013): "IS",  # InstanceNumber
    (0x0020, 0x0032): "DS",  # ImagePositionPatient
    (0x0020, 0x0037): "DS",  # ImageOrientationPatient
    (0x0028, 0x0002): "US",  # SamplesPerPixel
    (0x0028, 0x0004): "CS",  # PhotometricInterpretation
    (0x0028, 0x0008): "IS",  # NumberOfFrames
    (0x0028, 0x0010): "US",  # Rows
    (0x0028, 0x0011): "US",  # Columns
    (0x0028, 0x0030): "DS",  # PixelSpacing
    (0x0028, 0x0100): "US",  # BitsAllocated
    (0x0028, 0x0101): "US",  # BitsStored
    (0x0028, 0x0103): "US",  # PixelRepresentation
    (0x0028, 0x1050): "DS",  # WindowCenter
    (0x0028, 0x1051): "DS",  # WindowWidth
    (0x0028, 0x1052): "DS",  # RescaleIntercept
    (0x0028, 0x1053): "DS",  # RescaleSlope
}


class DicomError(Exception):
    pass


@dataclass
class DicomSlice:
    path: str = ""
    series_uid: str = ""
    series_desc: str = ""
    modality: str = ""
    patient: str = ""
    instance: Optional[int] = None
    ipp: Optional[list] = None
    iop: Optional[list] = None
    pixel_spacing: Optional[tuple] = None  # (行方向=y, 列方向=x)
    thickness: Optional[float] = None
    spacing_between: Optional[float] = None
    slope: float = 1.0
    intercept: float = 0.0
    wc: Optional[float] = None
    ww: Optional[float] = None
    photometric: str = "MONOCHROME2"
    frames: np.ndarray = field(default=None, repr=False)  # (n, rows, cols) 生のピクセル値


# ---------------------------------------------------------------- 低レベル
def _header(buf, pos, explicit):
    g, e = struct.unpack_from("<HH", buf, pos)
    if g == 0xFFFE:  # item / delimiter（常に暗黙VR形式）
        (ln,) = struct.unpack_from("<I", buf, pos + 4)
        return g, e, None, ln, pos + 8
    if explicit:
        vr = bytes(buf[pos + 4:pos + 6])
        if vr in LONG_VRS:
            (ln,) = struct.unpack_from("<I", buf, pos + 8)
            return g, e, vr, ln, pos + 12
        (ln,) = struct.unpack_from("<H", buf, pos + 6)
        return g, e, vr, ln, pos + 8
    (ln,) = struct.unpack_from("<I", buf, pos + 4)
    return g, e, None, ln, pos + 8


def _skip_sequence(buf, pos, explicit):
    """長さ不定のシーケンスを読み飛ばし、終端の次の位置を返す"""
    n = len(buf)
    while pos + 8 <= n:
        g, e, _, ln, p = _header(buf, pos, explicit)
        if (g, e) == (0xFFFE, 0xE0DD):
            return p
        if (g, e) != (0xFFFE, 0xE000):
            raise DicomError("シーケンス構造が不正です")
        pos = _skip_item(buf, p, explicit) if ln == UNDEF else p + ln
    raise DicomError("シーケンスが終了していません")


def _skip_item(buf, pos, explicit):
    n = len(buf)
    while pos + 8 <= n:
        g, e, _, ln, p = _header(buf, pos, explicit)
        if (g, e) == (0xFFFE, 0xE00D):
            return p
        pos = _skip_sequence(buf, p, explicit) if ln == UNDEF else p + ln
    raise DicomError("アイテムが終了していません")


def _text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip(" \x00")


def _decode(raw: bytes, vr: str):
    if vr == "US":
        k = len(raw) // 2
        return list(struct.unpack("<%dH" % k, raw[:2 * k]))
    s = _text(raw)
    if vr == "DS":
        return [float(v) for v in s.split("\\") if v.strip()]
    if vr == "IS":
        return [int(float(v)) for v in s.split("\\") if v.strip()]
    return s


def _parse_dataset(buf, pos, explicit, vals):
    n = len(buf)
    while pos + 8 <= n:
        g, e, _, ln, p = _header(buf, pos, explicit)
        if (g, e) == (0x7FE0, 0x0010):
            if ln == UNDEF:
                raise DicomError("圧縮(encapsulated)ピクセルデータです")
            vals["pixel"] = (p, ln)
            return
        if ln == UNDEF:
            pos = _skip_sequence(buf, p, explicit)
            continue
        if (g, e) in WANTED:
            vals[(g, e)] = _decode(bytes(buf[p:p + ln]), WANTED[(g, e)])
        pos = p + ln


def _first(vals, tag, default=None):
    v = vals.get(tag)
    if v is None or v == [] or v == "":
        return default
    return v[0] if isinstance(v, list) else v


# ---------------------------------------------------------------- 公開API
def read_dicom(path: str) -> DicomSlice:
    with open(path, "rb") as f:
        buf = f.read()
    vals: dict = {}
    pos, explicit = 0, False
    if len(buf) > 132 and buf[128:132] == b"DICM":
        pos = 132
        # ファイルメタ情報(グループ0002)は常に Explicit VR LE
        while pos + 8 <= len(buf) and struct.unpack_from("<H", buf, pos)[0] == 0x0002:
            g, e, _, ln, p = _header(buf, pos, True)
            if (g, e) in WANTED:
                vals[(g, e)] = _decode(bytes(buf[p:p + ln]), WANTED[(g, e)])
            pos = p + ln
        ts = _first(vals, (0x0002, 0x0010), TS_EXPLICIT_LE)
        if ts == TS_EXPLICIT_LE:
            explicit = True
        elif ts != TS_IMPLICIT_LE:
            return _read_with_pydicom(path, f"転送構文 {ts} は内蔵リーダ未対応")
    else:
        # プリアンブル無し: 先頭がVR文字なら Explicit とみなす
        if len(buf) >= 6 and buf[4:6].isalpha() and buf[4:6].isupper():
            explicit = True
    try:
        _parse_dataset(buf, pos, explicit, vals)
    except (DicomError, struct.error) as ex:
        return _read_with_pydicom(path, str(ex))
    if "pixel" not in vals:
        raise DicomError("ピクセルデータがありません: " + path)

    rows = _first(vals, (0x0028, 0x0010))
    cols = _first(vals, (0x0028, 0x0011))
    if not rows or not cols:
        raise DicomError("Rows/Columns がありません: " + path)
    if _first(vals, (0x0028, 0x0002), 1) != 1:
        raise DicomError("カラー画像(SamplesPerPixel>1)には未対応です")
    bits = _first(vals, (0x0028, 0x0100), 16)
    stored = _first(vals, (0x0028, 0x0101), bits)
    signed = _first(vals, (0x0028, 0x0103), 0) == 1
    nfr = _first(vals, (0x0028, 0x0008), 1) or 1
    dt = {8: "u1", 16: "u2", 32: "u4"}.get(bits)
    if dt is None:
        raise DicomError(f"BitsAllocated={bits} には未対応です")
    if signed:
        dt = dt.replace("u", "i")
    off, ln = vals["pixel"]
    count = rows * cols * nfr
    need = count * (bits // 8)
    if ln < need or off + need > len(buf):
        raise DicomError("ピクセルデータが不足しています: " + path)
    arr = np.frombuffer(buf, dtype="<" + dt, count=count, offset=off).reshape(nfr, rows, cols)
    if not signed and stored < bits:
        arr = arr & ((1 << stored) - 1)
    arr = arr.astype(arr.dtype.newbyteorder("="), copy=True)

    ps = vals.get((0x0028, 0x0030))
    return DicomSlice(
        path=path,
        series_uid=_first(vals, (0x0020, 0x000E), "") or "",
        series_desc=_first(vals, (0x0008, 0x103E), "") or "",
        modality=_first(vals, (0x0008, 0x0060), "") or "",
        patient=_first(vals, (0x0010, 0x0010), "") or "",
        instance=_first(vals, (0x0020, 0x0013)),
        ipp=vals.get((0x0020, 0x0032)) if len(vals.get((0x0020, 0x0032), [])) == 3 else None,
        iop=vals.get((0x0020, 0x0037)) if len(vals.get((0x0020, 0x0037), [])) == 6 else None,
        pixel_spacing=(ps[0], ps[1]) if ps and len(ps) >= 2 else None,
        thickness=_first(vals, (0x0018, 0x0050)),
        spacing_between=_first(vals, (0x0018, 0x0088)),
        slope=_first(vals, (0x0028, 0x1053), 1.0),
        intercept=_first(vals, (0x0028, 0x1052), 0.0),
        wc=_first(vals, (0x0028, 0x1050)),
        ww=_first(vals, (0x0028, 0x1051)),
        photometric=_first(vals, (0x0028, 0x0004), "MONOCHROME2") or "MONOCHROME2",
        frames=arr,
    )


def _read_with_pydicom(path: str, reason: str) -> DicomSlice:
    """内蔵リーダで読めない場合のみ pydicom を使う（任意依存）"""
    try:
        import pydicom  # noqa: WPS433
    except ImportError as ex:
        raise DicomError(f"{reason}（pydicom をインストールすると読める可能性があります）: {path}") from ex
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    ps = getattr(ds, "PixelSpacing", None)
    g = lambda name, d=None: getattr(ds, name, d)  # noqa: E731
    return DicomSlice(
        path=path,
        series_uid=str(g("SeriesInstanceUID", "")),
        series_desc=str(g("SeriesDescription", "")),
        modality=str(g("Modality", "")),
        patient=str(g("PatientName", "")),
        instance=int(g("InstanceNumber")) if g("InstanceNumber") is not None else None,
        ipp=[float(v) for v in g("ImagePositionPatient")] if g("ImagePositionPatient") is not None else None,
        iop=[float(v) for v in g("ImageOrientationPatient")] if g("ImageOrientationPatient") is not None else None,
        pixel_spacing=(float(ps[0]), float(ps[1])) if ps is not None else None,
        thickness=float(g("SliceThickness")) if g("SliceThickness") is not None else None,
        spacing_between=float(g("SpacingBetweenSlices")) if g("SpacingBetweenSlices") is not None else None,
        slope=float(g("RescaleSlope", 1.0)),
        intercept=float(g("RescaleIntercept", 0.0)),
        wc=float(np.atleast_1d(g("WindowCenter"))[0]) if g("WindowCenter") is not None else None,
        ww=float(np.atleast_1d(g("WindowWidth"))[0]) if g("WindowWidth") is not None else None,
        photometric=str(g("PhotometricInterpretation", "MONOCHROME2")),
        frames=np.ascontiguousarray(arr),
    )
