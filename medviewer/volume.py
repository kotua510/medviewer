from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np
from .dicomlite import DicomError, DicomSlice, read_dicom

@dataclass
class Volume:
    data: np.ndarray                 # (Z, Y, X) 実値（CTならHU）
    spacing: tuple                   # (sz, sy, sx) [mm]
    wl: float = 0.0
    ww: float = 1.0
    name: str = ""
    modality: str = ""
    info: str = ""
    vmin: float = field(init=False)
    vmax: float = field(init=False)

    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data)
        self.vmin = float(self.data.min())
        self.vmax = float(self.data.max())

    @property
    def shape(self):
        return self.data.shape


# ------------------------------------------------------------------ DICOM
def collect_files(paths: Sequence[str]) -> list:
    """ファイル/フォルダのリストからファイルを列挙（フォルダは再帰）"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                out += [os.path.join(root, n) for n in sorted(names) if not n.startswith(".")]
        elif os.path.isfile(p):
            out.append(p)
    return sorted(out)


def read_series(paths: Sequence[str], progress: Optional[Callable[[int, int], None]] = None):
    """DICOMとして読めたファイルをシリーズごとに分類して返す: {uid: [DicomSlice, ...]}"""
    files = collect_files(paths)
    series: dict = {}
    errors = []
    for i, f in enumerate(files):
        if progress:
            progress(i, len(files))
        try:
            s = read_dicom(f)
        except (DicomError, OSError, ValueError, KeyError) as ex:
            errors.append(f"{os.path.basename(f)}: {ex}")
            continue
        series.setdefault(s.series_uid or "(no uid)", []).append(s)
    if not series:
        detail = errors[0] if errors else "ファイルがありません"
        raise DicomError("読み込めるDICOMがありませんでした（" + detail + "）")
    return series


def describe_series(uid: str, slices: list) -> str:
    s = slices[0]
    n = sum(f.frames.shape[0] for f in slices)
    return f"{s.modality or '?'} | {s.series_desc or '(説明なし)'} | {n}枚 | {s.frames.shape[2]}x{s.frames.shape[1]}"


def _slice_key(slices: list):
    """各スライスの並び替えキー（スライス法線方向の位置）。無ければ None"""
    if all(s.ipp is not None and s.iop is not None for s in slices):
        iop = np.array(slices[0].iop, float)
        normal = np.cross(iop[:3], iop[3:])
        return np.array([np.dot(s.ipp, normal) for s in slices])
    return None


def build_volume(slices: list, name: str = "") -> Volume:
    shapes = [s.frames.shape[1:] for s in slices]
    major = max(set(shapes), key=shapes.count)
    slices = [s for s in slices if s.frames.shape[1:] == major]
    first = slices[0]

    # --- 並び替え
    flat = []  # (配列2D, 位置 or None, instance)
    key = _slice_key(slices) if all(s.frames.shape[0] == 1 for s in slices) else None
    if key is not None:
        order = np.argsort(key, kind="stable")
        arrays = [slices[i].frames[0] for i in order]
        sorted_slices = [slices[i] for i in order]
        pos = key[order]
    else:
        if all(s.instance is not None for s in slices):
            slices = sorted(slices, key=lambda s: s.instance)
        arrays = [fr for s in slices for fr in s.frames]
        sorted_slices = [s for s in slices for _ in range(s.frames.shape[0])]
        pos = None
    del flat

    # --- スライス間隔
    if pos is not None and len(pos) > 1 and np.median(np.abs(np.diff(pos))) > 1e-6:
        sz = float(np.median(np.abs(np.diff(pos))))
    else:
        sz = first.spacing_between or first.thickness or 1.0
    py, px = first.pixel_spacing or (1.0, 1.0)

    # --- 値の変換（Rescale）
    slope, intercept = first.slope, first.intercept
    raw = np.stack(arrays)
    if float(slope).is_integer() and float(intercept).is_integer():
        vol = raw.astype(np.int32) * int(slope) + int(intercept)
        if vol.min() >= -32768 and vol.max() <= 32767:
            vol = vol.astype(np.int16)
    else:
        vol = raw.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    if first.photometric == "MONOCHROME1":
        vol = (vol.max() + vol.min()) - vol
    del raw

    # --- 初期WL/WW
    if first.wc is not None and first.ww:
        wl, ww = first.wc, first.ww
    elif first.modality == "CT":
        wl, ww = 40.0, 400.0
    else:
        lo, hi = np.percentile(vol[:: max(1, len(vol) // 16)], [0.5, 99.5])
        wl, ww = (lo + hi) / 2.0, max(hi - lo, 1.0)

    info = (f"{first.modality or '?'}  {vol.shape[2]}x{vol.shape[1]}x{vol.shape[0]}  "
            f"間隔 x={px:.3g} y={py:.3g} z={sz:.3g} mm")
    return Volume(vol, (sz, py, px), float(wl), float(ww), name or first.series_desc or "DICOM",
                  first.modality, info)


def load_dicom(paths: Sequence[str],
               choose: Optional[Callable[[list], int]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> Volume:
    """DICOMフォルダ/ファイルを読み込む。複数シリーズあれば choose(説明リスト)->index で選ぶ"""
    series = read_series(paths, progress)
    uids = sorted(series, key=lambda u: -sum(s.frames.shape[0] for s in series[u]))
    idx = 0
    if len(uids) > 1 and choose is not None:
        idx = choose([describe_series(u, series[u]) for u in uids])
        if idx is None or idx < 0:
            raise DicomError("キャンセルされました")
    uid = uids[idx]
    first = series[uid][0]
    name = " / ".join(x for x in (first.patient, first.series_desc) if x) or os.path.basename(
        os.path.dirname(first.path))
    return build_volume(series[uid], name)


# ------------------------------------------------------------------ TIFF
ORIENTATIONS = ("axial", "coronal", "sagittal")


def load_tiff_stack(path: str, orientation: str = "axial", spacing=(1.0, 1.0, 1.0),
                    top_is_superior: bool = False) -> Volume:
    """マルチページTIFFを読み込む。

    orientation: ファイル内の各ページ(フレーム)の断面の向き。
      axial    : frame=Z, 行=Y, 列=X
      coronal  : frame=Y, 行=Z, 列=X
      sagittal : frame=X, 行=Z, 列=Y
    spacing: 画素間隔 (sx, sy, sz) [mm]
    top_is_superior: coronal/sagittal で「画像の上端が頭側」ならTrue。
      Falseなら行番号=Zインデックス（行0が足側）とみなす（添付のCT_coronal/sagittal.tifはこちら）。
    """
    from PIL import Image, ImageSequence

    frames = []
    with Image.open(path) as im:
        for fr in ImageSequence.Iterator(im):
            if fr.mode in ("RGB", "RGBA", "P", "LA"):
                fr = fr.convert("L")
            frames.append(np.array(fr))
    if not frames:
        raise ValueError("画像がありません")
    st = np.stack(frames)
    if st.dtype == np.uint32 or st.dtype == np.int32:
        st = st.astype(np.float32)
    if orientation == "axial":
        data = st
    elif orientation == "coronal":      # (y, z, x) -> (z, y, x)
        data = st.transpose(1, 0, 2)
    elif orientation == "sagittal":     # (x, z, y) -> (z, y, x)
        data = st.transpose(1, 2, 0)
    else:
        raise ValueError("orientation は " + "/".join(ORIENTATIONS))
    if top_is_superior and orientation != "axial":
        data = data[::-1]
    sx, sy, sz = spacing
    lo, hi = np.percentile(data[:: max(1, len(data) // 16)], [0.5, 99.5])
    base = os.path.basename(path)
    info = (f"TIFF({orientation})  {data.shape[2]}x{data.shape[1]}x{data.shape[0]}  "
            f"間隔 x={sx:g} y={sy:g} z={sz:g} mm")
    return Volume(data, (sz, sy, sx), float((lo + hi) / 2), float(max(hi - lo, 1.0)), base, "OT", info)
