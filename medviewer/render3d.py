from __future__ import annotations
import itertools
import math
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage as ndi
from .volume import Volume

# ------------------------------------------------------------------ 前処理
def prepare_grid(vol: Volume, max_dim: int = 160) -> tuple:
    """最大辺が max_dim ボクセルになる等方グリッドへリサンプルする。(grid float32, voxel_mm)"""
    ext = np.array(vol.shape, float) * np.array(vol.spacing)       # 物理サイズ[mm] (z,y,x)
    vox = ext.max() / max_dim
    factors = np.array(vol.spacing) / vox
    grid = ndi.zoom(vol.data, factors, output=np.float32, order=1, mode="nearest", prefilter=False)
    return grid, float(vox)


def otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m / w0
        mu1 = (m[-1] - m) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var[~np.isfinite(var)] = 0
    return float(centers[int(np.argmax(var))])


# ------------------------------------------------------------------ 領域抽出
def _fill_holes_axial(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for i in range(mask.shape[0]):
        if out[i].any():
            out[i] = ndi.binary_fill_holes(out[i])
    return out


def segment_lung(grid: np.ndarray, air_thr: float = -400.0, n_keep: int = 2) -> np.ndarray:

    air = grid < air_thr
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air)
    border = np.unique(np.concatenate([
        lab[:, 0, :].ravel(), lab[:, -1, :].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    sizes[border] = 0
    if sizes.max() == 0:
        return np.zeros_like(air)
    keep = np.argsort(sizes)[::-1][:n_keep]
    keep = keep[sizes[keep] >= 0.05 * sizes.max()]
    mask = np.isin(lab, keep)
    mask = _fill_holes_axial(mask)
    opened = ndi.binary_opening(mask, structure=ndi.generate_binary_structure(3, 1))
    return opened if opened.any() else mask


def segment_threshold(grid: np.ndarray, lo: float, hi: float = np.inf, largest: bool = True) -> np.ndarray:
    """輝度範囲 [lo, hi] の領域。最大連結成分のみ残し、スライス毎に穴を埋める"""
    mask = (grid >= lo) & (grid <= hi)
    mask = ndi.binary_opening(mask, structure=ndi.generate_binary_structure(3, 1))
    if largest and mask.any():
        lab, n = ndi.label(mask)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == int(np.argmax(sizes))
    return _fill_holes_axial(mask)


def segment(grid, mode: str, thr_lo: float, thr_hi: float = np.inf) -> np.ndarray:
    if mode == "lung":
        return segment_lung(grid, thr_lo)
    if mode == "threshold":
        return segment_threshold(grid, thr_lo, thr_hi)
    return np.ones(grid.shape, bool)


# ------------------------------------------------------------------ カラーマップ
_CMAPS = {
    "gray": [(0.0, (0.05, 0.05, 0.05)), (1.0, (1.0, 1.0, 1.0))],
    "lung": [(0.0, (0.30, 0.18, 0.30)), (0.45, (0.85, 0.50, 0.50)), (1.0, (1.0, 0.95, 0.85))],
    "bone": [(0.0, (0.25, 0.18, 0.12)), (0.6, (0.90, 0.84, 0.72)), (1.0, (1.0, 1.0, 1.0))],
    "hot": [(0.0, (0.1, 0.0, 0.0)), (0.4, (0.9, 0.1, 0.0)), (0.75, (1.0, 0.85, 0.1)), (1.0, (1.0, 1.0, 0.9))],
}
CMAP_NAMES = tuple(_CMAPS)


def colormap_lut(name: str, n: int = 256) -> np.ndarray:
    pts = _CMAPS[name]
    xs = [p[0] for p in pts]
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, xs, [p[1][c] for p in pts]) for c in range(3)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ レンダリング
def view_matrix(az_deg: float, el_deg: float) -> np.ndarray:
    """出力座標 (d=奥行, v=下向き, u=右向き) -> 入力ボリューム座標 (z, y, x) の3x3行列。
    az=el=0 で患者の正面（前方）から見た像（上=頭側、画面右=患者左=x増加方向）になる。"""
    a, e = math.radians(az_deg), math.radians(el_deg)
    rv = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])   # 縦軸まわり
    ru = np.array([[math.cos(e), -math.sin(e), 0], [math.sin(e), math.cos(e), 0], [0, 0, 1]])   # 横軸まわり
    m0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # (d,v,u)->(z,y,x): z=-v, y=d, x=u
    return m0 @ ru @ rv


_POOL = ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1)))


def image_size(shape) -> int:
    """どの視点でも一定の出力画像サイズ（ボリュームの対角長）"""
    return int(math.ceil(np.linalg.norm(shape))) + 6


def _rotated_shape(shape, m) -> tuple:
    """回転後のボリュームを過不足なく収める出力ボックス（中心対称）の大きさ"""
    half = (np.array(shape) - 1) / 2.0
    corners = np.array(list(itertools.product(*[(-h, h) for h in half])))
    ext = np.abs(corners @ m).max(axis=0)              # 出力軸方向の半幅
    return tuple((np.ceil(ext).astype(int) * 2 + 3).tolist())


def rotate_volume(g: np.ndarray, az: float, el: float, fill: float) -> np.ndarray:
    """gを視点(az, el)へ回転した配列 (奥行, 縦, 横)。必要最小のボックスだけ計算し、奥行方向に分割して並列実行"""
    m = view_matrix(az, el)
    shape = _rotated_shape(g.shape, m)
    c_in = (np.array(g.shape) - 1) / 2.0
    c_out = (np.array(shape) - 1) / 2.0
    offset = c_in - m @ c_out
    out = np.empty(shape, np.float32)
    n = min(_POOL._max_workers, shape[0])
    bounds = np.linspace(0, shape[0], n + 1).astype(int)

    def work(a, b):
        if b > a:
            ndi.affine_transform(g, m, offset=offset + m[:, 0] * a, output_shape=(b - a,) + shape[1:],
                                 output=out[a:b], order=1, mode="constant", cval=fill, prefilter=False)

    list(_POOL.map(lambda ab: work(*ab), zip(bounds[:-1], bounds[1:])))
    return out


def _embed(img: np.ndarray, size: int) -> np.ndarray:
    """(h,w,3)を size x size の黒画像の中央に置く"""
    out = np.zeros((size, size, img.shape[2]), img.dtype)
    y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def composite(rot: np.ndarray, size: int, tf_lo: float, tf_hi: float, opacity: float = 0.5,
              gamma: float = 1.0, cmap: str = "lung", mode: str = "composite",
              shading: bool = True) -> np.ndarray:

    span = max(tf_hi - tf_lo, 1e-6)
    lut = colormap_lut(cmap)

    if mode == "mip":
        t = np.clip((rot.max(axis=0) - tf_lo) / span, 0, 1)
        img = lut[(t * 255).astype(np.uint8)] * (t > 0)[..., None]
        return _embed((np.clip(img, 0, 1) * 255).astype(np.uint8), size)

    alut = (np.linspace(0, 1, 256) ** gamma * opacity).astype(np.float32)
    trans = np.ones(rot.shape[1:], np.float32)
    acc = np.zeros(rot.shape[1:] + (3,), np.float32)
    n0 = rot.shape[0]
    has = rot.max(axis=(1, 2)) > tf_lo          # 透明なスライスは丸ごと飛ばす
    for i in np.nonzero(has)[0]:
        t = np.clip((rot[i] - tf_lo) / span, 0, 1)
        idx = (t * 255).astype(np.uint8)
        a = alut[idx]
        col = lut[idx]
        if shading:
            gz = (rot[min(i + 1, n0 - 1)] - rot[max(i - 1, 0)]) * 0.5
            gy, gx = np.gradient(rot[i])
            mag = np.sqrt(gz * gz + gy * gy + gx * gx)
            w = np.clip(mag / (0.04 * span), 0, 1)
            cos = np.abs(gz) / (mag + 1e-6)
            col = col * (((1 - w) + w * (0.25 + 0.75 * cos))[..., None])
        acc += (trans * a)[..., None] * col
        trans *= (1 - a)
        if trans.max() < 0.01:
            break
    return _embed((np.clip(acc, 0, 1) * 255).astype(np.uint8), size)


def render(g: np.ndarray, az: float, el: float, tf_lo: float, tf_hi: float, opacity: float = 0.5,
           gamma: float = 1.0, cmap: str = "lung", mode: str = "composite", shading: bool = True,
           fill: float = None) -> np.ndarray:
    """マスク適用済みグリッド g（領域外は fill）を指定視点で描画し (L,L,3) uint8 を返す"""
    if fill is None:
        fill = float(g.min())
    return composite(rotate_volume(g, az, el, fill), image_size(g.shape), tf_lo, tf_hi, opacity, gamma,
                     cmap, mode, shading)


def crop_to_mask(grid: np.ndarray, mask: np.ndarray, margin: int = 2):
    """マスクのバウンディングボックス(+margin)に切り出す"""
    sl = ndi.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - margin), min(n, s.stop + margin)) for s, n in zip(sl, mask.shape))
    return grid[sl], mask[sl]
