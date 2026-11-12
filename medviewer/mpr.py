from __future__ import annotations
from dataclasses import dataclass
import numpy as np

AXIAL, CORONAL, SAGITTAL = "axial", "coronal", "sagittal"
PLANES = (AXIAL, CORONAL, SAGITTAL)

# 各断面に引く線の色（その線が表す“相手の断面”の色）
PLANE_COLOR = {AXIAL: "#ff6b4a", CORONAL: "#3ddc84", SAGITTAL: "#4da3ff"}
PLANE_LABEL = {AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}


def n_slices(shape, plane):
    z, y, x = shape
    return {AXIAL: z, CORONAL: y, SAGITTAL: x}[plane]


def plane_axis(plane):
    """断面のスライス位置が pos のどの要素か: pos=(z,y,x)"""
    return {AXIAL: 0, CORONAL: 1, SAGITTAL: 2}[plane]


def extract_slice(data, spacing, plane, idx):
    """表示向きの2D配列と (行方向の画素間隔, 列方向の画素間隔) を返す"""
    sz, sy, sx = spacing
    if plane == AXIAL:
        return data[idx], (sy, sx)
    if plane == CORONAL:
        return data[::-1, idx, :], (sz, sx)
    return data[::-1, :, idx], (sz, sy)


def crosshair(plane, pos, shape):
    """この断面画像上での (縦線のcol, 横線のrow)（画素中心=+0.5）"""
    z, y, x = pos
    nz = shape[0]
    if plane == AXIAL:
        return x + 0.5, y + 0.5
    if plane == CORONAL:
        return x + 0.5, (nz - 1 - z) + 0.5
    return y + 0.5, (nz - 1 - z) + 0.5


def lines_for(plane):
    """この断面に描く線: [(向き'v'|'h', その線が表す断面)]"""
    return {
        AXIAL: [("v", SAGITTAL), ("h", CORONAL)],
        CORONAL: [("v", SAGITTAL), ("h", AXIAL)],
        SAGITTAL: [("v", CORONAL), ("h", AXIAL)],
    }[plane]


def pos_from_image(plane, col, row, pos, shape):
    """断面画像上の位置(col,row: 小数可)から新しい pos=(z,y,x) を返す（範囲内にクランプ）"""
    z, y, x = pos
    nz, ny, nx = shape
    c, r = int(np.floor(col)), int(np.floor(row))
    if plane == AXIAL:
        x, y = c, r
    elif plane == CORONAL:
        x, z = c, nz - 1 - r
    else:
        y, z = c, nz - 1 - r
    return (int(np.clip(z, 0, nz - 1)), int(np.clip(y, 0, ny - 1)), int(np.clip(x, 0, nx - 1)))


def apply_window(img, wl, ww):
    """WL/WWで 0-255 のuint8に変換"""
    ww = max(float(ww), 1e-6)
    lo = float(wl) - ww / 2.0
    out = (img.astype(np.float32) - lo) * (255.0 / ww)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Geom:
    ox: float
    oy: float
    dw: int
    dh: int
    ncols: int
    nrows: int

    def to_canvas(self, col, row):
        return self.ox + col * self.dw / self.ncols, self.oy + row * self.dh / self.nrows

    def to_image(self, cx, cy):
        return (cx - self.ox) * self.ncols / self.dw, (cy - self.oy) * self.nrows / self.dh

    def inside(self, cx, cy):
        return self.ox <= cx < self.ox + self.dw and self.oy <= cy < self.oy + self.dh


def fit_geom(cw, ch, shape, pixel_size, margin=4):
    """キャンバス(cw,ch)に物理的な縦横比を保って画像を収める配置を計算"""
    nrows, ncols = shape
    W, H = ncols * pixel_size[1], nrows * pixel_size[0]
    s = max(min((cw - 2 * margin) / W, (ch - 2 * margin) / H), 1e-6)
    dw, dh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    return Geom((cw - dw) / 2.0, (ch - dh) / 2.0, dw, dh, ncols, nrows)
