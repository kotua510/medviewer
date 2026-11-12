"""サンプルDICOMを生成する。

  python tools/make_sample_dicom.py

sample_data/dicom_ct_phantom/  : 胸部CT風の合成ファントム（肺・気管・血管入り, HU値）
sample_data/dicom_mri_head/    : sample_data/mri-stack.tif をDICOM化したもの
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medviewer.dicom_writer import write_dicom  # noqa: E402


def ct_phantom(n=64, size=128, dz=4.0, dxy=2.8, seed=0):
    """単位はHU。(Z,Y,X)。体=40HU、肺=-800HU、気管=-1000HU、脊椎=300HU、肺内血管=-100HU"""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(np.arange(n), np.arange(size), np.arange(size), indexing="ij")
    zc = (z - n / 2) * dz
    yc = (y - size / 2) * dxy
    xc = (x - size / 2) * dxy
    hu = np.full((n, size, size), -1000.0)
    body = (xc / 150) ** 2 + (yc / 100) ** 2 + (zc / 140) ** 2 <= 1
    hu[body] = 40.0
    lung = np.zeros_like(body)
    for sx in (-1, 1):
        lung |= ((xc - sx * 60) / 45) ** 2 + ((yc + 5) / 65) ** 2 + (zc / 100) ** 2 <= 1
    hu[lung] = -800.0
    trachea = (xc ** 2 + (yc + 30) ** 2 <= 8 ** 2) & (zc > 20) & body
    hu[trachea] = -1000.0
    hu[(xc ** 2 + (yc - 70) ** 2 <= 14 ** 2) & body] = 300.0            # 脊椎
    for k in range(40):                                                     # 血管（ランダムな短い管）
        c = np.array([rng.uniform(-90, 90), rng.uniform(-50, 50), rng.uniform(-70, 70)])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        for s in np.linspace(-25, 25, 30):
            p = c + s * d
            m = ((xc - p[0]) ** 2 + (yc - p[1]) ** 2 + (zc - p[2]) ** 2 <= 3.5 ** 2) & lung
            hu[m] = -100.0
    hu = np.clip(hu + rng.normal(0, 8, hu.shape), -1024, 3000)
    return hu.astype(np.int16), (dz, dxy, dxy)


def main():
    out = os.path.join(ROOT, "sample_data", "dicom_ct_phantom")
    os.makedirs(out, exist_ok=True)
    hu, (dz, dy, dx) = ct_phantom()
    uid = "1.2.826.0.1.3680043.8.498.1000"
    for i in range(hu.shape[0]):
        z = i * dz
        fields = [
            ((0x0008, 0x0060), "CS", "CT"),
            ((0x0008, 0x103E), "LO", "Chest phantom"),
            ((0x0010, 0x0010), "PN", "PHANTOM^CT"),
            ((0x0018, 0x0050), "DS", f"{dz}"),
            ((0x0020, 0x000E), "UI", uid),
            ((0x0020, 0x0013), "IS", str(i + 1)),
            ((0x0020, 0x0032), "DS", [-(hu.shape[2] / 2) * dx, -(hu.shape[1] / 2) * dy, z]),
            ((0x0020, 0x0037), "DS", [1, 0, 0, 0, 1, 0]),
            ((0x0028, 0x0030), "DS", [dy, dx]),
            ((0x0028, 0x1050), "DS", "-600"),
            ((0x0028, 0x1051), "DS", "1500"),
            ((0x0028, 0x1052), "DS", "-1024"),
            ((0x0028, 0x1053), "DS", "1"),
        ]
        write_dicom(os.path.join(out, f"IM{i + 1:04d}.dcm"), fields, (hu[i].astype(np.int32) + 1024).astype(np.uint16))
    print("wrote", out)

    tif = os.path.join(ROOT, "sample_data", "mri-stack.tif")
    if os.path.exists(tif):
        from PIL import Image, ImageSequence
        out = os.path.join(ROOT, "sample_data", "dicom_mri_head")
        os.makedirs(out, exist_ok=True)
        with Image.open(tif) as im:
            frames = [np.array(f) for f in ImageSequence.Iterator(im)]
        uid = "1.2.826.0.1.3680043.8.498.2000"
        for i, f in enumerate(frames):
            fields = [
                ((0x0008, 0x0060), "CS", "MR"),
                ((0x0008, 0x103E), "LO", "Head MRI (from mri-stack.tif)"),
                ((0x0010, 0x0010), "PN", "SAMPLE^MRI"),
                ((0x0018, 0x0050), "DS", "5"),
                ((0x0020, 0x000E), "UI", uid),
                ((0x0020, 0x0013), "IS", str(i + 1)),
                ((0x0020, 0x0032), "DS", [0, 0, i * 5.0]),
                ((0x0020, 0x0037), "DS", [1, 0, 0, 0, 1, 0]),
                ((0x0028, 0x0030), "DS", [1.0, 1.0]),
            ]
            write_dicom(os.path.join(out, f"IM{i + 1:04d}.dcm"), fields, f.astype(np.uint16))
        print("wrote", out)


if __name__ == "__main__":
    main()
