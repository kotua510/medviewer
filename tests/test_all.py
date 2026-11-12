import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medviewer import mpr, render3d as r3                                   # noqa: E402
from medviewer.dicom_writer import write_dicom                              # noqa: E402
from medviewer.volume import load_dicom, load_tiff_stack, Volume           # noqa: E402
from medviewer.dicomlite import read_dicom                                  # noqa: E402


def make_series(folder, vol, spacing=(3.0, 0.8, 0.7), uid="1.2.3", explicit=True, shuffle=False,
                intercept=-1024, slope=1, desc="T", start_instance=1):
    """vol: (Z,Y,X) int HU。ファイル名順をZ順と無関係にもできる"""
    os.makedirs(folder, exist_ok=True)
    n = vol.shape[0]
    names = list(range(n))
    if shuffle:
        names = list(np.random.default_rng(0).permutation(n))
    for i in range(n):
        fields = [
            ((0x0008, 0x0060), "CS", "CT"), ((0x0008, 0x103E), "LO", desc),
            ((0x0020, 0x000E), "UI", uid), ((0x0020, 0x0013), "IS", str(start_instance + i)),
            ((0x0020, 0x0032), "DS", [-10.5, -20.25, 100.0 + i * spacing[0]]),
            ((0x0020, 0x0037), "DS", [1, 0, 0, 0, 1, 0]),
            ((0x0028, 0x0030), "DS", [spacing[1], spacing[2]]),
            ((0x0028, 0x1052), "DS", str(intercept)), ((0x0028, 0x1053), "DS", str(slope)),
        ]
        px = ((vol[i] - intercept) // slope).astype(np.uint16)
        write_dicom(os.path.join(folder, f"f{names[i]:04d}.dcm"), fields, px, explicit=explicit)


class TestDicom(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.vol = rng.integers(-900, 1500, (6, 9, 11)).astype(np.int16)

    def _roundtrip(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            make_series(d, self.vol, **kw)
            return load_dicom([d])

    def test_explicit(self):
        v = self._roundtrip()
        np.testing.assert_array_equal(v.data, self.vol)
        self.assertEqual(v.spacing, (3.0, 0.8, 0.7))

    def test_implicit_vr(self):
        np.testing.assert_array_equal(self._roundtrip(explicit=False).data, self.vol)

    def test_shuffled_filenames_sorted_by_position(self):
        np.testing.assert_array_equal(self._roundtrip(shuffle=True).data, self.vol)

    def test_instance_numbers_ignored_when_position_exists(self):
        # InstanceNumber が逆順でも位置で並ぶ
        with tempfile.TemporaryDirectory() as d:
            make_series(d, self.vol)
            files = sorted(os.listdir(d))
            s = read_dicom(os.path.join(d, files[0]))
            self.assertEqual(s.ipp, [-10.5, -20.25, 100.0])
            self.assertEqual(s.pixel_spacing, (0.8, 0.7))

    def test_largest_series_selected_and_chooser(self):
        with tempfile.TemporaryDirectory() as d:
            make_series(os.path.join(d, "a"), self.vol, uid="1.1")
            make_series(os.path.join(d, "b"), self.vol[:3], uid="1.2")
            self.assertEqual(load_dicom([d]).shape[0], 6)
            v = load_dicom([d], choose=lambda descs: 1)
            self.assertEqual(v.shape[0], 3)

    def test_non_dicom_files_ignored_and_error(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "readme.txt"), "w") as f:
                f.write("hello")
            with self.assertRaises(Exception):
                load_dicom([d])
            make_series(d, self.vol)
            self.assertEqual(load_dicom([d]).shape, self.vol.shape)

    def test_default_window_for_ct(self):
        v = self._roundtrip()
        self.assertEqual((v.wl, v.ww), (40.0, 400.0))


class TestTiff(unittest.TestCase):
    def test_orientations_consistent(self):
        s = os.path.join(ROOT, "sample_data")
        c = load_tiff_stack(os.path.join(s, "CT_coronal.tif"), "coronal", (1, 1, 6))
        g = load_tiff_stack(os.path.join(s, "CT_sagittal.tif"), "sagittal", (1, 1, 6))
        self.assertEqual(c.shape, (80, 608, 500))
        self.assertEqual(g.shape, c.shape)
        self.assertLess(np.abs(c.data.astype(int) - g.data).mean(), 5)  # 同じボリュームの別断面
        a = load_tiff_stack(os.path.join(s, "mri-stack.tif"), "axial", (1, 1, 5))
        self.assertEqual(a.shape, (27, 226, 186))
        # 添付TIFFは行0が足側。表示(上が頭側)は元フレームを上下反転したものになる
        img, _ = mpr.extract_slice(c.data, c.spacing, mpr.CORONAL, 100)
        from PIL import Image
        with Image.open(os.path.join(s, "CT_coronal.tif")) as im:
            im.seek(100)
            np.testing.assert_array_equal(img, np.array(im)[::-1])
        f = load_tiff_stack(os.path.join(s, "CT_coronal.tif"), "coronal", (1, 1, 6), top_is_superior=True)
        np.testing.assert_array_equal(f.data, c.data[::-1])


class TestMPR(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 7, 9)
        self.data = np.arange(np.prod(self.shape)).reshape(self.shape)
        self.sp = (3.0, 1.0, 2.0)

    def test_slice_shapes_and_marker(self):
        # 各断面で crosshair が指す画素 = pos のボクセル値
        pos = (1, 4, 6)
        want = self.data[pos]
        for plane in mpr.PLANES:
            img, ps = mpr.extract_slice(self.data, self.sp, plane, pos[mpr.plane_axis(plane)])
            col, row = mpr.crosshair(plane, pos, self.shape)
            self.assertEqual(img[int(row), int(col)], want, plane)

    def test_pos_roundtrip(self):
        for plane in mpr.PLANES:
            for pos in [(0, 0, 0), (4, 6, 8), (2, 3, 5)]:
                col, row = mpr.crosshair(plane, pos, self.shape)
                self.assertEqual(mpr.pos_from_image(plane, col, row, pos, self.shape), pos)

    def test_pos_clamped(self):
        self.assertEqual(mpr.pos_from_image(mpr.AXIAL, -5, 100, (2, 3, 3), self.shape), (2, 6, 0))

    def test_geom_aspect(self):
        g = mpr.fit_geom(400, 400, (5, 9), (3.0, 1.0))      # 物理サイズ 9 x 15
        self.assertAlmostEqual(g.dh / g.dw, 15 / 9, delta=0.02)
        cx, cy = g.to_canvas(4.5, 2.5)
        c, r = g.to_image(cx, cy)
        self.assertAlmostEqual(c, 4.5)
        self.assertAlmostEqual(r, 2.5)

    def test_window(self):
        out = mpr.apply_window(np.array([-1000, 0, 40, 240, 1000]), 40, 400)
        self.assertEqual((out[0], out[1], out[3], out[4]), (0, 102, 255, 255))
        self.assertTrue(125 <= out[2] <= 130)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import make_sample_dicom as m
        hu, sp = m.ct_phantom(n=48, size=96)
        cls.vol = Volume(hu, sp, -600, 1500, "phantom", "CT")

    def test_lung_segmentation(self):
        g, vox = r3.prepare_grid(self.vol, 96)
        mask = r3.segment(g, "lung", -400)
        from scipy import ndimage as ndi
        lab, n = ndi.label(mask)
        self.assertEqual(n, 2)                                     # 左右の肺
        self.assertFalse(mask[0].any() and mask[:, 0].any())
        # 体の外の空気(隅)は含まれない
        self.assertFalse(mask[:, :3, :3].any())
        # 肺の体積 ≈ 2つの楕円体 (4/3*pi*45*65*100) * 2 [mm^3]
        vol_ml = mask.sum() * vox ** 3 / 1000
        true_ml = 2 * 4 / 3 * np.pi * 45 * 65 * 100 / 1000
        self.assertAlmostEqual(vol_ml / true_ml, 1.0, delta=0.12)

    def test_render_views_differ_and_nonblank(self):
        g, _ = r3.prepare_grid(self.vol, 64)
        mask = r3.segment(g, "lung", -400)
        gm = np.where(mask, g, g.min())
        a = r3.render(gm, 0, 0, -950, -300, 0.5, 1, "lung")
        b = r3.render(gm, 90, 0, -950, -300, 0.5, 1, "lung")
        m = r3.render(gm, 0, 0, -950, -300, mode="mip")
        self.assertEqual(a.dtype, np.uint8)
        self.assertGreater(a.mean(), 1)
        self.assertGreater(m.mean(), 1)
        self.assertFalse(np.array_equal(a, b))
        # 正面像: 肺は左右に2つ離れて見える。側面(90°)では重なり1つの塊
        def ncomp(img):
            from scipy import ndimage as ndi
            return ndi.label(ndi.binary_opening(img.sum(2) > 30))[1]
        self.assertEqual(ncomp(a), 2)
        self.assertEqual(ncomp(b), 1)

    def test_orientation_upright(self):
        # 上(頭側)に小さな球を置く -> 正面像で画像の上半分に出る
        d = np.zeros((40, 40, 40), np.float32)
        d[34:38, 18:22, 18:22] = 1000          # z大 = 頭側
        img = r3.render(d, 0, 0, 0, 1000, 1.0, 1, "gray", shading=False, fill=0)
        ys = np.nonzero(img.sum(2) > 0)[0]
        self.assertLess(ys.mean(), img.shape[0] / 2)
        # x大(患者左)は画像の右側
        d = np.zeros((40, 40, 40), np.float32)
        d[18:22, 18:22, 34:38] = 1000
        img = r3.render(d, 0, 0, 0, 1000, 1.0, 1, "gray", shading=False, fill=0)
        xs = np.nonzero(img.sum(2) > 0)[1]
        self.assertGreater(xs.mean(), img.shape[1] / 2)
        # y大(後方)は奥: 前に置いた物体が後ろの物体を隠す（手前=y小）
        d = np.zeros((40, 40, 40), np.float32)
        d[18:22, 5:9, 18:22] = 1000          # 前方（赤く見せる代わりに値で区別）
        d[18:22, 30:34, 18:22] = 500
        img = r3.render(d, 0, 0, 0, 1000, 1.0, 1, "gray", shading=False, fill=0)
        c = img[img.shape[0] // 2, img.shape[1] // 2].mean()
        self.assertGreater(c, 150)           # 手前の明るい物体(値1000)が奥の物体(値500)より優先して見えている


class TestGUISmoke(unittest.TestCase):
    """偽tkinterで GUI コードを一通り動かす（実描画の確認ではない）"""

    def test_flow(self):
        import fake_tk
        tk = fake_tk.install()
        for m in [k for k in sys.modules if k.startswith("medviewer.viewer") or k.startswith("medviewer.render_window")]:
            del sys.modules[m]
        from medviewer import viewer as vw
        from medviewer.volume import load_dicom
        root = tk.Tk()
        app = vw.Viewer(root)
        vol = load_dicom([os.path.join(ROOT, "sample_data", "dicom_ct_phantom")])
        app.set_volume(vol)
        self.assertEqual(app.pos, (32, 64, 64))
        # スライダー連動（観点6）: axialスライダー -> z更新、coronal/sagittal は位置線のみ更新
        app.panels["axial"].s_pos.set(10)
        self.assertEqual(app.pos[0], 10)
        app.panels["coronal"].s_pos.set(20)
        app.panels["sagittal"].s_pos.set(30)
        self.assertEqual(app.pos, (10, 20, 30))
        # WL/WW は断面ごとに独立
        w0 = (vol.wl, vol.ww)
        app.panels["axial"].s_wl.set(100)
        app.panels["axial"].s_ww.set(777)
        self.assertEqual((app.panels["axial"].wl, app.panels["axial"].ww), (100, 777))
        for k in ("coronal", "sagittal"):
            self.assertEqual((app.panels[k].wl, app.panels[k].ww), w0)
        app.panels["coronal"].s_wl.set(-50)
        self.assertEqual(app.panels["axial"].wl, 100)
        # リセット: 押した断面だけ初期値に戻る
        app.reset_window("axial")
        self.assertEqual((app.panels["axial"].wl, app.panels["axial"].ww), w0)
        self.assertEqual(app.panels["axial"].s_wl.get(), w0[0])
        self.assertEqual(app.panels["coronal"].wl, -50)
        app.reset_window(None)
        self.assertEqual(app.panels["coronal"].wl, w0[0])
        # クリック
        ev = type("E", (), {"x": 170, "y": 170, "delta": 120})()
        app.panels["axial"]._on_click(ev)
        app.panels["coronal"]._on_wheel(ev)
        app.panels["sagittal"]._on_motion(ev)
        app.reset()
        # TIFF読み込みと3Dウィンドウ
        app._open_sample_tiff("mri-stack.tif", "axial")
        self.assertEqual(app.vol.shape, (27, 226, 186))
        app.open_3d()
        w = app.win3d
        w.set_volume(app.vol)
        self.assertIsNotNone(w.grid)
        w.az, w.el = 30, 20
        w._draw(final=False)
        w._draw(final=True)
        w.mode.set("lung")
        w.extract()                          # MRIに肺抽出: 領域なしでもクラッシュしない
        app.set_volume(vol)                  # CTに差し替え -> 3D側も更新
        self.assertEqual(w.mode.get(), "lung")
        self.assertIsNotNone(w.grid)
        self.assertEqual(sys.modules["tkinter.messagebox"].errors, [])
        # 作っただけで配置(pack/grid)していないウィジェットが無いこと（画面に出ないバグの検出）
        for w in fake_tk._W.registry:
            if type(w).__name__ in ("Scale", "Canvas", "Label", "Button", "Radiobutton", "Checkbutton",
                                    "Entry", "Combobox", "LabelFrame", "Listbox"):
                self.assertTrue(w.managed, f"{type(w).__name__} {w.opts.get('label') or w.opts.get('text')} が未配置")


if __name__ == "__main__":
    unittest.main()
