from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage as ndi
from . import render3d as r3
from .volume import Volume

MODES = (("lung", "肺領域 (CT, HU値)"), ("threshold", "閾値領域 (輝度)"), ("all", "全体 (抽出なし)"))
CANVAS = 560


class RenderWindow:
    def __init__(self, master, vol: Volume):
        self.top = tk.Toplevel(master)
        self.top.title("ボリュームレンダリング")
        self.vol = None
        self.grid = self.grid_lo = None       # マスク適用済みグリッド（フル / 半解像度）
        self.az = self.el = 0.0
        self.zoom = 1.0
        self._photo = None
        self._after = None
        self._drag = None
        self._lock = False
        self._grid_cache = None
        self._rot_key = None
        self._pending = None
        self._build()
        self.set_volume(vol)

    # ------------------------------------------------------------ UI
    def _build(self):
        t = self.top
        self.canvas = tk.Canvas(t, width=CANVAS, height=CANVAS, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._schedule_final())
        self.canvas.bind("<MouseWheel>", lambda e: self._zoom(1.1 if e.delta > 0 else 1 / 1.1))
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1))

        side = tk.Frame(t)
        side.grid(row=0, column=1, sticky="n", padx=6, pady=6)

        reg = tk.LabelFrame(side, text="領域抽出")
        reg.pack(fill="x", pady=3)
        self.mode = tk.StringVar(value="lung")
        for k, label in MODES:
            tk.Radiobutton(reg, text=label, variable=self.mode, value=k, command=self._mode_changed).pack(anchor="w")
        self.thr_lo = self._scale(reg, "閾値(下限) [肺: 空気とみなすHU上限]", None)
        self.thr_hi = self._scale(reg, "閾値(上限) [閾値領域のみ]", None)
        row = tk.Frame(reg)
        row.pack(fill="x")
        tk.Label(row, text="解像度(最大辺):").pack(side="left")
        self.res = tk.StringVar(value="160")
        ttk.Combobox(row, textvariable=self.res, values=("96", "128", "160", "192", "256"), width=5,
                     state="readonly").pack(side="left")
        tk.Button(reg, text="領域抽出 / 再計算", command=self.extract).pack(fill="x", pady=3)

        tfr = tk.LabelFrame(side, text="表示（伝達関数）")
        tfr.pack(fill="x", pady=3)
        self.tf_lo = self._scale(tfr, "輝度下限（透明になる値）", self._redraw)
        self.tf_hi = self._scale(tfr, "輝度上限（最も不透明な値）", self._redraw)
        self.opacity = self._scale(tfr, "不透明度 (%)", self._redraw, frm=1, to=100, res=1)
        self.gamma = self._scale(tfr, "ガンマ (×0.1)", self._redraw, frm=2, to=40, res=1)
        self.opacity.set(50)
        self.gamma.set(10)
        row = tk.Frame(tfr)
        row.pack(fill="x")
        self.cmap = tk.StringVar(value="lung")
        cb = ttk.Combobox(row, textvariable=self.cmap, values=r3.CMAP_NAMES, width=7, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        self.rmode = tk.StringVar(value="composite")
        for k, label in (("composite", "合成"), ("mip", "MIP")):
            tk.Radiobutton(row, text=label, variable=self.rmode, value=k, command=self._redraw).pack(side="left")
        self.shading = tk.BooleanVar(value=True)
        tk.Checkbutton(row, text="陰影", variable=self.shading, command=self._redraw).pack(side="left")

        btn = tk.Frame(side)
        btn.pack(fill="x", pady=3)
        tk.Button(btn, text="視点リセット", command=self.reset_view).pack(side="left", expand=True, fill="x")
        tk.Button(btn, text="PNG保存…", command=self.save_png).pack(side="left", expand=True, fill="x")

        self.msg = tk.StringVar(value="")
        tk.Label(t, textvariable=self.msg, anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew")

    def _scale(self, parent, label, cmd, frm=0, to=1, res=1):
        s = tk.Scale(parent, from_=frm, to=to, orient="horizontal", label=label, resolution=res, length=270)
        if cmd is not None:
            s.config(command=lambda _v: cmd())
        s.pack(fill="x")
        return s

    # ------------------------------------------------------------ データ設定
    def set_volume(self, vol: Volume):
        self.vol = vol
        self._lock = True
        lo, hi = vol.vmin, vol.vmax
        res = 1 if hi - lo > 20 else (hi - lo) / 200.0
        for s in (self.thr_lo, self.thr_hi, self.tf_lo, self.tf_hi):
            s.config(from_=lo, to=hi, resolution=res)
        if vol.modality == "CT" or lo < -500:
            self.mode.set("lung")
            self.thr_lo.set(-400)
            self.thr_hi.set(hi)
            self.tf_lo.set(max(lo, -950))
            self.tf_hi.set(-300)
            self.cmap.set("lung")
        else:
            self.mode.set("threshold")
            sample = vol.data[:: max(1, vol.shape[0] // 16)]
            self.thr_lo.set(r3.otsu(sample))
            self.thr_hi.set(hi)
            self.tf_lo.set(self.thr_lo.get())
            self.tf_hi.set(float(np.percentile(sample, 99.5)))
            self.cmap.set("bone")
        self._lock = False
        self.reset_view(redraw=False)
        self.extract()

    # ------------------------------------------------------------ 処理
    def _mode_changed(self):
        self.extract()

    def extract(self):
        """等方リサンプル -> 領域抽出 -> マスク適用グリッドを作る"""
        if self.vol is None:
            return
        self.top.config(cursor="watch")
        self.msg.set("領域抽出中…")
        self.top.update_idletasks()
        try:
            key = (id(self.vol), int(self.res.get()))
            if self._grid_cache is None or self._grid_cache[0] != key:      # 閾値だけ変えた再計算ではリサンプルを省く
                self._grid_cache = (key,) + r3.prepare_grid(self.vol, key[1])
            _, grid, vox = self._grid_cache
            mask = r3.segment(grid, self.mode.get(), float(self.thr_lo.get()), float(self.thr_hi.get()))
            if not mask.any():
                self.msg.set("領域が見つかりませんでした。閾値や領域抽出モードを変更してください")
                self.grid = self.grid_lo = None
                self.canvas.delete("all")
                return
            fill = float(self.vol.vmin)
            vol_ml = mask.sum() * (vox ** 3) / 1000.0
            grid, mask = r3.crop_to_mask(grid, mask)                        # 領域の外接ボックスだけを以後の計算対象にする
            self.grid = np.where(mask, grid, fill).astype(np.float32)
            self.grid_lo = ndi.zoom(self.grid, 0.5, order=1, prefilter=False)
            self._fill = fill
            self._rot_key = None
            self.msg.set(f"領域体積 約 {vol_ml:.0f} mL   (グリッド {self.grid.shape}, {vox:.2f} mm/voxel)")
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("領域抽出エラー", str(ex), parent=self.top)
            return
        finally:
            self.top.config(cursor="")
        self._draw(final=True)

    def _redraw(self):
        if self._lock:
            return
        self._schedule_final(preview_now=True)

    def _params(self):
        return dict(tf_lo=float(self.tf_lo.get()), tf_hi=float(self.tf_hi.get()),
                    opacity=self.opacity.get() / 100.0, gamma=self.gamma.get() / 10.0,
                    cmap=self.cmap.get(), mode=self.rmode.get())

    def _draw(self, final: bool):
        if self.grid is None:
            return
        g = self.grid if final else self.grid_lo
        key = (id(g), round(self.az, 3), round(self.el, 3))
        if key != self._rot_key:               # 視点が同じ(伝達関数だけ変更)なら回転結果を再利用
            self._rot, self._rot_key = r3.rotate_volume(g, self.az, self.el, self._fill), key
        p = self._params()
        img = r3.composite(self._rot, r3.image_size(g.shape), p["tf_lo"], p["tf_hi"], p["opacity"], p["gamma"],
                           p["cmap"], p["mode"], self.shading.get() and final)
        side = max(2, int(CANVAS * self.zoom))
        pil = Image.fromarray(img).resize((side, side), Image.BILINEAR)
        self._last = pil
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_image(max(w, 2) // 2, max(h, 2) // 2, image=self._photo, anchor="center")

    # ------------------------------------------------------------ 操作
    def _schedule_final(self, preview_now=False):
        if self._after is not None:
            self.top.after_cancel(self._after)
        if preview_now and self._pending is None:      # 連続イベントは1回の描画にまとめる
            self._pending = self.top.after_idle(self._flush)
        self._after = self.top.after(250, self._final)

    def _flush(self):
        self._pending = None
        self._draw(final=False)

    def _final(self):
        self._after = None
        self._draw(final=True)

    def _press(self, e):
        self._drag = (e.x, e.y)

    def _motion(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.az = (self.az + dx * 0.6) % 360
        self.el = max(-90.0, min(90.0, self.el + dy * 0.6))
        self._schedule_final(preview_now=True)

    def _zoom(self, f):
        self.zoom = max(0.3, min(4.0, self.zoom * f))
        self._schedule_final(preview_now=True)

    def reset_view(self, redraw=True):
        self.az = self.el = 0.0
        self.zoom = 1.0
        if redraw:
            self._schedule_final(preview_now=True)

    def save_png(self):
        if getattr(self, "_last", None) is None:
            return
        p = filedialog.asksaveasfilename(parent=self.top, defaultextension=".png", filetypes=[("PNG", "*.png")])
        if p:
            self._last.save(p)
