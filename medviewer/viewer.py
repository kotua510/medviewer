from __future__ import annotations
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from . import mpr
from .dicomlite import DicomError
from .volume import Volume, load_dicom, load_tiff_stack

APP_TITLE = "医用画像ビューア (MPR + ボリュームレンダリング)"
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


class SlicePanel:
    """1断面ぶんの表示領域: キャンバス + スライス位置/WL/WWスライダー"""

    def __init__(self, parent, app: "Viewer", plane: str):
        self.app, self.plane = app, plane
        self.wl, self.ww = 0.0, 1.0
        self._photo = None
        self._geom = None
        self._img2d = None

        color = mpr.PLANE_COLOR[plane]
        self.frame = tk.Frame(parent, bg="#1e1e1e", highlightthickness=2, highlightbackground=color)
        self.canvas = tk.Canvas(self.frame, bg="black", highlightthickness=0, width=340, height=340)
        self.canvas.pack(fill="both", expand=True)

        ctl = tk.Frame(self.frame, bg="#1e1e1e")
        ctl.pack(fill="x")
        self.s_pos = self._scale(ctl, "スライス位置", self._on_pos)
        self.s_wl = self._scale(ctl, "WL (ウィンドウレベル)", self._on_window)
        self.s_ww = self._scale(ctl, "WW (ウィンドウ幅)", self._on_window)
        tk.Button(ctl, text="WL/WWをリセット", command=lambda: self.app.reset_window(self.plane)).pack(fill="x", padx=4, pady=2)

        c = self.canvas
        c.bind("<Configure>", lambda e: self.draw())
        c.bind("<Button-1>", self._on_click)
        c.bind("<B1-Motion>", self._on_click)
        c.bind("<Motion>", self._on_motion)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", lambda e: self._step(+1))
        c.bind("<Button-5>", lambda e: self._step(-1))

    def _scale(self, parent, label, cmd):
        sc = tk.Scale(parent, from_=0, to=1, orient="horizontal", label=label, resolution=1,
                      command=cmd, bg="#1e1e1e", fg="#dddddd", troughcolor="#444444",
                      highlightthickness=0, length=300)
        sc.pack(fill="x")
        return sc

    # ---- ボリューム設定
    def set_volume(self, vol: Volume):
        n = mpr.n_slices(vol.shape, self.plane)
        self.s_pos.config(from_=0, to=n - 1)
        span = vol.vmax - vol.vmin
        res = 1 if span > 20 else span / 200.0
        self.s_wl.config(from_=vol.vmin, to=vol.vmax, resolution=res)
        self.s_ww.config(from_=1 if span > 20 else res, to=max(span, 1), resolution=res)
        self.set_window(vol.wl, vol.ww)

    def set_window(self, wl, ww):
        self.wl, self.ww = float(wl), float(ww)
        self.s_wl.set(self.wl)
        self.s_ww.set(self.ww)

    # ---- イベント
    def _on_pos(self, v):
        if self.app.busy:
            return
        self.app.set_slice(self.plane, int(float(v)))

    def _on_window(self, _v):
        if self.app.busy:
            return
        self.app.set_window(self.plane, float(self.s_wl.get()), float(self.s_ww.get()))

    def _step(self, d):
        if self.app.vol is None:
            return
        idx = self.app.pos[mpr.plane_axis(self.plane)] + d
        n = mpr.n_slices(self.app.vol.shape, self.plane)
        self.app.set_slice(self.plane, max(0, min(n - 1, idx)))

    def _on_wheel(self, e):
        self._step(+1 if e.delta > 0 else -1)

    def _on_click(self, e):
        if self.app.vol is None or self._geom is None:
            return
        col, row = self._geom.to_image(e.x, e.y)
        col = min(max(col, 0), self._geom.ncols - 1e-6)
        row = min(max(row, 0), self._geom.nrows - 1e-6)
        self.app.set_pos(mpr.pos_from_image(self.plane, col, row, self.app.pos, self.app.vol.shape))

    def _on_motion(self, e):
        if self.app.vol is None or self._geom is None or self._img2d is None:
            return
        if not self._geom.inside(e.x, e.y):
            return
        col, row = self._geom.to_image(e.x, e.y)
        c, r = int(col), int(row)
        if 0 <= r < self._img2d.shape[0] and 0 <= c < self._img2d.shape[1]:
            self.app.status_cursor(f"{mpr.PLANE_LABEL[self.plane]} ({c}, {r})  値 = {self._img2d[r, c]:g}")

    # ---- 描画
    def draw(self, image=True):
        """image=False のときは位置線だけ描き直す（スライス不変の他パネル用）"""
        vol = self.app.vol
        if vol is None:
            self.canvas.delete("all")
            return
        cw, ch = max(self.canvas.winfo_width(), 16), max(self.canvas.winfo_height(), 16)
        if image or self._img2d is None:
            idx = self.app.pos[mpr.plane_axis(self.plane)]
            self._img2d, ps = mpr.extract_slice(vol.data, vol.spacing, self.plane, idx)
            self._geom = mpr.fit_geom(cw, ch, self._img2d.shape, ps)
            g = self._geom
            u8 = mpr.apply_window(self._img2d, self.wl, self.ww)
            pil = Image.fromarray(u8).resize((g.dw, g.dh), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(g.ox, g.oy, image=self._photo, anchor="nw")
            n = mpr.n_slices(vol.shape, self.plane)
            self.canvas.create_text(8, 6, anchor="nw", fill="#ffd54a", font=("TkDefaultFont", 10, "bold"),
                                    text=f"{mpr.PLANE_LABEL[self.plane]}  {idx + 1}/{n}")
            self.canvas.create_text(8, ch - 6, anchor="sw", fill="#ffd54a", font=("TkDefaultFont", 9),
                                    text=f"WL {self.wl:g}  WW {self.ww:g}")
        self.draw_lines()

    def draw_lines(self):
        self.canvas.delete("line")
        g = self._geom
        if g is None or self.app.vol is None or not self.app.show_lines.get():
            return
        col, row = mpr.crosshair(self.plane, self.app.pos, self.app.vol.shape)
        x, y = g.to_canvas(col, row)
        for orient, other in mpr.lines_for(self.plane):
            color = mpr.PLANE_COLOR[other]
            if orient == "v":
                self.canvas.create_line(x, g.oy, x, g.oy + g.dh, fill=color, width=1, tags="line")
            else:
                self.canvas.create_line(g.ox, y, g.ox + g.dw, y, fill=color, width=1, tags="line")


class Viewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.vol: Volume | None = None
        self.pos = (0, 0, 0)  # (z, y, x)
        self.busy = False
        self.show_lines = tk.BooleanVar(value=True)
        self.win3d = None
        self._tiff_opts = {"orientation": "axial", "spacing": (1.0, 1.0, 1.0), "top_is_superior": False}

        root.title(APP_TITLE)
        root.geometry("1250x720")
        root.configure(bg="#1e1e1e")
        self._build_menu()

        body = tk.Frame(root, bg="#1e1e1e")
        body.pack(fill="both", expand=True)
        self.panels = {}
        for i, plane in enumerate(mpr.PLANES):
            p = SlicePanel(body, self, plane)
            p.frame.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
            body.columnconfigure(i, weight=1, uniform="p")
            self.panels[plane] = p
        body.rowconfigure(0, weight=1)

        self.status = tk.StringVar(value="ファイル > DICOMフォルダを開く… でデータを読み込んでください")
        self.cursor = tk.StringVar(value="")
        bar = tk.Frame(root, bg="#2b2b2b")
        bar.pack(fill="x")
        tk.Label(bar, textvariable=self.status, anchor="w", bg="#2b2b2b", fg="#dddddd").pack(side="left", padx=6)
        tk.Label(bar, textvariable=self.cursor, anchor="e", bg="#2b2b2b", fg="#9ad").pack(side="right", padx=6)

    # ------------------------------------------------------------ メニュー（観点2）
    def _build_menu(self):
        mb = tk.Menu(self.root)
        f = tk.Menu(mb, tearoff=0)
        f.add_command(label="DICOMフォルダを開く…", command=self.open_dicom_folder, accelerator="Ctrl+O")
        f.add_command(label="DICOMファイルを開く…（複数選択可）", command=self.open_dicom_files)
        f.add_command(label="TIFFスタックを開く…", command=self.open_tiff)
        f.add_separator()
        sm = tk.Menu(f, tearoff=0)
        sm.add_command(label="胸部CTファントム (DICOM)", command=lambda: self._open_sample_dicom("dicom_ct_phantom"))
        sm.add_command(label="頭部MRI (DICOM)", command=lambda: self._open_sample_dicom("dicom_mri_head"))
        sm.add_command(label="CT_coronal.tif", command=lambda: self._open_sample_tiff("CT_coronal.tif", "coronal"))
        sm.add_command(label="CT_sagittal.tif", command=lambda: self._open_sample_tiff("CT_sagittal.tif", "sagittal"))
        sm.add_command(label="mri-stack.tif", command=lambda: self._open_sample_tiff("mri-stack.tif", "axial"))
        f.add_cascade(label="サンプルデータ", menu=sm)
        f.add_separator()
        f.add_command(label="終了", command=self.root.destroy)
        mb.add_cascade(label="ファイル", menu=f)

        v = tk.Menu(mb, tearoff=0)
        v.add_checkbutton(label="位置線を表示", variable=self.show_lines, command=self._refresh_lines)
        v.add_command(label="全断面のWL/WWをリセット", command=lambda: self.reset_window(None))
        v.add_command(label="位置・WL/WWをすべてリセット", command=self.reset)
        mb.add_cascade(label="表示", menu=v)

        t = tk.Menu(mb, tearoff=0)
        t.add_command(label="ボリュームレンダリング(3D)を開く…", command=self.open_3d)
        mb.add_cascade(label="3D", menu=t)

        h = tk.Menu(mb, tearoff=0)
        h.add_command(label="操作方法", command=self.show_help)
        mb.add_cascade(label="ヘルプ", menu=h)
        self.root.config(menu=mb)
        self.root.bind("<Control-o>", lambda e: self.open_dicom_folder())

    # ------------------------------------------------------------ 読み込み
    def open_dicom_folder(self):
        d = filedialog.askdirectory(title="DICOMフォルダを選択")
        if d:
            self._load(lambda: load_dicom([d], self._choose_series, self._progress))

    def open_dicom_files(self):
        fs = filedialog.askopenfilenames(title="DICOMファイルを選択（複数可）")
        if fs:
            self._load(lambda: load_dicom(list(fs), self._choose_series, self._progress))

    def open_tiff(self):
        path = filedialog.askopenfilename(title="TIFFスタックを選択",
                                          filetypes=[("TIFF", "*.tif *.tiff"), ("すべて", "*.*")])
        if not path:
            return
        opts = self._ask_tiff_options(os.path.basename(path))
        if opts is None:
            return
        self._tiff_opts = opts
        self._load(lambda: load_tiff_stack(path, opts["orientation"], opts["spacing"], opts["top_is_superior"]))

    def _open_sample_dicom(self, name):
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.isdir(path):
            messagebox.showerror("サンプルなし", f"{path} がありません。tools/make_sample_dicom.py で生成できます。")
            return
        self._load(lambda: load_dicom([path], self._choose_series, self._progress))

    def _open_sample_tiff(self, fname, orientation):
        path = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(path):
            messagebox.showerror("サンプルなし", f"{path} がありません")
            return
        # TIFFには画素間隔の情報が無いので、3Dできれいに見える値を設定する
        sp = (0.781, 0.781, 2.5)
        self._load(lambda: load_tiff_stack(path, orientation, sp))

    def _load(self, fn):
        self.root.config(cursor="watch")
        self.status.set("読み込み中…")
        self.root.update_idletasks()
        try:
            vol = fn()
        except Exception as ex:  # noqa: BLE001 - 利用者に原因を表示する
            self.root.config(cursor="")
            self.status.set("読み込みに失敗しました")
            if not isinstance(ex, DicomError) or "キャンセル" not in str(ex):
                messagebox.showerror("読み込みエラー", str(ex))
            return
        self.root.config(cursor="")
        self.set_volume(vol)

    def _progress(self, i, n):
        if i % 10 == 0:
            self.status.set(f"読み込み中… {i}/{n}")
            self.root.update_idletasks()

    def _choose_series(self, descriptions):
        top = tk.Toplevel(self.root)
        top.title("シリーズを選択")
        top.transient(self.root)
        tk.Label(top, text="複数のシリーズがあります。表示するシリーズを選んでください。").pack(padx=10, pady=6)
        lb = tk.Listbox(top, width=80, height=min(12, len(descriptions)), exportselection=False)
        for d in descriptions:
            lb.insert("end", d)
        lb.selection_set(0)
        lb.pack(padx=10)
        result = {"i": -1}

        def ok():
            sel = lb.curselection()
            result["i"] = sel[0] if sel else 0
            top.destroy()

        tk.Button(top, text="OK", command=ok).pack(pady=8)
        lb.bind("<Double-Button-1>", lambda e: ok())
        top.grab_set()
        self.root.wait_window(top)
        return result["i"]

    def _ask_tiff_options(self, fname):
        top = tk.Toplevel(self.root)
        top.title("TIFFスタックの設定")
        top.transient(self.root)
        tk.Label(top, text=f"{fname}\nページ(フレーム)の断面の向き:").grid(row=0, column=0, columnspan=2, padx=10, pady=6)
        ori = tk.StringVar(value=self._tiff_opts["orientation"])
        for i, (k, t) in enumerate([("axial", "Axial（ページ=Z）"), ("coronal", "Coronal（ページ=Y、行=Z）"),
                                    ("sagittal", "Sagittal（ページ=X、行=Z）")]):
            tk.Radiobutton(top, text=t, variable=ori, value=k).grid(row=1 + i, column=0, columnspan=2, sticky="w", padx=14)
        sp = self._tiff_opts["spacing"]
        vars_ = [tk.StringVar(value=str(v)) for v in sp]
        for i, lab in enumerate(("画素間隔 X [mm]", "画素間隔 Y [mm]", "スライス間隔 Z [mm]")):
            tk.Label(top, text=lab).grid(row=4 + i, column=0, sticky="e", padx=6, pady=2)
            tk.Entry(top, textvariable=vars_[i], width=10).grid(row=4 + i, column=1, sticky="w")
        flip = tk.BooleanVar(value=self._tiff_opts["top_is_superior"])
        tk.Checkbutton(top, text="画像の上端が頭側（Coronal/Sagittalのみ）", variable=flip).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=14)
        result = {}

        def ok():
            try:
                vals = tuple(float(v.get()) for v in vars_)
                if min(vals) <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("入力エラー", "間隔は正の数値で入力してください", parent=top)
                return
            result["v"] = {"orientation": ori.get(), "spacing": vals, "top_is_superior": bool(flip.get())}
            top.destroy()

        tk.Button(top, text="OK", command=ok).grid(row=8, column=0, columnspan=2, pady=10)
        top.grab_set()
        self.root.wait_window(top)
        return result.get("v")

    # ------------------------------------------------------------ 状態更新
    def set_volume(self, vol: Volume):
        self.vol = vol
        z, y, x = vol.shape
        self.pos = (z // 2, y // 2, x // 2)
        self.busy = True
        for p in self.panels.values():
            p.set_volume(vol)
        self._sync_sliders()
        self.busy = False
        self.status.set(f"{vol.name}   {vol.info}")
        self.root.update_idletasks()
        self.redraw_all()
        if self.win3d is not None:
            try:
                self.win3d.set_volume(vol)
            except tk.TclError:
                self.win3d = None

    def _sync_sliders(self):
        z, y, x = self.pos
        for plane, v in ((mpr.AXIAL, z), (mpr.CORONAL, y), (mpr.SAGITTAL, x)):
            self.panels[plane].s_pos.set(v)

    def redraw_all(self):
        for p in self.panels.values():
            p.draw(image=True)

    def _refresh_lines(self):
        for p in self.panels.values():
            p.draw_lines()

    def set_slice(self, plane, idx):
        """スライダー操作: 該当断面の画像を更新し、他断面は位置線だけ動かす（観点6）"""
        if self.vol is None:
            return
        pos = list(self.pos)
        pos[mpr.plane_axis(plane)] = idx
        self._apply_pos(tuple(pos))

    def set_pos(self, pos):
        """画像クリック: 3つ全てのスライス位置を更新"""
        self._apply_pos(pos)

    def _apply_pos(self, pos):
        old, self.pos = self.pos, tuple(pos)
        self.busy = True
        self._sync_sliders()
        self.busy = False
        for plane, p in self.panels.items():
            k = mpr.plane_axis(plane)
            p.draw(image=(old[k] != self.pos[k]))

    def set_window(self, plane, wl, ww):
        if self.vol is None:
            return
        p = self.panels[plane]          # WL/WWは断面ごとに独立（他の断面には影響しない）
        p.wl, p.ww = wl, ww
        p.draw(image=True)

    def reset_window(self, plane=None):
        """WL/WWを読み込み時の値に戻す。plane=None なら全断面"""
        if self.vol is None:
            return
        for t in ([plane] if plane else list(mpr.PLANES)):
            p = self.panels[t]
            self.busy = True
            p.set_window(self.vol.wl, self.vol.ww)
            self.busy = False
            p.draw(image=True)

    def reset(self):
        if self.vol is not None:
            self.set_volume(self.vol)

    def status_cursor(self, text):
        self.cursor.set(text)

    # ------------------------------------------------------------ その他
    def open_3d(self):
        if self.vol is None:
            messagebox.showinfo("3D", "先にデータを読み込んでください")
            return
        from .render_window import RenderWindow
        if self.win3d is not None:
            try:
                self.win3d.top.lift()
                return
            except tk.TclError:
                self.win3d = None
        self.win3d = RenderWindow(self.root, self.vol)

    def show_help(self):
        messagebox.showinfo("操作方法", (
            "・各断面の下のスライダーでスライス位置 / WL / WW を変更\n"
            "・画像上をクリック/ドラッグすると3断面の交点が移動\n"
            "・画像上でマウスホイール: スライス送り\n"
            "・色付きの線: 他の断面の位置（赤=Axial, 緑=Coronal, 青=Sagittal）\n"
            "・3D > ボリュームレンダリング: 領域抽出して3D表示"))
