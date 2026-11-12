import sys
import types


class _W:
    registry = []

    def __init__(self, *a, **k):
        self.opts = dict(k)
        self.val = k.get("value", 0)
        self.cmd = k.get("command")
        self.items = []
        _W.registry.append(self)

    managed = False

    def __getattr__(self, name):          # pack/grid/bind/config/... を全て受け付ける
        def f(*a, **k):
            if name in ("pack", "grid", "place"):
                self.managed = True
            if name in ("config", "configure"):
                self.opts.update(k)
                if "command" in k:
                    self.cmd = k["command"]
            return None
        return f

    def winfo_width(self):
        return 340

    def winfo_height(self):
        return 340

    def set(self, v):
        self.val = v
        if self.cmd:
            self.cmd(str(v))

    def get(self):
        return self.val

    def curselection(self):
        return (0,)

    def after(self, ms, fn=None):
        return "id"

    def after_idle(self, fn):
        fn()
        return None


class _Var:
    def __init__(self, master=None, value=None):
        self.v = value

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


def install():
    tk = types.ModuleType("tkinter")
    for n in ("Tk", "Toplevel", "Frame", "Canvas", "Label", "Button", "Scale", "Menu", "Entry", "Listbox",
              "Radiobutton", "Checkbutton", "LabelFrame"):
        setattr(tk, n, type(n, (_W,), {}))
    tk.StringVar = tk.BooleanVar = tk.IntVar = _Var
    tk.TclError = type("TclError", (Exception,), {})
    fd = types.ModuleType("tkinter.filedialog")
    mb = types.ModuleType("tkinter.messagebox")
    ttk = types.ModuleType("tkinter.ttk")
    ttk.Combobox = type("Combobox", (_W,), {})
    mb.errors = []
    mb.showerror = lambda *a, **k: mb.errors.append(a)
    mb.showinfo = lambda *a, **k: None
    tk.filedialog, tk.messagebox, tk.ttk = fd, mb, ttk
    sys.modules.update({"tkinter": tk, "tkinter.filedialog": fd, "tkinter.messagebox": mb, "tkinter.ttk": ttk})

    # ImageTk も偽物に
    from PIL import ImageTk
    class PhotoImage:
        def __init__(self, img):
            self.img = img
    ImageTk.PhotoImage = PhotoImage
    return tk
