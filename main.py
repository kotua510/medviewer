"""医用画像ビューア起動スクリプト:  python main.py [DICOMフォルダ]"""
import sys
import tkinter as tk

from medviewer.viewer import Viewer
from medviewer.volume import load_dicom


def main():
    root = tk.Tk()
    app = Viewer(root)
    if len(sys.argv) > 1:
        app._load(lambda: load_dicom([sys.argv[1]], app._choose_series, app._progress))
    root.mainloop()


if __name__ == "__main__":
    main()
