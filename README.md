# 医用画像ビューア（MPR 3断面 + ボリュームレンダリング）

知能情報実験実習２ 医用画像処理実験 最終課題（Python / tkinter 製）

DICOM を読み込み、Axial / Coronal / Sagittal の 3 断面を表示する自作ビューアです。
抽出した領域（肺など）のボリュームレンダリング（3D 表示）も付けています。

## 動作環境

| 項目 | 内容 |
|---|---|
| Python | **3.13** で動作確認（3.10 以上を想定。`.python-version` に記載） |
| GUI | tkinter（Python 標準。Windows / macOS の公式インストーラ版には最初から含まれる） |
| ライブラリ | `requirements.txt` の通り（numpy・scipy・Pillow）。動作確認は numpy 2.5.3 / scipy 1.18.1 / Pillow 12.3.0 |

Linux で `No module named 'tkinter'` と出る場合は `sudo apt install python3-tk` を実行してください。
DICOM は内蔵リーダ（`medviewer/dicomlite.py`）で読むため **pydicom は不要**です
（JPEG 等で圧縮された DICOM を読みたい場合のみ、任意で `pip install pydicom`）。

## インストールと起動

```bash
# リポジトリのフォルダへ移動してから

# インストール済みPythonの確認（Windows）
py -0p

# 仮想環境の作成（いずれか1つを使用）
python -m venv .venv
python3 -m venv .venv
py -m venv .venv
py -3.12 -m venv .venv  # Python 3.12を指定する場合
```

```bash
# 仮想環境の有効化
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (コマンドプロンプト)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

```bash
# ライブラリのインストール
python -m pip install -r requirements.txt

# アプリケーションの起動
python main.py
```

起動後、メニュー **ファイル > サンプルデータ** からすぐに動作を確認できます。
`python main.py <DICOMフォルダ>` でフォルダを指定して起動することもできます。

## 操作方法

### ファイルメニュー

| メニュー | 内容 |
|---|---|
| ファイル > DICOMフォルダを開く… (Ctrl+O) | フォルダ（サブフォルダ含む）内の DICOM を読み込む。複数シリーズがあれば選択ダイアログが出る |
| ファイル > DICOMファイルを開く… | ファイルを複数選択して読み込む |
| ファイル > TIFFスタックを開く… | マルチページ TIFF を読み込む（断面の向き・画素間隔・上端が頭側かどうかを指定。下記「TIFF の読み込み設定」参照） |
| ファイル > サンプルデータ | 同梱の胸部 CT ファントム / 頭部 MRI（DICOM）と、TIFF 3 枚（TIFF は下記の推奨設定で読み込まれる） |

DICOM はスライス位置（ImagePositionPatient）でソートし、RescaleSlope/Intercept を適用して HU 値にします。
画素間隔・スライス間隔から縦横比を正しく保って表示します。

### 3断面表示とスライダー

画面は左から **Axial / Coronal / Sagittal** の 3 パネルです。各パネルの下に 3 本のスライダーがあります。

- **スライス位置**: その断面のスライスを送る
- **WL（ウィンドウレベル） / WW（ウィンドウ幅）**: 濃淡の調整。**Axial / Coronal / Sagittal それぞれ独立**して変更できる
- **WL/WWをリセット**ボタン: その断面のWL/WWを読み込み時の値に戻す（メニュー「表示 > 全断面のWL/WWをリセット」で一括リセット）

各断面画像には **他の 2 断面の位置を表す線** が表示されます（赤＝Axial、緑＝Coronal、青＝Sagittal の位置）。
スライダーを動かすと、他の断面上の線も連動して動きます（例: Axial のスライダー → Coronal / Sagittal の横線が動く）。

その他の操作:

- 画像上をクリック／ドラッグ: 3 断面の交点を移動（スライダーも連動）
- 画像上でマウスホイール: スライス送り
- 画像上にマウスを置くと、右下に画素位置と値（CT なら HU）を表示
- 表示 > 位置線を表示: 線の表示／非表示
- 表示 > 位置・WL/WWをすべてリセット: スライス位置と全断面の WL/WW を読み込み直後の状態に戻す

### ボリュームレンダリング

メニュー **3D > ボリュームレンダリング(3D)を開く…** で別ウィンドウが開きます。

1. **領域抽出**
   - *肺領域 (CT・HU値)*: HU < 閾値（初期値 −400）を空気とし、体外の空気（画像の辺に接する成分）を除去して大きい 2 成分（左右の肺）を残し、スライスごとに穴埋め（血管を領域に含める）→ 3D オープニング
   - *閾値領域 (輝度)*: 下限〜上限の範囲の最大連結成分（頭部 MRI の頭部など）
   - *全体*: 抽出なし
2. **表示**: 伝達関数（輝度の下限/上限、不透明度、ガンマ）、カラーマップ、合成 / MIP、陰影の有無
3. **操作**: 左ドラッグで回転、ホイールで拡大縮小。ドラッグ中は低解像度で描画し、止めると高画質で再描画
4. 「領域抽出 / 再計算」で閾値や解像度を変えて再抽出、「PNG保存…」で画像保存

## ファイル構成

```
main.py                   起動スクリプト
requirements.txt          必要ライブラリ
medviewer/
  dicomlite.py            DICOM リーダ（依存なし）
  volume.py               ボリュームの読み込み（DICOM / TIFF）
  mpr.py                  断面の切り出し・位置線・座標変換・ウィンドウ処理
  viewer.py               MPR ビューア GUI
  render3d.py             領域抽出とボリュームレンダリング
  render_window.py        3D 表示ウィンドウ GUI
  dicom_writer.py         テスト・サンプル生成用の DICOM ライタ
tools/make_sample_dicom.py
tests/                    単体テスト
sample_data/
```