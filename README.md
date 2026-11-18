# Cheat Detection on Online Chess Games — 再現および改善実験プロジェクト

本プロジェクトは，オンラインチェスにおけるニューラルネットワークを用いたチート検出に関する下記論文の再現実験（`ex001`）および過学習抑制・特徴量エンジニアリングによる改善実験（`ex002`）を PyTorch（GPU 環境）で実装・検証したものである．

- **対象論文**: Reyhan Patria, Sean Favian, Anggoro Caturdewa, Derwin Suhartono, *"Cheat Detection on Online Chess Games using Convolutional and Dense Neural Network"*, 2021 4th International Seminar on Research of Information Technology and Intelligent Systems (ISRITI), pp. 389–395, 2021.

---

## ディレクトリ構成

```text
.
├─ ex001/                  # 実験1: 論文再現実験（4モデル比較）
│  ├─ model.py             # Unanalyzed/Analyzed Dense & 3D CNN のモデル定義
│  ├─ utils.py             # データ分割・学習・評価・可視化関数
│  ├─ ex001.ipynb          # 実験1の実行済みノートブック
│  └─ results_summary.csv  # 実験1の定量評価結果サマリ
├─ ex002/                  # 実験2: 過学習抑制・拡張解析特徴量による改善実験
│  ├─ model.py             # BatchNorm・Pool圧縮・1D Temporal CNN 等のモデル定義
│  ├─ utils.py             # Z-score標準化・学習・アブレーション重要度分析関数
│  ├─ ex002.ipynb          # 実験2の実行済みノートブック
│  └─ results_summary.csv  # 実験2の定量評価結果サマリ
├─ report/                 # レポートおよび実験結果図表
│  ├─ img/                 # 各実験の学習曲線・混同行列・比較図
│  ├─ ex001.md             # 実験1レポート
│  ├─ ex002.md             # 実験2レポート
│  └─ report.md            # プロジェクト総合レポート
├─ requirements.txt        # Python 依存パッケージ一覧
├─ LICENSE                 # MIT License
└─ .gitignore              # Git 除外設定
```

※ `datasets/`, `ref/`, `venv/` 等の大容量データ・参考文献・仮想環境ディレクトリは `.gitignore` により除外されている．

---

## 環境構築と実行手順

1. **仮想環境の作成と依存ライブラリのインストール**（Python 3.10.11 / CUDA 11.8 環境で検証）

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

2. **データセット構築（Lichess 公開 DB + Lichess API + Stockfish 19）**
   - 論文の `Cheat-Net` 由来ユーザーリストは非公開であり，かつ Lichess の `tosViolation` ユーザーに対する個別対局 API（`/api/games/user/{id}`）が 404 を返すため，本プロジェクトでは **Lichess Open Database（2021年8月）** と **Lichess User バッチ照会 API（`POST /api/users`）**，および **Stockfish 19** による全手評価値補完を組み合わせた 5,000 対局の実データセットを構築した．

3. **実験ノートブックの実行**
   - `ex001/ex001.ipynb`（論文再現実験）
   - `ex002/ex002.ipynb`（過学習抑制・拡張解析特徴量実験）

---

## 実験結果サマリ

NVIDIA GeForce RTX 4080 を用いて 5,000 対局（Train 3,600 / Val 400 / Test 1,000）で評価した主要結果は以下の通りである．

| 実験 | モデル名 | パラメータ数 | 最終訓練正解率 (%) | 最高検証正解率 (%) | テスト正解率 (%) | Macro F1 (%) | 過学習ギャップ (%pt) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ex001 | Unanalyzed Dense | 36,848,643 | 99.92 | 50.75 | 41.70 | 36.00 | 53.42 |
| ex001 | Unanalyzed CNN | 1,501,347 | 50.11 | 51.00 | 49.80 | 24.31 | 0.61 |
| ex001 | Analyzed Dense | 37,226,755 | 99.67 | 50.50 | 43.50 | 35.94 | 52.67 |
| ex001 | Analyzed CNN | 1,879,459 | 50.06 | 50.00 | 50.00 | 22.22 | 0.06 |
| ex002 | Analyzed-Only MLP | 223,491 | 83.00 | 51.25 | 49.10 | 27.23 | 37.00 |
| ex002 | Temporal ConvNet | **130,051** | 57.19 | **53.75** | 48.50 | **33.16** | **5.19** |
| ex002 | Improved Analyzed Dense | 18,603,459 | 98.67 | 51.75 | 49.60 | 32.58 | 53.67 |
| ex002 | Improved Analyzed CNN | 464,803 | 61.75 | 51.50 | **50.20** | 29.99 | 14.75 |

詳細な分析と考察は各レポートを参照されたい：
- [総合レポート (report/report.md)](report/report.md)
- [実験1レポート (report/ex001.md)](report/ex001.md)
- [実験2レポート (report/ex002.md)](report/ex002.md)

---

## ライセンス

本プロジェクトは [MIT License](LICENSE) の下で公開されている．
