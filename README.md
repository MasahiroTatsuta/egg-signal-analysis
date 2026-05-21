# EGG Signal Analysis with EEMD

胃電図（Electrogastrogram: EGG）からの胃電気活動抽出を目的とした
信号処理・解析パイプラインの Python 実装です。

## 概要

本リポジトリは東京都市大学大学院 医用工学専攻における修士研究
「経験的モード分解を中心とした信号解析による胃電気活動の抽出」の
実装コードを整理・公開したものです。

- **優秀研究賞** 受賞（東京都市大学大学院 医用工学専攻）
- 国内外学会発表 10 件（IEEE EMBC 2019 Berlin 含む）
- 筆頭著者論文 1 件（生体医工学誌, Vol.57, No.6, 2019）

> **Note:** 実測 EGG データは被験者プライバシー保護のため非公開です。
> 本リポジトリでは Van der Pol 方程式による数値シミュレーションデータを使用します。

---

## 研究背景

EGG は胃の蠕動運動に伴う微弱な電気活動（約 0.05 Hz, 100 μV）を
体表から非侵襲的に計測する手法です。
機能性ディスペプシアの診断や自律神経活動評価への応用が期待されています。

EGG 信号には他臓器由来のアーチファクトが重畳するため、
胃電気活動成分の精度よい抽出が技術的課題でした。
本研究ではアンサンブル経験的モード分解（EEMD）の
AWGN パラメータを最適化することで、従来手法より高い抽出精度を実証しました。

---

## リポジトリ構造

```
egg-signal-analysis/
├── README.md
├── requirements.txt
├── simulation/
│   └── van_der_pol.py              # 胃電気活動数理モデル（Van der Pol 方程式）
├── preprocessing/
│   └── bandpass_filter.py          # 基線補正・バンドパスフィルタ (0.02–0.15 Hz)
├── signal_decomposition/
│   ├── emd_analysis.py             # EMD による IMF 抽出
│   └── eemd_analysis.py            # EEMD による eIMF 抽出（メイン手法）
├── analysis/
│   ├── fft_spectrum.py             # FFT パワースペクトル算出
│   ├── hilbert_transform.py        # ヒルベルト変換・瞬時周波数時間推移
│   └── coherence.py                # 2 チャンネル間コヒーレンス解析
├── data_utils/
│   └── data_splitter.py            # 時間窓分割ユーティリティ
└── data/sample/                    # シミュレーション生成サンプルデータ（CSV）
    ├── gastric_activity_clean.csv      # Van der Pol 数値解（ノイズなし、256 s）
    ├── egg_model_noisy.csv             # EGG モデル（ノイズあり、256 s）
    ├── egg_preprocessed.csv            # バンドパスフィルタ後（256 s）
    ├── egg_long_1024s.csv              # 長時間シミュレーション（1024 s）
    ├── time_axis_256s.csv              # 時間軸
    ├── power_spectrum.csv              # FFT パワースペクトル
    ├── instantaneous_frequency.csv     # 瞬時周波数時間推移
    └── amplitude_envelope.csv          # 瞬時振幅包絡線
```

---

## 処理フロー

```
[胃電気活動数理モデル]            [実測 EGG]
  simulation/van_der_pol.py        （被験者データにつき非公開）
          ↓
[前処理]
  preprocessing/bandpass_filter.py
  基線補正 → バンドパスフィルタ (0.02–0.15 Hz)
          ↓
        ┌─────────────────────────┐
        │                         │
[EMD]                         [EEMD]  ← メイン手法
signal_decomposition/         signal_decomposition/
emd_analysis.py               eemd_analysis.py
IMF 抽出                      AWGN パラメータ最適化 (AWGNL = 4 dB)
                              eIMF 自動選択
        └──────────┬──────────────┘
                   ↓
[評価・解析]
  analysis/fft_spectrum.py        # パワースペクトル比較
  analysis/hilbert_transform.py   # 瞬時周波数時間推移・相対標準偏差
  analysis/coherence.py           # 2ch コヒーレンス最大値
```

---

## 主な実装内容

| モジュール | 内容 |
|---|---|
| `simulation/van_der_pol.py` | Van der Pol 方程式の数値シミュレーション（RK45 法） |
| `preprocessing/bandpass_filter.py` | バターワースバンドパスフィルタ（ゼロ位相処理） |
| `signal_decomposition/emd_analysis.py` | EMD・IMF 抽出・胃電気活動帯域での自動選択 |
| `signal_decomposition/eemd_analysis.py` | EEMD・AWGNL 最適化・eIMF 自動選択 |
| `analysis/fft_spectrum.py` | FFT パワースペクトル・ピーク周波数算出 |
| `analysis/hilbert_transform.py` | 瞬時振幅包絡線・瞬時周波数時間推移・相対標準偏差 |
| `analysis/coherence.py` | 2ch コヒーレンス算出・胃電気活動帯域最大値・手法比較 |
| `data_utils/data_splitter.py` | 安定区間抽出・重複あり時間窓分割・CSV 保存・読み込み |

---

## サンプルデータについて

`data/sample/` に Van der Pol 方程式で生成したサンプル CSV が含まれています。
クローン後すぐに各モジュールを動かして確認できます。

| ファイル | 説明 | カラム |
|---|---|---|
| `gastric_activity_clean.csv` | ノイズなし数値解（256 s） | time_s, amplitude |
| `egg_model_noisy.csv` | ノイズあり EGG モデル（256 s） | time_s, amplitude |
| `egg_preprocessed.csv` | バンドパスフィルタ後（256 s） | time_s, amplitude |
| `egg_long_1024s.csv` | 長時間シミュレーション（1024 s） | time_s, amplitude |
| `time_axis_256s.csv` | 時間軸のみ | time_s |
| `power_spectrum.csv` | FFT パワースペクトル | frequency_hz, psd |
| `instantaneous_frequency.csv` | 瞬時周波数時間推移 | time_s, inst_freq_hz |
| `amplitude_envelope.csv` | 瞬時振幅包絡線 | time_s, envelope |

---

## インストール

```bash
pip install numpy scipy matplotlib PyEMD
```

---

## 使い方

### シミュレーションデータの生成から解析まで

```python
import numpy as np
from simulation.van_der_pol import simulate_gastric_activity
from preprocessing.bandpass_filter import preprocess
from signal_decomposition.eemd_analysis import extract_gastric_activity
from analysis.fft_spectrum import compute_power_spectrum
from analysis.hilbert_transform import compute_instantaneous_frequency

# 1. シミュレーションデータの生成
t, x_clean, x_noisy = simulate_gastric_activity(duration=256, noise_std=0.01)

# 2. 前処理（基線補正 + バンドパスフィルタ）
x_preprocessed = preprocess(x_noisy, fs=1.0)

# 3. EEMD で胃電気活動を抽出（AWGNL = 4 dB が最適）
extracted, eimfs, idx = extract_gastric_activity(
    x_preprocessed, t, awgnl_db=4.0, trials=100
)

# 4. パワースペクトル算出
freqs, psd = compute_power_spectrum(extracted, fs=1.0)

# 5. 瞬時周波数時間推移の算出
envelope, inst_freq, _ = compute_instantaneous_frequency(extracted, fs=1.0)
```

### サンプル CSV を使う場合

```python
import numpy as np

data = np.genfromtxt("data/sample/egg_preprocessed.csv",
                     delimiter=",", skip_header=1)
t, x = data[:, 0], data[:, 1]
```

### 各モジュールの単体実行

```bash
python simulation/van_der_pol.py        # シミュレーション + サンプルデータ生成
python preprocessing/bandpass_filter.py # フィルタ処理の確認
python signal_decomposition/eemd_analysis.py  # EEMD 解析
python analysis/fft_spectrum.py         # パワースペクトル
python analysis/hilbert_transform.py    # 瞬時周波数
python analysis/coherence.py            # コヒーレンス比較
python data_utils/data_splitter.py      # 時間窓分割
```

---

## 主な結果

EGG 数値解・実測 EGG の両評価において、AWGNL = 4 dB に最適化した EEMD が
バンドパスフィルタ・従来 EMD を上回る胃電気活動抽出性能を示した。

- **パワースペクトル RMSE**：EEMD が最小（EMD 比で約 7% 改善）
- **瞬時周波数時間推移 RMSE**：EEMD が最小（BPF 比で約 85% 減）
- **2ch コヒーレンス**：BPF に次いで EEMD が高く維持（胃電気活動を除去せずに雑音のみ除去）

---

## 参考文献

- Wu Z, Huang NE: Ensemble empirical mode decomposition — A noise-assisted data analysis method.
  *Advances in Adaptive Data Analysis*, 1(1), pp.1–41, 2009.
- Sarna SK, Daniel EE, Kingma YJ: Simulation of slow-wave electrical activity of small intestine.
  *Am J Physiol*, 221, pp.166–175, 1971.
- 辰田昌洋ら: 胃電図の数理モデルと健常者からの計測データを用いた妥当性評価.
  *生体医工学*, Vol.57, No.6, pp.198–205, 2019.
- 辰田昌洋ら: Parameter Optimization in Ensemble Empirical Mode Decomposition
  Applied to Electrogastrography. *IEEE EMBC 2019*, Berlin.
