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
├── simulation/
│   └── van_der_pol.py          # 胃電気活動数理モデル（Van der Pol方程式）
├── preprocessing/
│   └── bandpass_filter.py      # 基線補正・バンドパスフィルタ
├── signal_decomposition/
│   └── eemd_analysis.py        # EEMD による胃電気活動抽出（メイン手法）
├── analysis/
│   └── signal_analysis.py      # FFT / EMD / ヒルベルト変換 / コヒーレンス
├── data/
│   └── sample/                 # シミュレーション生成サンプルデータ
└── requirements.txt
```

---

## 処理フロー

```
[胃電気活動数理モデル]        [実測 EGG]
  van_der_pol.py              （非公開）
        ↓
[前処理]
  bandpass_filter.py
  基線補正 → バンドパスフィルタ (0.02–0.15 Hz)
        ↓
[EEMD による成分抽出]
  eemd_analysis.py
  AWGN パラメータ最適化 (AWGNL = 4 dB)
        ↓
[評価・解析]
  signal_analysis.py
  FFT パワースペクトル / ヒルベルト変換 / コヒーレンス
```

---

## 主な実装内容

| モジュール | 内容 |
|---|---|
| `van_der_pol.py` | Van der Pol 方程式の数値シミュレーション（4次ルンゲクッタ法） |
| `bandpass_filter.py` | バターワースバンドパスフィルタ（ゼロ位相処理） |
| `eemd_analysis.py` | EEMD・AWGNL 最適化・eIMF 自動選択 |
| `signal_analysis.py` | FFT / EMD / ヒルベルト変換（瞬時周波数）/ 2ch コヒーレンス |

---

## インストール

```bash
pip install numpy scipy matplotlib PyEMD
```

---

## 使い方

```python
# シミュレーションデータの生成
from simulation.van_der_pol import simulate_gastric_activity
t, x_clean, x_noisy = simulate_gastric_activity(duration=256, noise_std=0.01)

# 前処理
from preprocessing.bandpass_filter import preprocess
x_preprocessed = preprocess(x_noisy, fs=1.0)

# EEMD で胃電気活動を抽出
from signal_decomposition.eemd_analysis import extract_gastric_activity
extracted, eimfs, idx = extract_gastric_activity(x_preprocessed, t, awgnl_db=4.0)
```

---

## 主な結果

EGG 数値解・実測 EGG の両評価において、  
AWGNL = 4 dB に最適化した EEMD がバンドパスフィルタ・従来 EMD を  
上回る胃電気活動抽出性能を示した（RMSE 最小、コヒーレンス維持）。

---

## 参考文献

- Wu Z, Huang NE: Ensemble empirical mode decomposition. *Advances in Adaptive Data Analysis*, 1(1), 2009.  
- Sarna SK et al.: Simulation of slow-wave electrical activity. *Am J Physiol*, 221, 1971.  
- 辰田昌洋ら: 胃電図の数理モデルと健常者からの計測データを用いた妥当性評価. *生体医工学*, 57(6), 2019.
