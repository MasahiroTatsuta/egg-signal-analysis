"""
アンサンブル経験的モード分解（EEMD）による胃電気活動抽出
==========================================================
Wu & Huang (2009) が提案したEEMDを用いて、EGG信号から
胃電気活動成分を抽出する。

EEMDはEMDの周波数混合（モードミキシング）問題を解決するため、
AWGN（加算性白色ガウス雑音）を加えた信号に対してEMDを繰り返し、
アンサンブル平均を取ることで各周波数成分を安定して分離する。

AWGNの標準偏差パラメータ（AWGNL）の最適化により、
胃電気活動抽出精度が向上することが示されている（辰田ら, 2019）。

参考文献:
    Wu Z, Huang NE: Ensemble empirical mode decomposition —
    A noise-assisted data analysis method.
    Advances in Adaptive Data Analysis, 1(1), pp.1-41, 2009.

    辰田昌洋ら: Parameter Optimization in Ensemble Empirical Mode
    Decomposition Applied to Electrogastrography.
    IEEE EMBC 2019.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD as PyEEMD


# 胃電気活動の正常周波数帯 [Hz]
GASTRIC_BAND_LOW = 0.03
GASTRIC_BAND_HIGH = 0.08


def compute_awgn_std(signal, awgnl_db):
    """
    AWGNLパラメータ（dB）から、信号振幅に対応するAWGN標準偏差を算出する。

    EEMDに付加するAWGNの標準偏差は、入力信号の振幅レンジに対する
    相対値として定義される（式14: σ_wi = σ_i × |max(X) - min(X)|）。

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号
    awgnl_db : float
        AWGNの標準偏差を制御するパラメータ [dB]
        本研究の最適値: 0〜6 dB（推奨: 4 dB）

    Returns
    -------
    float
        EEMD付加用AWGN標準偏差（noise_widthとして使用）
    """
    amplitude_range = np.max(signal) - np.min(signal)
    sigma_i = 10 ** (-awgnl_db / 20.0)
    sigma_wi = sigma_i * amplitude_range
    return sigma_wi


def apply_eemd(signal, t, awgnl_db=4.0, trials=100):
    """
    EEMDを適用し、アンサンブルIMF（eIMF）を算出する。

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号（EGG数値解または実測EGG）
    t : numpy.ndarray
        時間軸 [s]
    awgnl_db : float, optional
        AWGNパラメータ [dB]（デフォルト: 4.0 dB）
        本研究では0〜6 dBが最適であることを示した
    trials : int, optional
        EEMDの試行回数（デフォルト: 100）
        試行回数が多いほど安定するが計算時間が増加する

    Returns
    -------
    eimfs : numpy.ndarray
        アンサンブルIMF群（形状: [eIMF数, 信号長]）
    """
    noise_width = compute_awgn_std(signal, awgnl_db)

    eemd = PyEEMD(trials=trials, noise_width=noise_width)
    eimfs = eemd.eemd(signal, t)
    return eimfs


def select_gastric_eimf(eimfs, fs=1.0):
    """
    eIMF群の中から胃電気活動帯域（0.03〜0.08 Hz）の
    主成分を含むeIMFを自動選択する。

    各eIMFのパワースペクトルを算出し、
    胃電気活動帯域内にピークパワーを持つeIMFを選択する。

    Parameters
    ----------
    eimfs : numpy.ndarray
        アンサンブルIMF群（形状: [eIMF数, 信号長]）
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）

    Returns
    -------
    selected_eimf : numpy.ndarray
        選択されたeIMF（胃電気活動抽出信号）
    selected_idx : int
        選択されたeIMFのインデックス（0始まり）
    """
    n = eimfs.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band_mask = (freqs >= GASTRIC_BAND_LOW) & (freqs <= GASTRIC_BAND_HIGH)

    band_powers = []
    for eimf in eimfs:
        psd = np.abs(np.fft.rfft(eimf)) ** 2
        band_powers.append(np.sum(psd[band_mask]))

    selected_idx = int(np.argmax(band_powers))
    return eimfs[selected_idx], selected_idx


def extract_gastric_activity(signal, t, awgnl_db=4.0, trials=100, fs=1.0):
    """
    EGG信号からEEMDにより胃電気活動を抽出する（一括処理）。

    Parameters
    ----------
    signal : numpy.ndarray
        入力EGG信号
    t : numpy.ndarray
        時間軸 [s]
    awgnl_db : float, optional
        AWGNパラメータ [dB]（デフォルト: 4.0）
    trials : int, optional
        EEMD試行回数（デフォルト: 100）
    fs : float, optional
        サンプリング周波数（デフォルト: 1.0）

    Returns
    -------
    extracted : numpy.ndarray
        抽出された胃電気活動信号
    eimfs : numpy.ndarray
        全eIMF群
    selected_idx : int
        選択されたeIMFのインデックス
    """
    print(f"  Running EEMD (trials={trials}, AWGNL={awgnl_db} dB)...")
    eimfs = apply_eemd(signal, t, awgnl_db=awgnl_db, trials=trials)
    extracted, selected_idx = select_gastric_eimf(eimfs, fs=fs)
    print(f"  Selected eIMF index: {selected_idx + 1} (1-indexed)")
    return extracted, eimfs, selected_idx


def plot_eemd_results(t, original, extracted, eimfs, selected_idx):
    """
    EEMD解析結果を可視化する。

    Parameters
    ----------
    t : numpy.ndarray
        時間軸 [s]
    original : numpy.ndarray
        元のEGG信号
    extracted : numpy.ndarray
        抽出された胃電気活動信号
    eimfs : numpy.ndarray
        全eIMF群
    selected_idx : int
        選択されたeIMFのインデックス
    """
    n_imfs = eimfs.shape[0]
    fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(10, 3 * (n_imfs + 2)))

    axes[0].plot(t, original, color="black")
    axes[0].set_title("Original EGG signal")
    axes[0].set_ylabel("Amplitude")

    for i, eimf in enumerate(eimfs):
        color = "steelblue" if i == selected_idx else "gray"
        label = f"eIMF{i+1} ← selected" if i == selected_idx else f"eIMF{i+1}"
        axes[i + 1].plot(t, eimf, color=color, label=label)
        axes[i + 1].legend(loc="upper right")
        axes[i + 1].set_ylabel("Amplitude")

    axes[-1].plot(t, extracted, color="steelblue")
    axes[-1].set_title("Extracted gastric activity (EEMD)")
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("eemd_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity
    from preprocessing.bandpass_filter import preprocess

    print("Running EEMD on simulated EGG data...")

    t, x_clean, x_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )
    x_preprocessed = preprocess(x_noisy, fs=1.0)

    extracted, eimfs, selected_idx = extract_gastric_activity(
        x_preprocessed, t, awgnl_db=4.0, trials=100, fs=1.0
    )

    print(f"  Number of eIMFs: {eimfs.shape[0]}")
    plot_eemd_results(t, x_preprocessed, extracted, eimfs, selected_idx)

    np.savetxt("eemd_extracted.csv", extracted, delimiter=",")
    print("Saved: eemd_extracted.csv")
