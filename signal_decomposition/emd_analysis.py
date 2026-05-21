"""
経験的モード分解（EMD）による胃電気活動抽出
============================================
Huang et al. (1998) が提案した EMD を用いて、EGG 信号を
複数の固有モード関数（IMF: Intrinsic Mode Function）に分解し、
胃電気活動帯域（0.03〜0.08 Hz）を含む IMF を抽出する。

EMD の課題:
    IMF の帯域制御が困難なため、複数の IMF 間で周波数混合
    （モードミキシング）が生じやすい。
    この問題を解決した手法が EEMD（eemd_analysis.py 参照）。

参考文献:
    Huang NE et al.: The empirical mode decomposition and the Hilbert
    spectrum for nonlinear and non-stationary time series analysis.
    Proc. R. Soc. Lond. A, 454, pp.903-995, 1998.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD as PyEMD


# 胃電気活動の正常周波数帯 [Hz]
GASTRIC_BAND_LOW = 0.03
GASTRIC_BAND_HIGH = 0.08


def apply_emd(signal, t):
    """
    EMD を適用して IMF 群と残余信号を算出する。

    信号を高周波成分から順に IMF へと分解する。
    原信号は全 IMF と残余信号の和として再構成できる:
        x(t) = Σ IMF_i(t) + r(t)

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号（EGG 数値解または実測 EGG）
    t : numpy.ndarray
        時間軸 [s]

    Returns
    -------
    imfs : numpy.ndarray
        IMF 群（形状: [IMF 数, 信号長]）
        IMF1 が最高周波数成分、番号が大きいほど低周波成分
    residual : numpy.ndarray
        残余信号
    """
    emd = PyEMD()
    imfs = emd.emd(signal, t)
    residual = signal - np.sum(imfs, axis=0)
    return imfs, residual


def select_gastric_imf(imfs, fs=1.0):
    """
    IMF 群から胃電気活動帯域（0.03〜0.08 Hz）の
    主成分を含む IMF を自動選択する。

    各 IMF のパワースペクトルを算出し、
    胃電気活動帯域内のパワーが最大の IMF を選択する。

    Parameters
    ----------
    imfs : numpy.ndarray
        IMF 群（形状: [IMF 数, 信号長]）
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）

    Returns
    -------
    selected_imf : numpy.ndarray
        選択された IMF（胃電気活動抽出信号）
    selected_idx : int
        選択された IMF のインデックス（0 始まり）
    """
    n = imfs.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band_mask = (freqs >= GASTRIC_BAND_LOW) & (freqs <= GASTRIC_BAND_HIGH)

    band_powers = [np.sum(np.abs(np.fft.rfft(imf)) ** 2 * band_mask)
                   for imf in imfs]

    selected_idx = int(np.argmax(band_powers))
    return imfs[selected_idx], selected_idx


def extract_gastric_activity(signal, t, fs=1.0):
    """
    EMD により EGG 信号から胃電気活動を抽出する（一括処理）。

    Parameters
    ----------
    signal : numpy.ndarray
        入力 EGG 信号
    t : numpy.ndarray
        時間軸 [s]
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）

    Returns
    -------
    extracted : numpy.ndarray
        抽出された胃電気活動信号
    imfs : numpy.ndarray
        全 IMF 群
    selected_idx : int
        選択された IMF のインデックス
    """
    print("  Running EMD...")
    imfs, _ = apply_emd(signal, t)
    extracted, selected_idx = select_gastric_imf(imfs, fs=fs)
    print(f"  Selected IMF index: {selected_idx + 1} (1-indexed), "
          f"total IMFs: {imfs.shape[0]}")
    return extracted, imfs, selected_idx


def plot_imfs(t, original, imfs, residual, selected_idx=None):
    """
    EMD で分解した IMF 群を可視化する。

    Parameters
    ----------
    t : numpy.ndarray
        時間軸 [s]
    original : numpy.ndarray
        元の EGG 信号
    imfs : numpy.ndarray
        IMF 群
    residual : numpy.ndarray
        残余信号
    selected_idx : int or None, optional
        ハイライト表示する IMF のインデックス（None の場合はハイライトなし）
    """
    n_imfs = imfs.shape[0]
    n_plots = n_imfs + 2  # 原信号 + IMF 群 + 残余信号

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots))

    # 原信号
    axes[0].plot(t, original, color="black")
    axes[0].set_title("Original signal (x)")
    axes[0].set_ylabel("Amplitude [V]")

    # 各 IMF
    for i, imf in enumerate(imfs):
        is_selected = (i == selected_idx)
        color = "steelblue" if is_selected else "gray"
        label = f"IMF{i+1} ← selected" if is_selected else f"IMF{i+1}"
        axes[i + 1].plot(t, imf, color=color, label=label)
        axes[i + 1].legend(loc="upper right", fontsize=8)
        axes[i + 1].set_ylabel("Amplitude")

    # 残余信号
    axes[-1].plot(t, residual, color="dimgray")
    axes[-1].set_title("Residual (r)")
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("emd_imfs.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity
    from preprocessing.bandpass_filter import preprocess

    print("Running EMD on simulated EGG data...")

    t, _, x_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )
    x_preprocessed = preprocess(x_noisy, fs=1.0)

    extracted, imfs, selected_idx = extract_gastric_activity(
        x_preprocessed, t, fs=1.0
    )

    imfs_full, residual = apply_emd(x_preprocessed, t)
    plot_imfs(t, x_preprocessed, imfs_full, residual, selected_idx)

    np.savetxt("emd_extracted.csv", extracted, delimiter=",")
    print("Saved: emd_extracted.csv")
