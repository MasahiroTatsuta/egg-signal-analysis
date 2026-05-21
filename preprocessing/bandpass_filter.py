"""
EGG信号の前処理モジュール
=========================
計測されたEGG信号に対して以下の前処理を行う。

1. 基線補正（トレンド除去）
2. バンドパスフィルタ処理（0.02〜0.15 Hz）

EGGの胃電気活動は約0.03〜0.08 Hzの周波数帯に存在するため、
この帯域を含む0.02〜0.15 Hzのバンドパスフィルタを適用する。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def detrend_signal(x):
    """
    信号から線形トレンド（基線）を除去する。

    計測中の体動などにより生じるゆっくりとしたドリフト成分を除去し、
    信号の平均値をゼロに近づける。

    Parameters
    ----------
    x : numpy.ndarray
        入力信号

    Returns
    -------
    numpy.ndarray
        基線補正後の信号
    """
    return signal.detrend(x)


def design_bandpass_filter(fs, f_low=0.02, f_high=0.15, gpass=3.0, gstop=40.0):
    """
    バターワースバンドパスフィルタの係数を設計する。

    EGGの解析に適した通過域・阻止域を設定する。
    デフォルトは0.02〜0.15 Hzで、胃電気活動帯域（0.03〜0.08 Hz）を含む。

    Parameters
    ----------
    fs : float
        サンプリング周波数 [Hz]
    f_low : float, optional
        低域通過端周波数 [Hz]（デフォルト: 0.02）
    f_high : float, optional
        高域通過端周波数 [Hz]（デフォルト: 0.15）
    gpass : float, optional
        通過域最大損失量 [dB]（デフォルト: 3.0）
    gstop : float, optional
        阻止域最小減衰量 [dB]（デフォルト: 40.0）

    Returns
    -------
    b : numpy.ndarray
        フィルタの分子係数
    a : numpy.ndarray
        フィルタの分母係数
    """
    fn = fs / 2.0  # ナイキスト周波数

    # 通過域・阻止域の正規化周波数
    wp_low = f_low / fn
    ws_low = max(f_low - 0.01, 0.001) / fn   # 阻止域端（通過域より低め）
    wp_high = f_high / fn
    ws_high = min(f_high + 0.02, fn * 0.99) / fn  # 阻止域端（通過域より高め）

    # ローパスフィルタ
    n_lp, wn_lp = signal.buttord(wp_high, ws_high, gpass, gstop)
    b_lp, a_lp = signal.butter(n_lp, wn_lp, btype="low")

    # ハイパスフィルタ
    n_hp, wn_hp = signal.buttord(wp_low, ws_low, gpass, gstop)
    b_hp, a_hp = signal.butter(n_hp, wn_hp, btype="high")

    return (b_lp, a_lp), (b_hp, a_hp)


def apply_bandpass_filter(x, fs, f_low=0.02, f_high=0.15):
    """
    EGG信号にバンドパスフィルタを適用する。

    ローパスフィルタとハイパスフィルタを順に適用し、
    指定した周波数帯域の成分を抽出する。
    位相ひずみを防ぐためにゼロ位相フィルタ（filtfilt）を使用する。

    Parameters
    ----------
    x : numpy.ndarray
        入力信号
    fs : float
        サンプリング周波数 [Hz]
    f_low : float, optional
        低域通過端周波数 [Hz]（デフォルト: 0.02）
    f_high : float, optional
        高域通過端周波数 [Hz]（デフォルト: 0.15）

    Returns
    -------
    numpy.ndarray
        フィルタ処理後の信号
    """
    (b_lp, a_lp), (b_hp, a_hp) = design_bandpass_filter(
        fs, f_low=f_low, f_high=f_high
    )
    x_lp = signal.filtfilt(b_lp, a_lp, x)
    x_bp = signal.filtfilt(b_hp, a_hp, x_lp)
    return x_bp


def preprocess(x, fs=1.0, f_low=0.02, f_high=0.15):
    """
    EGG信号の前処理を一括実行する。

    基線補正とバンドパスフィルタ処理を順に適用する。

    Parameters
    ----------
    x : numpy.ndarray
        入力信号
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）
    f_low : float, optional
        バンドパスフィルタの低域端 [Hz]（デフォルト: 0.02）
    f_high : float, optional
        バンドパスフィルタの高域端 [Hz]（デフォルト: 0.15）

    Returns
    -------
    numpy.ndarray
        前処理後の信号
    """
    x_detrended = detrend_signal(x)
    x_filtered = apply_bandpass_filter(x_detrended, fs, f_low, f_high)
    return x_filtered


def plot_preprocessing_result(t, x_raw, x_preprocessed):
    """
    前処理の前後を比較表示する。

    Parameters
    ----------
    t : numpy.ndarray
        時間軸 [s]
    x_raw : numpy.ndarray
        前処理前の信号
    x_preprocessed : numpy.ndarray
        前処理後の信号
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, x_raw, color="gray", label="Raw signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].set_title("Before preprocessing")

    axes[1].plot(t, x_preprocessed, color="black", label="Preprocessed signal")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].set_title("After bandpass filtering (0.02–0.15 Hz)")

    plt.tight_layout()
    plt.savefig("preprocessing_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # サンプルデータ（シミュレーション信号）で前処理を確認
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity

    print("Running preprocessing on simulated EGG data...")

    t, _, x_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )

    x_preprocessed = preprocess(x_noisy, fs=1.0)

    print(f"  Input length : {len(x_noisy)} samples")
    print(f"  Output length: {len(x_preprocessed)} samples")

    plot_preprocessing_result(t, x_noisy, x_preprocessed)
    np.savetxt("preprocessed_egg.csv", x_preprocessed, delimiter=",")
    print("Saved: preprocessed_egg.csv")
