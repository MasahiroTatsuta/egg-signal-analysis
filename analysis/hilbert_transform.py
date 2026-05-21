"""
ヒルベルト変換による瞬時周波数時間推移の算出
=============================================
ヒルベルト変換を用いて EGG 信号の解析信号を生成し、
瞬時振幅包絡線 A(t) と瞬時周波数時間推移 H(t) を算出する。

FFT や STFT では時間分解能と周波数分解能にトレードオフが存在するが、
ヒルベルト変換はこの制約を受けず、高い時間・周波数分解能を同時に実現できる。

ただし、ヒルベルト変換は単一周波数成分の信号に適用することで
正確な瞬時周波数が得られる。そのため EMD や EEMD で分解した
IMF / eIMF に対して適用することが一般的である。

解析信号の定義:
    Z(t) = x(t) + j·y(t) = A(t)·exp(j·θ(t))

    A(t) = sqrt(x(t)² + y(t)²)       # 瞬時振幅包絡線
    θ(t) = arctan(y(t) / x(t))        # 瞬時位相
    H(t) = dθ(t)/dt / (2π)            # 瞬時周波数 [Hz]

    ここで y(t) は x(t) のヒルベルト変換。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# 胃電気活動の正常周波数帯 [Hz]
GASTRIC_BAND_LOW = 0.03
GASTRIC_BAND_HIGH = 0.08


def compute_instantaneous_frequency(signal, fs=1.0):
    """
    ヒルベルト変換を用いて瞬時振幅包絡線と瞬時周波数時間推移を算出する。

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号。EMD / EEMD で抽出した単一成分信号が望ましい。
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）

    Returns
    -------
    envelope : numpy.ndarray
        瞬時振幅包絡線 A(t)（信号長と同じ長さ）
    inst_freq : numpy.ndarray
        瞬時周波数時間推移 H(t) [Hz]（長さ: 信号長 - 1）
    inst_phase : numpy.ndarray
        瞬時位相 θ(t) [rad]（信号長と同じ長さ）
    """
    # ヒルベルト変換で解析信号を生成
    analytic_signal = hilbert(signal)

    # 瞬時振幅包絡線: A(t) = |Z(t)|
    envelope = np.abs(analytic_signal)

    # 瞬時位相: θ(t) = arg(Z(t))（位相アンラップで不連続を除去）
    inst_phase = np.unwrap(np.angle(analytic_signal))

    # 瞬時周波数: H(t) = dθ(t)/dt / (2π) [Hz]
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs

    return envelope, inst_freq, inst_phase


def compute_relative_std(inst_freq, freq_range=None):
    """
    瞬時周波数時間推移の相対標準偏差を算出する。

    雑音除去性能の評価指標として使用する。
    値が小さいほど瞬時周波数が安定しており、雑音の影響が少ない。

    Parameters
    ----------
    inst_freq : numpy.ndarray
        瞬時周波数時間推移 [Hz]
    freq_range : tuple of float or None, optional
        評価する周波数範囲 (f_low, f_high) [Hz]。
        None の場合は全データを使用する。

    Returns
    -------
    float
        相対標準偏差（標準偏差 / 平均値の絶対値）
    """
    data = inst_freq
    if freq_range is not None:
        mask = (inst_freq >= freq_range[0]) & (inst_freq <= freq_range[1])
        data = inst_freq[mask]

    if len(data) == 0 or np.abs(np.mean(data)) < 1e-12:
        return np.nan

    return float(np.std(data) / np.abs(np.mean(data)))


def plot_hilbert_results(t, signal, envelope, inst_freq,
                         save_path="hilbert_result.png"):
    """
    ヒルベルト変換の結果（振幅包絡線・瞬時周波数時間推移）を可視化する。

    Parameters
    ----------
    t : numpy.ndarray
        時間軸 [s]
    signal : numpy.ndarray
        入力信号
    envelope : numpy.ndarray
        瞬時振幅包絡線 A(t)
    inst_freq : numpy.ndarray
        瞬時周波数時間推移 H(t)（長さ: len(t) - 1）
    save_path : str, optional
        保存先ファイルパス（デフォルト: "hilbert_result.png"）
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # 瞬時振幅包絡線
    axes[0].plot(t, signal, color="steelblue", linewidth=0.8, label="Signal")
    axes[0].plot(t, envelope, color="orange", linewidth=1.5,
                 label="Amplitude envelope A(t)")
    axes[0].set_ylabel("Amplitude [V]")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Signal and Amplitude Envelope")

    # 瞬時周波数時間推移
    axes[1].plot(t[1:], inst_freq, color="black", linewidth=0.8)
    axes[1].axhline(GASTRIC_BAND_LOW, color="gray", linestyle="--",
                    linewidth=0.8, label=f"Gastric band ({GASTRIC_BAND_LOW}–{GASTRIC_BAND_HIGH} Hz)")
    axes[1].axhline(GASTRIC_BAND_HIGH, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim([-0.05, 0.20])
    axes[1].legend(loc="upper right")
    axes[1].set_title("Instantaneous Frequency H(t)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity
    from preprocessing.bandpass_filter import preprocess
    from signal_decomposition.eemd_analysis import extract_gastric_activity

    print("Computing instantaneous frequency of simulated EGG data...")

    t, _, x_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )
    x_preprocessed = preprocess(x_noisy, fs=1.0)

    # EEMD で胃電気活動成分を抽出してからヒルベルト変換を適用
    extracted, _, _ = extract_gastric_activity(
        x_preprocessed, t, awgnl_db=4.0, trials=100, fs=1.0
    )

    envelope, inst_freq, inst_phase = compute_instantaneous_frequency(
        extracted, fs=1.0
    )

    rel_std = compute_relative_std(inst_freq)
    print(f"  Relative std of instantaneous frequency: {rel_std:.4f}")

    plot_hilbert_results(t, extracted, envelope, inst_freq)

    np.savetxt("instantaneous_frequency.csv", inst_freq, delimiter=",")
    np.savetxt("amplitude_envelope.csv", envelope, delimiter=",")
    print("Saved: instantaneous_frequency.csv, amplitude_envelope.csv")
