"""
FFT によるパワースペクトル算出
==============================
高速フーリエ変換（FFT）を用いて EGG 信号のパワースペクトルを算出する。

EGG の評価では、胃電気活動の正常周波数帯（0.03〜0.08 Hz, 3cpm）の
パワー分布を周波数領域で確認することが標準的な解析手法である。

ハニング窓を乗算してスペクトルリークを抑制し、
必要に応じてゼロパディングにより周波数分解能を向上させる。
"""

import numpy as np
import matplotlib.pyplot as plt


# 胃電気活動の正常周波数帯 [Hz]
GASTRIC_BAND_LOW = 0.03
GASTRIC_BAND_HIGH = 0.08


def compute_power_spectrum(x, fs=1.0, n_fft=None, apply_window=True):
    """
    FFT を用いてパワースペクトルを算出する。

    Parameters
    ----------
    x : numpy.ndarray
        入力信号
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）
    n_fft : int or None, optional
        FFT ポイント数。None の場合は信号長と同じ。
        ゼロパディングには信号長より大きい値を指定する。
        例: len(x) * 2 で周波数分解能が 2 倍になる。
    apply_window : bool, optional
        ハニング窓を適用するか（デフォルト: True）
        窓関数によりスペクトルリークを抑制できる。

    Returns
    -------
    freqs : numpy.ndarray
        周波数軸 [Hz]（0 から fs/2 まで）
    psd : numpy.ndarray
        パワースペクトル密度 [V²rms/Hz]
    """
    n = len(x)
    if n_fft is None:
        n_fft = n

    # ハニング窓の乗算
    if apply_window:
        window = np.hanning(n)
        x = x * window

    # ゼロパディング（n_fft > n の場合）
    x_padded = np.zeros(n_fft)
    x_padded[:n] = x

    yf = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd = np.abs(yf) ** 2 / n

    return freqs, psd


def get_peak_frequency(freqs, psd, freq_range=None):
    """
    指定した周波数範囲内のピーク周波数を返す。

    Parameters
    ----------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    psd : numpy.ndarray
        パワースペクトル密度
    freq_range : tuple of float or None, optional
        探索する周波数範囲 (f_low, f_high) [Hz]。
        None の場合は全範囲を探索する。

    Returns
    -------
    float
        ピーク周波数 [Hz]
    """
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        peak_idx = np.argmax(psd[mask])
        return float(freqs[mask][peak_idx])
    return float(freqs[np.argmax(psd)])


def plot_power_spectrum(freqs, psd, title="Power Spectrum",
                        freq_range=(0.0, 0.15), highlight_gastric_band=True,
                        save_path="power_spectrum.png"):
    """
    パワースペクトルを可視化する。

    Parameters
    ----------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    psd : numpy.ndarray
        パワースペクトル密度
    title : str, optional
        グラフタイトル
    freq_range : tuple of float, optional
        表示する周波数範囲 [Hz]（デフォルト: (0.0, 0.15)）
    highlight_gastric_band : bool, optional
        胃電気活動帯域（0.03〜0.08 Hz）をグレーでハイライトするか
        （デフォルト: True）
    save_path : str, optional
        保存先ファイルパス（デフォルト: "power_spectrum.png"）
    """
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, color="black", linewidth=1.0)

    if highlight_gastric_band:
        plt.axvspan(
            GASTRIC_BAND_LOW, GASTRIC_BAND_HIGH,
            color="gray", alpha=0.3, label="Gastric band (3 cpm, 0.03–0.08 Hz)"
        )
        plt.legend(fontsize=10)

    plt.xlim(freq_range)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [V²rms/Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity
    from preprocessing.bandpass_filter import preprocess

    print("Computing power spectrum of simulated EGG data...")

    t, _, x_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )
    x_preprocessed = preprocess(x_noisy, fs=1.0)

    # ゼロパディングで周波数分解能を向上
    freqs, psd = compute_power_spectrum(
        x_preprocessed, fs=1.0, n_fft=len(x_preprocessed) * 2
    )

    peak_freq = get_peak_frequency(
        freqs, psd, freq_range=(GASTRIC_BAND_LOW, GASTRIC_BAND_HIGH)
    )
    print(f"  Peak frequency in gastric band: {peak_freq:.4f} Hz")

    plot_power_spectrum(freqs, psd, title="EGG Power Spectrum (simulated)")

    np.savetxt("fft_frequencies.csv", freqs, delimiter=",")
    np.savetxt("fft_psd.csv", psd, delimiter=",")
    print("Saved: fft_frequencies.csv, fft_psd.csv")
