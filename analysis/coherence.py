"""
2 チャンネル EGG 信号間のコヒーレンス解析
==========================================
2 チャンネルで計測した EGG 信号のコヒーレンスを算出し、
各雑音除去法の胃電気活動抽出性能を評価する。

コヒーレンスとは 2 信号の周波数ごとの位相同期性を表す指標（0〜1）。
    Cxy(f) = |Wxy(f)|² / (Wxx(f) × Wyy(f))

    Wxy: クロスパワースペクトル密度
    Wxx, Wyy: 各信号のパワースペクトル密度

胃電気活動帯域（0.03〜0.08 Hz）内のコヒーレンス最大値が高いほど、
2 チャンネル間の胃電気活動成分が同期しており、
雑音の混入が少ないことを示す。

評価の解釈:
    - コヒーレンス高い → 胃電気活動が保持され、雑音が除去されている
    - コヒーレンス低い → 雑音の混入、または胃電気活動自体が除去されている
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence


# 胃電気活動の正常周波数帯 [Hz]
GASTRIC_BAND_LOW = 0.03
GASTRIC_BAND_HIGH = 0.08


def compute_coherence(ch1, ch2, fs=1.0, nperseg=256, noverlap=0):
    """
    2 チャンネル信号間のコヒーレンスを算出する。

    Parameters
    ----------
    ch1 : numpy.ndarray
        チャンネル 1 の信号
    ch2 : numpy.ndarray
        チャンネル 2 の信号
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）
    nperseg : int, optional
        セグメント長（デフォルト: 256）
        短くすると時間分解能が上がり、長くすると周波数分解能が上がる。
    noverlap : int, optional
        セグメント間のオーバーラップサンプル数（デフォルト: 0）

    Returns
    -------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    coh : numpy.ndarray
        コヒーレンス値（0〜1）
    """
    freqs, coh = coherence(
        ch1, ch2,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg * 2,   # ゼロパディングで周波数分解能を向上
    )
    return freqs, coh


def get_gastric_band_coherence_max(freqs, coh):
    """
    胃電気活動帯域（0.03〜0.08 Hz）内のコヒーレンス最大値を返す。

    雑音除去法の性能比較指標として使用する。

    Parameters
    ----------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    coh : numpy.ndarray
        コヒーレンス値

    Returns
    -------
    float
        胃電気活動帯域内のコヒーレンス最大値
    """
    band_mask = (freqs >= GASTRIC_BAND_LOW) & (freqs <= GASTRIC_BAND_HIGH)
    if not np.any(band_mask):
        return np.nan
    return float(np.max(coh[band_mask]))


def compare_coherence(freqs, coh_dict, freq_range=(0.0, 0.14),
                       save_path="coherence_comparison.png"):
    """
    複数の雑音除去法のコヒーレンスを比較表示する。

    Parameters
    ----------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    coh_dict : dict
        雑音除去法名をキー、コヒーレンス配列を値とする辞書。
        例: {"BPF": coh_bpf, "EMD": coh_emd, "EEMD": coh_eemd}
    freq_range : tuple of float, optional
        表示する周波数範囲 [Hz]（デフォルト: (0.0, 0.14)）
    save_path : str, optional
        保存先ファイルパス（デフォルト: "coherence_comparison.png"）
    """
    colors = ["dimgray", "steelblue", "tomato", "seagreen", "goldenrod"]

    plt.figure(figsize=(8, 4))

    for (label, coh), color in zip(coh_dict.items(), colors):
        max_val = get_gastric_band_coherence_max(freqs, coh)
        plt.plot(freqs, coh, color=color, linewidth=1.2,
                 label=f"{label} (max={max_val:.3f})")

    plt.axvspan(GASTRIC_BAND_LOW, GASTRIC_BAND_HIGH,
                color="gray", alpha=0.2, label="Gastric band (3 cpm)")
    plt.xlim(freq_range)
    plt.ylim([0.0, 1.0])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Coherence")
    plt.title("Coherence Comparison between Noise Removal Methods")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_coherence(freqs, coh, title="Coherence (Ch1 vs Ch2)",
                   freq_range=(0.0, 0.14), save_path="coherence.png"):
    """
    単一のコヒーレンスを可視化する。

    Parameters
    ----------
    freqs : numpy.ndarray
        周波数軸 [Hz]
    coh : numpy.ndarray
        コヒーレンス値
    title : str, optional
        グラフタイトル
    freq_range : tuple of float, optional
        表示する周波数範囲 [Hz]（デフォルト: (0.0, 0.14)）
    save_path : str, optional
        保存先ファイルパス
    """
    max_val = get_gastric_band_coherence_max(freqs, coh)

    plt.figure(figsize=(7, 4))
    plt.plot(freqs, coh, color="black", linewidth=1.0)
    plt.axvspan(GASTRIC_BAND_LOW, GASTRIC_BAND_HIGH,
                color="gray", alpha=0.3,
                label=f"Gastric band (max coherence = {max_val:.3f})")
    plt.xlim(freq_range)
    plt.ylim([0.0, 1.0])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Coherence")
    plt.title(title)
    plt.legend(loc="upper right")
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
    from signal_decomposition.emd_analysis import extract_gastric_activity as extract_emd

    print("Computing coherence on simulated 2-channel EGG data...")

    # 2 チャンネルのシミュレーションデータを生成（微妙に異なるノイズを付加）
    t, _, ch1_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )
    _, _, ch2_noisy = simulate_gastric_activity(
        duration=256, dt=1.0, A=11.0, noise_std=0.01
    )

    ch1_pre = preprocess(ch1_noisy, fs=1.0)
    ch2_pre = preprocess(ch2_noisy, fs=1.0)

    # EEMD で各チャンネルの胃電気活動を抽出
    ch1_eemd, _, _ = extract_gastric_activity(ch1_pre, t, awgnl_db=4.0, trials=50)
    ch2_eemd, _, _ = extract_gastric_activity(ch2_pre, t, awgnl_db=4.0, trials=50)

    # EMD で各チャンネルの胃電気活動を抽出
    ch1_emd, _, _ = extract_emd(ch1_pre, t)
    ch2_emd, _, _ = extract_emd(ch2_pre, t)

    # コヒーレンスの算出と比較
    freqs_bpf, coh_bpf = compute_coherence(ch1_pre, ch2_pre, fs=1.0)
    freqs_emd, coh_emd = compute_coherence(ch1_emd, ch2_emd, fs=1.0)
    freqs_eemd, coh_eemd = compute_coherence(ch1_eemd, ch2_eemd, fs=1.0)

    print(f"  BPF  coherence max: {get_gastric_band_coherence_max(freqs_bpf, coh_bpf):.3f}")
    print(f"  EMD  coherence max: {get_gastric_band_coherence_max(freqs_emd, coh_emd):.3f}")
    print(f"  EEMD coherence max: {get_gastric_band_coherence_max(freqs_eemd, coh_eemd):.3f}")

    compare_coherence(
        freqs_bpf,
        {"BPF": coh_bpf, "EMD": coh_emd, "EEMD": coh_eemd},
    )

    np.savetxt("coherence_eemd.csv", coh_eemd, delimiter=",")
    print("Saved: coherence_eemd.csv")
