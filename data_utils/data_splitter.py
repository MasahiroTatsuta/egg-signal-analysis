"""
EGG 信号の時間窓分割ユーティリティ
====================================
長時間計測した EGG 信号を、指定した時間窓サイズで分割する。

実測 EGG の解析では、計測全体（例: 2400 s）の中から
胃活動が安定した後半区間（例: 最後の 1024 s）を切り出し、
さらに重複率 50% の時間窓（例: 256 s）で分割して解析する。

この手法により、短い窓で局所的な時間変動を捉えながら、
隣接窓との連続性も確保できる（短時間フーリエ変換的なアプローチ）。
"""

import numpy as np
import os


def extract_stable_segment(signal, fs=1.0, stable_duration=1024.0):
    """
    計測信号の末尾から安定区間を抽出する。

    計測開始直後は体動などのアーチファクトが混入しやすいため、
    胃活動が活発化した末尾の安定区間を使用する。

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号（全計測時間分）
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）
    stable_duration : float, optional
        抽出する末尾区間の長さ [s]（デフォルト: 1024.0）

    Returns
    -------
    numpy.ndarray
        末尾から抽出した安定区間の信号

    Raises
    ------
    ValueError
        信号長が stable_duration より短い場合
    """
    n_stable = int(stable_duration * fs)
    if len(signal) < n_stable:
        raise ValueError(
            f"Signal length ({len(signal)} samples) is shorter than "
            f"stable_duration ({n_stable} samples). "
            f"Reduce stable_duration or use a longer signal."
        )
    return signal[-n_stable:]


def split_into_windows(signal, window_size, overlap_ratio=0.5, fs=1.0):
    """
    信号を重複ありの時間窓に分割する。

    Parameters
    ----------
    signal : numpy.ndarray
        入力信号
    window_size : int
        1 窓あたりのサンプル数
    overlap_ratio : float, optional
        隣接窓間の重複率（0.0〜1.0 未満、デフォルト: 0.5 = 50%）
        0.5 の場合、窓を window_size / 2 ずつずらしながら分割する。
    fs : float, optional
        サンプリング周波数 [Hz]（デフォルト: 1.0）

    Returns
    -------
    windows : list of numpy.ndarray
        分割された窓信号のリスト
    window_times : list of float
        各窓の開始時刻 [s]

    Raises
    ------
    ValueError
        overlap_ratio が 0.0 未満または 1.0 以上の場合
    """
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio must be in [0.0, 1.0). Got {overlap_ratio}.")

    step = int(window_size * (1.0 - overlap_ratio))
    n_windows = (len(signal) - window_size) // step + 1

    windows = []
    window_times = []

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > len(signal):
            break
        windows.append(signal[start:end])
        window_times.append(start / fs)

    return windows, window_times


def save_windows(windows, output_dir="split_data", prefix="window"):
    """
    分割した窓信号を個別の CSV ファイルに保存する。

    Parameters
    ----------
    windows : list of numpy.ndarray
        分割された窓信号のリスト
    output_dir : str, optional
        保存先ディレクトリ（デフォルト: "split_data"）
    prefix : str, optional
        ファイル名のプレフィックス（デフォルト: "window"）

    Returns
    -------
    list of str
        保存したファイルパスのリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, window in enumerate(windows):
        filename = f"{prefix}_{i + 1:03d}.csv"
        filepath = os.path.join(output_dir, filename)
        np.savetxt(filepath, window, delimiter=",", fmt="%.6f")
        saved_paths.append(filepath)

    print(f"  Saved {len(windows)} windows to '{output_dir}/'")
    return saved_paths


def load_windows(output_dir="split_data", prefix="window"):
    """
    保存した窓信号ファイルを読み込む。

    Parameters
    ----------
    output_dir : str, optional
        読み込み元ディレクトリ（デフォルト: "split_data"）
    prefix : str, optional
        ファイル名のプレフィックス（デフォルト: "window"）

    Returns
    -------
    list of numpy.ndarray
        読み込んだ窓信号のリスト
    """
    import glob
    pattern = os.path.join(output_dir, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found."
        )

    windows = [np.genfromtxt(f, delimiter=",", dtype=float) for f in files]
    print(f"  Loaded {len(windows)} windows from '{output_dir}/'")
    return windows


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from simulation.van_der_pol import simulate_gastric_activity

    print("Splitting simulated EGG signal into windows...")

    # 長めのシミュレーション信号を生成（実測 EGG の代わり）
    t_full, _, x_noisy = simulate_gastric_activity(
        duration=2400, dt=1.0, A=11.0, noise_std=0.01
    )

    # 末尾 1024 s を安定区間として抽出
    x_stable = extract_stable_segment(x_noisy, fs=1.0, stable_duration=1024.0)
    print(f"  Stable segment length: {len(x_stable)} samples")

    # 256 s 窓・重複率 50% で分割
    windows, times = split_into_windows(
        x_stable, window_size=256, overlap_ratio=0.5, fs=1.0
    )
    print(f"  Number of windows: {len(windows)}")
    print(f"  Window size: {len(windows[0])} samples")
    print(f"  Window start times: {[f'{t:.0f} s' for t in times[:5]]} ...")

    # 保存と再読み込みの確認
    saved = save_windows(windows, output_dir="split_data", prefix="window")
    reloaded = load_windows(output_dir="split_data", prefix="window")
    print(f"  Reload check: {len(reloaded)} windows, shape={reloaded[0].shape}")
