"""
Van der Pol方程式による胃電気活動数理モデルの数値シミュレーション
=====================================================================
Sarna et al. (1971) が提案した胃電気活動の力学的モデルに基づき、
4次ルンゲクッタ法で数値解を算出する。

数理モデル式:
    x'' - A(1 - x^2)x' + x = 0

リエナール型に変換した連立1階微分方程式:
    dx/dt = y + A(x - x^3/3)
    dy/dt = -x

パラメータAにより胃電気活動の周期を制御する。
正常な胃電気活動は約0.05Hz（3cpm）に相当する。

参考文献:
    Sarna SK, Daniel EE, Kingma YJ: Simulation of slow-wave electrical
    activity of small intestine. Am J Physiol. 221, pp.166-175, 1971.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def van_der_pol(t, z, A=11.0):
    """
    Van der Pol方程式をリエナール型の連立1階微分方程式として定義する。

    Parameters
    ----------
    t : float
        時刻（solve_ivpが内部で使用するため引数として必要）
    z : list of float
        状態変数 [x, y]
        x: 振動座標（胃電気活動を模擬）
        y: xの時間微分
    A : float, optional
        周期制御パラメータ（デフォルト: 11.0）
        大きいほど周期が長くなる

    Returns
    -------
    list of float
        微分方程式の右辺 [dx/dt, dy/dt]
    """
    x, y = z
    dxdt = y + A * (x - x**3 / 3)
    dydt = -x
    return [dxdt, dydt]


def simulate_gastric_activity(
    duration=1024,
    dt=1.0,
    A=11.0,
    initial_state=None,
    noise_std=0.0
):
    """
    胃電気活動数理モデルの数値シミュレーションを実行する。

    Parameters
    ----------
    duration : int, optional
        シミュレーション時間 [s]（デフォルト: 1024）
    dt : float, optional
        サンプリング周期 [s]（デフォルト: 1.0）
    A : float, optional
        Van der Pol方程式の周期制御パラメータ（デフォルト: 11.0）
    initial_state : list of float or None, optional
        初期値 [x0, y0]（デフォルト: [0, 1]）
    noise_std : float, optional
        加算するガウス白色雑音の標準偏差（デフォルト: 0.0、雑音なし）

    Returns
    -------
    t : numpy.ndarray
        時間軸 [s]
    x : numpy.ndarray
        胃電気活動数値解（雑音なし）
    x_noisy : numpy.ndarray
        ガウス白色雑音を加算した胃電気活動数値解（EGGモデル）
    """
    if initial_state is None:
        initial_state = [0.0, 1.0]

    t_span = (0, duration)
    t_eval = np.arange(0, duration, dt)

    result = solve_ivp(
        fun=lambda t, z: van_der_pol(t, z, A=A),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    t = result.t
    x = result.y[0]  # 胃電気活動数値解

    # ガウス白色雑音の加算（EGGモデルの生成）
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=len(x))
        x_noisy = x + noise
    else:
        x_noisy = x.copy()

    return t, x, x_noisy


def plot_results(t, x, x_noisy=None, show_spectrum=True):
    """
    シミュレーション結果を可視化する。

    Parameters
    ----------
    t : numpy.ndarray
        時間軸 [s]
    x : numpy.ndarray
        胃電気活動数値解
    x_noisy : numpy.ndarray or None, optional
        雑音加算後の信号（Noneの場合は表示しない）
    show_spectrum : bool, optional
        パワースペクトルを表示するか（デフォルト: True）
    """
    fig, axes = plt.subplots(2 if show_spectrum else 1, 1,
                             figsize=(10, 8 if show_spectrum else 4))
    if not show_spectrum:
        axes = [axes]

    # 時系列信号
    axes[0].plot(t, x, color="black", label="Gastric activity (model)")
    if x_noisy is not None:
        axes[0].plot(t, x_noisy, color="gray", alpha=0.6, label="EGG model (with noise)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude [arb.unit]")
    axes[0].set_xlim([t[0], t[-1]])
    axes[0].legend()
    axes[0].set_title("Gastric Electrical Activity Model (Van der Pol)")

    # パワースペクトル
    if show_spectrum:
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0)  # サンプリング周期 1.0s
        psd = np.abs(np.fft.rfft(x)) ** 2 / n

        axes[1].plot(freqs, psd, color="black")
        axes[1].axvspan(0.03, 0.08, color="gray", alpha=0.3, label="Normal EGG band (3cpm)")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Power Spectral Density [arb.unit²/Hz]")
        axes[1].set_xlim([0, 0.15])
        axes[1].legend()
        axes[1].set_title("Power Spectrum")

    plt.tight_layout()
    plt.savefig("van_der_pol_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # サンプルデータの生成と保存
    print("Simulating gastric electrical activity...")

    t, x_clean, x_noisy = simulate_gastric_activity(
        duration=1024,
        dt=1.0,
        A=11.0,
        noise_std=0.01,
    )

    print(f"  Duration : {t[-1]:.0f} s")
    print(f"  Samples  : {len(t)}")
    print(f"  Peak frequency: ~0.05 Hz (3 cpm, normal gastric activity)")

    # 結果の保存（サンプルデータとして使用可能）
    np.savetxt("../data/sample/gastric_activity_clean.csv", x_clean, delimiter=",")
    np.savetxt("../data/sample/gastric_activity_noisy.csv", x_noisy, delimiter=",")
    np.savetxt("../data/sample/time_axis.csv", t, delimiter=",")
    print("Sample data saved to data/sample/")

    plot_results(t, x_clean, x_noisy)
