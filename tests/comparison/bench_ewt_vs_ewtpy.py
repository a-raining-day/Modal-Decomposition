"""
自研 EWT (EWT_new) 与 ewtpy 的全面对比基准 —— 生成 docs/EWT_Native_Report.md 的数据。

  §1 重构保真       : reconstruct() 与输入的偏差 (相对/最大)
  §2 频带划分质量   : 用滤波器组把**纯音能量**按带分配 -> capture(单带吃下比例)/覆盖带数
  §3 边界质量       : 与"理想边界"(相邻音中点)的距离 + 噪声下的稳定性
  §4 边界策略对比   : maximum / max-min / scale-space / envelope
  §5 预处理分支增益 : no-dc / no-trend / window / Slepian-Optimize
  §6 速度与峰值内存
  §7 鲁棒性边界情形

用法::

    python tests/comparison/bench_ewt_vs_ewtpy.py [--quick]

只读脚本: 不修改任何模块; scipy/ewtpy 仅作参考实现。
"""

import argparse
import time
import tracemalloc
import warnings

warnings.simplefilter("ignore")

import numpy as np
from ewtpy import EWT1D

import Modal_Decomposition as M

FS = 1000.0
EWTPY_KW = dict(log=0, detect="locmax", completion=0, reg="average",
                lengthFilter=10, sigmaFilter=5)
K_DEFAULT = 5
_QUICK = False


# --------------------------------------------------------------------------- #
# 信号
# --------------------------------------------------------------------------- #
def tones_signal(N, tones, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = sum(a * np.sin(2 * np.pi * f * t)
            for a, f in zip([1.0 / (i + 1) for i in range(len(tones))], tones))
    return x + (noise * rng.standard_normal(N) if noise else 0.0)


def make_signals(N=2048):
    """{名字: (信号, 已知分量频率 Hz 或 None)}"""
    rng = np.random.default_rng(1)
    t = np.arange(N) / FS
    carrier = np.sin(2 * np.pi * 300 * t)
    impulse = np.zeros(N)
    impulse[::40] = 1.0
    return {
        "multitone": (tones_signal(N, (50, 150, 320)), (50, 150, 320)),
        "close_tones": (tones_signal(N, (50, 70), noise=0.02), (50, 70)),
        "chirp": (np.sin(2 * np.pi * (20 * t + 200 * t * t)), None),
        "amfm": ((1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 120 * t), (120,)),
        "white": (rng.standard_normal(N), None),
        "dc_trend": (5.0 + 0.8 * np.sin(2 * np.pi * 40 * t) + 0.2 * rng.standard_normal(N), (40,)),
        "bearing": (carrier * (0.2 + np.convolve(impulse, np.hanning(8), mode="same"))
                    + 0.05 * rng.standard_normal(N), (300, 25)),
        "spike": (np.where(np.arange(N) == N // 4, 1.0, 0.0) + 0.02 * rng.standard_normal(N), None),
    }


# --------------------------------------------------------------------------- #
# 三个被比较的实现 -> (IMFs, Res, mfb, boundaries_Hz|None, info)
# --------------------------------------------------------------------------- #
def run_ours(S, K=None, **kw):
    """自研实现 (注册键 "EWT", 即 Modal_Decomposition.EWT)。"""
    if K is not None:
        kw.setdefault("num_imfs", int(K))
    kw.setdefault("mirror", False)
    r = M.Class.EWT(**kw).decompose(S, fs=FS)
    return r.IMFs, r.Res, r.info.get("mfb"), r.info["boundaries"], r.info


def run_ewtpy(S, K=K_DEFAULT):
    """ewtpy 原始调用 (参考实现; 直接 import, 不走本库适配层)。"""
    ewt, mfb, bnd = EWT1D(S, K, **EWTPY_KW)
    bands = np.asarray(ewt).T
    return (bands, S - bands.sum(axis=0), np.asarray(mfb).T,
            np.concatenate(([0.0], np.asarray(bnd) / np.pi * (FS / 2), [FS / 2])), {})


def run_adapter(S, K=K_DEFAULT):
    """本库的 ewtpy 适配层 (可选入口 "EWTpy"): 模态应与 run_ewtpy 逐位一致。"""
    r = M.Class.EWTpy(N=K).decompose(S)
    return r.IMFs, r.Res, r.info.get("mfb"), None, r.info


METHODS = {"native": run_ours, "ewtpy": run_ewtpy, "adapter": run_adapter}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def rel_err(x, ref):
    n = float(np.linalg.norm(ref))
    return float(np.linalg.norm(x - ref) / n) if n else float("nan")


def max_err(x, ref):
    return float(np.max(np.abs(x - ref)))


def tone_split(H, tone, axis_n):
    """纯音能量按滤波器组分配: E_i = sum_f |T(f)|^2 H_i(f)^2 / sum_f |T(f)|^2"""
    P = np.abs(np.fft.rfft(tone, n=axis_n)) ** 2
    W = np.asarray(H, dtype=np.float64) ** 2
    nb = min(P.size, W.shape[1])
    P, W = P[:nb], W[:, :nb]
    tot = float(P.sum())
    return np.zeros(W.shape[0]) if tot <= 0 else (W * P[None, :]).sum(axis=1) / tot


def capture_metrics(H, tones, N):
    """(最小 capture, 平均 capture, 平均覆盖带数): capture = 单带吃下的最大占比"""
    if H is None or not tones:
        return float("nan"), float("nan"), float("nan")
    axis_n = (H.shape[1] - 1) * 2
    caps, cover = [], []
    for f in tones:
        e = tone_split(H, np.sin(2 * np.pi * f * np.arange(N) / FS), axis_n)
        caps.append(float(e.max()))
        cover.append(int(np.sum(e > 0.01)))
    return float(np.min(caps)), float(np.mean(caps)), float(np.mean(cover))


def ideal_boundaries(tones):
    if not tones or len(tones) < 2:
        return None
    return np.array([0.5 * (tones[i] + tones[i + 1]) for i in range(len(tones) - 1)])


def boundary_error(boundaries_hz, tones):
    """
    (平均"最近匹配"距离 Hz, 漏检数, 多检数)。

    理想边界 = 相邻音中点; 检测边界按最近距离匹配 (容差 = 最小音间距的 25%),
    未被任何理想边界匹配上的检测边界计为"多检"(spurious)。
    """
    ideal = ideal_boundaries(tones)
    if boundaries_hz is None or ideal is None:
        return float("nan"), 0, 0
    inner = np.asarray(boundaries_hz)[1:-1]
    if inner.size == 0:
        return float("nan"), int(ideal.size), 0
    tol = 0.25 * float(np.min(np.diff(np.sort(tones))))
    dist = np.array([np.min(np.abs(inner - x)) for x in ideal])
    matched = dist <= tol
    spurious = int(sum(1 for v in inner if np.min(np.abs(ideal - v)) > tol))
    return float(np.mean(dist)), int((~matched).sum()), spurious


def boundary_stability(fn, tones, N, reps=8):
    """同一信号 + 不同噪声实现下, 内部边界的标准差均值 (越小越稳)。"""
    ideal = ideal_boundaries(tones)
    if ideal is None:
        return float("nan")
    base = tones_signal(N, tones, noise=0.05)
    vals = []
    rng = np.random.default_rng(7)
    for _ in range(reps):
        S = base + 0.15 * rng.standard_normal(N)
        try:
            b = fn(S)[3]
        except Exception:
            return float("nan")
        if b is None:
            return float("nan")
        inner = np.asarray(b)[1:-1]
        n = min(inner.size, ideal.size)
        if n:
            vals.append(inner[:n])
    if len(vals) < 2:
        return float("nan")
    return float(np.mean(np.std(np.stack(vals), axis=0)))


# --------------------------------------------------------------------------- #
# 报告段落
# --------------------------------------------------------------------------- #
def sec1_recon(N=2048, K=K_DEFAULT):
    """
    两个指标要分开看 (否则会误判):

    * **带和保真** ``||S - ΣIMFs|| / ||S||``: 滤波器组幅度划分的紧致程度
      (ΣH_i 与 1 的偏离), 与"残差怎么定义"无关 -> 跨实现可比;
    * **重构保真** ``||S - (ΣIMFs + Res)||``: 各实现**自己**的残差约定下的保真度。
      native: ``Res = S - ΣIMFs`` (恒精确); ewtpy 无残差概念, 故只给带和;
      adapter (本库 EWTpy 入口) 同样按库内契约定残差, 故亦恒精确, 其
      ``info["ewtpy_band_sum_error"]`` 即 ewtpy 原始约定的带和偏差。
    """
    print(f"\n## §1 带和保真 vs 重构保真 (N={N}, K={K})\n")
    print("| 信号 | native 带和误差 | native 重构误差 | ewtpy 带和误差 | adapter 重构误差 | ewtpy 原始带和(适配层 info) |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, (S, _) in make_signals(N).items():
        n_imfs, n_res, _, _, n_info = run_ours(S, K)
        e_imfs, _, _, _, _ = run_ewtpy(S, K)
        a_imfs, a_res, _, _, a_info = run_adapter(S, K)
        band_native = rel_err(n_imfs.sum(axis=0), S)
        band_ewtpy = rel_err(e_imfs.sum(axis=0), S)
        rec_native = rel_err(n_imfs.sum(axis=0) + (n_res if n_res is not None else 0.0), S)
        rec_adapter = rel_err(a_imfs.sum(axis=0) + (a_res if a_res is not None else 0.0), S)
        print(f"| {name} | {band_native:.3e} | {rec_native:.1e} | {band_ewtpy:.3e} | "
              f"{rec_adapter:.1e} | {a_info['ewtpy_band_sum_error']:.3e} |")


def sec2_capture(N=2048, K=K_DEFAULT):
    print(f"\n## §2 频带划分质量 (纯音能量按带分配, N={N}, K={K})\n")
    print("| 信号 | 方法 | 最小 capture | 平均 capture | 平均覆盖带数 |")
    print("|---|---|---:|---:|---:|")
    for name in ("multitone", "close_tones", "bearing"):
        S, tones = make_signals(N)[name]
        for key in ("native", "ewtpy", "adapter"):
            imfs, res, mfb, _, _ = METHODS[key](S, K)
            cmin, cmean, cover = capture_metrics(mfb, tones, N)
            print(f"| {name} | {key} | {cmin:.4f} | {cmean:.4f} | {cover:.2f} |")


def sec3_boundaries(N=2048, K=K_DEFAULT):
    print(f"\n## §3 边界质量 (最近匹配距离 Hz / 漏检 / 多检 / 噪声稳定性 Hz)\n")
    print("| 信号 | 方法 | 平均匹配距离 | 漏检 | 多检 | 稳定性(std) | 检测到的内部边界 (Hz) |")
    print("|---|---|---:|---:|---:|---:|---|")
    for name in ("multitone", "close_tones", "bearing"):
        S, tones = make_signals(N)[name]
        for key in ("native", "ewtpy"):
            fn = METHODS[key]
            b = fn(S, K)[3]
            err, miss, spur = boundary_error(b, tones)
            stab = boundary_stability(lambda x, f=fn: f(x, K), tones, N)
            inner = "—" if b is None else np.round(np.asarray(b)[1:-1], 1).tolist()
            print(f"| {name} | {key} | {err:.2f} | {miss} | {spur} | {stab:.2f} | {inner} |")


def sec4_strategies(N=2048, K=K_DEFAULT):
    print(f"\n## §4 边界策略 (native, N={N}, K={K})\n")
    print("| 信号 | 策略 | n_bands | 最小 capture | 匹配距离 Hz | 漏检 | 多检 | 稳定性 Hz | 耗时 ms |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("multitone", "close_tones", "bearing"):
        S, tones = make_signals(N)[name]
        for bm in ("maximum", "max-min", "scale-space", "envelope"):
            t0 = time.perf_counter()
            imfs, res, mfb, b, info = run_ours(S, K, boundary_mod=bm)
            dt = (time.perf_counter() - t0) * 1e3
            cmin = capture_metrics(mfb, tones, N)[0]
            err, miss, spur = boundary_error(b, tones)
            stab = boundary_stability(lambda x: run_ours(x, K, boundary_mod=bm), tones, N)
            print(f"| {name} | {bm} | {info['n_bands']} | {cmin:.4f} | {err:.2f} | {miss} | "
                  f"{spur} | {stab:.2f} | {dt:.1f} |")


def sec5_pre_deal(N=2048, K=K_DEFAULT):
    """预处理分支的**可测增益**: DC/趋势去除、强噪下的边界稳定性、加窗的边界振铃。"""
    print(f"\n## §5 预处理分支增益 (native, N={N}, K={K})\n")
    sigs = make_signals(N)
    t = np.arange(N) / FS

    print("### 5.1 DC / 趋势去除 (信号 = 5.0 + 0.8·sin(40Hz) + 0.02·坡度 + 0.2·噪声)\n")
    S = 5.0 + 0.8 * np.sin(2 * np.pi * 40 * t) + 0.02 * (np.arange(N) / N) + 0.2 * np.random.default_rng(3).standard_normal(N)
    print("| pre_deal | IMF0 均值 | 残差(vs 原始) | 残差(vs 预处理) | 低频带能量占比 |")
    print("|---|---:|---:|---:|---:|")
    for pd in (None, ["no-dc"], ["no-trend"], ["no-dc", "no-trend"]):
        imfs, res, _, _, info = run_ours(S, K, pre_deal=pd)
        tot = float(np.sum(imfs ** 2)) or 1.0
        print(f"| {pd} | {imfs[0].mean():.4f} | {info['residual_ratio']:.4f} | "
              f"{info['residual_ratio_preprocessed']:.4f} | {float(np.sum(imfs[0] ** 2)) / tot:.4f} |")

    print("\n### 5.2 强噪下的边界稳定性 (multitone + σ=1.0 噪声, 8 次实现)\n")
    base = tones_signal(N, (50, 150, 320), noise=0.0)
    print("| pre_deal | 平均匹配距离 Hz | 稳定性(std) Hz |")
    print("|---|---:|---:|")
    for pd in (None, ["Slepian-Optimize"], ["no-trend", "Slepian-Optimize"]):
        def fn(S):
            return run_ours(S, K, pre_deal=pd)
        rng = np.random.default_rng(11)
        dists, inners = [], []
        for _ in range(8):
            Sx = base + 1.0 * rng.standard_normal(N)
            b = fn(Sx)[3]
            e, _, _ = boundary_error(b, (50, 150, 320))
            dists.append(e)
            inners.append(np.asarray(b)[1:-1])
        n = min(len(v) for v in inners)
        print(f"| {pd} | {np.nanmean(dists):.2f} | "
              f"{float(np.mean(np.std(np.stack([v[:n] for v in inners]), axis=0))):.2f} |")

    print("\n### 5.3 加窗对边界振铃的影响 (chirp, 端点 2% 与内部峰值比)\n")
    S = sigs["chirp"][0]

    def ring(x, frac=0.02):
        k = max(1, int(x.size * frac))
        den = float(np.max(np.abs(x[k:-k]))) or 1.0
        return float(max(np.max(np.abs(x[:k])), np.max(np.abs(x[-k:]))) / den)

    print("| pre_deal | 平均振铃比 (越小越好) | 重构(vs 原始) |")
    print("|---|---:|---:|")
    for pd in (None, ["window"], ["no-trend", "window"]):
        imfs, res, _, _, info = run_ours(S, K, pre_deal=pd, window_kind="hann")
        rings = float(np.mean([ring(row) for row in imfs]))
        rec = imfs.sum(axis=0) + (res if res is not None else 0.0)
        print(f"| {pd} | {rings:.4f} | {rel_err(rec, S):.2e} |")


def sec6_speed(sizes=(2048, 8192, 32768, 65536), K=K_DEFAULT, rep=3):
    print(f"\n## §6 速度与峰值内存 (K={K})\n")
    print("| N | native ms | ewtpy ms | adapter ms | 加速(vs ewtpy) | native MB | ewtpy MB |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    def timed(fn, S):
        best = float("inf")
        for _ in range(rep):
            t0 = time.perf_counter()
            fn(S, K)
            best = min(best, time.perf_counter() - t0)
        tracemalloc.start()
        fn(S, K)
        peak = tracemalloc.get_traced_memory()[1] / 1024 ** 2
        tracemalloc.stop()
        return best * 1e3, peak

    for N in sizes:
        S = tones_signal(N, (50, 150, 320))
        tn, pn = timed(run_ours, S)
        te, pe = timed(lambda x, k: run_ewtpy(x, k), S)
        tw, _ = timed(lambda x, k: run_adapter(x, k), S)
        print(f"| {N} | {tn:.1f} | {te:.1f} | {tw:.1f} | {te / tn:.1f}x | {pn:.1f} | {pe:.1f} |")


def sec7_edge(K=K_DEFAULT):
    print(f"\n## §7 鲁棒性边界情形 (K={K})\n")
    print("| 情形 | native | ewtpy | adapter |")
    print("|---|---|---|---|")
    short = [4, 5, 6, 7, 8, 10, 16]
    cases = [("N=%d" % n, np.sin(2 * np.pi * 50 * np.arange(n) / FS)) for n in short]
    cases += [
        ("constant", np.ones(2048) * 3.0),
        ("zeros", np.zeros(2048)),
        ("spike", make_signals(2048)["spike"][0]),
        ("DC=1e3", 1e3 + np.sin(2 * np.pi * 40 * np.arange(2048) / FS)),
        ("pure_tone", np.sin(2 * np.pi * 100 * np.arange(2048) / FS)),
    ]
    for label, S in cases:
        cells = []
        for key in ("native", "ewtpy", "adapter"):
            try:
                imfs, res, _, _, _ = METHODS[key](S, K)
                rec = imfs.sum(axis=0) + (res if res is not None else 0.0)
                cells.append(f"ok {imfs.shape} recon {rel_err(rec, S):.1e}")
            except Exception as exc:
                cells.append(f"{type(exc).__name__}: {str(exc)[:26]}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    global _QUICK
    ap = argparse.ArgumentParser(description="EWT native vs ewtpy benchmark")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    _QUICK = args.quick

    print("# 自研 EWT vs ewtpy —— 基准数据 (自动生成)")
    print(f"\nnative = Modal_Decomposition.Class.EWT (自研实现, mirror=False, boundary_mod=maximum) | "
          f"ewtpy = EWT1D(N={K_DEFAULT}, detect='locmax') | adapter = 本库可选入口 Class.EWTpy")
    sec1_recon()
    sec2_capture()
    sec3_boundaries()
    sec4_strategies()
    sec5_pre_deal()
    sec6_speed(sizes=(2048, 8192) if _QUICK else (2048, 8192, 32768, 65536))
    sec7_edge()


if __name__ == "__main__":
    main()
