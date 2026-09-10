# 3GB 大数据实验 — 推迟记录 (Postponed)

> 记录日期: 2026-09-06 (会话快照)
> 范围: EMD 对比套件中 "数据尺寸达到 2~3 GB" 的记忆/吞吐格子。
> 状态: **推迟，不执行** — 本文件是它的计划与理由记录。
>
> **2026-09-10 更新**: 内存轴已在**原生引擎**上重跑
> （`docs/EMD_Large_Signal_Memory_Report.md`）:
> * increasing 轴原生引擎快 6–12×（1 GB: 0.91 s, 峰值 4257 MB），本机 16 GB
>   完全可承载 —— 与 §2 的"包装时代先例"结论已不同（当时 1 GB 需 10.8 s /
>   8119 MB）;
> * random 500 MB 重跑仍 timeout（MD 峰值 7522 MB vs 旧 7789 MB）；
> * random 1 GB 与 3 GB **保持 deferred**（§4 理由不变）。

## 1. 这套格子是什么

`bench_memory.py` 的记忆矩阵沿用了 `tests/test_memory` 的轴约定:

```
implementation × data size × pattern × dtype
{MD-EMD, PyEMD, PySDKit} × {1MB 20MB 100MB 500MB 1GB 3GB} × {increasing, random} × float64
```

本文件推迟的格子:

| 格子 | 处理 |
|---|---|
| size >= 2GB (即 3GB 一列, 全部 pattern) | 一律记录为 `deferred`, 不执行 |
| random noise 且 size >= 512MB | 同上(见 §4 本机内存约束) |

## 2. 为什么推迟 —— 先例证据

`tests/test_memory` 已用同一机器/同一 PyEMD 引擎跑过完整的
`{1MB … 3GB} × {f16,f32,f64} × {increasing, random}` 网格
(`run_matrix.py`, 每格 20 s 预算, 结果见 `tests/test_memory/result_for_each_decomposition/EMD.csv`):

* 3GB 全部 12 格 (3 dtype × 2 pattern × …) **全部只有 phase-1(输入建成) 记录, 无一产出最终报告** —— 即全部 timeout/crash;
* 随机噪声从 **20MB 起就无法在 20 s 内完成** (1MB 噪声 ok, 20MB/100MB/500MB/1GB/3GB 全部 timeout);
* 只有单调递增 (zero-extrema) 输入能在 1GB/3GB 量级跑完 —— 那不是有代表性的 EMD 计算。

纯 Python 筛分对噪声输入的代价 ~ O(N × sift_iter × 样条), 3GB float64
(≈3.75 亿采样) 的一阶 IMF 就要成百上千次全数组样条, 预期数小时~数十小时,
对三方实现 (我的封装/PyEMD/pysdkit) 都成立。跑它唯一的新信息是"RSS 增长轨迹",
而这一信息在 1GB 及以下已经能测到。

## 3. 恢复执行的步骤 (换机器后)

```powershell
# 先确认目标机 RAM >= 64GB, %TEMP% 剩余 >= 3×3GB
.venv\Scripts\python.exe tests\comparison\bench_memory.py --sizes 3GB --budget 60 --no-defer
```

* `--budget` 建议 60 s 起 (输入构建后单独计时, 父进程按 phase-1 报告计时);
* 预期结果: `random` 格全部 timeout + RSS 心跳轨迹 (记忆增长斜率),
  `increasing` 格可以 ok 并给出真实 delta RSS;
* 产出照旧落到 `tests/comparison/results/memory/*.json` 和 `_runs/*.hb.log`。

## 4. 附加推迟: random noise >= 512MiB (本机 16GB RAM)

本机总内存 15.6GB。噪声 sift 每一步的驻留工作集 ≈ 5~10 × N(全尺寸 imf/mean/
上下包络/样条临时量), 500MB/1GB 噪声 → 数 GB~10GB 峰值, 在 16GB 机器上会
直接换页, 使"超时"记录失去可比性且拖垮整机。

本套件实际执行边界 (与阈值的字节换算):

* random **500MB(十进制 MB, 即 476.8MiB, 低于 512MiB 阈值)** 已执行并记录:
  三家均在 20s 预算内 timeout。**2026-09-10 原生引擎重跑**: MD-EMD kill 时
  峰值 7522 MB（旧引擎 7789 MB），PyEMD 8439 MB、PySDKit 8730 MB —— 已贴近
  本机换页边界, 属于"跑得动但已无比较价值"的极限格;
* random **>= 512MiB (如 1GB token)** 默认记录为 `deferred` 不执行
  (1GB random 曾在 `tests/test_memory` 以 20s 预算尝试过, 仅留下 phase-1);
* 换大内存机器: `python tests\comparison\bench_memory.py --sizes 500MB 1GB --patterns random --no-defer`

## 5. 相关产出

* 本套件可执行部分的结果: `tests/comparison/results/`
* 之前的 3GB 先例数据: `tests/test_memory/result_for_each_decomposition/EMD.csv`
  (3GB 格仅 `*.report.phase1.json`, 全部无最终报告)
