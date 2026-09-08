# EMD 对比套件 — Modal_Decomposition vs PySDKit (ref/pysdkit)

目标: 回答"我的 EMD 和他的 EMD, 性能(时间/内存)谁更好"。
方法: 三列黑盒对比 —— `MD-EMD`(我的封装, 内部是 PyEMD sifter)、
`PyEMD`(直用同一 sifter, 隔离封装开销的基线)、`PySDKit`(ref/pysdkit v0.5.0
的独立移植)。三列全部默认参数, 同一台机器、同一输入数组。

## 目录

| 文件 | 内容 |
|---|---|
| `REPORT_EMD_MD_vs_PySDKit.md` | **结论报告** (速度/内存/质量/引擎同源性分析 + 复现命令) |
| `POSTPONED_3GB.md` | 3GB 及超大随机噪声实验的**推迟记录** (理由/先例/恢复步骤) |
| `signals.py` | 确定性基准信号 (A 双音+噪声 / B AM-FM+趋势 / C 白噪声) |
| `workers.py` | 三个实现的统一 worker + 参数对齐说明 |
| `quality.py` | 质量指标 (重构误差/正交性 IO/模式恢复) |
| `bench_timing.py` | 常规长度计时轴 (n=256..65536, 3 重复中位数) |
| `bench_memory.py` | 大数据内存轴驱动 (1MB..1GB, 子进程+预算+心跳) |
| `case_worker.py` | 内存轴单格子进程 (输入构建 → 预热 → 测量 → 心跳 RSS) |
| `summarize.py` | 汇总为 markdown/CSV 表格 |
| `bench_plot.py` | 出图 (`figs/*.png`) |
| `results/` | 原始记录: `timing_raw.csv`, `timing_metrics.json`,
  `memory/<impl>.json/csv`, `env.json`, `summary_*.md`, `memory_flat.csv` |

## 快速开始

```powershell
# 1) 常规长度计时+质量 (约 5 分钟)
.venv\Scripts\python.exe tests\comparison\bench_timing.py

# 2) 大数据内存网格 (1MB..1GB; 20s/格; 3GB 与 >=1GB 随机默认记为 deferred 不跑)
.venv\Scripts\python.exe tests\comparison\bench_memory.py

# 3) 汇总 + 图
.venv\Scripts\python.exe tests\comparison\summarize.py
.venv\Scripts\python.exe tests\comparison\bench_plot.py
```

子集/调参例子:

```powershell
python tests\comparison\bench_timing.py --cases A B --ns 4096 16384
python tests\comparison\bench_memory.py --sizes 1GB --patterns increasing
python tests\comparison\bench_memory.py --budget 60            # 放宽分解预算
python tests\comparison\bench_memory.py --no-defer            # 大内存机器上执行 deferred 格
```

## 公平性约定

* 计时: 同进程、轮转顺序、3 次重复取中位数; import/预热不计时;
  每次运行前 gc, 并校验输入未被实现改写。
* 内存: 每格独立子进程; **输入完全建成后**才开始计分解预算; RSS 以 50 ms
  心跳采样 (被 kill 的超时格也有增长轨迹, 见 `results/memory/_runs/*.hb.log`)。
* 质量: 三种实现的重构/正交性/模式恢复指标在同一代码路径计算。
* 限制: 单机 16GB RAM, 系统负载波动 ±5–10%; 亚毫秒格子 (n=256) 的比值无意义。

## 主要结论 (详见 REPORT)

* 常规长度: PySDKit 慢约 5–20% (中位 ~1.1x), 我的封装 ~0–4% 开销, 两者都 ≈ PyEMD;
* 大数据: 峰值 RSS 三者基本一致 (~8–9× 输入); pysdkit 无内存优势;
* 质量: 逐位级一致 (尾部边界偶尔 ±1 个低能 IMF);
* 我库的可优化点已按指示**以 EMD 为代表修复** (REPORT §8): `Check_Time_and_Signal`
  新增 `default_T` 开关, EMD 不再分配引擎用不到的默认时间轴 —— 峰值 RSS
  100MB/500MB/1GB 分别降 ~0.1–0.2GB / ~0.5–1.5GB / ~1.5GB; 其余算法未动,
  留待各自优化轮次。

## 已推迟实验 (只记录不执行)

3GB 全网格、随机噪声 ≥1GB: 见 `POSTPONED_3GB.md`;
先前 `tests/test_memory` 的 3GB 尝试 (EMD.csv) 全部为 timeout。
