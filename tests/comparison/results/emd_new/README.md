# `results/emd_new/` —— EMD（原生引擎）验证轮的原始数据

本目录是 [`docs/test/methods/EMD_Native_Engine_Report.md`](../../../../docs/test/methods/EMD_Native_Engine_Report.md)
的全部底层数据。脚本在 `.scratch/emd_new/`，图片在 `fig/`。

> **命名说明**：除 `three_way.json` 由脚本直接写入本目录外，其余 JSON 是脚本写到
> `%TEMP%/emd_new_test/` 后的归档副本，归档时做了重命名（见下表"脚本原生输出名"）。
> 重跑脚本不会自动覆盖本目录，需要手动复制重命名。

---

## 1. 文件清单

| 本目录文件 | 脚本 | 脚本原生输出名 | 报告章节 | 内容 |
|---|---|---|---|---|
| `three_way.json` | `t27_three_way.py` | `three_way.json`（直接写入） | §3 | 四种实现 × A/B/C 9 格的三方对照 |
| `head_to_head.json` | `t06_headtohead.py` | `t06_headtohead.json` | （早期轮次，被 `three_way.json` 取代） | 更早一轮的四方对照 |
| `faster_ab.json` | `t23_faster.py` | `t23_faster.json` | §6 | 默认参数下 `faster=True/False` 逐格对比 |
| `pareto_sweep.json` | `t25_pareto.py` | `t25_pareto.json` | §5.2 / §5.3 | 48 个配置的扫描 + Pareto 前沿标签 |
| `pareto_refine.json` | `t26_refine.py` | `t26_refine.json` | §5.5 | 20 个配置的细化（填 15–123 ms 空档） |
| `memory_axis.json` | `t18_memory_axis.py` | `t18_memory_axis.json` | §9.1 | 每格独立子进程的 ΔRSS / peak RSS |

---

## 2. 字段说明

### 2.1 `three_way.json`（`t27`）

顶层是对象，三个键：

| 键 | 内容 |
|---|---|
| `env` | `harness.env_info()` 的环境快照，另含 `legacy` 子对象（wheel 路径、版本、解包目录、`EMD.py` 字节数与能力标记） |
| `grid` | `[[case, n], ...]`，本文件用的 9 格 |
| `rows` | 36 条记录（9 格 × 4 实现） |

每条 `rows` 记录：

| 字段 | 含义 |
|---|---|
| `case` / `n` | `"A"` / `"B"` / `"C"` 与样本数 |
| `impl` | 实现标签：`EMD (新, 默认=拐点)` / `EMD (新, 质量档)` / `EMD_OLD (0.3.0)` / `PyEMD 1.9` |
| `t_ms` | 同进程中位墙钟（ms，预热 1 次 + 3 次重复取中位） |
| `k` | IMF 行数（**不含**残差行） |
| `valid` / `valid_frac` | 满足 `\|zc − ext\| ≤ 1` 的 **IMF 行**数与占比 |
| `c1` / `c2` | case A = 37 Hz / 113 Hz 的最佳单行 \|corr\|；case B = AM-FM / 89 Hz；case C 为 `null` |
| `io` | Huang 正交性指数 |
| `recon` | `max\|Σ IMFs + Res − S\|` |
| `n_rows` | `k + 1`（含残差行） |
| `iters` | 该实现所有 IMF 的筛分迭代次数之和（PyEMD 不暴露，恒为 0） |
| `zcext` | 逐行 `\|zc − ext\|`，长度 `n_rows`；末位是残差行 |

实现对应的参数（写在 `t27_three_way.py` 顶部）：

```python
IMPLS = [
    ("EMD (新, 默认=拐点)", lambda: NEW()),
    ("EMD (新, 质量档)",    lambda: NEW(faster=False, internal_delta=0.002, internal_iter=150)),
    ("EMD_OLD (0.3.0)",    lambda: OLD(nbsym=2, sd_thr=0.01, max_iter=100, faster=False)),
    ("PyEMD 1.9",          lambda: H.engines()["PyEMD"]()),
]
```

> `EMD (新, 默认=拐点)` 取的是**运行时的模块默认值**。本轮归档时的默认是
> `nbsym=4, internal_delta=0.2, internal_iter=50, faster=True, mirror_mod="extrema"`；
> 若以后改了默认值，重跑 `t27` 得到的就不再是同一列，请对照报告 §5.4。

### 2.2 `head_to_head.json`（`t06`）

顶层是数组。每条记录字段：`case`、`n`、`label`、`t_ms`、`c1`、`c2`、
`valid_imf`（**仅 IMF 行**，如 `"7/7"`）、`n_rows`、`recon_max_abs_err`、
`orthogonality_index`、`mode_recovery{tone_37hz, tone_113hz, amfm, tone_89hz, trend}{best_abs_corr, best_nrmse}`、
`k_imfs`、`valid`（**含残差行**，如 `"8/8"`）、`valid_ok`、`valid_n`、`finite`、`zcext`。

### 2.3 `faster_ab.json`（`t23`）

顶层是数组。字段与 `head_to_head.json` 相同（少了 `mode_recovery` 的内层展开），
即 `case`、`n`、`impl`、`t_ms`、`k`、`valid`（**含残差行**）、`c1`、`c2`、`io`、
`iters`、`recon`。case C 的 `c1`/`c2` 是 `NaN`（已序列化为 `NaN`，读取时注意）。

`impl` 取值：`OLD  nbsym=2 faster=F`、`PyEMD nbsym=2`、
`NEW  faster=False (默认)`、`NEW  faster=True  (默认)`、`OLD  nbsym=2 faster=T`。

### 2.4 `pareto_sweep.json`（`t25`）

顶层是对象：

| 键 | 内容 |
|---|---|
| `all` | 48 条配置记录 |
| `front` | 13 个 Pareto 前沿点的 `label` 列表（按耗时升序、Q 严格递增） |

每条 `all` 记录：`label`（如 `"T nb4 d0.2 i50"`，`T`=`faster=True`、`F`=`faster=False`）、
`nbsym`、`internal_delta`、`internal_iter`、`faster`、`q_A`、`q_C`、`q_B`、
`Q`（`= (q_A+q_C+q_B)/3`）、`t_ms`（case A / n=16384 的中位耗时，重复 2 次）、
`k_A16`、`io_A16`。失败配置带 `error` 字段。

**Q 的定义**（`t25` 顶部注释与报告 §5.1 一致）：
`q_A` = n∈{4096,16384} 上 `0.5·(corr37 + corr113)` 的均值；
`q_C` = 白噪声 n=4096 的 IMF 行级合规率；`q_B` = 干净 AM-FM n=4096 的 `0.5·(corr_amfm + corr_89Hz)`。

### 2.5 `pareto_refine.json`（`t26`）

顶层是数组，每条记录：`label`、`t_ms`、`Q`、`q_A`、`q_C`、`q_B`、
`nbsym`、`internal_delta`、`internal_iter`、`faster`。
覆盖 `nbsym ∈ {2,3,4,5}` × `internal_delta ∈ {0.08,0.1,0.12,0.15,0.18}`，`internal_iter=50`。

### 2.6 `memory_axis.json`（`t18`）

顶层是数组。每条记录由被测子进程以一行 JSON 打印、驱动侧收集：

| 字段 | 含义 |
|---|---|
| `pattern` | `increasing`（`linspace(-1,1,n)` 单调斜坡）/ `random`（种子白噪声）/ `tones`（7+50+211 Hz 三音调 + 轻噪） |
| `size_mb` / `dtype` | 输入体积（MiB）与 dtype |
| `impl` | `OLD 0.3.0` / `NEW_ORIG (未改)` / `NEW num_imf=-1` / `NEW num_imf=20` |
| `status` | `ok` / `killed`（撞 RSS 上限或时间预算）/ `error` / `crash` / `timeout` |
| `wall` | `decompose` 墙钟（s） |
| `base_mb` / `peak_mb` / `delta_mb` | 输入建好后的基线 RSS、峰值 RSS、二者之差 |
| `n` / `n_imfs` / `imfs_dtype` | 样本数、IMF 行数、输出 dtype |
| `policy` | 该格 `should_use_memmap(S.nbytes)` 的返回值（`true`/`false`/`null`） |
| `samples` | 心跳采样次数 |
| `recon` | 重构误差 |
| `memmap_files` | 运行期间新建的 `md_memmap_*.dat` 个数 |

---

## 3. 如何重新生成

### 3.1 前提

```powershell
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "$PWD\.scratch\emd_new"
```

- `bootstrap.py` 会装一个 `Modal_Decomposition` 桩包绕开损坏的包级 `__init__.py`，
  并在**内存里**修复被污染的 scipy（`scipy_repair.py`，不改磁盘文件）。
  若已重装过 scipy，`scipy_repair.install()` 会自动跳过。
- `legacy.py` 会把 `dist/modal_decomposition-0.3.0-*.whl` 解包并重命名为 `Legacy_MD`
  （缓存在 `%TEMP%/emd_new_test/`），即 **EMD_OLD (0.3.0)** 基线。
- `t18` 支持环境变量 `MEM_BUDGET`（每格时间预算，默认 60 s）、`MEM_CAP_MB`
  （RSS 硬上限，默认 2000 MB）、`ONLY_B=1`（跳过 Part A，只跑 RSS 网格）。

### 3.2 逐文件重跑

```powershell
# three_way.json —— 直接写入本目录
& $py -X utf8 -u .scratch\emd_new\t27_three_way.py

# 其余五个：先写进 %TEMP%\emd_new_test\，再改名放进本目录
& $py -X utf8 -u .scratch\emd_new\t06_headtohead.py   # -> t06_headtohead.json  => head_to_head.json
& $py -X utf8 -u .scratch\emd_new\t23_faster.py       # -> t23_faster.json      => faster_ab.json
& $py -X utf8 -u .scratch\emd_new\t25_pareto.py       # -> t25_pareto.json      => pareto_sweep.json
& $py -X utf8 -u .scratch\emd_new\t26_refine.py       # -> t26_refine.json      => pareto_refine.json
& $py -X utf8 -u .scratch\emd_new\t18_memory_axis.py  # -> t18_memory_axis.json => memory_axis.json
```

一键改名（PowerShell）：

```powershell
$src = Join-Path $env:TEMP "emd_new_test"; $dst = "tests\comparison\results\emd_new"
@{ "t06_headtohead.json"="head_to_head.json"; "t23_faster.json"="faster_ab.json";
   "t25_pareto.json"="pareto_sweep.json"; "t26_refine.json"="pareto_refine.json";
   "t18_memory_axis.json"="memory_axis.json" }.GetEnumerator() |
  ForEach-Object { Copy-Item (Join-Path $src $_.Key) (Join-Path $dst $_.Value) -Force }
```

### 3.3 图片

```powershell
& $py -X utf8 -u .scratch\emd_new\t28_figures.py
```

该脚本只读本目录的 JSON，写出 `fig/methods/emd/emd_pareto.png`、`fig/methods/emd/emd_three_way.png`、
`fig/methods/emd/emd_faster.png`、`fig/methods/emd/emd_memory.png`、`fig/methods/emd/emd_architecture.png`。
注意 `t28` 里的图注用英文（默认 matplotlib 字体无中文字形）。

---

## 4. 注意事项

1. **`valid` 的口径不统一**：`three_way.json` 的 `valid` 只数 IMF 行；
   `head_to_head.json` / `faster_ab.json` 的 `valid` **含残差行**（同文件里另有
   `valid_imf` 或可从 `k` 推算）。跨文件比较前先对齐口径。
2. **计时只在同一文件内可比**：跨脚本、跨日期的绝对值差异可达 10×（机器负载）。
3. `faster_ab.json` 的 `NEW faster=True/False (默认)` 两列取的是**运行时的模块默认值**
   （当时为 `nbsym=3, internal_delta=0.002, internal_iter=300`），不是当前默认值。
4. `memory_axis.json` 的 `policy` 列在本机全为 `false`：`should_use_memmap` 在 ≤100 MB
   时未触发，因此这些数字是**纯内存路径**；memmap 路径由 `t18` Part A 单独验证
   （强制策略时 EMD 新建 1 个 `md_memmap_*.dat`，EMD_OLD 为 0 个）。
5. `NaN` 是合法取值（case C 无音调基准），读取时用 `math.isnan` 而不是相等比较。
