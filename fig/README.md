# `fig/` — 图片目录（分级）

本目录存放**全部**实验图片。规则只有三条：

1. **一张图只属于一个"对象"**：分解法进 `methods/`，工具层进 `utils/`，Base 层进 `base/`；
   跨对象/归档类图进 `comparison/`，规模轴（1 MB–600 MB）图进 `size_axis/`。
2. **目录名 = 对象名的小写**（`EMD` → `methods/emd/`，`Monotonicity` → `utils/monotonicity/`）。
3. **文件名保留生成脚本里的原名**（便于 `grep` 反查生成器），不在这里重命名。

```
fig/
├── methods/                  15 个分解法 + 跨方法内存报告
│   ├── emd/                  emd_pareto / emd_three_way / emd_faster / emd_memory / emd_architecture
│   ├── eemd/                 eemd_pareto / eemd_speed / eemd_quality / eemd_recon / eemd_inner
│   ├── ceemd/ ceemdan/ iceemdan/ rpsemd/    <method>_compare / <method>_pareto
│   └── （其余方法暂无图，见 docs/test/methods/）
├── utils/                    Utils/ 下 8 个工具的深度测试图
│   ├── check/ poly/ envelope/ peaks/ mirror/ monotonicity/ hilbert/ slepian/
├── base/                     Base/ 下 5 个低成本工具的图
│   ├── cache/ chunk/ cpu/ memory/ seed/
├── comparison/               跨方法对照与历史（PyEMD 包装时代）计时图
├── size_axis/                1 MiB / 50 / 100 / 200 / 500 / 600 MiB 规模轴图
└── explore/denoise/          去噪专题图（原先放在 docs/explore/denoise/figures/，本轮迁入）
```

## 生成脚本一览

| 图片 | 生成脚本 |
|---|---|
| `methods/emd/*.png` | `.scratch/emd_new/t28_figures.py` |
| `methods/eemd/*.png` | `.scratch/eemd_new/figs.py`、`fig_pareto.py` |
| `methods/{ceemd,ceemdan,iceemdan,rpsemd}/*.png` | `.scratch/eemd_new/fig_{ceemd,ceemdan,iceemdan,rpsemd}.py` |
| `comparison/*.png` | `tests/comparison/bench_plot.py`、`summarize_{vmd,lmd}.py` |
| `utils/**`、`base/**` | `tests/utils/test_<Tool>_deep.py`、`tests/base/test_<Tool>_deep.py` 的 `--export` 分支 |
| `explore/denoise/*.png` | 历史生成脚本（`.scratch` 已清理）；报告在 `docs/explore/denoise/` |
| `size_axis/**` | `tests/size_axis/run_scale.py --tool <tool> --export` |

> 判据：**图必须能被重跑脚本再生成一次**。没有生成脚本的一次性图片不放本目录
> （历史图见 `fig/comparison/`，它们由 `bench_plot.py` 的 legacy 分支生成）。
