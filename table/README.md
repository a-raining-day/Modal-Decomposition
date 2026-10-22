# `table/` — 表格数据目录（分级）

本目录存放**全部** `csv` / `xlsx`。分工如下（三者互不重叠）：

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `fig/` | 图片（png） | 数据 |
| `table/` | 机器可读表格（csv、xlsx） | 图、原始 JSON |
| `tests/…/results/` | 原始逐格 JSON（实验产物、可被脚本再读取） | 汇总表 |
| `docs/` | 叙述性报告（md） | 数据 |

```
table/
├── index.xlsx                 # 全库汇总工作簿（每个 csv 一个 sheet + 总览 sheet）
├── methods/                  分解法的逐 run 计时表
│   ├── efd_fixed/ fmd/ lmd_fixed/ vmd/ vmd_pysdkit/   timing_raw.csv
│   └── _legacy_pyemd_wrapper/<method>/timing_raw.csv  # 历史（PyEMD 包装时代）归档
├── utils/                    Utils/ 工具深度测试的汇总表
│   └── <tool>/               <tool>_cases.csv / <tool>_backends.csv / …
├── base/                     Base/ 低成本工具的汇总表
│   └── <tool>/
├── comparison/               跨方法对照汇总
└── size_axis/                大规模规模轴（1 MB–600 MB）
    ├── <tool>_<op>.csv       逐规模一行：n、nbytes、耗时、吞吐、峰值内存
    └── summary.csv           所有工具 × 所有规模的合并表（供画图）
```

## 约定

- **列名用 ASCII 小写下划线**（`n`, `nbytes`, `wall_ms`, `mb_per_s`, `peak_mib`），
  中文只出现在 `docs/` 的叙述里，避免 `csv` 在不同 locale 下乱码。
- **每个 csv 必须能被 `table/index.xlsx` 的生成脚本枚举**：文件名 = 内容主题，
  不再套一层 `data/`、`output/` 之类无信息目录。
- 规模轴里的 **"MB" 一律指 MiB（2²⁰ B）**：`1 MiB` = 1 048 576 B = 131 072 个 float64。
- 逐 run 的 `timing_raw.csv` 由 `tests/comparison/bench_timing.py` 写入（`DEFAULT_TABLE`），
  `summarize_*.py` 读同一路径，JSON 指标仍留在 `tests/comparison/results/<method>/`。
