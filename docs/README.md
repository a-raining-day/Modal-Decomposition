# `docs/` — 文档目录（分级）

仓库的文字材料全部在这里。**代码、数据、图片不在本目录**。

```
docs/
├── README.md                     # 本文件：目录导航与写作约定
├── test/                         # 验证类文档（与 tests/ 一一对应）
│   ├── README.md                 # 包说明（API、依赖、致谢、许可证声明）
│   ├── EXPERIMENTS_INDEX.md      # 实验索引：脚本 ↔ 原始数据 ↔ 报告
│   ├── methods/                  # 15 个分解法：一方法一报告
│   │   ├── EMD_Native_Engine_Report.md … SVMD_Native_Engine_Report.md
│   │   └── Large_Array_Memory_Report.md   # 跨方法的大数组内存报告
│   ├── utils/                    # Utils/ 工具层
│   │   ├── Check_Engine_Report.md … Slepian_Engine_Report.md   # 模块报告
│   │   ├── Hilbert_Backend_Report.md / FFT_Backend_Report.md / Slepian_Backend_Report.md
│   │   └── 每个模块报告的末尾「深度测试」小节（本轮新增，含发现表与复现命令）
│   ├── base/                     # Base/ 低成本工具
│   │   ├── Cache_Engine_Report.md / Chunk / CPU / Memory / Seed（各含深度测试小节）
│   │   └── Utils_Base_API_Report.md   # 工具层三层 API 总索引
│   ├── Scale_Axis_Report.md      # 1 MiB–600 MiB 规模轴（8 个低成本工具）
│   ├── LICENSE                   # 与仓库根 LICENSE 同源
│   └── _template/                # 报告模板与生成器（FunctionTemplate / Template_Generate.py）
└── explore/                      # 探索性研究（非契约性结论）
    └── denoise/                  # 去噪专题：只放报告（图在 fig/explore/denoise/）
```

## 材料落位规则

| 类型 | 位置 | 例子 |
|---|---|---|
| 契约/实测报告 | `docs/test/{methods,utils,base}/` | `EMD_Native_Engine_Report.md` |
| 原始逐格数据（json） | `tests/…/results/` | `tests/comparison/results/emd_new/pareto_sweep.json` |
| 汇总表格（csv/xlsx） | `table/` | `table/size_axis/monotonicity_scale.csv` |
| 图片（png） | `fig/` | `fig/methods/emd/emd_pareto.png` |
| 可执行测试 | `tests/` | `tests/utils/test_Monotonicity_deep.py` |

## 写作约定

1. **每个数字都要能追溯**：报告里出现的数值，必须能在 `tests/` 的某个 JSON、
   或 `table/` 的某个 csv、或报告内直接给出的复现命令里找到。
2. **图必须带相对链接**（`../../../fig/...`），不要写绝对路径；`tests/tools/check_links.py` 会检查。
3. **不得美化回归**：变慢、变差、失败都要如实写进"已知问题"。
4. 报告文件名与对象名对齐：`<Object>_<Kind>_Report.md`，`Kind ∈ {Native_Engine, Engine,
   Backend, Deep, Scale}`。

## 快速导航

- 想知道**有什么实验**：`docs/test/EXPERIMENTS_INDEX.md`
- 想知道**怎么跑一个方法**：`docs/test/README.md`（API）→ `docs/test/methods/<M>_*.md`
- 想知道**工具层契约**：`docs/test/base/Utils_Base_API_Report.md` → 具体模块报告
- 想知道**大数组能不能跑**：`docs/test/methods/Large_Array_Memory_Report.md`（分解法）、
  `docs/test/Scale_Axis_Report.md`（工具层 1 MiB–600 MiB 规模轴）
