# Pareto 扫描结果（15 个方法）

由 `.scratch/eemd_new/pareto_all.py` 串行生成，每个方法一个 `<METHOD>.json`。

## 度量

| 字段 | 含义 |
|---|---|
| `q_rec` | 真实分量（A 用例的 37 Hz / 113 Hz）的最大绝对相关均值 |
| `q_io` | `min(1, max(0, 1 - 正交性指数))` |
| `q` | `0.5*q_rec + 0.5*q_io` |
| `leak` | `‖Res‖ / ‖S‖`；无残差概念的方法为 `null` |
| `t_ms` | A/1024（FMD 用 A/512）单次耗时 |

信号：`tests/comparison/signals.py::build_case("A", n)`。选点规则：可行域（`leak <= 0.05` 或无残差）内 Q 最高，再取 Q ≥ 99.5% 最优点中最省时的那个。每个 JSON 里另有一行 `current: true`，是**当前默认参数**的参照结果。

## 结论

- **12/15 的方法，其当前默认参数已经落在 Pareto 选点的 2% 以内**（多数方法的 Q 对旋钮不敏感，或选点就是默认值本身）。
- **4 个方法（FMD / LMD / MEMD / SVMD）全部组合都不可行**（`leak > 0.05`）。这不是旋钮问题，而是算法特性：它们的残差里保留了大量能量。
  - LMD 另有幅值钳制（`1e4 × 原信号尺度`），残差会被放大。
  - FMD 的残差与模态能量接近，`leak` 常年在 0.2–0.4。
  - MEMD / SVMD 的模态数固定，剩余能量本就无处可去。
- 在 Q 上有可测差异的三个方法及其问题：

| 方法 | 选点 | ΔQ | 与既有文档的关系 |
|---|---|---|---|
| RPSEMD | `sin_smaple_num=4, separation_ratio=1.2` | +24.5% | **与 `docs/test/RPSEMD_Native_Engine_Report.md` 的结论相反**：那份报告用 A/B/C 三例复合 Q，得出 `sin_smaple_num=8` 最优、`separation_ratio` 在 {1.2,1.5,2.0} 上**逐位相同**。本表的提升来自单例 A 的度量，不足以推翻它。 |
| EWT | `num_imfs=8` | +6.0% | 无既有 Pareto 记录，但把默认模式数从 5 改成 8 会改变**默认输出维度**，属可见的 API 变更。 |
| SSA | `window_size=128, rank_rule="svht"` | +2.2% | `rank_rule="svht"` 是**有损截断**；默认 `None` 才是"完美重构"。Q 只因为 IO 变好，不宜作为默认。 |

## 采纳情况（用户决定）

- **RPSEMD**：采纳 → 默认改为 `separation_ratio=1.2`、`sin_smaple_num=4`（Q 0.4728 → **0.5891**）。
- **EWT**：采纳 → 默认改为 `num_imfs=8`（Q 0.9269 → **0.9829**）。
- **SSA**：不采纳（`rank_rule="svht"` 是有损截断，保持 `None` = 完美重构）。

## 原先的保留理由（存档）

理由：单例（A/1024）复合度量不足以推翻已经用 A/B/C 三例复合 Q 定下的默认值；而 EWT 与 SSA 的两个候选改动会改变默认语义（输出维度 / 有损截断）。若要采纳，建议先补 A/B/C 三例复合扫描再定。

## 复现

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py            # 全部 15 个
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py EMD EEMD   # 指定方法
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_report.py         # 决策表
```
