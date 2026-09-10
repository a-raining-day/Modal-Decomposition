# SSA（stride 参数化）信号级验证报告

> 对象: `src/Modal_Decomposition/SSA.py` —— stride 参数化 SSA（Hankel 矩阵按
> ``stride`` 抽列）、rank 规则族（`energy` / `svht` / `svht_clip`）、窗
> （`rect` / `hann` / `hamming`）与加权 OLA 重构。
> 数据: `tests/ssa/results/ssa_signal_results.json` + `.md`（由
> `tests/ssa/test_ssa_signal.py::test_ssa_signal_results_recorded` 生成）。
> 复现: `pytest tests/ssa -q`（三个测试文件；结果文件会被重写）。
> 关联: 组件统计与窗/权重的单元级验证见 `tests/ssa/test_ssa_rules.py`
> （19 项：SVHT omega 参考值、energy/svht/svht_clip 秩规则、去噪增益对比、
> 窗向量形状与 Hann 端点、groups∩rank、参数校验与 config 快照）。

## 1. 实验设置

* 信号（N=2048, fs=1000 Hz，确定性种子）：
  * `multitone` —— 37/113/231 Hz 三纯音（幅 1/0.8/0.5）+ 0.02 白噪；
  * `wideband` —— 纯白噪（平坦谱，无主导模式）；
  * `nonstationary` —— 中点频率跳变（37→90 Hz）+ 20% AM（分段窄带）。
* 扫描 `stride ∈ {1, 4, 16}`，全分解（`n_components == window_size`）。
* 指标：组件数、重构 `max|reconstruct()−S|`、首组件与第一真值分量的 |corr|
  （宽带用首组件能量占比），墙钟时间。

## 2. 信号级结果（N=2048）

| case | stride | window_size | n_components | recon max abs | 首组件指标 | wall (s) |
|---|---:|---:|---:|---:|---|---:|
| multitone | 1 | 682 | 682 | 1.31e-14 | corr 0.9965 | 8.45 |
| multitone | 4 | 343 | 343 | 6.38e-15 | corr 0.9957 | 0.93 |
| multitone | 16 | 87 | 87 | 3.55e-15 | corr 0.9834 | 0.06 |
| wideband | 1 | 682 | 682 | 5.77e-15 | energy share 0.0028 | 7.06 |
| wideband | 4 | 343 | 343 | 4.96e-15 | energy share 0.0038 | 0.98 |
| wideband | 16 | 87 | 87 | 5.77e-15 | energy share 0.0059 | 0.06 |
| nonstationary | 1 | 682 | 682 | 5.77e-15 | corr 0.9517 | 7.40 |
| nonstationary | 4 | 343 | 343 | 8.55e-15 | corr 0.9525 | 0.98 |
| nonstationary | 16 | 87 | 87 | 7.55e-15 | corr 0.9339 | 0.05 |

（墙钟为 2026-09-10 复跑值；质量类数字跨次逐位一致，仅墙钟随机器负载浮动
——同一网格前一次运行 stride=1 为 15.9–16.9 s，即绝对值可有 ~2× 波动。）

## 3. 解读

1. **重构精确性**: 全部 9 格 `max|reconstruct()−S| ≤ 1.3e-14`（浮点级），
   与 stride/窗/rank 参数无关——加权 OLA 与 Hankel 反投影的契约成立。
2. **stride 的加速与压缩**: stride 1→4→16 使组件数 682→343→87、墙钟
   8.45 s→0.93 s→0.06 s（**~150×** 端到端）。代价是频率分辨率下降：多音调
   首组件捕获从 0.9965 降到 0.9834（仍 >0.98），非平稳段从 0.9517 到 0.9339。
   即 stride 是"时间/分辨率"旋钮，不是质量开关——低 stride 用于精细分析，
   高 stride 用于快速筛查。
3. **分组对（sine/cosine pair）语义**: `groups=[[0,1],[2,3],[4,5]]` 时三个
   音调各自落入一对组件（测试断言 |corr| > 0.95）；非平稳信号的两段各自
   落入一对（断言 > 0.9，实测 0.93–0.95）——SSA 对分段窄带信号的分离能力
   由 stride=1 的实测支撑。
4. **宽带（白噪声）能量分散**: 首组件能量占比 0.0028–0.0059（≪ 0.5 的
   判据），说明对平坦谱不会产生虚假主导分量——SSA 输出在该场景下退化为
   正交基底展开，符合预期。
5. **测试覆盖闭环**: 结果记录测试外，`test_ssa_stride.py` 覆盖 stride=1 与
   默认位级一致、stride 缩小组件数、非法 stride/window_size 报错、
   `window_size=N−1` 边界与单样本拒绝；`test_ssa_rules.py` 覆盖三种秩规则
   与窗的数值契约、`groups` 与 rank 规则的交集语义。

## 4. 局限与备注

* 本报告只覆盖 `Class.SSA` 的信号级行为；不含与外部 SSA 实现（如
  `pyts`/`spectrum`）的横向对比——当前仓库未引入第三方 SSA 参考实现，
  因此采用"性质断言 + 数值记录"而非三方对照。
* 全分解的 `n_components == window_size` 会把信号分解成大量成对组件
  （682/343/87），实际使用应配合 `rank_rule` 或 `groups` 取前若干分量；
  本报告刻意保留全分解以验证重构契约的完整性。
* 结果文件位于 `tests/ssa/results/`（测试运行时自动重写）；本报告为其
  可读化版本，数值与该 JSON 一一对应。
