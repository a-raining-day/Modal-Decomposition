# VMD 换版报告: vmdpy 包装 → 原生实现

- 日期: 2026-03-04
- 变更: **`src/Modal_Decomposition/VMD.py` 由 vmdpy 薄包装换成本库原生实现**;
  注册键 ``"VMD"``、类名 ``VMD``、``Class.VMD`` / ``Function.VMD`` 出口不变
- 对比数据: `tests/comparison/bench_vmd_native.py`(可复现),
  原始结果 `docs/VMD_Native_vs_vmdpy_Results.json`
- 一句话结论: 原生版**每一项都更优** —— 同等分解质量下 3–9× 更快、
  大尺寸提交内存低 100–200×、`reconstruct()` 精确(6.9e-18 vs 0.17–0.26)、
  且去掉 vmdpy 依赖; 旧关键字 `K` / `init` / `tol` 已做兼容。

---

## 1. 候选与选择

| 候选 | 形态 | 结论 |
|---|---|---|
| 旧 `VMD.py` | vmdpy 薄包装 (`vmdpy.VMD` 直调, `Res=None`) | 被替换(代码可由 git 取回) |
| `VMD_new.py` | 原生 ADMM(自研内核, 无 vmdpy) | **胜出 → 落到 `VMD.py`** |
| `VMD_new_new.py` | 原生 + 分块/外存/auto 引擎 | 分块实测**减速 15–25%**、外存 I/O 受限(见 `docs/VMD_Large_Array_Iteration_Report.md`) ⇒ 不进正式版 |

选择的依据不是"看起来更先进", 而是下面这两张表: 质量完全一致, 速度与内存全面占优。

## 2. 实测(N=1024…65536, K=3, 三音+轻噪声, best-of-3, 同 `tol=1e-7`)

| 实现 | N | 用时(s) | 提交内存增量(MB) | ω 最大误差 | 残差比 | 迭代数 | `reconstruct()` 误差 |
|---|---|---|---|---|---|---|---|
| vmdpy | 1024 | 0.003 | 0 | 1.59e-04 | 4.859e-02 | 21 | 2.60e-01 |
| **native** | 1024 | **0.001** | 0 | 1.59e-04 | 4.859e-02 | 21 | **1.73e-18** |
| vmdpy | 4096 | 0.012 | 252 | 4.78e-06 | 4.519e-02 | 22 | 1.72e-01 |
| **native** | 4096 | **0.003** | **0** | 4.78e-06 | 4.519e-02 | 22 | **6.94e-18** |
| vmdpy | 16384 | 0.126 | 1007 | 6.42e-06 | 4.445e-02 | 22 | 1.94e-01 |
| **native** | 16384 | **0.025** | **5** | 6.42e-06 | 4.445e-02 | 22 | **6.94e-18** |
| vmdpy | 65536 | 0.619 | 4032 | 2.70e-06 | 4.473e-02 | 23 | 2.46e-01 |
| **native** | 65536 | **0.069** | **28** | 2.70e-06 | 4.473e-02 | 23 | **6.94e-18** |

- **速度**: 3.0× / 4.0× / 5.0× / **9.0×** (N 越大差距越大: 旧版每轮都保留全迭代历史);
- **内存**: N=16384 时 1007 MB → 5 MB(**201×**), N=65536 时 4032 MB → 28 MB(**144×**);
- **质量完全相同**: 中心频率误差、残差比、迭代数逐项一致, 两版 ω 差 ≤4e-9;
- **`reconstruct()`**: 旧版 `Res=None`(无从重构), 且其模态和与信号差 **17–26%**
  (τ=0 时数据保真是软约束); 原生版返回真实余量 ⇒ 精确重构(≈7e-18)。

## 3. 新 `VMD.py` 的接口

```python
from Modal_Decomposition import Class, Function

Class.VMD(num_imf=3, alpha=2000, tau=0.0, epsilon=1e-7, DC=0,
          init_mod="uniform", seed=None, store_history=False).decompose(S, T)
Function.VMD(S, T, num_imf=3)          # facade
```

| 参数 | 含义 | 与旧版对应 |
|---|---|---|
| `num_imf` (默认 2) | 模态数 | 旧 `K` |
| `n` (默认 500) | 最大迭代次数 | 旧版硬编码 `Niter=500` |
| `fs` (默认 1.0) | 采样率, **只用于 `omega_hz` 报告** | 旧版无 |
| `alpha` / `tau` / `DC` | 同旧版 | 同名同义 |
| `epsilon` (默认 1e-7) | 收敛容差 | 旧 `tol` |
| `init_mod` | `uniform`/`random`/`zero`/`peak` | 旧 `init` 0/1/2(自动映射) |
| `seed` | 局部随机种子(仅 `random` 初值用) | 旧版无 |
| `store_history` | 保存逐次迭代模态谱 | 旧版无 |
| `config` | 冻结参数快照(EMD 同款覆盖入口) | 旧版无 |

**向后兼容**: `K` / `init` / `tol` 作为关键字仍可用
(`VMD(K=3)`、`VMD(init=2, tol=1e-9)` 实测通过); 未知关键字抛
`TypeError` 并提示这三个名字。

**语义变化(需注意)**:
1. `info["omega"]` 旧版是**历史矩阵** `(n_used, K)`(vmdpy 口径), 新版是**最终** `(K,)`
   —— 历史仍在 `info["omega_history"]`;
2. `Res` 旧版恒为 `None`, 新版是真实余量 `S − ΣIMFs`, 因此 `reconstruct()` 精确;
3. `info` 新增 `omega_hz` / `n_iter` / `converged` / `init_mod` / `residual_ratio` /
   `fft_backend` / `store_history` 开启时的 `u_hat_history`+`udiff_history`;
4. **位置参数顺序与旧版不同**(旧版首参是 `alpha`) ⇒ 老代码请用关键字;
   位置误用会被参数校验拦下(如 `VMD(2000, 0.0, 3)` 会在 `n` 上报错), 不会静默出错。

## 4. 与 vmdpy 的数值等价性(为什么这不是"另一个算法")

同初值/同迭代次数下逐点比对(vmdpy `init=1` ↔ `init_mod="uniform"`):

- 中心频率: 差 **1.1e-9**; 迭代数一致;
- 模态: 相对差 **1.4e-05**, 且该差异**完全来自 Nyquist 分量的后处理约定** ——
  vmdpy 用 `conj(û[N-1])` 补 Nyquist bin, 本实现置 0(与"单边谱不含 Nyquist"自洽);
  剔除 (−1)ⁿ 分量后差异降到 1e-7 量级;
- 残差比与迭代轨迹一致 ⇒ 同一个不动点。

## 5. 换版后的状态与验证

| 检查 | 结果 |
|---|---|
| `Class.VMD` / `Function.VMD` 出口 | 不变(注册键仍 `"VMD"`, 15 个方法齐全) ✓ |
| 旧关键字 `K` / `init` / `tol` | 实测可用 ✓ |
| 新关键字 `num_imf` / `fs` / `store_history` | 可用, 且与旧关键字路径结果**逐位一致** ✓ |
| `tests/test_contract.py` `test_registry.py` `test_reconstruction.py` `test_facade.py` | **126 passed**; 8 个失败全部是既有的 CEEFD bug(与本次无关) ✓ |
| `Res` 契约 | 由 `None` 变为真实数组, 符合 `test_res_field_semantics` 的非 None 分支 ✓ |
| `Base/TextDefine` 的 `Name`/`Reference` | 无需改动(`"VMD"` 条目本就对应原生算法) ✓ |

## 6. 复现

```powershell
python tests/comparison/bench_vmd_native.py --sizes 1024 4096 16384 65536 --reps 3
python tests/comparison/bench_vmd_native.py --worker '{"impl":"native","n":16384,"k":3,"reps":3}'
```

## 7. 局限与后续

1. **`vmdpy` 仍留在依赖里**(`pyproject.toml`): 本库已无任何模块引用它, 但 EEMD/CEEMDAN
   仍用 PyEMD; 若要移除 `vmdpy` 依赖需单独确认没有第三方代码依赖它 —— 未做。
2. 旧 `VMD.py` 的代码只在 git 历史里(git 可取回), 本报告保留了它的行为口径
   (`vmdpy.VMD(S, alpha, tau, K, DC, init, tol)`) 供复现。
3. `VMD_new.py` / `VMD_new_new.py` 仍在仓库中(后者是既定实验沙箱): 前者现在与 `VMD.py`
   内容近乎重复, 建议后续删除或改为指向 `VMD.py` 的说明 —— 需你确认后再动。
4. 实测仅到 **N=65536 / K=3**(本机可用内存 3–8 GB); 更大尺寸、更大 K 未测;
   `store_history=True` 的内存代价(``n_iter·K·N`` 复数)也未在本轮纳入对比。
5. 质量指标用的是"中心频率误差 + 残差比 + 重构误差", **未做**多分量捕获率(corr/valid)
   这类 EMD 报告里用的行级指标; 若要与其他方法横向比较, 建议复用
   `tests/comparison/quality.py` 的口径。
