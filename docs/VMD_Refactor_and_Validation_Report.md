# VMD 重构完成报告: 去依赖 + 引擎化 + 完整验证

- 日期: 2026-03-04
- 变更: `src/Modal_Decomposition/VMD.py` **重构为正式版**(976 行): 原生 ADMM 内核 +
  RAM/分块/外存三态引擎 + `store_history` + vmdpy 的 `DC` 语义 + 旧关键字兼容;
  **`VMD_new.py` / `VMD_new_new.py` / 旧 `bench_vmd_large.py` 已删除**;
  `pyproject.toml` 移除 `vmdpy` 依赖
- 实验: `tests/comparison/bench_vmd_native.py`(质量/速度/内存 vs vmdpy)、
  `tests/comparison/bench_vmd_engine.py`(引擎与大数组可行, 受到严格内存上限约束)
- 一句话结论: 与 vmdpy **同质量**(ω 差 ≤4e-9、残差比与迭代数逐项相同) 而
  **3–9× 更快、大尺寸提交内存低 100–200×、`reconstruct()` 精确**;
  受限内存下 `engine="auto"` 自动落盘, 内存再降 25–38%, 代价约 1.25× 时间。

---

## 1. 六项要求与结果

| # | 要求 | 结果 |
|---|---|---|
| 1 | 去除依赖 | `VMD.py` **零 vmdpy 引用**(仅文档提及作为参考实现); `pyproject.toml` 已移除 `vmdpy` |
| 2 | 增加 vmdpy 的 `DC` 选项 | 已支持并实测: `DC=1` 时模态 0 中心频率**锁定 0**(`omega[0] == 0.0`), `DC=0` 时不锁 |
| 3 | `store_history`(默认 False) | False: 无额外分配、`info` 键不变; True: `info["u_hat_history"] (n_iter,K,N)` + `udiff_history (n_iter,)`; 分块引擎**逐块写入**不额外拷贝 |
| 4 | 外存分块在大数据下不崩 | `engine="auto"` 由 `Utils.Memory` 预算自动分流; 受限预算(256MB)下实测落到 `chunked+out_of_core` 并完成, 见 §3 |
| 5 | 删除 `VMD_new` / `VMD_new_new` | 已删除, 全库无残留引用(仅 `VMD.py` 文档提到这两个名字的历史) |
| 6 | 进一步完整实验 + 报告 | 本文 + `docs/VMD_Native_vs_vmdpy_Results.json` + `docs/VMD_Engine_Results.json` |

## 2. 质量与速度(vs vmdpy, K=3, 三音+轻噪声, best-of-3, 同 `tol=1e-7`)

| 实现 | N | 用时(s) | 提交增量(MB) | ω 最大误差 | 残差比 | 迭代数 | `reconstruct()` 误差 |
|---|---|---|---|---|---|---|---|
| vmdpy | 1024 | 0.004 | 63 | 1.59e-04 | 4.859e-02 | 21 | 2.60e-01 |
| **VMD(新)** | 1024 | **0.001** | **0** | 1.59e-04 | 4.859e-02 | 21 | **1.73e-18** |
| vmdpy | 4096 | 0.012 | 252 | 4.78e-06 | 4.519e-02 | 22 | 1.72e-01 |
| **VMD(新)** | 4096 | **0.003** | **0** | 4.78e-06 | 4.519e-02 | 22 | **6.94e-18** |
| vmdpy | 16384 | 0.143 | 1009 | 6.42e-06 | 4.445e-02 | 22 | 1.94e-01 |
| **VMD(新)** | 16384 | **0.051** | **4** | 6.42e-06 | 4.445e-02 | 22 | **6.94e-18** |
| vmdpy | 65536 † | 0.619 | 4032 | 2.70e-06 | 4.473e-02 | 23 | 2.46e-01 |
| **VMD(新)** † | 65536 | **0.069** | **28** | 2.70e-06 | 4.473e-02 | 23 | **6.94e-18** |

† N=65536 一行取自上一轮同脚本实测(本机内存紧张, 本轮未重跑; 原始数据仍在 JSON 里)。

- **质量完全一致**: 两版 ω 差 ≤4e-9(排序对齐后), 残差比与迭代数逐项相同 ⇒ 同一不动点;
- **`reconstruct()`**: 旧版 `Res=None` 且模态和与信号差 **17–26%**(τ=0 软约束),
  新版返回真实余量 ⇒ 精确重构;
- **速度/内存**: N 越大优势越大(旧版每轮保留全迭代历史)。

## 3. 引擎与大数组可行性(受限内存下, 提交上限 0.9 GB, 预算 256 MB)

| config | N | 状态 | 秒 | 提交峰值(MB) | 生效引擎 | chunk | 残差比 |
|---|---|---|---|---|---|---|---|
| ram | 2 097 152 (16MB) | ok | 3.81 | **1387** | ram | 2097152 | 4.46e-02 |
| chunked | 2 097 152 | ok | 8.39 | **1045** | chunked+ooc | 466033 | 4.46e-02 |
| auto | 2 097 152 | ok | 5.71 | **1045** | chunked+ooc | 466033 | 4.46e-02 |
| auto+ooc | 2 097 152 | ok | 5.68 | 1045 | chunked+ooc | 466033 | 4.46e-02 |
| ram | 4 194 304 (32MB) | ok | 9.60 | **2012** | ram | 4194304 | 4.46e-02 |
| chunked | 4 194 304 | ok | 12.11 | **1254** | chunked+ooc | 466033 | 4.46e-02 |
| auto | 4 194 304 | ok | 11.77 | **1254** | chunked+ooc | 466033 | 4.46e-02 |
| auto+ooc | 4 194 304 | ok | 11.79 | 1254 | chunked+ooc | 466033 | 4.46e-02 |

- **`auto` 在受限预算下确实分流**: 生效引擎变成 `chunked+out_of_core`,
  提交峰值 2012 → **1254 MB(−38%)**, 16MB 时 1387 → 1045 MB(−25%);
- 代价: 时间约 **1.25×**(12.11 vs 9.60 s), 残差比**逐位一致**(4.46e-02) ⇒ 分块不改结果;
- **诚实的边界**: 分块/外存只能压"模态谱 + 镜像 + λ̂"这几项; **单边频谱与返回的
  `IMFs`/`Res` 仍是 O(N·K) 且必须常驻**(FFT 是全局变换、返回契约要求结果在内存),
  所以内存不是无限可压 —— 这条在 `docs/VMD_Large_Array_Iteration_Report.md` 有完整推导。

## 4. 接口与兼容性

```python
from Modal_Decomposition import Class, Function

Class.VMD(num_imf=3, alpha=2000, tau=0.0, epsilon=1e-7, DC=0, init_mod="uniform",
          seed=None, engine="auto", chunk_size=None, out_of_core=None,
          store_history=False).decompose(S, T)
Function.VMD(S, T, num_imf=3)
```

| 参数 | 说明 | 与旧版/参考对应 |
|---|---|---|
| `num_imf` (默认 2) | 模态数 | 旧 `K` |
| `n` (默认 500) | 最大迭代数 | vmdpy 硬编码 `Niter=500` |
| `fs` | 采样率, 仅用于 `omega_hz` 报告 | — |
| `alpha` / `tau` | 同 vmdpy | 同名同义 |
| `epsilon` (默认 1e-7) | 收敛容差 | 旧 `tol` |
| `DC` (默认 0) | 模态 0 锁定 0 频 | vmdpy `DC` |
| `init_mod` | `uniform`/`random`/`zero`/`peak` | vmdpy `init` 0/1/2(自动映射) |
| `seed` / `store_history` | 局部种子 / 保存迭代过程 | — |
| `engine` / `chunk_size` / `out_of_core` | 存储引擎(默认 `auto`) | — |
| `config` | 冻结参数快照(EMD 同款) | — |

**兼容性实测**: 旧关键字 `K` / `init` / `tol` 可用(`VMD(K=3)`、`VMD(init=2, tol=1e-9)`),
未知关键字抛 `TypeError` 并提示这三个名字; `Function.VMD(S, K=2)` 工作正常。

**语义变化(消费 `info` 时注意)**: ① `info["omega"]` 由历史矩阵改为最终 `(K,)`
(历史仍在 `omega_history`); ② `Res` 由 `None` 变为真实余量(故 `reconstruct()` 精确);
③ 新增 `omega_hz`/`n_iter`/`converged`/`residual_ratio`/`fft_backend`/`engine`/
`chunk_size`/`out_of_core`/`projected_bytes`(以及开启 `store_history` 时的两个历史键);
④ **位置参数顺序与旧版不同** — 老代码请用关键字(位置误用会被校验拦下, 不会静默出错)。

## 5. 与 vmdpy 的数值等价性(为什么不是"另一个算法")

同初值、同迭代次数逐点比对: 中心频率差 **1.1e-9**, 迭代数一致; 模态相对差 **1.4e-05**,
且该差异**全部来自 Nyquist bin 的后处理约定**(vmdpy 用 `conj(û[N-1])` 补, 本实现置 0,
与"单边谱不含 Nyquist"自洽); 剔除 (−1)ⁿ 分量后降到 1e-7 量级。

## 6. 验证清单

| 检查 | 结果 |
|---|---|
| `Class.VMD` / `Function.VMD` / 15 个注册方法 | 正常 ✓ |
| 旧关键字 `K`/`init`/`tol` | 可用 ✓ |
| `DC=1` → `omega[0] == 0.0` | ✓ |
| `store_history` 开/关 | shapes 正确; 关闭时 `info` 无历史键、结果与关闭前逐位一致 ✓ |
| 三种引擎结果一致 | ram 与 chunked 差 ≤1.1e-15; 受限预算下残差比逐位相同 ✓ |
| `reconstruct()` | 6.94e-18 ✓ |
| `tests/test_contract.py` `test_registry.py` `test_reconstruction.py` `test_facade.py` | **126 passed**; 8 个失败全部是既有的 CEEFD bug(与本次无关) ✓ |

## 7. 局限

1. **提交内存上限是采样式**, 不是硬限: 表中 `ram` 一行峰值 1387/2012 MB 超过了脚本设定
   的 0.9 GB 而未被拦下(采样间隔漏过瞬时峰值)。要硬限需 Windows Job Object / cgroup。
2. **分块压不动的部分**: 单边频谱 + 返回的 `IMFs`/`Res` 恒为 O(N·K);
   因此超大输入仍需足够内存, 或改用 float32/降 K(见 FFT 报告 §4.1 与 VMD 大数组报告 §2.4)。
3. **外存是 I/O 受限**: 每轮迭代读写 `û`(K·N·16×2 字节), 迭代数上百时不可行;
   本轮实测仅到 N=4.2M。
4. 质量指标为"中心频率误差 + 残差比 + 重构误差", **未做行级捕获率(corr/valid)**;
   横向比较建议复用 `tests/comparison/quality.py` 口径。
5. 未测更大 K(仅 K=3); `store_history=True` 的内存代价(n_iter·K·N 复数)未纳入本轮对比。

## 8. 复现

```powershell
python tests/comparison/bench_vmd_native.py --sizes 1024 4096 16384 --reps 3
python tests/comparison/bench_vmd_engine.py --sizes 2097152 4194304 --budget-mb 256 --commit-cap-gb 0.9
```
