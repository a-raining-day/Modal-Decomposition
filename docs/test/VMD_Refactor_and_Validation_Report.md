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
| **`vmdpy`** (默认 False) | **True 时改用可选第三方 `vmdpy` 分解**(经 import cache 惰性导入) | = 旧版 `VMD` 的实现 |
| `config` | 冻结参数快照(EMD 同款) | — |

### 4.1 `vmdpy=True` 分支(可选依赖, 不强制安装)

- 经 ``cache.import_module("vmdpy", ...)`` 惰性导入 ⇒ **未安装只在使用该分支时报错**并给出
  `pip install vmdpy` 命令, 不影响其余功能; `pyproject.toml` 里已把它放进
  `optional-dependencies["vmdpy"]`(不再是必装依赖)。
- 与原生分支的差异(已在 docstring 写明): 迭代上限固定 500(`n` 不生效)、初值只支持
  `zero`/`uniform`/`random`(`peak` 仅原生)、**不支持奇数长度**、`fs`/`seed`/
  `store_history`/`engine`/`chunk_size`/`out_of_core` 不生效。
- 返回契约保持一致: `IMFs` = vmdpy 的 `u`, `Res` = `S − ΣIMFs`(真实余量) ⇒ `reconstruct()` 精确;
  `info` 带 `impl="vmdpy"`, `omega_history` 为其历史矩阵, `n_iter`/`converged`(vmdpy 不报告,
  置 None)/`fft_backend`(None, 它内部自调 `np.fft`)。
- 实测一致性(N=4096, DC 偏置+两音): 两分支 ω 差 **9.4e-10**, IMFs 相对差 **8.0e-06**
  (该差异即 §5 的 Nyquist 约定), 残差比**完全相同**(1.3784e-02)。

**兼容性(第二轮已按需收紧)**: 第一轮曾提供旧关键字 `K`/`init`/`tol` 的兼容层; 第二轮
**整体删除**(参数元表、别名表与 `**legacy` 一起去掉), 现在只认规范名 `num_imf`/`init_mod`/
`epsilon`(外加 `engine`/`chunk_size`/`out_of_core`/`store_history`/`vmdpy`), 未知名直接
`TypeError`。数字式模式选择器也不再接受: `init_mod` 只收字符串, 数字→vmdpy 的 0/1/2 转换
只在 `vmdpy=True` 分支内部做; `DC` 收紧为真正的 `bool`(`DC=1` 现在报错)。

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
| 旧关键字 `K`/`init`/`tol` | 第一轮可用; **第二轮已删除**(见 §9) |
| `DC: bool` → `omega[0] == 0.0` | ✓ (`DC=1` 第二轮起报 `ValueError`) |
| `store_history` 开/关 | shapes 正确; 关闭时 `info` 无历史键、结果与关闭前逐位一致 ✓ |
| 三种引擎结果一致 | ram 与 chunked 差 ≤1.1e-15; 受限预算下残差比逐位相同 ✓ |
| `reconstruct()` | 6.94e-18 ✓ |
| `tests/test_contract.py` `test_registry.py` `test_reconstruction.py` `test_facade.py` | **154 passed**(含第二轮新增的 `test_vmd.py`); 8 个失败全部是既有的 CEEFD bug(与本次无关) ✓ |

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

---

# 第二轮重构 (五项整改): 复用 Utils + Literal/match + `DC: bool`

第一轮把算法做对、做了引擎化; 第二轮按"**能复用工具就不要再造**"和"**不要向后兼容包袱**"
的要求做了五项整改。本轮**不改算法**: 全部改动都以"与整改前逐位一致"为验收条件。

## 9. 五项整改与落点

| # | 要求 | 落点 |
|---|---|---|
| 1 | 外存不用自建 memmap, 改用 `Utils.Chunk` | 删掉本地 `_temp_memmap`/`_cleanup_memmaps`/`_drop_memmap`/`_MEM_FILES`/`atexit` 登记; 改调 `Chunk.temp_memmap(shape, dtype)` + `Chunk.drop_memmap(mm)`; 引擎结束时**显式释放**外存工作区, 不再只靠 `atexit` 兜底 |
| 2 | 镜像改用 `Utils.Mirror` | 删掉本地 `_mirror_signal`/`_fill_mirror`; `__init__` 里 `self.mirror = get_mirror().mirror_signal`, 调用点统一 `self.mirror(f, out=..., chunk_size=...)`(整条信号语义, 与极值镜像 `mirror_extrema` 区分) |
| 3 | `_init_omega` 移入 `vmd` 类 | `_init_omega`/`_project_bytes`/`_work_dtype`/`_spans` 由模块级函数改为 `VMD` 的 `@staticmethod`(调用点为 `self.*`, 静态者用 `VMD.*`), 模块级只剩 `VMDConfig` 与 `VMD` 两个名字 |
| 4 | 去掉参数元表, 用 `Literal` + `match` | 删掉 `_VMD_LEGACY`/`_VMD_INIT_MAP`/`_INIT_TO_VMDPY`/`_INIT_MODES`/`_ENGINES`/`_INIT_ALIASES`/`_VMDPY_CACHE_KEY` 与 `**legacy`; `init_mod`/`engine` 形参标 `Literal[...]`, 运行期用 `match` 归一化并对非法值报错; vmdpy 缓存键改字面量 `"vmdpy"` |
| 5 | 模式选择不用数字 | 对外只收字符串; vmdpy 分支内部用 `match self.init_mod` 现转 0/1/2(不再有全局映射表); 常量化: `VMD_MIN_SAMPLES`/`VMD_UHAT_INFO_LIMIT`/`VMD_PEAK_INIT_LIMIT`/`VMD_CHUNK_WORK_BYTES` 移入 `Base/ConstDefine.py` 并进 `__all__` 与 `Base/__init__.py` |

**顺带修掉的一个真实缺陷**: `Chunk.temp_memmap` 在 `np.memmap` 建图失败(如参数顺序写错
传入非法 shape)时会遗弃 `mkstemp` 留下的空文件 —— 已加 `try/except` 清理并配回归测试。
(本轮调试期间正是踩到它, 在系统临时目录留下过文件, 现场已清理。)

## 10. 验收结果(全部以"整改前"为基准)

| 检查 | 结果 |
|---|---|
| `engine="ram"` 与整改前**逐位一致** | N ∈ {64, 256, 1024, 3001, 4096} × 4 组参数(zero/uniform/random/peak × DC 开关) → `max|ΔIMF| = 0`, `max|Δω| = 0` ✓ |
| 三引擎一致性 | ram vs chunked vs chunked+out_of_core: `≤2.2e-15`(仅求和顺序) ✓ |
| 外存走 `Utils.Chunk` 后的大数组行为不变 | N=4 194 304 复测: `ram` 2012 MB/8.90 s、`chunked+ooc` 1254 MB/15.12 s、`auto` 1254 MB/12.85 s, 残差比三行同为 `4.46e-02` —— 与整改前报告数字一致 ✓ |
| 外存文件用完即删 | 运行前后 `%TEMP%\md_store_*.dat` 集合不变 ✓ |
| `store_history` / `DC` / 非法参数 / 短信号 | `u_hat_history=(n_iter,K,N)`、`udiff_history` 同长; `DC=True → ω[0]=0`; `init_mod`/`engine`/`DC`/`num_imf`/`n`/`alpha`/`chunk_size` 非法值全部报错 ✓ |
| `engine="auto"` 分流 | 预算内 → `ram`; 预算受迫 → `chunked` + `out_of_core`, 且与 `ram` 差 `1.6e-15` ✓ |
| 独立参考实现 | `tests/test_vmd.py::test_ram_matches_plain_numpy_reference` 用"镜像+单边谱+顺序 ADMM"的朴素参考实现逐点比对, 差 `<1e-12` ✓ |
| 测试套件 | `tests/test_vmd.py` **29 passed**; `test_contract`+`test_registry`+`test_reconstruction`+`test_facade`+`test_vmd` **154 passed / 8 failed**, 8 个失败全是既有 CEEFD bug ✓ |

## 11. 本轮查清的一件事: 与 vmdpy 对比必须 `n >= 500`

**`vmdpy.VMD` 的签名里没有迭代数参数**(`f, alpha, tau, K, DC, init, tol`), 它内部硬编码
`Niter = 500`。所以拿 `n=200` 去比对时, 被截断的只有**本实现**, 差异是迭代预算差而非实现差:

| `n` | 本实现 n_iter | vmdpy n_iter | `max|ΔIMF|` | `max|Δω|` |
|---|---|---|---|---|
| 200 | 200(未收敛, 被截断) | 255 | 2.13e-02 | 1.43e-02 |
| 500 | 255 | 255 | **2.54e-05** | **1.44e-05** |
| 600 | 255 | 255 | 2.54e-05 | 1.44e-05 |

`n>=500` 后两者的 ω 差回到 `1.4e-05` 量级(即 §5 的 Nyquist 约定差), 迭代数**完全相同**。
另外 `vmdpy` 的 `init=2` 用**全局** `np.random.rand`, 与本实现的局部 `Generator(seed)`
不可对齐 ⇒ `init_mod="random"` **不做跨实现逐点比对**, 只验证各自同种子可复现(已验证 ✓)。
`tests/test_vmd.py` 已把"必须 `n>=500`"写进用例注释与基准参数。

## 12. 过程失误(留档, 避免重犯)

1. **批量替换打到了函数定义行**: 把 `_project_bytes(` → `self._project_bytes(` 全局替换,
   连类里 `def _project_bytes(` 一起改成了 `def self._project_bytes(` ⇒ 语法错误。改为只替换
   **调用点独有串**(带实参形态), 并把 `ast.parse` 自检放到**落盘之前**(先前先写文件再自检,
   把坏文件写进了仓库 —— 靠 `git` 备份恢复)。*教训: 改写脚本必须"先自检后落盘"。*
2. **`str.replace` 默认只替第一处**: 4 个常量重命名只改了第一次出现(恰好是新插的 import 块),
   于是 `import VMDVMD_MIN_SAMPLES` 反而把常量名改坏。改用逐处精确编辑 + 事后全量 grep 复查。
3. **删除区段划得过宽**: 删"旧关键字兼容块"时把 `__init__` 里的 `self.config = config` 与
   config/None 两个赋值分支一起删了(症状: `AttributeError: 'VMD' object has no attribute
   'num_imf'`)。现已补回, 并由 §10 的逐位一致性检查兜住。
4. **参数顺序假设错误**: 沿用旧助手的 `(dtype, shape)` 顺序调 `Chunk.temp_memmap(shape, dtype)`
   ⇒ 外存分支全崩; 已按新签名改正, 并给该函数加了失败清理(见 §9)。
5. **`python` 不是项目解释器**: 系统 `python` 是 3.8(`match` 语法不支持), 项目解释器是
   `.venv\Scripts\python.exe`(3.10.11)。本轮所有命令都改用 venv 解释器。
6. **测试脚本自身的错误**曾一度伪装成实现缺陷: `info` 历史键名(`u_hat_history`)、
   `Check_Time_and_Signal` 返回三元组、`**BASE` 与显式关键字重复传参 —— 已逐个修正;
   `tests/_cases.py` 里 VMD 的参数也从旧名 `{"K": 3}` 改为 `{"num_imf": 3, "n": 100}`。

## 13. 本轮改动的文件

- `src/Modal_Decomposition/VMD.py` — 五项整改主体(现 871 行, 模块级只剩 2 个类名)。
- `src/Modal_Decomposition/Utils/Chunk.py` — `temp_memmap` 失败清理。
- `src/Modal_Decomposition/Base/ConstDefine.py`、`Base/__init__.py` — 4 个 `VMD_*` 常量入
  `__all__` 与包导出。
- `tests/test_vmd.py` — 新增 29 个回归用例(三引擎一致、auto 分流、外存释放、逐点参考实现、
  vmdpy 等价、非法参数、`store_history`、`DC`)。
- `tests/_cases.py` — VMD 用例参数改为新 API。
- 删除: `VMD_new.py` / `VMD_new_new.py`(第一轮已删)、本轮所有一次性重构脚本与
  `.VMD.py.bak` 备份。
- `docs/VMD_Engine_Results.json` — 引擎基准复测(8 行, 4 配置 × 2 尺寸)覆盖写入。

## 14. 仍待办(已知, 未做)

1. 行级质量指标(corr/valid)仍未补; 横向比较建议用 `tests/comparison/quality.py` 口径。
2. `Check_Time_and_Signal` 只看到输入 `S.nbytes`, **不做工作集投影**, 因此"输入很小但
   `K`/`N` 组合很贵"的情形仍要靠 `VMD` 内部的 `_project_bytes` 判定。
3. `test_utils.py` 的 5 个既有失败(`Check_Time_and_Signal` 不对 `S` 应用
   `_to_default_float`/`_to_keep_dtype`)与 CEEFD 的 `range()` 浮点 bug 均与 VMD 无关,
   本轮未动。

