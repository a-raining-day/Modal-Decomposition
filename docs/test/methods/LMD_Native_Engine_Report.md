# LMD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Local Mean Decomposition

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `max_pf` | `None` | `int | None` |
| `max_iter` | `37` | `<class 'int'>` |
| `eps` | `0.05` | `<class 'float'>` |
| `eps_stable` | `1e-12` | `<class 'float'>` |
| `min_amp` | `1e-12` | `<class 'float'>` |
| `max_amp` | `1000000000000.0` | `<class 'float'>` |
| `converge_mean` | `0.001` | `<class 'float'>` |
| `smooth_window` | `5` | `<class 'int'>` |
| `envelope` | `'midpoint'` | `typing.Union[typing.Literal['midpoint', 'hilbert'], Modal_Decomposition.LMD.LMD_ENVELOPE_MODS]` |
| `hilbert_mod` | `'Scipy'` | `typing.Union[typing.Literal['scipy', 'Fourth-order Wave', 'FHT', 'SB-Hilbert', 'Kramers-Kronig', 'FIR', 'HST', 'FFT', 'Optional-order FIR'], Modal_Decomposition.Utils._Hilbert.HILBERT_MODS]` |
| `hilbert_backend` | `None` | `typing.Optional[typing.Literal['Scipy', 'scipy', 'FHT']]` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 9 | 1.66e+03 | -0.9928 | 180.7154 | 1e+04 | 56 ms |
| 4096 | 12 | 1.73e+04 | -0.8429 | 1000.1168 | 1e+04 | 136 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.6113**（q_rec 0.2227 / q_io 1.0000），耗时 68 ms
- **扫描的每一种参数组合都不可行**（`leak > 0.05`）：残差保留了大部分能量，这是算法特性而非旋钮问题。
- 完整数据：`tests/comparison/results/pareto/LMD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_LMD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_LMD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py LMD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py LMD
```

## 已知失败（本轮体检发现，**非本轮引入**）

工作区里 `src/Modal_Decomposition/LMD.py` 存在**未提交的改动**（`git diff` 显示 -198 行）。
本轮按指示只做了两处最小修改：

1. `hilbert_mod` 默认值 `"Scipy"` → **`"scipy"`**（不做大小写归一）—— `LMD(...)` 与
   `Function.LMD(...)` 的构造阶段不再抛
   `ValueError: the parameter of mod can only be [...]`；
2. `f_LMD` 不再把已废弃的 `hilbert_backend` 透传进类（类已不接受该参数，否则每次调用
   都会 `TypeError: unexpected keyword arguments`）。

**仍未修**（属在途重构，未擅自改动）：`LMD.decompose` 仍调用已被删除的模块级
`mirror_extrema`，因此：

```
src/Modal_Decomposition/LMD.py:283  ext_idx, ext_vals = mirror_extrema(...)
src/Modal_Decomposition/LMD.py:352  ext_idx, ext_vals = mirror_extrema(...)
NameError: name 'mirror_extrema' is not defined
```

受影响：`tests/methods/test_LMD.py` 6 条 + `tests/test_memmap_output.py` 4 条（LMD 参数化），
共 **10 条**；其余 **1499 条**测试全绿（`pytest tests -q`）。两种修法（都需你确认走哪条）：

* **A**：把 HEAD（提交 `d01fa2a`）里的 `mirror_extrema`（约 80 行）恢复回来；
* **B**：把这两处调用改成 `Utils.Mirror.mirror(...)`（`mirror()` 返回 `(T, A, is_max)`，
  注意它需要整条信号而非极值表）。

