# SVMD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Successive Variational Mode Decomposition

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `num_modes` | `3` | `<class 'int'>` |
| `alpha` | `2000.0` | `<class 'float'>` |
| `tau` | `0.0` | `<class 'float'>` |
| `tol` | `1e-07` | `<class 'float'>` |
| `max_iter` | `500` | `<class 'int'>` |
| `backend` | `'numpy'` | `typing.Union[typing.Literal['numpy', 'numba'], Modal_Decomposition.SVMD.SVMD_BACKEND_MODS]` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 3 | 4.58e-01 | +0.9085 | 0.2289 | 0.33 | 1 ms |
| 4096 | 3 | 5.88e-01 | +0.9093 | 0.2272 | 0.31 | 2 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.5438**（q_rec 0.9967 / q_io 0.0910），耗时 1 ms
- **扫描的每一种参数组合都不可行**（`leak > 0.05`）：残差保留了大部分能量，这是算法特性而非旋钮问题。
- 完整数据：`tests/comparison/results/pareto/SVMD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_SVMD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_SVMD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py SVMD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py SVMD
```
