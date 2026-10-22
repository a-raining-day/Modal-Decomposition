# MEMD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Multivariate Empirical Mode Decomposition

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `d` | `None` | `int | None` |
| `k` | `None` | `int | None` |
| `max_imf` | `None` | `int | None` |
| `sd_thresh` | `0.2` | `<class 'float'>` |
| `max_iter` | `10` | `<class 'int'>` |
| `spline_kind` | `'linear'` | `typing.Union[typing.Literal['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'], Modal_Decomposition.MEMD.MEMD_SPLINE_MODS]` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 7 | 9.87e-01 | +0.6479 | 0.3645 | 0.572 | 148 ms |
| 4096 | 9 | 7.49e-01 | +0.3528 | 0.2578 | 0.709 | 532 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.6597**（q_rec 0.8250 / q_io 0.4944），耗时 312 ms
- **扫描的每一种参数组合都不可行**（`leak > 0.05`）：残差保留了大部分能量，这是算法特性而非旋钮问题。
- 完整数据：`tests/comparison/results/pareto/MEMD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_MEMD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_MEMD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py MEMD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py MEMD
```
