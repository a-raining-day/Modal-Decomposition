# FMD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Filtered Mode Decomposition

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `M` | `None` | `int | None` |
| `L_factor` | `10` | `<class 'int'>` |
| `K` | `10` | `<class 'int'>` |
| `max_iter` | `50` | `<class 'int'>` |
| `tol` | `1e-06` | `<class 'float'>` |
| `num_hand` | `None` | `int | None` |
| `seed` | `None` | `int | None` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 10 | 1.71e+00 | -0.5213 | 0.9219 | 0.443 | 9843 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.5956**（q_rec 0.1912 / q_io 1.0000），耗时 9761 ms
- **扫描的每一种参数组合都不可行**（`leak > 0.05`）：残差保留了大部分能量，这是算法特性而非旋钮问题。
- 完整数据：`tests/comparison/results/pareto/FMD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_FMD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_FMD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py FMD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py FMD
```
