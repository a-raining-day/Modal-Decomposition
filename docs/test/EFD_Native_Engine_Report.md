# EFD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Empirical Fourier Decomposition

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `max_IMFs` | `-1` | `<class 'int'>` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 86 | 1.16e-01 | +0.0048 | 0.0493 | 0.639 | 393 ms |
| 4096 | 677 | 2.76e-01 | +0.0121 | 0.0781 | 0.667 | 705 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.9931**（q_rec 0.9937 / q_io 0.9925），耗时 174 ms
- Pareto 选点：`{"max_IMFs": 5}` → Q = **0.9921**，耗时 2 ms
- 相对默认：**-0.1%**（< 2%，不改变默认）
- 完整数据：`tests/comparison/results/pareto/EFD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_EFD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_EFD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py EFD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py EFD
```
