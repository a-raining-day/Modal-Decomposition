# EWT（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Empirical Wavelet Transform —— 本库默认实现 (自研经验小波滤波器组)。

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `num_imfs` | `8` | `<class 'int'>` |
| `band_factor` | `'auto'` | `typing.Union[float, typing.Literal['auto'], Modal_Decomposition.EWT.EWT_BAND_FACTOR_MODS]` |
| `boundary_mod` | `'maximum'` | `typing.Union[typing.Literal['maximum', 'max-min', 'scale-space', 'envelope'], Modal_Decomposition.EWT.EWT_BOUNDARY_MODS]` |
| `pre_deal` | `None` | `typing.Union[typing.Literal['no-dc', 'no-trend', 'window', 'Slepian-Optimize', 'Smooth'], typing.Sequence[typing.Literal['no-dc', 'no-trend', 'window', 'Slepian-Optimize', 'Smooth']], NoneType]` |
| `mirror` | `True` | `<class 'bool'>` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 8 | 1.51e-01 | +0.0298 | 0.0480 | 0.68 | 140 ms |
| 4096 | 8 | 5.92e-02 | +0.0034 | 0.0268 | 0.673 | 4 ms |

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.9269**（q_rec 0.9877 / q_io 0.8661），耗时 1 ms
- Pareto 选点：`{"num_imfs": 8}` → Q = **0.9829**，耗时 2 ms
- 相对默认：**+6.0%**
- 完整数据：`tests/comparison/results/pareto/EWT.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_EWT.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_EWT.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py EWT
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py EWT
```
