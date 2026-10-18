# SSA（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Singular Spectrum Analysis with rank-rule / window / stride control.

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `window_size` | `None` | `int | None` |
| `groups` | `None` | `typing.Optional[typing.Sequence[typing.Sequence[int]]]` |
| `stride` | `1` | `<class 'int'>` |
| `window` | `'rect'` | `typing.Union[typing.Literal['rect', 'hann', 'hamming'], Modal_Decomposition.SSA.SSA_WINDOW_MODS]` |
| `rank_rule` | `None` | `typing.Union[typing.Literal['energy', 'svht', 'svht_clip'], Modal_Decomposition.SSA.SSA_RANK_MODS, NoneType]` |
| `energy_frac` | `0.995` | `<class 'float'>` |
| `svht_cap_frac` | `0.95` | `<class 'float'>` |
| `dtype` | `None` | `type | numpy.dtype | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 170 | 5.88e-15 | +0.9422 | — | 0.548 | 55 ms |
| 4096 | 1365 | 2.99e-14 | +1.0250 | — | 0.334 | 35179 ms |

> 该方法没有残差概念，`Res` 为 `None`；`IMFs` 之和即重构信号本身。

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.5047**（q_rec 0.9956 / q_io 0.0138），耗时 564 ms
- Pareto 选点：`{"window_size": 128, "rank_rule": "svht"}` → Q = **0.5159**，耗时 42 ms
- 相对默认：**+2.2%**
- 完整数据：`tests/comparison/results/pareto/SSA.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_SSA.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_SSA.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py SSA
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py SSA
```
