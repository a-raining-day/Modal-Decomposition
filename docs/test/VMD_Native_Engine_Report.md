# VMD（原生引擎）验证与调参报告

> 本文件由 `.scratch/eemd_new/gen_method_docs.py` 从**当前代码的实测**生成；所有数字可由第 6 节的命令复现，无手写数值。

## 1. 方法概要

Variational Mode Decomposition —— 原生实现 (自研 ADMM 内核)。

## 2. 参数与默认值

| 参数 | 默认值 | 注解 |
|---|---|---|
| `num_imf` | `2` | `<class 'int'>` |
| `n` | `500` | `<class 'int'>` |
| `fs` | `1.0` | `<class 'float'>` |
| `alpha` | `2000.0` | `<class 'float'>` |
| `tau` | `0.0` | `<class 'float'>` |
| `epsilon` | `1e-07` | `<class 'float'>` |
| `DC` | `False` | `<class 'bool'>` |
| `init_mod` | `'uniform'` | `typing.Union[typing.Literal['zero', 'uniform', 'random', 'peak'], Modal_Decomposition.VMD.VMD_INIT_MODS]` |
| `seed` | `None` | `int | None` |
| `engine` | `'auto'` | `typing.Union[typing.Literal['auto', 'ram', 'chunked'], Modal_Decomposition.VMD.VMD_ENGINE_MODS]` |
| `chunk_size` | `None` | `int | None` |
| `out_of_core` | `None` | `bool | None` |
| `store_history` | `False` | `<class 'bool'>` |
| `vmdpy` | `False` | `<class 'bool'>` |
| `dtype` | `None` | `type | numpy.dtype | None` |
| `config` | `None` | `Modal_Decomposition.VMD.VMDConfig | None` |

## 3. 实测（A 用例：37 Hz + 113 Hz 双音 + 噪声）

| N | k | 重构误差 max|ΣIMF+Res−S| | 正交性指数 | leak=‖Res‖/‖S‖ | max|IMF|/max|S| | 耗时 |
|---|---|---|---|---|---|---|
| 512 | 2 | 5.29e-01 | +0.0020 | — | 0.644 | 1 ms |
| 4096 | 2 | 5.41e-01 | +0.0004 | — | 0.608 | 2 ms |

> 该方法没有残差概念，`Res` 为 `None`；`IMFs` 之和即重构信号本身。

## 4. 参数扫描（Pareto）

- 当前默认：Q = **0.9975**（q_rec 0.9960 / q_io 0.9990），耗时 1 ms
- Pareto 选点：`{"num_imf": 2, "alpha": 1000.0}` → Q = **0.9974**，耗时 1 ms
- 相对默认：**-0.0%**（< 2%，不改变默认）
- 完整数据：`tests/comparison/results/pareto/VMD.json`；口径见该目录 `README.md`。

## 5. 契约与验证

`tests/methods/test_VMD.py` 覆盖 9 项统一契约：注册与门面 `__qualname__`、返回 `DecompositionResult`、`config.to_dict()` 可 JSON 序列化、重构恒等式、门面与类等价、退化输入（常量/零/ramp）、输入不变异、确定性、非法参数拒绝。

运行：

```
.\.venv\Scripts\python.exe -m pytest tests/methods/test_VMD.py -q
```

## 6. 复现本文数字

```
.\.venv\Scripts\python.exe -u .scratch\eemd_new\gen_method_docs.py VMD
.\.venv\Scripts\python.exe -u .scratch\eemd_new\pareto_all.py VMD
```
