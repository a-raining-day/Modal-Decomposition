# `Utils.Poly` 模块报告

> 由 `.scratch/eemd_new/gen_tool_doc.py` 从**当前代码内省**生成；契约与 `tests/utils/test_Poly.py` 一一对应。

## 1. 模块定位



## 2. 公开 API

| 名字 | 签名 | 说明 |
|---|---|---|
| `fast_poly` | `(x_ref: numpy.ndarray, y_ref: numpy.ndarray, poly_order: int, x_new: numpy.ndarray, safe: Literal['safe', 'suspicious', 'unknown'] = 'unknown') -> numpy.ndarray` | 在 ``(x_ref, y_ref)`` 上拟合 ``poly_order`` 次多项式, 求 ``x_new`` 处的值。 |

## 3. 契约（由测试验证）

- 可导入且 `__all__` 与实际导出一致
- `mod` / `method` 等参数同时接受字符串与 Enum 成员
- 非法 `mod` / `backend` 抛 `ValueError`
- 返回值契约（形状 / dtype / 复数性 / 元组长度）
- 与参考实现（scipy / numpy）数值一致
- 边界输入不崩（空 / 单点 / 常量 / NaN）
- 确定性（同输入两次逐位相同）且不改写调用方输入

## 4. 复现

```
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Poly.py -q
```

`__all__` 共 1 项。

## 5. 深度测试（本轮）

> 测试：`tests/utils/test_Poly_deep.py`（**26 条**，全绿）；数据：
> `table/utils/poly/poly_vs_polyfit.csv`；图：`fig/utils/poly/poly_vs_polyfit.png`。

### 5.1 覆盖

* **精确性**：次数 0/1/2/3/5/8 的多项式在参考点上精确到机器精度；`order=0` 等于 `y.mean()`；
* **等价性**：与 `np.polyfit`+`np.polyval` 在 `order ∈ [0,8]` 上逐一比对，并**额外**与
  "手写中心化+lstsq"的独立实现比对；
* **病态**：1.7e9 量级自变量（时间戳）上的精确三次多项式；
* **欠定/退化**：`order > n_ref-1`、单点、常值 x（span=0）、空 `x_new`、远外推；
* **契约**：2-D/NaN 拒绝、非法 `safe`、dtype、不改输入、逐位确定性、O(n_new) 扩展性。

### 5.2 结论（docstring 的核心主张已验证）

| 主张 | 实测 |
|---|---|
| "归一化不改变最小二乘解" | `fast_poly` 与手写中心化 lstsq **逐位相同**（0.0 差异），3 个用例 × 9 个阶数 |
| "`np.polyfit` 在时间戳量级上会退化" | 精确三次 + `x≈1.7e9`：`np.polyfit` 最大误差 **245.8**（并发 `RankWarning`），`fast_poly` 误差 **3.4e-13** |
| "省掉 `np.poly1d` 构造开销" | 未单独量化（属微优化），但 6 个阶数的 1 Mi 点求值均 < 1 s |

### 5.3 发现

| 编号 | 现象 | 期望 |
|---|---|---|
| **F1** | `x_new` 的长度不受限，内部**先建 `(len(x_new), order+1)` 的 Vandermonde** 再乘：峰值分配 = **(order+1)× 输入**（600 MiB 上 deg3 = 3000 MiB、deg8 = **6000 MiB**，规模轴实测），deg8 耗时 4.2 s | 分块求值（`chunked_map`）可把峰值降到 ~1× |
| **F2** | `x_ref` / `y_ref` 长度不匹配时不校验，交由 `np.linalg.lstsq` 抛 `LinAlgError`（消息不含字段名） | 前置 `require_length_match` |
| **F3** | `safe="safe"` 时 list 输入在 `x_ref.mean()` 处 `AttributeError`；`safe="suspicious"` 放过 2-D 输入，直到 `vander`/`lstsq` 才炸 | 同 `Check` F7 |
| **F4** | 欠定（`order > n_ref-1`）不报错，返回**最小范数解**；在参考点外会剧烈振荡 | 文档已隐含，但建议在 docstring 里显式写"外推不可信" |

### 5.4 复现

```powershell
.\.venv\Scripts\python.exe -m pytest tests/utils/test_Poly_deep.py -q
.\.venv\Scripts\python.exe tests/utils/test_Poly_deep.py --export
.\.venv\Scripts\python.exe tests/size_axis/run_scale.py --tool poly --export
```
