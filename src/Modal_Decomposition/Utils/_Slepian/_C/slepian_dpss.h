/*
 * Slepian (DPSS) generation — C core, public API.
 *
 * 设计要点 (与 Python 端 ctypes 封装配合):
 *   - 不分配内存: 输出写进调用方预分配的 `out` (k*n 行主序 double), 工作区由调用方
 *     按 md_slepian_worklen() 预分配后传入 `work`;
 *   - 无全局状态: 纯函数, 可重入;
 *   - 算法: 三对角矩阵特征问题 + 中心对称性折半 + 二分(Sturm)定位特征值 +
 *     反迭代求特征向量; 复杂度 O(k*n) 量级, 内存 O(n) (不含调用方输出缓冲);
 *   - 集中比 (return_ratios) 用 FFT 自相关法, 与 scipy/Percival-Walden 同式。
 *
 * Python version: 3.10
 *
 * Lib and Version:
 *     (无第三方依赖; 仅 C99 标准库)
 *
 * Only accessed by: Utils/_Slepian/C.py (ctypes)
 *
 * Modify:
 *    2026.9.15
 */

#ifndef MD_SLEPIAN_DPSS_H
#define MD_SLEPIAN_DPSS_H

#ifdef _WIN32
#  define MD_SLEPIAN_API __declspec(dllexport)
#else
#  define MD_SLEPIAN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 版本字符串 (供 Python 端校验装载的确实是本库)。 */
MD_SLEPIAN_API const char *md_slepian_version(void);

/*
 * 生成前 k 阶 DPSS。
 *
 * n      : 序列长度 (>= 1)
 * nw     : 时间-带宽积 NW (> 0)
 * k      : 需要返回的阶数 (1 .. solve_n)
 * sym    : 1 = 长度 n 的对称序列; 0 = 周期性 (DFT-even: 内部按 n+1 求解,
 *          调用方取每行前 n 个样本即得结果, 本函数不做就地截断)
 * out    : 输出缓冲, 长度 k*solve_n (solve_n = sym ? n : n+1), 行主序
 *          (行 = 阶数, 集中比降序), 由调用方分配
 * ratios : 集中比输出 (长度 k), 可为 NULL 表示不需要
 * norm   : 0 = 单位能量 (L2); 1 = 峰值 1 并按 M^2/(M^2+NW) 修正 (scipy "approximate")
 * work   : 工作区, 长度 >= md_slepian_worklen(n, k, sym, ratios != NULL)
 * worklen: work 的元素个数
 *
 * 返回 0 成功; 负数表示参数错误 (见 .c 中的错误码)。
 */
MD_SLEPIAN_API int md_slepian_dpss(int n, double nw, int k, int sym,
                                   double *out, double *ratios, int norm,
                                   double *work, long worklen);

/* 所需工作区元素个数 (double 计)。 */
MD_SLEPIAN_API long md_slepian_worklen(int n, int k, int sym, int want_ratios);

#ifdef __cplusplus
}
#endif

#endif /* MD_SLEPIAN_DPSS_H */
