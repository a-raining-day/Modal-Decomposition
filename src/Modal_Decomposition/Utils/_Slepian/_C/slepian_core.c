/*
 * Slepian (DPSS) generation — C core implementation.
 *
 * 算法 (与 numpy 后端同一套数学, 互为参考实现):
 *   1) DPSS 三对角矩阵: d_i = ((n-1-2i)/2)^2 cos(2*pi*W), e_i = i(n-i)/2, W = NW/n;
 *      e 的约定与全长数组一致: e[i] 连接 (i-1, i), e[0] 不用;
 *   2) 中心对称折半: 奇偶两支各约 n/2 规模 (权重 G_j = 2, 唯自映射点取 1;
 *      N 偶时折回项落在末主对角 ±e_{N/2}); 经 u -> G^{-1/2}u 相似变换后与对称
 *      三对角同谱: B[j][j] = d_j + fold_j, B[j][j+1] = e_{j+1}*sqrt(w_j/w_{j+1});
 *   3) 特征值: Gershgorin 区间上按 Sturm 计数 (LDL^T 负主元个数) 二分定位前若干阶;
 *   4) 特征向量: 以该特征值为移位做反迭代 (Thomas 解三对角; 移位加扰动、主元设
 *      相对下限、失败自动加大扰动重试, 并用 Rayleigh 商验收), 再用商迭代精化;
 *   5) 合并两支候选 -> 按特征值降序取前 k -> 按奇偶镜像展开;
 *   6) 符号按 Percival & Walden 1993 pg379 约定 (偶阶和为正 / 奇阶首瓣为正);
 *   7) 归一化 (单位能量 / 峰值 1 + 偶长度功率修正);
 *   8) 集中比: FFT(radix-2) 自相关法, l1 = sum_j r_j*autocorr_j, r_j = 4W*sinc(2Wj)。
 *
 * 复杂度: 时间 O(k*n*log n) 量级 (含集中比 FFT), 内存 O(k*n) (含调用方输出)。
 * 不分配堆内存 (仅诊断入口例外), 无全局状态。
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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#include "slepian_dpss.h"

#define MD_PI 3.14159265358979323846

/* ------------------------------------------------------------------ */
/* 基础工具                                                            */
/* ------------------------------------------------------------------ */
static double md_sinc(double x)
{
    double p;
    if (x == 0.0) return 1.0;
    p = MD_PI * x;
    return sin(p) / p;
}

static double vec_norm(const double *v, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

/* 归一化; 返回归一化前的范数。0 表示失败 (零向量 / NaN / Inf), 此时向量不改动。 */
static double vec_normalize(double *v, int n)
{
    double nrm = vec_norm(v, n);
    int i;
    if (!(nrm > 0.0) || nrm >= DBL_MAX) return 0.0;
    for (i = 0; i < n; i++) v[i] /= nrm;
    return nrm;
}

static void vec_scale(double *v, int n, double f)
{
    int i;
    for (i = 0; i < n; i++) v[i] *= f;
}

/* ------------------------------------------------------------------ */
/* 三对角: Gershgorin 界 / Sturm 计数 / 二分定位特征值                  */
/* ------------------------------------------------------------------ */
static void gershgorin(const double *d, const double *e, int n, double *lo, double *hi)
{
    double a, b, dd;
    int i;
    a = d[0]; b = d[0];
    if (n > 1) { a -= fabs(e[1]); b += fabs(e[1]); }
    for (i = 1; i < n; i++) {
        double rad = fabs(e[i]);
        if (i + 1 < n) rad += fabs(e[i + 1]);
        dd = d[i];
        if (dd - rad < a) a = dd - rad;
        if (dd + rad > b) b = dd + rad;
    }
    *lo = a; *hi = b;
}

/* 小于 x 的特征值个数 (LDL^T 负主元计数)。 */
static int sturm_count(const double *d, const double *e, int n, double x)
{
    int count = 0, i;
    double q = d[0] - x;
    if (q < 0.0) count++;
    for (i = 1; i < n; i++) {
        if (q == 0.0) q = DBL_MIN;              /* 防除零 (测度为零的情形) */
        q = d[i] - x - (e[i] * e[i]) / q;
        if (q < 0.0) count++;
    }
    return count;
}

/* 降序前 k 个特征值: 第 j 大为升序第 (n-j) 个, 在 [lo,hi] 上二分 Sturm 计数。 */
static void top_eigenvalues(const double *d, const double *e, int n, int k,
                            double lo, double hi, double *out)
{
    int j, it;
    for (j = 0; j < k; j++) {
        int target = n - j;
        double a = lo, b = hi, m;
        for (it = 0; it < 200; it++) {
            m = 0.5 * (a + b);
            if (!(m > a && m < b)) break;
            if (sturm_count(d, e, n, m) >= target) b = m; else a = m;
        }
        out[j] = 0.5 * (a + b);
    }
}

/* 归一化向量 x 对 (d,e) 的 Rayleigh 商: x^T T x。 */
static double rayleigh(const double *d, const double *e, int n, const double *x)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += d[i] * x[i] * x[i];
    for (i = 1; i < n; i++) s += 2.0 * e[i] * x[i - 1] * x[i];
    return s;
}

/* ------------------------------------------------------------------ */
/* 反迭代求特征向量                                                    */
/* ------------------------------------------------------------------ */
/*
 * 解 (T - shift I) x = b, 三对角 Thomas 消元。
 * pfloor 为主元下限 (按矩阵尺度给出): 接近奇异时把主元夹到 ±pfloor, 避免
 * 1/tiny 上溢成 Inf —— 这是 NaN 的根源。pfloor <= 0 时仅防精确 0。
 */
static void thomas_solve(const double *d, const double *e, int n, double shift,
                         const double *b, double *x, double *cp, double pfloor)
{
    int i;
    double piv = d[0] - shift;
    if (pfloor > 0.0 && fabs(piv) < pfloor) piv = (piv >= 0.0) ? pfloor : -pfloor;
    if (piv == 0.0) piv = DBL_MIN;
    x[0] = b[0] / piv;
    if (n > 1) cp[0] = e[1] / piv;
    for (i = 1; i < n; i++) {
        double denom = d[i] - shift - e[i] * cp[i - 1];
        if (pfloor > 0.0 && fabs(denom) < pfloor) denom = (denom >= 0.0) ? pfloor : -pfloor;
        if (denom == 0.0) denom = DBL_MIN;
        x[i] = (b[i] - e[i] * x[i - 1]) / denom;
        if (i + 1 < n) cp[i] = e[i + 1] / denom;
    }
    for (i = n - 2; i >= 0; i--) x[i] -= cp[i] * x[i + 1];
}

/*
 * 反迭代求移位 lam 对应的特征向量; 返回 1 成功 / 0 失败。
 *
 * 稳健性要点 (移位极接近特征值 -> (T-lam I) 几近奇异, 最容易出 Inf/NaN):
 *   1) 起始向量首选 1/(d_i - shift) (类 LAPACK dstein); 失败则换确定性伪随机向量
 *      (与任何特征向量都不正交);
 *   2) 主元下限 pfloor = 1e-12 * scale (矩阵尺度), 限制解的量级;
 *   3) 扰动逐级放大重试 (最多 4 次), 每次迭代后用 Rayleigh 商验收, 确认落在目标
 *      特征值上而不是邻居 (特征值密集时尤重要)。
 */
static int inverse_iteration(const double *d, const double *e, int n, double lam,
                             double span, double scale,
                             double *x, double *tmp, double *cp)
{
    double pfloor;
    int attempt, i, it, ok;

    if (!(span > 0.0)) span = 1.0;
    if (!(scale > 0.0)) scale = 1.0;
    pfloor = 1e-12 * scale;

    for (attempt = 0; attempt < 4; attempt++) {
        double pert = (1e-12 * span + DBL_EPSILON * fabs(lam)) * pow(4.0, (double)attempt);
        double shift = lam + (lam >= 0.0 ? pert : -pert);

        if (attempt == 0) {
            for (i = 0; i < n; i++) {
                double dd = d[i] - shift;
                if (fabs(dd) < pfloor) dd = (dd >= 0.0) ? pfloor : -pfloor;
                x[i] = 1.0 / dd;
            }
        } else {
            for (i = 0; i < n; i++)
                x[i] = sin(0.7 + 2.399963229728653 * (double)i + 0.37 * (double)attempt);
        }
        if (vec_normalize(x, n) == 0.0) continue;

        ok = 1;
        for (it = 0; it < 4 && ok; it++) {
            thomas_solve(d, e, n, shift, x, tmp, cp, pfloor);
            if (vec_normalize(tmp, n) == 0.0) { ok = 0; break; }
            memcpy(x, tmp, sizeof(double) * (size_t)n);
        }
        if (!ok) continue;

        for (it = 0; it < 2 && ok; it++) {          /* Rayleigh 商迭代 */
            double lam_rq = rayleigh(d, e, n, x);
            double p2 = 1e-13 * span + DBL_EPSILON * fabs(lam_rq);
            thomas_solve(d, e, n, lam_rq + (lam_rq >= 0.0 ? p2 : -p2), x, tmp, cp, pfloor);
            if (vec_normalize(tmp, n) == 0.0) { ok = 0; break; }
            memcpy(x, tmp, sizeof(double) * (size_t)n);
        }
        if (!ok) continue;

        if (fabs(rayleigh(d, e, n, x) - lam) <= 1e-6 * span) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 中心对称折半                                                        */
/* ------------------------------------------------------------------ */
/*
 * parity = +1 对称支 / -1 反对称支。返回半长 h, 填 diag[h], off[h], weight[h]:
 *   diag[j] = d_j + fold_j            (N 偶时 fold 落在末主对角: ±e_{N/2})
 *   off[i]  = e_i * sqrt(w_{i-1}/w_i) (i = 1..h-1; off[0] 不使用)
 * 特征向量还原: u_j = (B 的特征向量分量) / sqrt(w_j)。
 */
static int fold(const double *d, const double *e, int n, int parity,
                double *diag, double *off, double *weight)
{
    int h, j;
    if (parity > 0) {
        h = (n + 1) / 2;
        for (j = 0; j < h; j++) { weight[j] = 2.0; diag[j] = d[j]; }
        if (n % 2 == 1) weight[h - 1] = 1.0;
        else diag[h - 1] += e[h];
    } else {
        h = n / 2;
        for (j = 0; j < h; j++) { weight[j] = 2.0; diag[j] = d[j]; }
        if (h > 0 && n % 2 == 0) diag[h - 1] -= e[h];
    }
    if (h > 0) off[0] = 0.0;
    for (j = 1; j < h; j++)
        off[j] = e[j] * sqrt(weight[j - 1] / weight[j]);
    return h;
}

/* 半长向量 -> 全长向量 (按奇偶镜像; 奇数 N 的反对称支中间点强制 0)。 */
static void expand(const double *half, int h, int n, int parity, double *out)
{
    int i;
    for (i = 0; i < h; i++) out[i] = half[i];
    i = h;
    if (i < n && n % 2 == 1 && parity < 0) { out[i] = 0.0; i++; }
    for (; i < n; i++) out[i] = parity * out[n - 1 - i];
}

/* 符号约定 (Percival & Walden 1993 pg 379)。 */
static void sign_fix(double *v, int n, int parity)
{
    int i;
    if (parity > 0) {
        double s = 0.0;
        for (i = 0; i < n; i++) s += v[i];
        if (s < 0.0) vec_scale(v, n, -1.0);
    } else {
        double thresh = (1.0 / (double)n > 1e-7) ? 1.0 / (double)n : 1e-7;
        for (i = 0; i < n; i++) {
            if (v[i] * v[i] > thresh) {
                if (v[i] < 0.0) vec_scale(v, n, -1.0);
                break;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* radix-2 FFT 与集中比 (自相关法)                                      */
/* ------------------------------------------------------------------ */
static void fft_radix2(double *re, double *im, int n, int inverse)
{
    int i, j, len, half;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            tr = im[i]; im[i] = im[j]; im[j] = tr;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * MD_PI / (double)len * (inverse ? 1.0 : -1.0);
        double wr = cos(ang), wi = sin(ang);
        half = len >> 1;
        for (i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (j = 0; j < half; j++) {
                double ur = re[i + j], ui = im[i + j];
                double xr = re[i + j + half], xi = im[i + j + half];
                double vr = xr * cr - xi * ci;
                double vi = xr * ci + xi * cr;
                re[i + j] = ur + vr; im[i + j] = ui + vi;
                re[i + j + half] = ur - vr; im[i + j + half] = ui - vi;
                if ((j & 127) == 127) {              /* 周期性重算, 抑制递推漂移 */
                    double t = (double)(j + 1) * ang;
                    cr = cos(t); ci = sin(t);
                } else {
                    double ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr; cr = ncr;
                }
            }
        }
    }
    if (inverse) {
        double inv = 1.0 / (double)n;
        for (i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
    }
}

static int next_pow2(int x)
{
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* 真集中比: l1 = sum_j r_j * autocorr_j, r_j = 4W*sinc(2W*j), r_0 = 2W。 */
static double concentration_ratio(const double *v, int n, double W,
                                  double *re, double *im, int fn)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < fn; i++) { re[i] = (i < n) ? v[i] : 0.0; im[i] = 0.0; }
    fft_radix2(re, im, fn, 0);
    for (i = 0; i < fn; i++) { double p = re[i] * re[i] + im[i] * im[i]; re[i] = p; im[i] = 0.0; }
    fft_radix2(re, im, fn, 1);
    for (i = 0; i < n; i++) {
        double r = (i == 0) ? (2.0 * W) : (4.0 * W * md_sinc(2.0 * W * (double)i));
        sum += re[i] * r;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* 工作区 (调用方预分配, 主流程内部不 malloc)                            */
/* ------------------------------------------------------------------ */
typedef struct { double *base; long cap; long used; } md_arena;

static double *arena_take(md_arena *a, long n)
{
    double *p;
    if (n <= 0) return NULL;
    if (a->used + n > a->cap) return NULL;
    p = a->base + a->used;
    a->used += n;
    return p;
}

long md_slepian_worklen(int n, int k, int sym, int want_ratios)
{
    long solve_n, m, total, fn;
    if (n < 1) return 0;
    solve_n = sym ? (long)n : (long)n + 1;
    m = (solve_n + 1) / 2;
    total = 2 * solve_n        /* d_full, e_full */
          + 6 * m              /* diag, off, weight, vec, tmp, cp (两支复用) */
          + solve_n            /* 预留 */
          + 2L * (k + 2L)      /* cand_lam (每支 k+2 个候选) */
          + 2L * (k + 2L)      /* cand_par */
          + (long)k + 2L       /* branch eigenvalues (含 2 个备用) */
          + 2L * (k + 2L) * m  /* 两支各前 k+2 个候选向量 */
          + 2L * k;            /* sel_idx, sel_lam */
    if (want_ratios) {
        fn = next_pow2(2 * (int)solve_n - 1);
        total += 2 * fn;     /* fft re/im */
    }
    return total + 8;
}

/* ------------------------------------------------------------------ */
/* 主入口                                                             */
/* ------------------------------------------------------------------ */
int md_slepian_dpss(int n, double nw, int k, int sym, double *out, double *ratios,
                    int norm, double *work, long worklen)
{
    long solve_n, m, need, fn = 0;
    double *d, *e, *diag, *off, *weight, *vec, *tmp, *cp;
    double *cand_lam, *cand_par, *branch_eig, *cand_vec, *fre = NULL, *fim = NULL;
    double *sel_idx, *sel_lam;
    double W, span, scale, lo, hi, factor = 1.0, peak = 0.0;
    int i, j, h, parity, n_cand = 0, take, already;
    md_arena ar;

    if (n < 1) return -1;
    if (!(nw > 0.0) || !(nw == nw)) return -2;
    if (k < 1) return -3;
    if (out == NULL) return -4;
    if (sym != 0) sym = 1;

    solve_n = sym ? (long)n : (long)n + 1;
    if ((long)k > solve_n) return -3;
    m = (solve_n + 1) / 2;

    need = md_slepian_worklen(n, k, sym, ratios != NULL);
    if (work == NULL || worklen < need) return -5;

    ar.base = work; ar.cap = worklen; ar.used = 0;
    d = arena_take(&ar, solve_n);               if (!d) return -6;
    e = arena_take(&ar, solve_n);               if (!e) return -6;
    diag = arena_take(&ar, m);                  if (!diag) return -6;
    off = arena_take(&ar, m);                   if (!off) return -6;
    weight = arena_take(&ar, m);                if (!weight) return -6;
    vec = arena_take(&ar, m);                   if (!vec) return -6;
    tmp = arena_take(&ar, m);                   if (!tmp) return -6;
    cp = arena_take(&ar, m);                    if (!cp) return -6;
    cand_lam = arena_take(&ar, 2L * (k + 2L));  if (!cand_lam) return -6;
    cand_par = arena_take(&ar, 2L * (k + 2L));  if (!cand_par) return -6;
    branch_eig = arena_take(&ar, (long)k + 2L); if (!branch_eig) return -6;
    cand_vec = arena_take(&ar, 2L * (k + 2L) * m); if (!cand_vec) return -6;
    sel_idx = arena_take(&ar, k);               if (!sel_idx) return -6;
    sel_lam = arena_take(&ar, k);               if (!sel_lam) return -6;
    if (ratios != NULL) {
        fn = next_pow2(2 * (int)solve_n - 1);
        fre = arena_take(&ar, fn); if (!fre) return -6;
        fim = arena_take(&ar, fn); if (!fim) return -6;
    }

    W = nw / (double)solve_n;

    /* --- 全长三对角 --- */
    {
        double cosW = cos(2.0 * MD_PI * W);
        for (i = 0; i < (int)solve_n; i++) {
            double t = ((double)solve_n - 1.0 - 2.0 * (double)i) / 2.0;
            d[i] = t * t * cosW;
        }
        e[0] = 0.0;
        for (i = 1; i < (int)solve_n; i++)
            e[i] = (double)i * ((double)solve_n - (double)i) / 2.0;
    }

    /* --- 两支: 折半 -> 特征值 -> 反迭代, 各存前若干候选 (多取 2 个备用) --- */
    for (parity = 1; parity >= -1; parity -= 2) {
        h = fold(d, e, (int)solve_n, parity, diag, off, weight);
        if (h <= 0) continue;
        gershgorin(diag, off, h, &lo, &hi);
        span = hi - lo;
        if (!(span > 0.0)) span = 1.0;
        scale = 0.0;
        for (i = 0; i < h; i++) { double a = fabs(diag[i]); if (a > scale) scale = a; }
        for (i = 1; i < h; i++) { double a = fabs(off[i]); if (a > scale) scale = a; }
        take = (k + 2 < h) ? (k + 2) : h;
        top_eigenvalues(diag, off, h, take, lo, hi, branch_eig);
        for (j = 0; j < take; j++) {
            if (!inverse_iteration(diag, off, h, branch_eig[j], span, scale,
                                   vec, tmp, cp))
                continue;                        /* 极端退化: 跳过, 由备用候选补上 */
            for (i = 0; i < h; i++) vec[i] /= sqrt(weight[i]);   /* 去权重 */
            cand_lam[n_cand] = branch_eig[j];
            cand_par[n_cand] = (double)parity;
            memcpy(cand_vec + (long)n_cand * m, vec, sizeof(double) * (size_t)h);
            n_cand++;
        }
    }
    if (n_cand < k) return -6;

    /* --- 合并取全局前 k (按特征值降序) --- */
    for (i = 0; i < k; i++) {
        int best = -1;
        double best_lam = 0.0;
        for (j = 0; j < n_cand; j++) {
            already = 0;
            {
                int t;
                for (t = 0; t < i; t++) if (sel_idx[t] == (double)j) { already = 1; break; }
            }
            if (already) continue;
            if (best < 0 || cand_lam[j] > best_lam) { best = j; best_lam = cand_lam[j]; }
        }
        if (best < 0) return -6;
        sel_idx[i] = (double)best;
        sel_lam[i] = best_lam;
    }
    (void)sel_lam;

    /* --- 展开 + 符号 + 归一化 (+ 集中比) --- */
    for (i = 0; i < k; i++) {
        int src = (int)sel_idx[i];
        double *row = out + (long)i * solve_n;
        parity = (int)cand_par[src];
        expand(cand_vec + (long)src * m,
               (parity > 0) ? (int)((solve_n + 1) / 2) : (int)(solve_n / 2),
               (int)solve_n, parity, row);
        sign_fix(row, (int)solve_n, parity);
        vec_normalize(row, (int)solve_n);
        if (ratios != NULL)
            ratios[i] = concentration_ratio(row, (int)solve_n, W, fre, fim, (int)fn);
    }

    /* --- 归一化策略 (集中比与整体缩放无关, 故在其后调整) --- */
    if (norm == 1) {                                    /* "approximate": 峰值 1 */
        peak = 0.0;
        for (i = 0; i < k; i++)
            for (j = 0; j < (int)solve_n; j++) {
                double a = fabs(out[(long)i * solve_n + j]);
                if (a > peak) peak = a;
            }
        if (peak > 0.0) factor = 1.0 / peak;
        if (solve_n % 2 == 0)
            factor *= (double)solve_n * (double)solve_n
                    / ((double)solve_n * (double)solve_n + nw);
        for (i = 0; i < (int)((long)k * solve_n); i++) out[i] *= factor;
    }

    /* sym=False 不做就地截断: 由调用方取每行前 n 个样本 (见头文件说明)。 */
    return 0;
}

const char *md_slepian_version(void)
{
    return "md_slepian 1.0 (tridiagonal + symmetry-fold + bisection/inverse-iteration)";
}

/*
 * 仅为满足 setuptools 在 Windows 上"按扩展模块链接"时的 /EXPORT:PyInit_<name> 要求:
 * 本库以 ctypes 装载共享库使用, **不作为 Python 模块导入**, 故该桩返回 NULL。
 * 不引用 Python API, 因此编译无需 Python 头文件。
 */
MD_SLEPIAN_API void *PyInit__slepian_native(void) { return 0; }

/* ------------------------------------------------------------------ */
/* 诊断入口 (仅测试/开发用): 导出各支 h / Gershgorin 区间 / 前 k 个特征值  */
/* ------------------------------------------------------------------ */
MD_SLEPIAN_API int md_slepian_debug(int n, double nw, int k,
                                    double *buf, int buflen)
{
    double *work, *d, *e, *diag, *off, *weight, *vec, *tmp, *cp, *eig;
    double lo, hi, W, cosW, span, scale;
    long solve_n, m, need;
    int i, j, h, parity, take, pos = 0;
    md_arena ar;

    if (n < 1 || k < 1 || buf == NULL) return -1;
    solve_n = (long)n;
    m = (solve_n + 1) / 2;
    need = 2 * solve_n + 8 * m + 2L * k + 32;
    work = (double *)malloc(sizeof(double) * (size_t)need);
    if (!work) return -6;
    ar.base = work; ar.cap = need; ar.used = 0;
    d = arena_take(&ar, solve_n);
    e = arena_take(&ar, solve_n);
    diag = arena_take(&ar, m);
    off = arena_take(&ar, m);
    weight = arena_take(&ar, m);
    vec = arena_take(&ar, m);
    tmp = arena_take(&ar, m);
    cp = arena_take(&ar, m);
    eig = arena_take(&ar, k + 2);
    if (!d || !e || !diag || !off || !weight || !vec || !tmp || !cp || !eig) {
        free(work); return -6;
    }
    W = nw / (double)solve_n;
    cosW = cos(2.0 * MD_PI * W);
    for (i = 0; i < (int)solve_n; i++) {
        double t = ((double)solve_n - 1.0 - 2.0 * (double)i) / 2.0;
        d[i] = t * t * cosW;
    }
    e[0] = 0.0;
    for (i = 1; i < (int)solve_n; i++)
        e[i] = (double)i * ((double)solve_n - (double)i) / 2.0;

    if (2 + 2 * (4 + k) + k > buflen) { free(work); return -5; }
    buf[pos++] = (double)solve_n;
    buf[pos++] = (double)k;
    for (parity = 1; parity >= -1; parity -= 2) {
        h = fold(d, e, (int)solve_n, parity, diag, off, weight);
        buf[pos++] = (double)parity;
        buf[pos++] = (double)h;
        if (h <= 0) {
            buf[pos++] = 0.0; buf[pos++] = 0.0;
            for (j = 0; j < k; j++) buf[pos++] = -1.0;
            buf[pos++] = 0.0;
            continue;
        }
        gershgorin(diag, off, h, &lo, &hi);
        buf[pos++] = lo;
        buf[pos++] = hi;
        span = (hi - lo > 0.0) ? (hi - lo) : 1.0;
        scale = 0.0;
        for (i = 0; i < h; i++) { double a = fabs(diag[i]); if (a > scale) scale = a; }
        for (i = 1; i < h; i++) { double a = fabs(off[i]); if (a > scale) scale = a; }
        take = (k < h) ? k : h;
        top_eigenvalues(diag, off, h, take, lo, hi, eig);
        for (j = 0; j < k; j++) buf[pos++] = (j < take) ? eig[j] : -2.0;
        {
            double nn = 0.0;
            int okr = inverse_iteration(diag, off, h, eig[0], span, scale, vec, tmp, cp);
            if (okr) for (i = 0; i < h; i++) nn += vec[i] * vec[i];
            buf[pos++] = nn;
        }
    }
    free(work);
    return pos;
}
