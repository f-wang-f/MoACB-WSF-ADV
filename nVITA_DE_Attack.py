"""
nVITA_DE_Attack.py
==================
nVITA 论文中「基于差分进化 (DE) 的单目标对抗样本生成方法」的独立实现，
用于与 LBA-MOACB-WSF-FixedEncoding.py 中的 NSGA-II 多目标变长编码方法做对比实验。

------------------------------------------------------------------
复用（通过 importlib 加载同目录的 LBA-MOACB-WSF-FixedEncoding.py，
      因其文件名含连字符无法常规 import）：
  - HybridCNNBiLSTM 模型类（固定编码 FIXED_ENCODING，topo=[1,0,1,1,1,0,1,1,1,1]）
  - load_and_preprocess_data：winddata.xlsx 数据加载 / MinMax 归一化 / 训练-验证-测试划分
  - decode_hyperparams / encode_individual / is_valid_individual：编码解码与合法性校验
  - evaluate_attack：攻击前后 MAE/RMSE/MAPE/R2/RSE 评估
  - SEQUENCE_LENGTH / FEATURE_COLUMNS / DEVICE / FIXED_ENCODING / 训练超参常量

不复用（这些正是对比对象）：NSGA-II、变长编码、20 档离散编码、双目标优化。

------------------------------------------------------------------
核心算法（忠实复现 nVITA 的 DE 单目标攻击）：
  * 编码：固定扰动点数 N（= DE_CONFIG['n_points']），实数编码，个体 = [t1,f1,p1, ..., tN,fN,pN]（长度 3N）
          t_i∈[0, seq_len-1]、f_i∈[0, num_features-1]（评估时取整），p_i 为实数幅值
  * 去重：评估时若 (t,f) 重复，保留第一个、忽略后续（有效点数可能 < N）
  * 预算：训练集全局逐特征极差  bud_f = beta * global_feature_ranges[f]
          （global_feature_ranges = 训练集全部窗口逐特征 max-min，与 NSGA-II train_feature_ranges/FGSM 同口径）
          每个扰动点 p 裁剪到 [-bud_f, bud_f]；未传入全局 range 时回退到逐窗口极差
  * 优化目标：忠实复现 nVITA 论文——DE 最小化 fitness = -MSE_adv
          （MSE_adv = 攻击后预测与真实值的均方误差；进化过程中不算 RSE、不需 clean 预测）
  * 评估指标：进化结束后用与 NSGA-II 逐字一致的 RSE/L2/n_actual 评估最优个体
          （【优化与评估分离】：用 MSE_adv 搜索、用 RSE/L2 对比，同一把尺子衡量才公平）
  * 算子：DE/rand/1/bin，F=0.5、CR=0.9、pop_size/maxiter 见 DE_CONFIG（与 NSGA-II 对齐）

------------------------------------------------------------------
【关键·公平对比】最终评估的 RSE 定义与 NSGA-II 完全一致（注意事项 #1，直接复制其实现，不重新定义）：
      adv_mse   = (pred_adv   - y_true) ** 2
      clean_mse = (pred_clean - y_true) ** 2
      RSE = sqrt(adv_mse / clean_mse)  if clean_mse > 1e-10  else 10.0
  注意：这与本任务 prompt 正文示意的 (mse_adv - mse_clean)/mse_clean 不同；
  按注意事项 #1「RSE 必须和 NSGA-II 完全一致、直接复制现有逻辑、不要重新定义」，
  此处以 LBA-MOACB-WSF-FixedEncoding.py 中 _evaluate_objectives 的实际实现为准。
  （DE 优化用的 MSE_adv 与该 RSE 对固定样本单调等价：clean_mse 为样本内常量，
    故最小化 MSE_adv 即最大化 RSE，最优个体一致；收敛曲线可按此换算为 RSE 供对比。）
"""

import os
import sys
import time
import pickle
import random
import argparse
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')   # 无界面后端：保证脚本可独立、无窗口运行（须在 import pyplot 前设置）
import matplotlib.pyplot as plt

# 【Windows 控制台兼容】默认 GBK 编码无法输出 R² / δ / β 等字符（UnicodeEncodeError），
# 此处统一将控制台代码页与 stdout/stderr 切到 UTF-8，保证脚本可独立运行且中文不乱码。
if os.name == 'nt':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)   # 65001 = UTF-8 代码页
    except Exception:
        pass
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =============================================================================
# 通过 importlib 复用同目录的 LBA-MOACB-WSF-FixedEncoding.py（文件名含连字符）
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_HERE, 'LBA-MOACB-WSF-FixedEncoding.py')
_spec = importlib.util.spec_from_file_location('fixed_encoding_mod', _SRC_PATH)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)   # 仅执行其顶层定义（不触发 main，__name__ != '__main__'）

# 复用常量与工具
SEQUENCE_LENGTH = fe.SEQUENCE_LENGTH
FEATURE_COLUMNS = fe.FEATURE_COLUMNS
FIXED_ENCODING = fe.FIXED_ENCODING
DEVICE = fe.DEVICE
HybridCNNBiLSTM = fe.HybridCNNBiLSTM
load_and_preprocess_data = fe.load_and_preprocess_data
decode_hyperparams = fe.decode_hyperparams
encode_individual = fe.encode_individual
is_valid_individual = fe.is_valid_individual
evaluate_attack = fe.evaluate_attack

# 训练相关超参（与主文件完全一致，用于 checkpoint 缺失时重训）
FINAL_EPOCHS = fe.FINAL_EPOCHS
FINAL_PATIENCE = fe.FINAL_PATIENCE
GRADIENT_CLIP = fe.GRADIENT_CLIP
LAMBDA_REG = fe.LAMBDA_REG

# =============================================================================
# DE 攻击配置（与主文件 NSGA2_CONFIG 同款集中配置，作为唯一默认值来源）
# =============================================================================
# 命名说明（对齐原论文）：
#   beta: nVITA 的扰动预算系数（原论文中的 β），须与 NSGA2_CONFIG['beta'] 保持相同值以公平对比
# 【固定编码】n_points 取代 NSGA-II 的变长 n_min/n_max：每个个体固定扰动 n_points 个点（去重后可能更少）
# 【预算口径·已统一】DE 与 NSGA-II 均用训练集全局逐特征 range_f：bud_f=beta*global_feature_ranges[f]
#   （global_feature_ranges 在 main 中按训练集全部窗口逐特征 max-min 计算，与 NSGA-II train_feature_ranges/FGSM 同口径）；
#   nVITA_DE_Attack 仍保留逐窗口极差作为 global_feature_ranges=None 时的回退，beta 值三者一致
DE_CONFIG = {
    'n_points': 5,        # 固定扰动点数 N（非变长；与 NSGA2_CONFIG['n_max']=5 上界口径对齐）
    'beta': 0.01,         # nVITA perturbation budget factor (原论文 β)，与 NSGA2_CONFIG['beta'] 一致
    'maxiter': 20,        # DE 最大代数（与 NSGA-II 对比设置对齐）
    'pop_size': 10,       # DE 种群大小（与 NSGA2_CONFIG['pop_size'] 对齐）
    'de_f': 0.5,          # DE 缩放因子 F（nVITA 标准值）
    'de_cr': 0.9,         # DE 交叉概率 CR（nVITA 标准值）
    'num_eval_samples': None,  # 参加 DE 评估的样本数（None/<=0=全部，正整数则从测试集头部截取）
}

# 兼容旧引用的模块级常量（全部取自 DE_CONFIG，避免与配置不同步的"死配置"）
N_POINTS = DE_CONFIG['n_points']
POP_SIZE = DE_CONFIG['pop_size']
MAX_ITER = DE_CONFIG['maxiter']
DE_F = DE_CONFIG['de_f']
DE_CR = DE_CONFIG['de_cr']
BETA = DE_CONFIG['beta']

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【模型来源·控制变量】优先加载 output/pretrained_moacb_wsf.pt——该权重由主文件
#   LBA-MOACB-WSF-FixedEncoding.py 用同一固定编码训练并保存（--save_model），
#   保证 DE 与 NSGA-II / FGSM 在【同一基准模型】上对比（绝不逐次重训，避免 GPU 非确定性漂移）。
#   若 checkpoint 缺失，则用同一固定编码按主文件训练流程重训一次并保存。
PRETRAINED_CKPT = os.path.join(OUTPUT_DIR, 'pretrained_moacb_wsf.pt')

# 中文字体与负号显示
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# nVITA 的 DE 单目标攻击器
# =============================================================================
class nVITA_DE_Attack:
    """基于差分进化 (DE/rand/1/bin) 的单目标稀疏对抗攻击器（忠实复现 nVITA）。

    与 NSGA-II 版本的根本区别：
      - 固定 N 扰动点（N=DE_CONFIG['n_points']，非变长）
      - 实数编码（非 20 档离散）
      - 单目标优化 MSE_adv（非 (L2, -RSE) 双目标 + L0 约束）；最终评估仍用 RSE/L2/n_actual
    与 NSGA-II 保持一致的口径（非区别）：
      - 扰动预算 = beta * 训练集全局逐特征 range_f（同 NSGA-II train_feature_ranges / FGSM）
    【优化与评估分离】DE 用 MSE_adv 搜索（忠实复现 nVITA），评估用与 NSGA-II 逐字一致的
    RSE/L2/n_actual（同一把尺子），保证对比公平。
    """

    def __init__(self, model, beta=BETA, pop_size=POP_SIZE, maxiter=MAX_ITER,
                 de_f=DE_F, de_cr=DE_CR, n_points=N_POINTS, device=DEVICE,
                 global_feature_ranges=None):
        self.model = model
        self.beta = float(beta)
        self.pop_size = int(pop_size)
        self.maxiter = int(maxiter)
        self.de_f = float(de_f)
        self.de_cr = float(de_cr)
        self.n_points = int(n_points)
        self.device = device
        self.dim = self.n_points * 3
        self.seq_len = int(SEQUENCE_LENGTH)
        self.num_features = int(len(FEATURE_COLUMNS))
        self.query_count = 0     # 模型前向查询计数（clean 每样本缓存 1 次）
        # 【预算口径】传入训练集全局逐特征 range（与 NSGA-II train_feature_ranges 同口径）时，
        #   扰动预算基数由窗口极差改为全局 range；None 则回退到逐窗口极差。
        self.global_feature_ranges = np.asarray(global_feature_ranges, dtype=float).reshape(-1) \
            if global_feature_ranges is not None else None

    # ---------------- 编码边界 ----------------
    def _t_bounds(self):
        return 0.0, float(self.seq_len - 1)

    def _f_bounds(self):
        return 0.0, float(self.num_features - 1)

    # ---------------- 种群初始化 ----------------
    def _init_population(self, window_range):
        """随机初始化种群：每个个体是长度 n_points*3 的实数数组。
        t~U(0,seq_len-1)，f~U(0,num_features-1)，p~U(-bud_f, bud_f)
        （bud_f = beta*(global_feature_ranges[f] 优先，缺省回退 window_range[f])）。
        """
        t_lo, t_hi = self._t_bounds()
        f_lo, f_hi = self._f_bounds()
        pop = np.zeros((self.pop_size, self.dim), dtype=float)
        for i in range(self.pop_size):
            for k in range(self.n_points):
                t = np.random.uniform(t_lo, t_hi)
                f = np.random.uniform(f_lo, f_hi)
                f_int = int(np.clip(round(f), 0, self.num_features - 1))
                bud_base = self.global_feature_ranges[f_int] if self.global_feature_ranges is not None else window_range[f_int]
                bud = self.beta * bud_base
                p = np.random.uniform(-bud, bud) if bud > 0 else 0.0
                pop[i, 3 * k] = t
                pop[i, 3 * k + 1] = f
                pop[i, 3 * k + 2] = p
        return pop

    # ---------------- 解码：个体 -> 去重后的扰动点列表 ----------------
    def _decode_points(self, individual, window_range):
        """解析个体，取整 t/f，去重（保留第一个），并把 p 裁剪到该特征的预算 [-bud_f, bud_f]。
        预算基数 bud_f = beta * (global_feature_ranges[f] 优先，缺省回退 window_range[f])。"""
        points = []
        seen = set()
        for k in range(self.n_points):
            t = int(np.clip(round(individual[3 * k]), 0, self.seq_len - 1))
            f = int(np.clip(round(individual[3 * k + 1]), 0, self.num_features - 1))
            p = float(individual[3 * k + 2])
            if (t, f) in seen:
                continue          # 去重：忽略后续重复位置
            seen.add((t, f))
            bud_base = self.global_feature_ranges[f] if self.global_feature_ranges is not None else window_range[f]
            bud = self.beta * bud_base
            p = float(np.clip(p, -bud, bud))
            points.append((t, f, p))
        return points

    # ---------------- 生成对抗样本 ----------------
    @staticmethod
    def _make_adv(x_window, points):
        x_adv = x_window.copy()
        for (t, f, p) in points:
            x_adv[t, f] += p
        return x_adv

    # ---------------- DE 优化用评估：只用 MSE_adv（忠实复现 nVITA 论文，不算 RSE） ----------------
    def _evaluate_for_de(self, individual, x_window, y_true, window_range):
        """DE 进化过程中的适应度评估：只返回 fitness = -MSE_adv，不计算 RSE、不需 clean 预测。

        nVITA 论文非目标攻击目标函数：FF = (1/t) * Σ(y_true - y_pred_adv)^2；
        本模型为单步预测（t=1），故 MSE_adv = (pred_adv - y_true)^2。
        DE 为最小化算法，取 fitness = -MSE_adv（最小化 -MSE_adv = 最大化攻击后误差）。
        【优化与评估分离】此处只用 MSE_adv 引导搜索；RSE/L2 仅在进化结束后由 _evaluate_final 计算。
        """
        points = self._decode_points(individual, window_range)
        x_adv = self._make_adv(x_window, points)
        with torch.no_grad():
            pred_adv = self.model(torch.FloatTensor(x_adv[None]).to(self.device)).item()
        self.query_count += 1
        mse_adv = (pred_adv - y_true) ** 2      # 单步预测：.mean() 即该标量本身
        return -mse_adv                          # fitness（DE 最小化）

    # ---------------- 最终评估：用与 NSGA-II 逐字一致的指标（RSE / L2 / n_actual） ----------------
    def _evaluate_final(self, individual, x_window, y_true, window_range, pred_clean=None):
        """对最终最优个体做评估，返回 (rse, l2, n_actual, mse_clean, mse_adv, x_adv, points)。

        评估逻辑与 NSGA-II 评估个体完全相同（仅个体编码格式不同）：
          - 解析/去重（保留第一个）+ 按预算裁剪 p（beta*全局 range_f）+ 生成 x_adv（同 _decode_points/_make_adv）；
            注：NSGA-II 的 _apply_perturbation 不对 x_adv 做 [0,1] 裁剪，故此处也不裁剪（逐字一致）。
          - RSE 定义与 NSGA-II _evaluate_objectives（L695）逐字一致：
                adv_mse=(pred_adv-y)^2, clean_mse=(pred_clean-y)^2,
                rse = sqrt(adv_mse/clean_mse) if clean_mse>1e-10 else 10.0
            （注意：本任务 prompt 正文示意 (mse_adv-mse_clean)/mse_clean，但按“必须与 NSGA-II 逐字一致、
              直接复制、不要重新定义”的强制要求，此处以 NSGA-II 实际实现为准。）
          - L2 = sqrt(Σ p^2)（与 NSGA-II L690 一致）；n_actual = 去重后有效点数。
        pred_clean 可传入已缓存值以避免重复查询；缺省时自行前向计算（保证可独立调用验证）。
        """
        points = self._decode_points(individual, window_range)
        x_adv = self._make_adv(x_window, points)
        with torch.no_grad():
            if pred_clean is None:
                pred_clean = self.model(torch.FloatTensor(x_window[None]).to(self.device)).item()
                self.query_count += 1
            pred_adv = self.model(torch.FloatTensor(x_adv[None]).to(self.device)).item()
        self.query_count += 1

        mse_clean = (pred_clean - y_true) ** 2
        mse_adv = (pred_adv - y_true) ** 2
        # 【与 NSGA-II _evaluate_objectives 完全一致】
        rse = float(np.sqrt(mse_adv / mse_clean)) if mse_clean > 1e-10 else 10.0
        l2 = float(np.sqrt(sum(p ** 2 for (_, _, p) in points))) if points else 0.0
        return float(rse), float(l2), len(points), float(mse_clean), float(mse_adv), x_adv, points

    # ---------------- DE/rand/1 变异 ----------------
    def _mutation(self, population, i):
        idxs = [j for j in range(self.pop_size) if j != i]
        r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
        return population[r1] + self.de_f * (population[r2] - population[r3])

    # ---------------- 二项式交叉 (bin) ----------------
    def _crossover(self, parent, mutant):
        trial = parent.copy()
        j_rand = int(np.random.randint(0, self.dim))   # 保证至少一维来自 mutant
        cross_mask = np.random.random(self.dim) < self.de_cr
        cross_mask[j_rand] = True
        trial[cross_mask] = mutant[cross_mask]
        return trial

    # ---------------- 边界裁剪 ----------------
    def _clip_bounds(self, individual):
        """裁剪 t/f 到合法范围；p 不在此裁剪（评估时按各特征预算 beta*全局 range_f 裁剪）。"""
        ind = individual.copy()
        t_lo, t_hi = self._t_bounds()
        f_lo, f_hi = self._f_bounds()
        ind[0::3] = np.clip(ind[0::3], t_lo, t_hi)
        ind[1::3] = np.clip(ind[1::3], f_lo, f_hi)
        return ind

    # ---------------- 单样本 DE 攻击 ----------------
    def attack_single_sample(self, x_window, y_true, seed=None):
        """对单个时间窗口执行 DE 攻击（优化 MSE_adv，评估用 RSE/L2）。
        x_window: np.ndarray shape=(seq_len, num_features)（归一化空间）
        y_true:   float（归一化空间的下一时刻真实值）

        【优化与评估分离】
          - DE 主循环的 fitness = -MSE_adv（_evaluate_for_de，忠实复现 nVITA）；
          - 每代记录最优 MSE_adv（convergence_mse），并按 RSE=sqrt(mse_adv/mse_clean) 换算出
            每代最优 RSE（convergence_rse，供与 NSGA-II 收敛曲线对比；mse_clean 为样本内常量，
            故最小化 MSE_adv 与最大化 RSE 单调等价，最优个体一致，换算无需额外查询）；
          - 进化结束后用 _evaluate_final 对最优个体做最终评估，得到 RSE/L2/n_actual（与 NSGA-II 一致）。
        返回 dict：最优个体、RSE、L2、n_actual、mse_clean、mse_adv、对抗样本、扰动点、两条收敛曲线。
        """
        if seed is not None:
            np.random.seed(seed)

        x_window = np.asarray(x_window, dtype=float)
        window_range = x_window.max(axis=0) - x_window.min(axis=0)   # (num_features,) 逐窗口极差（仅作 global_feature_ranges=None 时的回退预算基数）
        window_range = np.asarray(window_range, dtype=float)

        # clean 预测缓存（每样本仅 1 次查询；eval 模式确定性）——仅用于把 MSE_adv 收敛曲线换算成 RSE 及最终评估
        with torch.no_grad():
            pred_clean = self.model(torch.FloatTensor(x_window[None]).to(self.device)).item()
        self.query_count += 1
        mse_clean = (pred_clean - y_true) ** 2

        def _to_rse(mse_adv):
            """按 NSGA-II 定义把 MSE_adv 换算成 RSE（mse_clean 为样本内常量）。"""
            return float(np.sqrt(mse_adv / mse_clean)) if mse_clean > 1e-10 else 10.0

        # 初始化种群并评估（fitness = -MSE_adv）
        pop = self._init_population(window_range)
        fits = np.zeros(self.pop_size, dtype=float)
        for i in range(self.pop_size):
            fits[i] = self._evaluate_for_de(pop[i], x_window, y_true, window_range)

        best_i = int(np.argmin(fits))
        best_mse = -float(fits[best_i])                    # 初始种群最优 MSE_adv
        convergence_mse = [best_mse]                        # index 0 = 初始种群
        convergence_rse = [_to_rse(best_mse)]

        # DE 主循环（只优化 MSE_adv，不计算 RSE）
        for gen in range(self.maxiter):
            for i in range(self.pop_size):
                mutant = self._mutation(pop, i)
                trial = self._crossover(pop[i], mutant)
                trial = self._clip_bounds(trial)
                fit_t = self._evaluate_for_de(trial, x_window, y_true, window_range)
                if fit_t <= fits[i]:      # 贪心选择（最小化 -MSE_adv = 最大化 MSE_adv）
                    pop[i] = trial
                    fits[i] = fit_t
            best_i = int(np.argmin(fits))
            best_mse = -float(fits[best_i])
            convergence_mse.append(best_mse)
            convergence_rse.append(_to_rse(best_mse))

        # 最终评估：用与 NSGA-II 一致的指标（RSE/L2/n_actual）评估最优个体
        best_ind = pop[best_i].copy()
        rse, l2, n_actual, mse_clean_f, mse_adv_f, x_adv, points = \
            self._evaluate_final(best_ind, x_window, y_true, window_range, pred_clean=pred_clean)

        return {
            'best_individual': best_ind,
            'rse': rse,
            'l2': l2,
            'n_actual': n_actual,
            'mse_clean': mse_clean_f,
            'mse_adv': mse_adv_f,
            'x_adv': x_adv,
            'points': points,
            'convergence_mse': convergence_mse,   # 每代最优 MSE_adv（DE 优化目标），长度 = maxiter + 1
            'convergence_rse': convergence_rse,   # 每代最优 RSE（换算，供对比画图），长度 = maxiter + 1
        }

    # ---------------- 批量攻击 ----------------
    def attack_batch(self, X_test, Y_test, sample_indices=None, print_info=True):
        """对一批测试样本执行 DE 攻击。
        X_test: torch.Tensor (N, seq_len, num_features)；Y_test: torch.Tensor (N, 1)
        返回 (X_adv_total, metrics, per_sample_records)。
        """
        self.model.to(self.device)
        self.model.eval()

        n_total = X_test.shape[0]
        idxs = list(range(n_total)) if sample_indices is None else list(sample_indices)

        X_adv_list, all_rse, all_l2, all_nact = [], [], [], []
        all_mse_clean, all_mse_adv = [], []
        all_conv_rse, all_conv_mse, all_query, all_wall = [], [], [], []
        per_sample_records = []

        for cnt, i in enumerate(idxs):
            x_window = X_test[i].detach().cpu().numpy()                       # (seq_len, F)
            y_true = float(Y_test[i].detach().cpu().numpy().reshape(-1)[0])   # 标量

            _t0 = time.perf_counter()
            q0 = self.query_count
            res = self.attack_single_sample(x_window, y_true, seed=int(i))
            all_wall.append(time.perf_counter() - _t0)
            all_query.append(self.query_count - q0)

            X_adv_list.append(res['x_adv'])
            all_rse.append(res['rse'])
            all_l2.append(res['l2'])
            all_nact.append(res['n_actual'])
            all_mse_clean.append(res['mse_clean'])
            all_mse_adv.append(res['mse_adv'])
            all_conv_rse.append(res['convergence_rse'])
            all_conv_mse.append(res['convergence_mse'])
            # 字段名与 NSGA-II 结果格式兼容，额外携带 mse_clean/mse_adv 作为辅助（DE 优化目标）
            per_sample_records.append({
                'sample_idx': int(i),
                'rse': res['rse'],                       # 与 NSGA-II 一致的 RSE
                'l2': res['l2'],                         # 与 NSGA-II 一致的 L2
                'n_actual': res['n_actual'],             # 实际扰动点数
                'mse_clean': res['mse_clean'],           # 干净样本 MSE（辅助）
                'mse_adv': res['mse_adv'],               # 攻击后 MSE（DE 优化目标，辅助）
                'convergence_curve': res['convergence_mse'],   # 每代最优 MSE_adv（改动4）
                'convergence_rse': res['convergence_rse'],     # 每代最优 RSE（换算，供画图）
                'points': res['points'],
                'best_individual': res['best_individual'],
            })
            if print_info and (cnt + 1) % 20 == 0:
                print(f"  DE progress: {cnt + 1}/{len(idxs)}")

        X_adv_total = torch.FloatTensor(np.stack(X_adv_list, axis=0)).to(self.device)

        rse_arr = np.asarray(all_rse, dtype=float)
        # 收敛曲线平均（同一样本内长度一致 = maxiter+1，直接按列平均）
        conv_rse_mat = np.asarray(all_conv_rse, dtype=float) if all_conv_rse else np.empty((0, 0))
        mean_conv_rse = conv_rse_mat.mean(axis=0) if conv_rse_mat.size > 0 else np.empty(0)
        conv_mse_mat = np.asarray(all_conv_mse, dtype=float) if all_conv_mse else np.empty((0, 0))
        mean_conv_mse = conv_mse_mat.mean(axis=0) if conv_mse_mat.size > 0 else np.empty(0)

        metrics = {
            'mean_sel_rse': float(rse_arr.mean()) if rse_arr.size else 0.0,
            'median_sel_rse': float(np.median(rse_arr)) if rse_arr.size else 0.0,
            'std_sel_rse': float(rse_arr.std()) if rse_arr.size else 0.0,
            'mean_sel_l2_norm': float(np.mean(all_l2)) if all_l2 else 0.0,
            'mean_sel_n_actual': float(np.mean(all_nact)) if all_nact else 0.0,
            'mean_mse_clean': float(np.mean(all_mse_clean)) if all_mse_clean else 0.0,
            'mean_mse_adv': float(np.mean(all_mse_adv)) if all_mse_adv else 0.0,
            'sel_rses': all_rse,
            'sel_l2_norms': all_l2,
            'sel_n_actuals': all_nact,
            'sel_mse_clean': all_mse_clean,
            'sel_mse_adv': all_mse_adv,
            'total_samples': len(idxs),
            'convergence_curves': all_conv_rse,        # 每样本每代最优 RSE（与 NSGA-II 收敛曲线同口径）
            'mean_convergence_rse': mean_conv_rse,
            'convergence_curves_mse': all_conv_mse,    # 每样本每代最优 MSE_adv（DE 优化目标）
            'mean_convergence_mse': mean_conv_mse,
            'query_counts': all_query,
            'mean_query_count': float(np.mean(all_query)) if all_query else 0.0,
            'wall_times': all_wall,
            'mean_wall_time': float(np.mean(all_wall)) if all_wall else 0.0,
        }
        return X_adv_total, metrics, per_sample_records


# =============================================================================
# 模型与数据准备（复用主文件训练/加载逻辑；对比实验须固定同一基准 checkpoint）
# =============================================================================
def set_seed(seed):
    """统一设定随机种子（与主文件 main() 顺序一致）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fixed_encoding_model(data_result):
    """使用固定编码 FIXED_ENCODING 构建并训练模型：优化器/损失/正则/梯度裁剪/早停
    均与主文件 main() 逐行一致。早停后使用【最后一个 epoch】的模型（不回滚 best-val 权重），
    以保证测试集指标可复现。调用前须已执行 set_seed(seed)，且构建模型前不得消耗随机数。
    """
    train_dataset = data_result['train_dataset']
    val_dataset = data_result['val_dataset']
    num_features = data_result['num_features']

    topo = FIXED_ENCODING['topo']
    cnn_params = FIXED_ENCODING['cnn_params']
    lstm_params = FIXED_ENCODING['lstm_params']
    setting = FIXED_ENCODING['setting']
    best_individual = encode_individual(topo, cnn_params, lstm_params, setting)
    if not is_valid_individual(best_individual):
        raise ValueError("固定编码的拓扑结构不合法: 存在没有输入连接的模块")

    batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)
    print('\n========== 使用固定编码构建并训练模型 ==========')
    print(f'  topo:    {topo}')
    print(f'  CNN:     {cnn_params}')
    print(f'  BiLSTM:  {lstm_params}')
    print(f'  Setting: {setting}')
    print(f'  batch_size={batch_size}, lr={learn_rate:.8f}, optimizer={opt_type}, regularizer={reg_type}')

    print('\n训练最终模型...')
    model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    criterion = nn.MSELoss()

    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    elif opt_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(FINAL_EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if reg_type is not None:
                l1_penalty = torch.tensor(0., device=DEVICE)
                l2_penalty = torch.tensor(0., device=DEVICE)
                for name, param in model.named_parameters():
                    if 'bias' in name or 'bn' in name:
                        continue
                    if reg_type in ['L1', 'L1L2']:
                        l1_penalty += torch.norm(param, 1)
                    if reg_type in ['L2', 'L1L2']:
                        l2_penalty += torch.norm(param, 2)
                loss += LAMBDA_REG * (l1_penalty + l2_penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
                val_samples += inputs.size(0)
        if val_samples > 0:
            val_loss /= val_samples

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch + 1}/{FINAL_EPOCHS}, 训练损失: {train_loss / len(train_loader):.6f}, '
                  f'验证损失: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= FINAL_PATIENCE:
            print(f'早停于 epoch {epoch + 1}')
            break

    model.eval()
    return model


def _ckpt_encoding(ck):
    """从 checkpoint 提取其自带的编码字段 (topo, cnn_params, lstm_params, setting)；
    若缺失任一字段则返回 None（表示无法核对编码）。"""
    if isinstance(ck, dict) and all(k in ck for k in ('topo', 'cnn_params', 'lstm_params', 'setting')):
        return ck['topo'], ck['cnn_params'], ck['lstm_params'], ck['setting']
    return None


def build_or_load_model(data_result, load_path=None, seed=42):
    """获取基准模型：优先加载 checkpoint（默认 output/pretrained_moacb_wsf.pt），
    缺失/失败/编码不匹配则用固定编码重训一次并保存。与 baseline_gradient_attacks.py 的
    load_or_train_model 同口径，确保 DE 与 NSGA-II / FGSM 在同一模型上对比。

    【编码一致性保证】攻击模型编码必须严格等于主文件 LBA-MOACB-WSF-FixedEncoding.py 的
    FIXED_ENCODING（topo=[1,0,1,1,1,0,1,1,1,1] 等，已通过 fe.FIXED_ENCODING 复用）。
    加载 checkpoint 时会核对其自带编码：一致才加载；不一致（或无编码字段）则视为无效权重，
    用 FIXED_ENCODING 重训并覆盖，杜绝静默加载到错误架构的模型。
    """
    num_features = data_result['num_features']
    topo = FIXED_ENCODING['topo']
    cnn_params = FIXED_ENCODING['cnn_params']
    lstm_params = FIXED_ENCODING['lstm_params']
    setting = FIXED_ENCODING['setting']
    target_enc = [topo, cnn_params, lstm_params, setting]

    print('\n>>> 攻击模型目标编码（= 主文件 FIXED_ENCODING，权威口径）:')
    print(f'    topo:    {topo}')
    print(f'    CNN:     {cnn_params}')
    print(f'    BiLSTM:  {lstm_params}')
    print(f'    Setting: {setting}')

    ckpt_path = load_path if load_path else PRETRAINED_CKPT

    # ---------- 1) 优先尝试加载已有权重（须编码核对通过） ----------
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=DEVICE)
            sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
            ck_enc = _ckpt_encoding(ck)
            if ck_enc is None:
                print(f"\n>>> checkpoint 未携带编码字段，无法核对，将用 FIXED_ENCODING 重训: {ckpt_path}")
            elif [ck_enc[0], ck_enc[1], ck_enc[2], ck_enc[3]] != target_enc:
                print(f"\n>>> 警告：checkpoint 编码与 FIXED_ENCODING 不一致，视为无效权重，将重训！")
                print(f'    checkpoint topo:    {ck_enc[0]}')
                print(f'    FIXED_ENCODING topo:{topo}')
            else:
                model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(DEVICE)
                model.load_state_dict(sd)
                model.eval()
                print(f"\n>>> 已加载预训练权重: {ckpt_path}（编码核对通过，跳过训练）")
                return model
        except Exception as e:
            print(f"\n>>> 预训练权重加载失败（{e}），将使用固定编码重新训练。")
    else:
        print(f"\n>>> 未找到 checkpoint（{ckpt_path}）。")

    # ---------- 2) 重新训练：先设定种子（与主文件顺序一致），再构建 + 训练 ----------
    print(">>> 使用固定编码 FIXED_ENCODING 重新训练 ...")
    set_seed(seed)
    model = train_fixed_encoding_model(data_result)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = PRETRAINED_CKPT   # 重训结果统一落到权威路径，供后续脚本复用
    torch.save({
        'model_state_dict': model.state_dict(),
        'topo': topo, 'cnn_params': cnn_params, 'lstm_params': lstm_params, 'setting': setting,
    }, save_path)
    print(f"\n>>> 训练完成，权重（含 FIXED_ENCODING 编码字段）已保存至: {save_path}")
    return model


def extract_test_data(test_dataset):
    """一次性取出整个测试集为张量（与主文件 main() 攻击数据池同口径）。
    返回 X_test:(N, seq_len, num_features)、Y_test:(N, 1)，均在 DEVICE 上。
    """
    loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, Y_test = next(iter(loader))
    return X_test.to(DEVICE), Y_test.to(DEVICE)


# =============================================================================
# NSGA-II 结果加载与收敛曲线提取（用于 DE vs NSGA-II 对比）
# =============================================================================
def load_nsga2_results(path):
    """加载 NSGA-II 结果 pickle（output/run{n}_nsga2_attack_results.pkl）。
    返回 dict 或 None（文件不存在时）。结构含 metrics / generation_pareto / config 等。
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f">>> NSGA-II 结果加载失败（{e}），跳过对比。")
        return None


def nsga2_best_rse_curve(generation_pareto):
    """从 NSGA-II 的逐代 Pareto 前沿提取【每代最优 RSE 的跨样本平均收敛曲线】。

    generation_pareto: list，每条 = {'sample_idx', 'generation_pareto': [每代 front (m,3)=[L2,-RSE,n_actual]], ...}
    对每个样本、每一代：best_rse = -min(front[:,1])（front[:,1] = -RSE，故 -min = max RSE）；
    再跨样本按代数 nanmean 平均。返回一维 np.ndarray（长度 = 最大代数）。
    """
    if not generation_pareto:
        return np.empty(0)
    per_sample = []
    for rec in generation_pareto:
        gp = rec.get('generation_pareto') or []
        curve = []
        for front in gp:
            front = np.asarray(front)
            if front.ndim != 2 or front.shape[0] == 0 or front.shape[1] < 2:
                curve.append(np.nan)
                continue
            curve.append(float(-np.min(front[:, 1])))   # -min(-RSE) = max RSE
        if curve:
            per_sample.append(curve)
    if not per_sample:
        return np.empty(0)
    max_len = max(len(c) for c in per_sample)
    mat = np.full((len(per_sample), max_len), np.nan)
    for i, c in enumerate(per_sample):
        mat[i, :len(c)] = c
    return np.nanmean(mat, axis=0)


def nsga2_global_front_xy(nsga2_res):
    """从 NSGA-II metrics 提取全局最终 Pareto 前沿的 (L2, RSE) 散点，用于散点对比图。
    记录格式为 [L2, -RSE, n_actual]，故 RSE = -col1。返回 (L2_array, RSE_array) 或 (None, None)。
    """
    if not nsga2_res:
        return None, None
    metrics = nsga2_res.get('metrics', {})
    front = metrics.get('global_final_pareto_obj')
    if not isinstance(front, np.ndarray) or front.shape[0] == 0 or front.shape[1] < 2:
        return None, None
    l2 = front[:, 0].astype(float)
    rse = (-front[:, 1]).astype(float)
    return l2, rse



# =============================================================================
# 对比可视化与汇总
# =============================================================================
def plot_convergence_comparison(de_mean_conv, nsga2_curve, out_path):
    """收敛曲线对比图：平均 RSE vs 代数（DE 单目标 vs NSGA-II 每代前沿最优 RSE）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    de_mean_conv = np.asarray(de_mean_conv, dtype=float)
    if de_mean_conv.size > 0:
        ax.plot(np.arange(de_mean_conv.size), de_mean_conv,
                color='#d62728', lw=2.0, marker='o', ms=3, label=f'nVITA-DE (单目标, N={N_POINTS})')
    has_nsga2 = nsga2_curve is not None and np.asarray(nsga2_curve).size > 0
    if has_nsga2:
        nsga2_curve = np.asarray(nsga2_curve, dtype=float)
        ax.plot(np.arange(nsga2_curve.size), nsga2_curve,
                color='#1f77b4', lw=2.0, marker='s', ms=3, ls='--', label='NSGA-II (双目标, 变长)')
    ax.set_xlabel('代数 (Generation)')
    ax.set_ylabel('平均最优 RSE')
    ax.set_title('DE vs NSGA-II 收敛曲线对比（平均 RSE vs 代数）')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f">>> 收敛曲线对比图已保存: {out_path}")
    return out_path


def plot_scatter_comparison(de_metrics, nsga2_res, out_path):
    """散点对比图：L2 vs RSE（DE 每样本最终解散点 vs NSGA-II 全局 Pareto 前沿）。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    de_l2 = np.asarray(de_metrics.get('sel_l2_norms', []), dtype=float)
    de_rse = np.asarray(de_metrics.get('sel_rses', []), dtype=float)
    if de_l2.size > 0:
        ax.scatter(de_l2, de_rse, s=28, c='#d62728', alpha=0.7,
                   edgecolors='none', label='nVITA-DE 最终解 (每样本)')

    n_l2, n_rse = nsga2_global_front_xy(nsga2_res)
    if n_l2 is not None:
        ax.scatter(n_l2, n_rse, s=34, c='#1f77b4', alpha=0.8, marker='^',
                   edgecolors='none', label='NSGA-II 全局 Pareto 前沿')

    ax.set_xlabel('扰动 L2 范数')
    ax.set_ylabel('RSE (攻击强度)')
    ax.set_title('DE vs NSGA-II：L2 vs RSE 散点对比')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f">>> 散点对比图已保存: {out_path}")
    return out_path


def build_comparison_table(de_metrics, nsga2_res):
    """构建 DE vs NSGA-II 对比表格（dict of dict），返回 (rows_dict, printable_str)。
    指标口径与主方法 LBA-MOACB-WSF 一致：批量 RSE=RMSE_adv/RMSE_clean，逐样本 RSE=sqrt(adv_mse/clean_mse)。"""
    n_metrics = nsga2_res.get('metrics', {}) if nsga2_res else {}
    n_attack = nsga2_res.get('attack_results', {}) if nsga2_res else {}

    de_rse = np.asarray(de_metrics.get('sel_rses', []), dtype=float)
    n_rse = np.asarray(n_metrics.get('sel_rses', []), dtype=float)

    def _mean(a):
        return float(np.mean(a)) if a.size else float('nan')

    # 批量 RSE（= 攻击后 RMSE / 干净 RMSE）：DE 从 de_metrics['batch_rse']；NSGA-II 从其 attack_results 重算
    de_batch_rse = float(de_metrics.get('batch_rse', float('nan')))
    if n_attack and n_attack.get('rmse_clean', 0) > 0:
        n_batch_rse = float(n_attack['rmse_adv'] / n_attack['rmse_clean'])
    else:
        n_batch_rse = float('nan')

    rows = {
        'RSE (批量=RMSE比)': (de_batch_rse, n_batch_rse),
        'RSE 均值(逐样本)': (_mean(de_rse), _mean(n_rse)),
        'RSE 中位数': (float(np.median(de_rse)) if de_rse.size else float('nan'),
                     float(np.median(n_rse)) if n_rse.size else float('nan')),
        'RSE 标准差': (float(de_rse.std()) if de_rse.size else float('nan'),
                     float(n_rse.std()) if n_rse.size else float('nan')),
        'L2 均值': (float(de_metrics.get('mean_sel_l2_norm', float('nan'))),
                   float(n_metrics.get('mean_sel_l2_norm', float('nan')))),
        'n_actual 均值': (float(de_metrics.get('mean_sel_n_actual', float('nan'))),
                        float(n_metrics.get('mean_sel_n_actual', float('nan')))),
        '平均查询次数': (float(de_metrics.get('mean_query_count', float('nan'))),
                      float(n_metrics.get('mean_query_count', float('nan')))),
        '平均单样本耗时(s)': (float(de_metrics.get('mean_wall_time', float('nan'))),
                          float(n_metrics.get('mean_wall_time', float('nan')))),
        '评估样本数': (float(de_metrics.get('total_samples', float('nan'))),
                    float(n_metrics.get('total_samples', float('nan')))),
    }

    lines = []
    lines.append(f"{'指标':<20}{'nVITA-DE':>16}{'NSGA-II':>16}")
    lines.append('-' * 52)
    for k, (dv, nv) in rows.items():
        ds = f"{dv:.4f}" if dv == dv else "N/A"      # NaN != NaN
        ns = f"{nv:.4f}" if nv == nv else "N/A"
        lines.append(f"{k:<20}{ds:>16}{ns:>16}")
    return rows, '\n'.join(lines)


def print_comparison_table(de_metrics, nsga2_res):
    """打印对比表格；若 NSGA-II 结果缺失则给出提示。"""
    print(f"\n{'=' * 60}")
    print("DE (nVITA 单目标) vs NSGA-II (双目标变长) 对比汇总")
    print(f"{'=' * 60}")
    if nsga2_res is None:
        print(">>> 未提供/未找到 NSGA-II 结果，仅展示 DE 统计：")
        de_rse = np.asarray(de_metrics.get('sel_rses', []), dtype=float)
        print(f"  评估样本数     : {de_metrics.get('total_samples', 0)}")
        print(f"  RSE (批量=RMSE比) : {de_metrics.get('batch_rse', float('nan')):.4f}")
        if de_rse.size:
            print(f"  RSE  均值/中位数/标准差 : {de_rse.mean():.4f} / "
                  f"{np.median(de_rse):.4f} / {de_rse.std():.4f}")
        else:
            print("  (无样本)")
        print(f"  L2   均值      : {de_metrics.get('mean_sel_l2_norm', 0.0):.6f}")
        print(f"  n_actual 均值  : {de_metrics.get('mean_sel_n_actual', 0.0):.3f}")
        print(f"  平均查询次数   : {de_metrics.get('mean_query_count', 0.0):.1f}")
        print(f"  平均单样本耗时 : {de_metrics.get('mean_wall_time', 0.0):.3f} s")
        print(f"{'=' * 60}")
        return
    _, table_str = build_comparison_table(de_metrics, nsga2_res)
    print(table_str)
    print(f"{'=' * 60}")


# =============================================================================
# MAIN：完整对比实验流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='nVITA DE 单目标稀疏对抗攻击（与 NSGA-II 双目标变长编码对比）')
    parser.add_argument('--data_file', type=str, default='winddata.xlsx', help='输入数据文件')
    parser.add_argument('--load_model', type=str, default=None,
                        help=f'基准模型 checkpoint 路径（默认 {PRETRAINED_CKPT}）')
    # 以下 DE 攻击超参均以 DE_CONFIG 为唯一默认值来源（与主文件 NSGA2_CONFIG 的做法一致），
    # 避免出现与配置不同步的"死配置"；命令行传参仍可临时覆盖，不传参时一律生效 DE_CONFIG 中的设置
    parser.add_argument('--beta', type=float, default=DE_CONFIG['beta'],
                        help='nVITA 扰动预算系数 β（须与 NSGA-II 对比时使用相同值）')
    parser.add_argument('--pop_size', type=int, default=DE_CONFIG['pop_size'],
                        help='DE 种群大小（与 NSGA-II 对齐）')
    parser.add_argument('--maxiter', type=int, default=DE_CONFIG['maxiter'],
                        help='DE 最大代数（与 NSGA-II 对齐）')
    parser.add_argument('--de_f', type=float, default=DE_CONFIG['de_f'], help='DE 缩放因子 F（nVITA 标准值 0.5）')
    parser.add_argument('--de_cr', type=float, default=DE_CONFIG['de_cr'], help='DE 交叉概率 CR（nVITA 标准值 0.9）')
    parser.add_argument('--n_points', type=int, default=DE_CONFIG['n_points'], help='固定扰动点数 N（非变长）')
    parser.add_argument('--num_eval_samples', type=int, default=DE_CONFIG['num_eval_samples'],
                        help='参加评估的测试样本数（默认 None=全部，正整数则从测试集头部截取）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--nsga2_results', type=str,
                        default=os.path.join(OUTPUT_DIR, 'run1_nsga2_attack_results.pkl'),
                        help='NSGA-II 结果 pkl 路径（用于对比图/表；不存在则跳过对比）')
    parser.add_argument('--out_dir', type=str, default=os.path.join(OUTPUT_DIR, 'nvita_de'),
                        help='DE 结果输出目录')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("nVITA 差分进化 (DE/rand/1/bin) 单目标稀疏对抗攻击 —— 与 NSGA-II 双目标变长编码对比")
    print(f"{'=' * 80}")
    print(f"Device        : {DEVICE}")
    print(f"Data file     : {args.data_file}")
    print(f"N (固定点数)  : {args.n_points}")
    print(f"pop_size      : {args.pop_size}   maxiter: {args.maxiter}")
    print(f"DE F / CR     : {args.de_f} / {args.de_cr}")
    print(f"beta (预算)   : {args.beta}")

    # ---------- 1) 加载数据 ----------
    data_result = load_and_preprocess_data(args.data_file)
    test_dataset = data_result['test_dataset']
    num_features = data_result['num_features']

    # 【预算口径】计算训练集全局逐特征 range_f（与 NSGA-II 主文件 train_feature_ranges 同口径：
    #   全部训练窗口在 N×T 上逐特征取 max-min，归一化空间），供 DE 扰动预算 beta*range_f 使用。
    train_X_all = np.concatenate(
        [x.numpy() for x, _ in DataLoader(data_result['train_dataset'], batch_size=512, shuffle=False)], axis=0)
    global_feature_ranges = (train_X_all.reshape(-1, num_features).max(axis=0)
                             - train_X_all.reshape(-1, num_features).min(axis=0))
    print(f"[扰动预算] 训练集逐特征 range_f = {np.round(global_feature_ranges, 4).tolist()}（与 NSGA-II/FGSM 同口径）")

    # ---------- 2) 获取基准模型（优先加载权威 checkpoint，控制变量） ----------
    model = build_or_load_model(data_result, load_path=args.load_model, seed=args.seed)

    # ---------- 3) 提取测试数据 ----------
    X_test, Y_test = extract_test_data(test_dataset)
    print(f"\n[测试集] 共 {X_test.shape[0]} 个样本，shape={tuple(X_test.shape)}")

    if args.num_eval_samples is not None and args.num_eval_samples > 0:
        n_take = min(args.num_eval_samples, X_test.shape[0])
        sample_indices = list(range(n_take))
        print(f"[评估样本截取] 取测试集前 {n_take} 个样本参与 DE 评估")
    else:
        sample_indices = None

    # ---------- 4) 运行 DE 单目标攻击 ----------
    print(f"\n{'=' * 80}")
    print("开始 nVITA-DE 单目标攻击（优化 MSE_adv（忠实复现 nVITA），评估用 RSE/L2，全局 range_f 预算）")
    print(f"{'=' * 80}")
    attacker = nVITA_DE_Attack(
        model, beta=args.beta, pop_size=args.pop_size, maxiter=args.maxiter,
        de_f=args.de_f, de_cr=args.de_cr, n_points=args.n_points, device=DEVICE,
        global_feature_ranges=global_feature_ranges,
    )
    t_start = time.perf_counter()
    X_adv_de, de_metrics, per_sample_records = attacker.attack_batch(
        X_test, Y_test, sample_indices=sample_indices, print_info=True)
    de_total_wall = time.perf_counter() - t_start
    print(f"\n>>> DE 攻击完成，总耗时 {de_total_wall:.2f} s，"
          f"累计模型查询 {attacker.query_count} 次")

    # DE 攻击前后精度破坏（反归一化到 m/s，复用主文件 evaluate_attack 口径）
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']
    n_take = X_adv_de.shape[0]
    X_eval = X_test[:n_take] if sample_indices is None else X_test[sample_indices]
    Y_eval = Y_test[:n_take] if sample_indices is None else Y_test[sample_indices]
    attack_results = evaluate_attack(
        model, X_eval, Y_eval, X_adv_de, min_speed, max_speed,
        f"nVITA-DE (N={args.n_points})", DEVICE)

    # ========== 与主文件 NSGA-II 汇总块（LBA-MOACB-WSF-FixedEncoding.py L2484-2509）逐项对齐 ==========
    # 批量 RSE = 攻击后 RMSE / 干净 RMSE（口径与主文件 main 完全一致；等于 sqrt(mean adv_mse/mean clean_mse)）
    clean_rmse = attack_results['rmse_clean']
    adv_rmse = attack_results['rmse_adv']
    rse_batch = adv_rmse / clean_rmse if clean_rmse > 0 else 0.0
    # 将批量 RSE 与逐样本选中解统计写入 de_metrics，保证与 NSGA-II metrics 同名同口径、可直接对比
    de_metrics['batch_rse'] = float(rse_batch)
    de_metrics['rmse_clean'] = float(clean_rmse)
    de_metrics['rmse_adv'] = float(adv_rmse)

    print(f"\n{'=' * 70}")
    print(f"nVITA-DE 单目标攻击汇总 · 测试样本攻击效果（评估指标与主方法 LBA-MOACB-WSF 一致）")
    print(f"{'=' * 70}")
    print(f"评估(测试)样本数: {n_take}")
    print(f"--- 预测精度破坏程度（clean → adv，反归一化到 m/s 口径） ---")
    print(f"  MAE :  {attack_results['mae_clean']:.4f} → {attack_results['mae_adv']:.4f} m/s "
          f"(Δ {attack_results['delta_mae']:+.4f}, {attack_results['chg_mae_pct']:+.2f}%)")
    print(f"  RMSE:  {clean_rmse:.4f} → {adv_rmse:.4f} m/s "
          f"(Δ {attack_results['delta_rmse']:+.4f}, {attack_results['chg_rmse_pct']:+.2f}%)")
    print(f"  MAPE:  {attack_results['mape_clean']:.2f}% → {attack_results['mape_adv']:.2f}% "
          f"(Δ {attack_results['delta_mape']:+.2f}, {attack_results['chg_mape_pct']:+.2f}%)")
    print(f"  R2  :  {attack_results['r2_clean']:.4f} → {attack_results['r2_adv']:.4f} "
          f"(下降率 {attack_results['drop_r2_pct']:.2f}%)")
    print(f"  RSE :  {rse_batch:.4f}  (= 攻击后RMSE / 干净RMSE)")
    print(f"--- 扰动规模（跨全部测试样本平均） ---")
    print(f"  平均 L2 范数（选中解实测）: {de_metrics['mean_sel_l2_norm']:.6f}")
    print(f"  平均 L0 范数（每样本扰动点数 n_actual 均值）: {de_metrics['mean_sel_n_actual']:.3f}")
    print(f"  选中解平均 RSE（逐样本 RSE 均值）: {de_metrics['mean_sel_rse']:.4f}")
    print(f"{'=' * 70}")

    # ---------- 5) 加载 NSGA-II 结果用于对比 ----------
    nsga2_res = load_nsga2_results(args.nsga2_results)
    if nsga2_res is not None:
        print(f"\n>>> 已加载 NSGA-II 结果用于对比: {args.nsga2_results}")
    else:
        print(f"\n>>> 未找到 NSGA-II 结果（{args.nsga2_results}），对比图/表将仅含 DE。")

    # ---------- 6) 打印对比表格 ----------
    print_comparison_table(de_metrics, nsga2_res)

    # ---------- 7) 画图 ----------
    nsga2_curve = nsga2_best_rse_curve(nsga2_res.get('generation_pareto')) if nsga2_res else np.empty(0)
    conv_png = os.path.join(args.out_dir, 'de_vs_nsga2_convergence.png')
    scat_png = os.path.join(args.out_dir, 'de_vs_nsga2_scatter_l2_rse.png')
    plot_convergence_comparison(de_metrics.get('mean_convergence_rse'), nsga2_curve, conv_png)
    plot_scatter_comparison(de_metrics, nsga2_res, scat_png)

    # ---------- 8) 保存结果（pickle 格式与 NSGA-II 结果兼容） ----------
    rows_dict = None
    if nsga2_res is not None:
        rows_dict, _ = build_comparison_table(de_metrics, nsga2_res)
    result_payload = {
        'attack_results': attack_results,        # 与 NSGA-II 结果同名键，口径一致
        'metrics': de_metrics,                   # 含 mean_sel_rse/sel_rses/sel_l2_norms 等同名键
        'per_sample_records': per_sample_records,
        'generation_pareto': None,               # DE 单目标无 Pareto 前沿，占位保持结构兼容
        'config': {
            'method': 'nVITA_DE',
            'n_points': args.n_points,
            'pop_size': args.pop_size,
            'maxiter': args.maxiter,
            'de_f': args.de_f,
            'de_cr': args.de_cr,
            'beta': args.beta,
            'budget': 'dynamic_window_range',
            # 【优化与评估分离】DE 优化目标为 MSE_adv（忠实复现 nVITA 论文）；
            #                最终评估指标与 NSGA-II 完全一致（RSE/L2/n_actual）
            'objective': 'MSE_adv (nVITA paper setting)',
            'final_metric': 'RSE/L2/n_actual (same as NSGA-II)',
            'rse_def': 'sqrt(adv_mse/clean_mse) (与 NSGA-II 一致)',
            'seed': args.seed,
            'total_wall_time': de_total_wall,
            'total_query_count': int(attacker.query_count),
        },
        # 改动4：汇总统计（字段名便于与 NSGA-II 统一分析）
        'summary': {
            'method': 'nVITA_DE',
            'config': {
                'objective': 'MSE_adv (nVITA paper setting)',
                'final_metric': 'RSE/L2 (same as NSGA-II)',
                'budget': 'dynamic window range',
                'n_points': args.n_points,
                'pop_size': args.pop_size,
                'maxiter': args.maxiter,
                'beta': args.beta,
                'de_f': args.de_f,
                'de_cr': args.de_cr,
            },
            'rse_mean': de_metrics['mean_sel_rse'],
            'rse_median': de_metrics['median_sel_rse'],
            'rse_std': de_metrics['std_sel_rse'],
            'l2_mean': de_metrics['mean_sel_l2_norm'],
            'n_actual_mean': de_metrics['mean_sel_n_actual'],
            'convergence_curve_mean': de_metrics.get('mean_convergence_mse'),   # 每代最优 MSE_adv（DE 优化目标）
            'convergence_rse_mean': de_metrics.get('mean_convergence_rse'),      # 每代最优 RSE（换算，供对比）
            'per_sample_results': per_sample_records,
        },
        'comparison_table': rows_dict,
        'nsga2_results_path': args.nsga2_results if nsga2_res is not None else None,
    }
    de_pkl = os.path.join(args.out_dir, 'nvita_de_attack_results.pkl')
    with open(de_pkl, 'wb') as f:
        pickle.dump(result_payload, f)
    print(f"\n>>> DE 攻击结果已保存: {de_pkl}")

    # 同时保存 DE 收敛曲线（供后续统一分析）：MSE_adv（优化目标）与 RSE（换算）两条
    conv_pkl = os.path.join(args.out_dir, 'de_convergence_curves.pkl')
    with open(conv_pkl, 'wb') as f:
        pickle.dump({
            'mean_convergence_mse': de_metrics.get('mean_convergence_mse'),
            'convergence_curves_mse': de_metrics.get('convergence_curves_mse'),
            'mean_convergence_rse': de_metrics.get('mean_convergence_rse'),
            'convergence_curves': de_metrics.get('convergence_curves'),
            'nsga2_mean_best_rse_curve': nsga2_curve,
        }, f)
    print(f">>> DE 收敛曲线已保存: {conv_pkl}")

    print(f"\n{'=' * 80}")
    print("nVITA-DE 对比实验全部完成！输出文件：")
    print(f"  - DE 结果 pkl     : {de_pkl}")
    print(f"  - 收敛曲线 pkl    : {conv_pkl}")
    print(f"  - 收敛曲线对比图  : {conv_png}")
    print(f"  - L2-RSE 散点对比 : {scat_png}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
