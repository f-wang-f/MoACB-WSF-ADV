"""
MoACB-WSF with Fixed Manual Encoding (NSGA-II 变长编码双目标稀疏攻击版，L0 作为约束)
=======================================================================
基于论文给出的最佳折中混合编码向量直接构建 CNN-BiLSTM 风速预测模型，
训练目标模型后执行 nVITA/NSGA-II 稀疏对抗攻击评估。

本版本相对 LBA 版的核心改动：
1.【删除 LBA】完整移除 LBA 学习式攻击管线：LBA_Dataset / BayesianConv1d / BayesianLinear /
   SEBlock / CNN_LBA_Model 等类，get_sensitive_point_and_value / get_LBA_Dataset /
   build_sensitive_labels / print_sign_distribution / train_lba_model / calc_sensitive_point_ar /
   calc_perturb_rmse / evaluate_lba_fitting_quality / run_lba_attack / run_lba_pipeline 等函数，
   全部 LBA 可视化函数与 LBA_CONFIG 全局配置，以及 main() 中的 LBA 分支与 LBA 命令行参数。
   保留：数据加载/归一化/划分、HybridCNNBiLSTM 固定编码训练与评估、evaluate_attack、
        run_nsga2_standalone_attack、plot_hv_convergence_curves、
        plot_pareto_front_evolution_grid、plot_perturb_position_heatmap。
2.【变长编码】NVITA_NSGA2 由固定 n 组三元组改为可变长度三元组编码：
   个体 = [t1,f1,p1, t2,f2,p2, ..., tn,fn,pn]，总长度 3n 随 n 动态变化，
   n ∈ [n_min, n_max]，由初始化随机采样、插入变异(n+1)、删除变异(n-1)共同驱动；
   交叉改为「变长单点交叉 + 位置去重 + 长度截断 + 长度补齐 + 预算截断」的合法性修复链。
3.【双目标 + L0 约束】优化目标为 (L2, -RSE) 双目标，全部最小化：
   obj1 = 扰动 L2 范数（越小越隐蔽）
   obj2 = -RSE，RSE = sqrt(攻击后MSE / 干净预测MSE)（越小代表攻击越强，替代原 1/RSE）
   n_actual（去重后真实扰动点数，即 L0 范数）不再作为目标维度，而是约束变量：
   要求 n_actual ∈ [n_min, n_max]，违约度 cv = max(0, n-n_max) + max(0, n_min-n)；
   非支配排序采用 Deb 约束支配（可行优先 → 二维目标支配 → 违约度比较）。
   删除冗余的第三目标维度可避免第一前沿过早吞没整个种群，拥挤距离 / 锦标赛 /
   环境选择均按 2 个目标计算。
4.【分析扩展】攻击高级分析模块（SECTION 5 / 5.5）：二维超体积 HV、双目标均值/最优值、每代 n_actual
   均值等收敛统计打印到控制台并持久化；本项目保留三张攻击可视化图——
   ① HV / max RSE / min L2 三行收敛曲线（跨样本聚合，plot_hv_convergence_curves）、
   ② 随机 10 样本 Pareto 前沿进化网格图（2×5 子图，y 轴 = -RSE，前沿向左下收敛，plot_pareto_front_evolution_grid）、
   ③ 全局 Pareto 前沿扰动位置 (t,f) 频次热力图（YlOrRd + 格内数字标注，plot_perturb_position_heatmap）；
   图片输出到 output/run{id}/，统计数据统一持久化到 output/run{id}/analysis_data.pkl。
   【可视化更新】原 plot_sample_pf_evolution_key_gens（2×5 多样本 PF 进化总览）已被 ② 替换删除；
   全局聚合 Pareto 前沿图（plot_global_pareto_front，y 轴 = RSE）已按需求删除，PF 前沿统一以 -RSE 为纵轴。
5.【删除结构搜索】MoACB-WSF 的「模型自动设计」部分已整体移除，本文件只保留「输入编码 → 具体模型」链路：
   删除 initialize_individual / initialize_population / generate_valid_topology /
   variable_length_mutation / crossover_population（随机种群与架构进化算子）、
   evaluate_individual / evaluate_population（逐个体训练一遍模型的 NAS 适应度评估）、
   以及 NAS 版 fast_non_dominated_sort / crowding_distance / tournament_selection /
   environmental_selection（攻击所需的约束支配排序由 NVITA_NSGA2 类内部方法独立实现）。
   保留 encode_individual / decode_individual / decode_hyperparams / is_valid_individual
   作为编码向量与模型结构/超参数之间的转换与合法性校验层，SECTION 4 起的章节号相应前移。

原文件 LBA-MOACB-WSF.py 完全保留，两个文件并存。
"""

## 编码来源（Sotavento 10-min Dataset）
# Topo:    [1,1,1,1,0,0,0,0,0,0]
# CNN:     [[0,0,1,0,1], [1,3,4,1,1], [3,0,3,1,1]]
# BiLSTM:  [[0,0,7,1,0], [0,0,0,1,2]]
# Setting: [0, 0, 0.0072, 3]

## 第一次运行：训练固定编码模型并保存
#python LBA-MOACB-WSF-FixedEncoding.py --save_model output/fixed_encoding_model.pt

# 后续消融实验：加载模型，仅跑 NSGA-II 变长双目标（L0 约束）攻击评估（跳过训练）
#python LBA-MOACB-WSF-FixedEncoding.py --load_model output/fixed_encoding_model.pt --n_min 1 --n_max 6
#python LBA-MOACB-WSF-FixedEncoding.py --load_model output/fixed_encoding_model.pt --n_min 3 --n_max 3 --pop_size 20 --maxiter 30

import os
import sys
import ast
import json
import random
import time
import warnings
import pickle
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ==================== Global Configuration ====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'attack_results'), exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MoACB-WSF Configuration ====================
FILENAME = 'winddata.xlsx'
FEATURE_COLUMNS = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
TARGET_COLUMN = 'Wind Speed (m/s)'
SEQUENCE_LENGTH =20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES = 100

# 编码向量描述的网络结构规模（HybridCNNBiLSTM 组网与 decode_individual 解码共用）
NUM_MODULES = 5
NUM_CNN_MODULES = 3
NUM_LSTM_MODULES = 2
TOPO_BITS_LENGTH = NUM_MODULES * (NUM_MODULES - 1) // 2

# 固定编码模型的训练/评估超参数
# 【结构搜索代码已删除】原专用常量 POP_SIZE / MAX_GEN / MUTATION_PROB / CROSSOVER_PROB /
# EVAL_EPOCHS / EVAL_PATIENCE / MAX_MODEL_SIZE / INDIVIDUAL_LENGTH / NUM_FEATURES 一并移除
NUM_RUNS = 1
LAMBDA_REG = 1e-4
FINAL_EPOCHS = 200
FINAL_PATIENCE = 20
GRADIENT_CLIP = 1.0

# 编码位的取值区间：decode_individual 解码时的越界钳制范围
KNUM_RANGE = (0, 7)
KSIZE_RANGE = (0, 3)
KACT_RANGE = (0, 7)
PT_RANGE = (0, 2)
PS_RANGE = (0, 3)
BS_RANGE = (0, 3)
OPT_RANGE = (0, 3)
LR_RANGE = (0.0001, 0.01)
REG_RANGE = (0, 3)

# 激活函数编码由 HybridCNNBiLSTM 内部的 act_map 直接构建，无需额外字符串映射表
OPTIMIZER_MAP = {0: 'SGD', 1: 'Adam', 2: 'AdaDelta', 3: 'RMSprop'}
REGULARIZER_MAP = {0: None, 1: 'L1', 2: 'L2', 3: 'L1L2'}
BATCH_SIZE_MAP = {0: 32, 1: 64, 2: 96, 3: 128}

# ==================== NSGA-II 变长双目标稀疏攻击配置（L0 作为约束） ====================
# 命名说明（对齐原论文）：
#   beta: nVITA 的扰动预算系数（原论文中的 β）
# 【变长编码】n_min / n_max 取代原固定 n：每个个体独立采样扰动点数 n ∈ [n_min, n_max]
NSGA2_CONFIG = {
    'n_min': 1,      # 变长编码：扰动点数下界（时序稀疏攻击场景默认 1）
    'n_max': 8,      # 变长编码：扰动点数上界（时序稀疏攻击场景默认 6）
    'beta': 0.01,     # nVITA perturbation budget factor (原论文 β)
    'maxiter':50, # NSGA-II max generations
    'pop_size': 20,  # NSGA-II population size（增大可提升搜索覆盖度，但每代计算量线性增加）
    'insert_prob': 0.1,  # 变长编码：插入变异概率（n+1）
    'delete_prob': 0.2,  # 变长编码：删除变异概率（n-1）【方案A】0.1→0.2：加速不可行解(n<n_min)在种群中积累，使 Deb 约束支配更快发挥作用
    'select_mode': 'knee',   # 最终解选择策略：'knee'=归一化折中膝点 / 'max_rse'=攻击强度上界端点
    'knee_min_rse': 1,     # 膝点有效性下界：仅 RSE>=该值的前沿点可作膝点候选
    'select_strategy': 'percentile',      # 折中解选点策略（在过滤后的候选集上二选一）：'knee'=距参考点最近 / 'percentile'=按L2分位数取点
    'knee_l2_weight': 0.5,   # knee策略：折中解参考点 L2 维分量（归一化空间 0~1）：越小越接近理想点(左下)
    'knee_rse_weight': 0.5,  # knee策略：折中解参考点 -RSE 维分量（归一化空间 0~1）：越大越偏攻击效果一侧
    'front_percentile': 0.6, # percentile策略：候选解按 L2 升序后取第 int(k*该值) 个位置（0~1，默认60%分位）
    'num_eval_samples': None, # 参加 NSGA-II 评估的样本数量（None 或 <=0 表示使用数据池全部样本，正整数则从数据池头部截取该数量的样本）

    # ===== 敏感度引导退火（永久启用）=====
    # 攻击前用 seq_len*num_features 次前向查询估计每个位置的敏感度，取 top-K 建候选池；
    # 初始化与变异选新位置时按 guided_ratio 概率从候选池选，其余随机；
    # guided_ratio 随代数从 guided_ratio_start 线性退火到 guided_ratio_end
    # （前 30% 代保持 start，后 70% 代线性衰减）。
    'candidate_pool_size': 40,
    'guided_ratio_start': 0.6,
    'guided_ratio_end': 0.3,
}

# 扰动幅值 p 的离散档位数（20 档位整数编码）
P_DISCRETE_LEVELS = 40

# ==================== FIXED MANUAL ENCODING FROM PAPER ====================
# 论文表 "BEST TRADE-OFF HYBRID ENCODING VECTORS OBTAINED BY THE PROPOSED MoACB-WSF"
# Dataset: Sotavento 10-min Dataset
FIXED_ENCODING = {
    'topo': [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    'cnn_params': [
        [0, 2, 3, 2, 0],   # CNN module 0
        [1, 1, 6, 0, 1],   # CNN module 1
        [2, 0, 4, 0, 2],   # CNN module 2
    ],
    'lstm_params': [
        [0, 0, 6, 0, 3],   # BiLSTM module 0 (module index 3)
        [0, 1, 4, 0, 0],   # BiLSTM module 1 (module index 4)
    ],
    'setting': [0, 1, 0.0007425602587, 0],  # batch_size=32, SGD, lr=0.0072, regularizer=L1L2
}


# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING (MoACB-WSF)
# =============================================================================

class WindDataset(Dataset):
    def __init__(self, data, sequence_length=SEQUENCE_LENGTH):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, -1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def load_and_preprocess_data(filename=FILENAME):
    print('正在读取风速数据...')
    # 【已删除合成数据兜底】读文件/缺列失败时直接抛出异常终止，不再生成示例数据
    if filename.endswith('.csv'):
        data = pd.read_csv(filename)
    else:
        data = pd.read_excel(filename)
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise ValueError(f'未找到特征列: {col}')
    feature_data = data[FEATURE_COLUMNS].values

    valid_rows = np.all(~np.isnan(feature_data) & ~np.isinf(feature_data), axis=1)
    feature_data = feature_data[valid_rows, :]
    if feature_data.shape[0] < MIN_SAMPLES:
        raise ValueError(f'数据不足(少于{MIN_SAMPLES}个样本)')

    scaler = MinMaxScaler()
    feature_data_norm = scaler.fit_transform(feature_data)
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_

    dataset = WindDataset(feature_data_norm, SEQUENCE_LENGTH)

    num_samples = len(dataset)
    num_train = int(TRAIN_RATIO * num_samples)
    num_val = int(VAL_RATIO * num_samples)
    num_test = num_samples - num_train - num_val

    train_dataset = Subset(dataset, range(num_train))
    val_dataset = Subset(dataset, range(num_train, num_train + num_val))
    test_dataset = Subset(dataset, range(num_train + num_val, num_samples))

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'min_speed': min_vals[-1],
        'max_speed': max_vals[-1],
        'num_features': len(FEATURE_COLUMNS),
        'feature_columns': FEATURE_COLUMNS,
        'scaler': scaler,
        'full_data_norm': feature_data_norm
    }


# =============================================================================
# SECTION 2: HYBRID CNN-BiLSTM MODEL (MoACB-WSF)
# =============================================================================

class HybridCNNBiLSTM(nn.Module):
    def __init__(self, topo, cnn_params, lstm_params, num_features, sequence_length):
        super(HybridCNNBiLSTM, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        n = NUM_MODULES
        self.n = n
        self.feature_weights = nn.Parameter(torch.ones(num_features))

        bits_length = n * (n - 1) // 2
        if len(topo) != bits_length:
            raise ValueError(f"拓扑长度不匹配: {len(topo)} != {bits_length}")
        self.adj = np.zeros((n, n), dtype=int)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                self.adj[i, j] = topo[k]
                k += 1

        in_channels_list = [0] * n
        in_channels_list[0] = num_features
        for i in range(1, n):
            num_incoming = np.sum(self.adj[:, i])
            if num_incoming == 0:
                num_incoming = 1
            incoming_channels = []
            for j in range(i):
                if self.adj[j, i] == 1:
                    if j < NUM_CNN_MODULES:
                        incoming_channels.append(16 * (cnn_params[j][0] + 1))
                    else:
                        incoming_channels.append(32 * (lstm_params[j - NUM_CNN_MODULES][0] + 1))
            if incoming_channels:
                in_channels_list[i] = sum(incoming_channels)
            else:
                in_channels_list[i] = 16

        self.convs = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.adaptive_pools = nn.ModuleList()

        act_map = {
            0: nn.Softplus, 1: nn.Softsign, 2: nn.ELU, 3: nn.Softmax,
            4: nn.Sigmoid, 5: nn.Tanh, 6: nn.ReLU, 7: nn.Identity
        }

        for i in range(n):
            in_channels = max(in_channels_list[i], 1)
            if i < NUM_CNN_MODULES:
                knum, ksize, kact, pt, ps = cnn_params[i]
                out_channels = 16 * (knum + 1)
                kernel_size = 2 * ksize + 3
                self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'))
            else:
                knum, lnum, kact, pt, ps = lstm_params[i - NUM_CNN_MODULES]
                hidden = 16 * (knum + 1)
                dropout = ps * 0.1
                num_layers = lnum + 1
                self.lstms.append(
                    nn.LSTM(in_channels, hidden, num_layers=num_layers, bidirectional=True,
                            batch_first=False, dropout=dropout)
                )
                out_channels = 2 * hidden

                self.bns.append(nn.BatchNorm1d(out_channels))
                if kact == 3:
                    self.acts.append(act_map[kact](dim=1))
                else:
                    self.acts.append(act_map[kact]())

                pool_size = 2 * ps + 3
                if pt == 0:
                    self.pools.append(nn.MaxPool1d(pool_size, stride=1, padding=(pool_size - 1) // 2))
                elif pt == 1:
                    self.pools.append(nn.AvgPool1d(pool_size, stride=1, padding=(pool_size - 1) // 2))
                else:
                    self.pools.append(nn.Identity())
                self.adaptive_pools.append(nn.AdaptiveAvgPool1d(sequence_length))

        total_out_channels = sum(
            [16 * (cnn_params[i][0] + 1) if i < NUM_CNN_MODULES else 32 * (lstm_params[i - NUM_CNN_MODULES][0] + 1)
             for i in range(n)]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(total_out_channels, 1)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("输入包含NaN或Inf值")
        x = x.transpose(1, 2)
        x = x * self.feature_weights.unsqueeze(0).unsqueeze(2)
        outputs = []
        for i in range(self.n):
            inputs = []
            if i == 0:
                inputs.append(x)
            else:
                for prev in range(i):
                    if self.adj[prev, i] == 1:
                        inputs.append(outputs[prev])
            if not inputs:
                inputs.append(x)
            input_i = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            if i < NUM_CNN_MODULES:
                out = self.convs[i](input_i)
            else:
                input_lstm = input_i.permute(2, 0, 1)
                out_lstm, _ = self.lstms[i - NUM_CNN_MODULES](input_lstm)
                out = out_lstm.permute(1, 2, 0)
                out = self.bns[i - NUM_CNN_MODULES](out)
                out = self.acts[i - NUM_CNN_MODULES](out)
                out = self.pools[i - NUM_CNN_MODULES](out)
                out = self.adaptive_pools[i - NUM_CNN_MODULES](out)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError(f"模块{i}输出包含NaN或Inf值")
            outputs.append(out)
        final = torch.cat(outputs, dim=1)
        final = self.global_pool(final).squeeze(-1)
        final = self.fc(final)
        return final



# =============================================================================
# SECTION 3: 混合编码向量的编码 / 解码 / 合法性校验（输入编码 → 具体模型）
# =============================================================================

def encode_individual(topo, cnn_params, lstm_params, setting):
    individual = topo + [item for sublist in cnn_params for item in sublist] + \
                 [item for sublist in lstm_params for item in sublist] + setting
    return np.array(individual, dtype=float)


def decode_individual(individual):
    idx = 0
    n = NUM_MODULES
    bits_length = TOPO_BITS_LENGTH
    topo = [int(round(individual[idx + i])) for i in range(bits_length)]
    idx += bits_length

    cnn_params = []
    for i in range(NUM_CNN_MODULES):
        knum = max(KNUM_RANGE[0], min(KNUM_RANGE[1], int(round(individual[idx]))))
        ksize = max(KSIZE_RANGE[0], min(KSIZE_RANGE[1], int(round(individual[idx + 1]))))
        kact = max(KACT_RANGE[0], min(KACT_RANGE[1], int(round(individual[idx + 2]))))
        pt = max(PT_RANGE[0], min(PT_RANGE[1], int(round(individual[idx + 3]))))
        ps = max(PS_RANGE[0], min(PS_RANGE[1], int(round(individual[idx + 4]))))
        cnn_params.append([knum, ksize, kact, pt, ps])
        idx += 5

    lstm_params = []
    for i in range(NUM_LSTM_MODULES):
        knum = max(KNUM_RANGE[0], min(KNUM_RANGE[1], int(round(individual[idx]))))
        lnum = max(KSIZE_RANGE[0], min(KSIZE_RANGE[1], int(round(individual[idx + 1]))))
        kact = max(KACT_RANGE[0], min(KACT_RANGE[1], int(round(individual[idx + 2]))))
        pt = max(PT_RANGE[0], min(PT_RANGE[1], int(round(individual[idx + 3]))))
        ps = max(PS_RANGE[0], min(PS_RANGE[1], int(round(individual[idx + 4]))))
        lstm_params.append([knum, lnum, kact, pt, ps])
        idx += 5

    bs = max(BS_RANGE[0], min(BS_RANGE[1], int(round(individual[idx]))))
    opt = max(OPT_RANGE[0], min(OPT_RANGE[1], int(round(individual[idx + 1]))))
    lr = max(LR_RANGE[0], min(LR_RANGE[1], individual[idx + 2]))
    reg = max(REG_RANGE[0], min(REG_RANGE[1], int(round(individual[idx + 3]))))
    setting = [bs, opt, lr, reg]

    return topo, cnn_params, lstm_params, setting


def decode_hyperparams(setting):
    bs, opt, lr, reg = setting
    return BATCH_SIZE_MAP[bs], lr, OPTIMIZER_MAP[opt], REGULARIZER_MAP[reg]


def is_valid_individual(individual):
    try:
        topo, _, _, _ = decode_individual(individual)
        n = NUM_MODULES
        for i in range(1, n):
            num_incoming = 0
            k = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if jj == i and topo[k] == 1:
                        num_incoming += 1
                    k += 1
            if num_incoming == 0:
                return False
        return True
    except:
        return False


# =============================================================================
# SECTION 4: NVITA_NSGA2 - NSGA-II 变长编码双目标稀疏对抗攻击（L0 作为约束）
# =============================================================================
# 【双目标】基于 NSGA-II 的 nVITA 稀疏黑盒攻击，同时优化两个目标（全部最小化）:
#   目标1: 扰动 L2 范数——值越小，扰动越隐蔽、越难检测
#   目标2: -RSE，其中 RSE = sqrt(攻击后MSE / 干净预测MSE)
#          RSE 越大代表攻击效果越强，取负号后统一为最小化方向（替代原双目标版的 1/RSE）
# 【L0 约束】n_actual 实际扰动点数（去重后真实数量，即 L0 范数）不再进入目标向量，
#          而是作为约束变量：要求 n_actual ∈ [n_min, n_max]，
#          违约度 cv = max(0, n_actual-n_max) + max(0, n_min-n_actual)，
#          非支配排序采用 Deb 约束支配（可行优先 → 二维目标支配 → 违约度比较）
# 【变长编码】个体为 n 组三元组 [t1,f1,p1, ..., tn,fn,pn] 拼接成的一维向量，长度 3n 随 n 变化，
#          n ∈ [n_min, n_max]；完全保留原 nVITA 的动态扰动预算（窗口极差）、特征约束等功能
# =============================================================================


class NVITA_NSGA2:
    """
    基于 NSGA-II 的双目标变长稀疏黑盒对抗攻击（L0 作为约束）

    【变长编码】n 组 (时间步t, 特征f, 扰动值p) 展平为 3n 维一维向量，
                n 在 [n_min, n_max] 内逐个体独立变化（不再是固定的 self.n）
    【双目标】（全部最小化）：
        obj[0] = ||p||_2   扰动 L2 范数（越小越隐蔽）
        obj[1] = -RSE      RSE = sqrt(攻击后MSE / 干净预测MSE)，取负号后越小=攻击越强
    【L0 约束】n_actual（去重后真实扰动点数）不进入目标向量，作为约束变量：
        要求 n_actual ∈ [n_min, n_max]，违约度 cv = max(0, n-n_max) + max(0, n_min-n)，
        非支配排序采用 Deb 约束支配（可行优先 → 二维目标支配 → 违约度比较）
    """

    def __init__(self, n_min, n_max, epsilon, model, feature_ranges, maxiter=60, pop_size=30,
                 eta_c=15, eta_m=20, crossover_prob=0.9,
                 feature_constraint=None, perturb_features=None,
                 use_window_range=True, init_strategy='random', select_mode='knee',
                 knee_min_rse=2.0, insert_prob=0.1, delete_prob=0.1,
                 candidate_pool_size=20,
                 guided_ratio_start=0.6, guided_ratio_end=0.3,
                 knee_l2_weight=0.3, knee_rse_weight=0.7,
                 select_strategy='knee', front_percentile=0.6):
        # ====== 【变长编码】用 [n_min, n_max] 区间取代原固定 self.n ======
        self.n_min = max(1, int(n_min))            # 扰动点数下界
        self.n_max = max(self.n_min, int(n_max))   # 扰动点数上界（自动纠正 n_min > n_max 的误配置）
        self.epsilon = epsilon          # 扰动预算系数
        self.model = model
        self.feature_ranges = feature_ranges  # 全局特征取值范围回退值
        self.maxiter = maxiter
        self.pop_size = pop_size
        self.eta_c = eta_c              # SBX 分布指数
        self.eta_m = eta_m              # PM 多项式变异分布指数
        self.crossover_prob = crossover_prob
        self.feature_constraint = feature_constraint
        self.perturb_features = perturb_features
        self.use_window_range = use_window_range
        self.init_strategy = init_strategy  # 'random' / 'sensitivity'
        self.select_mode = select_mode      # 最终解选择策略：'knee'=归一化折中膝点（默认）/ 'max_rse'=攻击强度上界端点（消融对比用）
        self.knee_min_rse = knee_min_rse    # 膝点有效性下界：仅 RSE>=该值的前沿点可作膝点候选，防止弱前沿上膝点退化为近零效果点
        self.knee_l2_weight = float(knee_l2_weight)    # 折中解参考点 L2 维分量（归一化空间 0~1）
        self.knee_rse_weight = float(knee_rse_weight)  # 折中解参考点 -RSE 维分量（归一化空间 0~1，越大越偏攻击效果）
        self.select_strategy = select_strategy    # 折中解选点策略：'knee'=距参考点最近 / 'percentile'=按L2分位数取点
        self.front_percentile = float(front_percentile)  # percentile策略：候选集按L2升序后的分位数位置（0~1）
        # ====== 【变长编码】长度变异的两个独立概率 ======
        self.insert_prob = insert_prob      # 插入变异概率：随机新增一个三元组，n+1
        self.delete_prob = delete_prob      # 删除变异概率：随机删除一个三元组，n-1

        # ====== 【查询计数 / clean 缓存】（第8点：clean 每样本只查一次并缓存）======
        self.query_count = 0                 # 当前样本累计模型前向查询数（attack 开头重置）
        self.first_success_query = -1        # 首次出现 RSE>=knee_min_rse 个体时的累计查询数（未达标记 -1）
        self._clean_pred_cache = None        # 当前样本 clean 预测缓存（attack 开头计算一次；None 时评估内回退现算）

        # ====== 【过程诊断】截断率/预算截断率/空个体率计数（attack 内累计，不影响数值）======
        self.diag = None

        # ====== 【敏感度引导退火】退火参数（机制永久启用，无总开关）======
        # attack() 每样本先探测 seq_len*num_features 个位置的敏感度，取 top-candidate_pool_size 建候选池；
        # _init_individual / _mutation 选新位置时按 guided_ratio 概率从候选池选，其余随机；
        # guided_ratio 随代数从 guided_ratio_start 线性退火到 guided_ratio_end。
        self.candidate_pool_size = int(candidate_pool_size)
        self.guided_ratio_start = float(guided_ratio_start)
        self.guided_ratio_end = float(guided_ratio_end)
        self.candidate_pool = []       # [(t,f), ...] 运行时由 attack() 逐样本填充
        self.current_gen = 0           # 当前进化代数，用于退火计算，由 attack() 主循环每代更新
        # 每样本上下文（由 attack() 在探测前赋值；此处给默认值防独立调用报错）
        self.seq_len = 0
        self.num_features = 0
        self.device = None

    def _get_window_range(self, x_np):
        """计算当前样本每个特征的窗口极差（对齐官方 calculate_test_window_ranges）"""
        if not self.use_window_range:
            return np.asarray(self.feature_ranges, dtype=float)
        return np.ptp(x_np[0], axis=0).astype(float)

    def _sample_features(self, num_features):
        """根据约束采样允许扰动的特征索引"""
        if self.perturb_features is not None:
            valid_features = [f for f in self.perturb_features if 0 <= f < num_features]
            if valid_features:
                return np.array(valid_features)
            if len(self.perturb_features) == 0:
                return np.array([], dtype=int)
        if self.feature_constraint is None or self.feature_constraint >= num_features:
            return np.arange(num_features)
        return np.random.choice(num_features, self.feature_constraint, replace=False)

    @staticmethod
    def _quantize_p(p_value, budget):
        """将实数 p 量化到 [-budget, budget] 上最近的离散值（P_DISCRETE_LEVELS 档等间隔）
        用于 _clip_tf_budget 修复、变异后约束，保证 p 恒为离散值之一。
        budget<=0 时返回 0.0。
        """
        if budget < 1e-12:
            return 0.0
        n = P_DISCRETE_LEVELS
        idx = int(np.clip(
            np.round((p_value + budget) / (2 * budget) * (n - 1)),
            0, n - 1))
        return float(-budget + idx * (2 * budget) / (n - 1))

    @staticmethod
    def _num_points(individual):
        """【变长编码】由向量长度反解当前个体的扰动点数 n = len(individual) // 3"""
        return len(individual) // 3

    def _random_new_point(self, seq_len, allowed_features, ranges, existing):
        """【变长编码】随机生成一个与 existing 中 (t,f) 不重复的新三元组
        供「插入变异」与「长度补齐」复用，幅值按该特征的动态预算 epsilon*range[f] 采样。
        返回: ([t, f, p], (t, f))
        """
        if len(allowed_features) == 0:
            return [0.0, 0.0, 0.0], (0, 0)
        for _ in range(200):
            t = int(np.random.randint(0, seq_len))
            f = int(np.random.choice(allowed_features))
            if (t, f) not in existing:
                budget = self.epsilon * ranges[f]
                if budget > 0:
                    values = np.linspace(-budget, budget, P_DISCRETE_LEVELS)
                    p = float(np.random.choice(values))
                else:
                    p = 0.0
                return [float(t), float(f), p], (t, f)
        # 兜底：位置空间被占满时允许重复，交由 _repair_unique 合并幅值
        t = int(np.random.randint(0, seq_len))
        f = int(np.random.choice(allowed_features))
        budget = self.epsilon * ranges[f]
        if budget > 0:
            values = np.linspace(-budget, budget, P_DISCRETE_LEVELS)
            p = float(np.random.choice(values))
        else:
            p = 0.0
        return [float(t), float(f), p], (t, f)

    def _compute_sensitivity(self, x_window, pred_clean, ranges, allowed_features=None):
        """【敏感度引导退火 / 探测】对每个位置 (t,f) 单独施加固定小扰动，观察模型输出变化的绝对值。
        仅用前向查询，完全黑盒；探测扰动大小 = 0.5 * epsilon * ranges[f]（中间档，避免饱和）。
        参数：
            x_window: shape=(seq_len, num_features) 的干净窗口（2D）
            pred_clean: 干净窗口下的模型预测标量
            ranges: shape=(num_features,) 逐特征预算基底（use_window_range=False 时为全局 range_f）
            allowed_features: 允许扰动的特征索引集合（None 表示全部），非法特征敏感度保持 0
        返回：敏感度矩阵 shape=(seq_len, num_features)
        """
        if allowed_features is None:
            allowed_set = set(range(self.num_features))
        else:
            allowed_set = set(int(f) for f in np.asarray(allowed_features).flatten().tolist())
        sensitivity = np.zeros((self.seq_len, self.num_features), dtype=float)
        x_base = np.asarray(x_window, dtype=float).copy()
        ranges_arr = np.asarray(ranges, dtype=float)
        for t in range(self.seq_len):
            for f in range(self.num_features):
                if f not in allowed_set:
                    continue
                delta = 0.5 * self.epsilon * float(ranges_arr[f])
                x_test = x_base.copy()
                x_test[t, f] += delta
                with torch.no_grad():
                    pred_test = self.model(
                        torch.FloatTensor(x_test[None]).to(self.device)).item()
                self.query_count += 1
                sensitivity[t, f] = abs(pred_test - pred_clean)
        return sensitivity
    
    def _build_candidate_pool(self, sensitivity, k):
        """【敏感度引导退火 / 候选池】把 seq_len × num_features 个位置按敏感度降序排列，取 top-k 组成候选池。
        返回：[(t, f), ...] 长度 <= k 的位置列表（敏感度相同时保持排序稳定性）。
        """
        positions = [(float(sensitivity[t, f]), int(t), int(f))
                     for t in range(self.seq_len)
                     for f in range(self.num_features)]
        positions.sort(reverse=True, key=lambda x: x[0])
        kk = max(0, min(int(k), len(positions)))
        return [(t, f) for (_s, t, f) in positions[:kk]]
    
    def _sample_new_position(self, seen, allowed_features=None):
        """【敏感度引导退火 / 采样】选择新位置的统一入口（含退火）：
            - guided_ratio 概率从候选池均匀随机选（排除 seen）
            - (1 - guided_ratio) 概率从全部 seq_len × |allowed_features| 位置随机选（排除 seen）
        guided_ratio 随代数退火：前 30% 代保持 guided_ratio_start，后 70% 代线性衰减到 guided_ratio_end。
        参数：
            seen: 已占用位置集合 set of (t,f)，采样时排除
            allowed_features: 允许扰动的特征索引（None / 空 表示全部特征）
        返回：(t, f) 二元组（int）
        """
        # ---- 退火计算 ----
        denom = max(int(self.maxiter), 1)
        gen_ratio = float(self.current_gen) / denom
        if gen_ratio < 0.3:
            guided_ratio = self.guided_ratio_start
        else:
            decay_progress = (gen_ratio - 0.3) / 0.7
            guided_ratio = self.guided_ratio_start - \
                decay_progress * (self.guided_ratio_start - self.guided_ratio_end)
    
        if allowed_features is not None and len(allowed_features) > 0:
            allowed_arr = np.asarray(allowed_features, dtype=int).flatten()
        else:
            allowed_arr = np.arange(self.num_features, dtype=int)
        if allowed_arr.size == 0 or self.seq_len <= 0:
            return (0, 0)
    
        # ---- 引导采样：从候选池均匀随机选（排除 seen）----
        if np.random.random() < guided_ratio and self.candidate_pool:
            pool = [(t, f) for (t, f) in self.candidate_pool if (t, f) not in seen]
            if pool:
                idx = int(np.random.randint(0, len(pool)))
                return (int(pool[idx][0]), int(pool[idx][1]))
    
        # ---- 随机采样：从全部位置随机选（限定 allowed_features、排除 seen）----
        for _ in range(200):
            t = int(np.random.randint(0, self.seq_len))
            f = int(np.random.choice(allowed_arr))
            if (t, f) not in seen:
                return (t, f)
        # 兜底：位置空间被占满时随机返回一个（交由调用方的 _repair_unique 合并重复幅值）
        t = int(np.random.randint(0, self.seq_len))
        f = int(np.random.choice(allowed_arr))
        return (t, f)
    
    def _init_individual(self, seq_len, num_features, allowed_features, ranges):
        """【变长编码】初始化一个个体，输出长度为 3n 的一维向量
        步骤:
          1) 先从 [n_min, n_max] 中均匀随机采样一个整数 n
          2) 再生成 n 个互不重复的 (t,f) 位置，每个位置按动态预算 epsilon*range[f] 采样幅值 p
             【敏感度引导退火】位置采样由 _sample_new_position 接管
             （前期高比例从敏感度 top-K 候选池选，后期退火到弱引导）。
          3) 拼接为 [t1,f1,p1, ..., tn,fn,pn]
        """
        n = int(np.random.randint(self.n_min, self.n_max + 1))   # 变长：逐个体随机采样扰动点数
        positions = set()
        ind = []
        max_attempts = n * 100
        attempts = 0
        while len(positions) < n and attempts < max_attempts:
            # 【敏感度引导退火】新位置由候选池引导 + 随机回退采样（排除已占用位置）
            t_new, f_new = self._sample_new_position(positions, allowed_features)
            t = int(t_new)
            f = int(f_new)
            pos_key = (t, f)
            if pos_key not in positions:
                positions.add(pos_key)
                budget = self.epsilon * ranges[f]
                if budget > 0:
                    values = np.linspace(-budget, budget, P_DISCRETE_LEVELS)
                    p = float(np.random.choice(values))
                else:
                    p = 0.0
                ind.extend([float(t), float(f), p])
            attempts += 1
        # 位置空间不足时兜底补齐，保证输出长度恒为 3n
        while len(ind) // 3 < n:
            new_point, pos_key = self._random_new_point(seq_len, allowed_features, ranges, positions)
            positions.add(pos_key)
            ind.extend(new_point)
        return np.array(ind, dtype=float)

    def _apply_perturbation(self, x_np, individual):
        """将个体编码的扰动应用到输入样本上（检测重复 (t,f) 并合并幅值）
        交叉/变异后两个三元组可能落到同一位置，此处合并幅值避免重复计数，
        保证 L0 稀疏性诚实反映。
        【变长编码适配】扰动点数 n 由 len(individual)//3 反解，其余逻辑不变。
        """
        x_adv = x_np.copy()
        seq_len = x_np.shape[1]
        num_features = x_np.shape[2]
        n_points = self._num_points(individual)   # 变长：n 随个体长度变化

        # 检测重复位置并合并幅值
        perturbation_map = {}
        for i in range(n_points):
            t = int(np.clip(round(individual[3 * i]), 0, seq_len - 1))
            f = int(np.clip(round(individual[3 * i + 1]), 0, num_features - 1))
            p = individual[3 * i + 2]
            key = (t, f)
            if key in perturbation_map:
                perturbation_map[key] += p  # 合并幅值
            else:
                perturbation_map[key] = p

        # 应用合并后的扰动
        for (t, f), p in perturbation_map.items():
            x_adv[0, t, f] += p

        return x_adv

    def _evaluate_objectives(self, x_np, y_val, individual, device):
        """【双目标 + L0 约束】计算目标向量与约束变量，返回 (obj, n_actual)：
        obj: shape=(2,)，两个目标全部为最小化方向
            目标1: 扰动 L₂ 范数（从合并后的 perturbation_map 计算，和 _apply_perturbation 口径一致）
            目标2: -RSE（RSE = sqrt(攻击后MSE / 干净预测MSE)，取负号后攻击越强目标值越小）
        n_actual: 实际扰动点数（用 perturbation_map 的真实键数，即去重后的 L0 范数，
               不直接用编码长度反解的 n，避免重复位置虚增点数）。
               【L0 约束】不再作为第三目标进入 obj，而是作为约束变量随 obj 一并返回，
               用于违约度计算、knee tie-break 与前沿记录的附带属性。
        """
        x_adv = self._apply_perturbation(x_np, individual)
        # 【查询计数】每次评估只做 1 次 adv 前向并计入 query_count（clean 已在 attack 开头缓存）
        with torch.no_grad():
            pred_adv = self.model(torch.FloatTensor(x_adv).to(device)).item()
        self.query_count += 1
        # 【clean 缓存】attack 开头对同一 x_np 计算并缓存一次 clean 预测（eval 模式确定性，逐位相同）；
        #              缓存缺失（如单元测试直接调用本方法）时回退现算并计数，保证数值与原实现等价
        if self._clean_pred_cache is not None:
            pred_clean = self._clean_pred_cache
        else:
            with torch.no_grad():
                pred_clean = self.model(torch.FloatTensor(x_np).to(device)).item()
            self.query_count += 1

        # 目标1: 扰动 L₂ 范数（重复位置幅值相加后平方，与扰动施加口径一致）
        seq_len = x_np.shape[1]
        num_features = x_np.shape[2]
        n_points = self._num_points(individual)   # 变长：n 由个体长度反解
        perturbation_map = {}
        for i in range(n_points):
            t = int(np.clip(round(individual[3 * i]), 0, seq_len - 1))
            f = int(np.clip(round(individual[3 * i + 1]), 0, num_features - 1))
            p = individual[3 * i + 2]
            key = (t, f)
            if key in perturbation_map:
                perturbation_map[key] += p
            else:
                perturbation_map[key] = p
        l2_norm = np.sqrt(sum(p ** 2 for p in perturbation_map.values()))

        # 目标2: -RSE（RSE 越大攻击越强 → 取负号后越小越好，替代原双目标版的 1/RSE）
        adv_mse = (pred_adv - y_val) ** 2
        clean_mse = (pred_clean - y_val) ** 2
        rse = np.sqrt(adv_mse / clean_mse) if clean_mse > 1e-10 else 10.0
        neg_rse = -rse

        # 约束变量: n_actual 实际扰动点数（去重后 perturbation_map 的真实键数 = L0 范数）
        n_actual = float(len(perturbation_map))

        return np.array([l2_norm, neg_rse]), n_actual

    def _constraint_violation(self, n_actual_arr):
        """【L0 约束】计算违约度 cv = max(0, n_actual - n_max) + max(0, n_min - n_actual)
        cv = 0 表示个体可行（n_actual ∈ [n_min, n_max]）；cv 越大违约越严重。
        变长交叉/变异本就把点数控制在 [n_min, n_max]，因此绝大多数个体可行；
        违约度主要用于 Deb 约束支配中不可行个体之间的比较。
        """
        n = np.asarray(n_actual_arr, dtype=float)
        return np.maximum(0.0, n - self.n_max) + np.maximum(0.0, self.n_min - n)

    @staticmethod
    def _constr_dominates(obj_i, cv_i, obj_j, cv_j):
        """【约束支配判定 / 公共改造】Deb constrained-domination：判定个体 i 是否约束支配个体 j。
        提取自 _fast_non_dominated_sort 内部闭包，供排序、算子1判父代优劣、算子3/8记功统一复用，
        禁止各处另写一套。参数：obj_i/obj_j shape=(2,) 二维目标 [L2,-RSE]；cv_i/cv_j 标量违约度。
        规则：(a) i 可行(cv_i<=0) 而 j 不可行(cv_j>0) → True；
              (b) 都可行 → 二维标准 Pareto 支配（all(obj_i<=obj_j) 且 any(obj_i<obj_j)）；
              (c) 都不可行 → cv_i < cv_j；(d) i 不可行而 j 可行 → False。
        """
        fi, fj = cv_i <= 0.0, cv_j <= 0.0
        if fi and not fj:
            return True                                    # (a) 可行支配不可行
        if fi and fj:
            # (b) 都可行：二维目标上的标准 Pareto 支配
            return bool(np.all(obj_i <= obj_j) and np.any(obj_i < obj_j))
        if not fi and not fj:
            return bool(cv_i < cv_j)                        # (c) 都不可行：违约度小者胜
        return False                                        # (d) i 不可行而 j 可行：必不支配

    @staticmethod
    def _fast_non_dominated_sort(objectives, cv):
        """快速非支配排序（最小化问题，Deb 约束支配版）
        【动机】目标向量由三维 (L2, -RSE, n_actual) 改回二维 (L2, -RSE)：
            n_actual 不再作为第三目标，而是以违约度 cv 的形式进入约束支配。
            真正缓解“第一前沿过早吞没种群”的是删除冗余的第三目标维度
            （三目标下几乎人人互不支配，front0 迅速≈pop_size）；
            变长交叉/变异本就把点数控制在 [n_min, n_max]，绝大多数个体可行，
            引入约束支配是为了框架正确并防止任何超长个体。
        约束支配规则（Deb constrained-domination）：个体 i 约束支配 j 当且仅当下列之一：
            (a) i 可行(cv_i=0) 而 j 不可行(cv_j>0)；
            (b) i、j 都可行，且 i 在二维目标上标准 Pareto 支配 j
                （all(obj_i<=obj_j) 且 any(obj_i<obj_j)）；
            (c) i、j 都不可行，且 cv_i < cv_j。
        分层递推流程与标准 NSGA-II 完全一致，只把两两比较替换为上述规则。
        参数：objectives shape=(N,2) 目标值矩阵；cv shape=(N,) 违约度向量
        返回值：(fronts, rank)
            fronts: list of list，每层 Pareto 前沿的索引列表
            rank: array，每个个体的非支配层级（0 为最优）
        """
        pop_size = objectives.shape[0]
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_set = [[] for _ in range(pop_size)]

        # 【公共改造】两两约束支配判定统一复用静态方法 _constr_dominates（规则 a/b/c）
        dominates = NVITA_NSGA2._constr_dominates
        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                if dominates(objectives[i], cv[i], objectives[j], cv[j]):
                    dominated_set[i].append(j)
                elif dominates(objectives[j], cv[j], objectives[i], cv[i]):
                    domination_count[i] += 1

        fronts = []
        current_front = [i for i in range(pop_size) if domination_count[i] == 0]
        rank = np.full(pop_size, -1)

        front_level = 0
        while current_front:
            fronts.append(current_front)
            for i in current_front:
                rank[i] = front_level
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_level += 1

        return fronts, rank

    @staticmethod
    def _crowding_distance(objectives, fronts):
        """拥挤距离计算
        每个目标按自身值域做归一化，避免两个目标量级差异（L2~1e-2、-RSE~1e0）
        导致某一目标主导选择结果；边界点距离设为无穷大，优先保留边界解
        说明：按 n_obj = objectives.shape[1] 循环实现，双目标下 n_obj=2；
              n_actual 为约束变量不参与拥挤距离
        """
        pop_size = objectives.shape[0]
        n_obj = objectives.shape[1]
        distance = np.zeros(pop_size)

        for front in fronts:
            if len(front) <= 2:
                distance[front] = np.inf
                continue
            front_size = len(front)
            front_obj = objectives[front]
            obj_ranges = np.max(front_obj, axis=0) - np.min(front_obj, axis=0)
            obj_ranges[obj_ranges < 1e-10] = 1.0

            for m in range(n_obj):
                front_arr = np.array(front)
                sorted_indices = front_arr[np.argsort(front_obj[:, m])]
                distance[sorted_indices[0]] = np.inf
                distance[sorted_indices[-1]] = np.inf
                for i in range(1, front_size - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    distance[idx] += (objectives[next_idx, m] - objectives[prev_idx, m]) / obj_ranges[m]

        return distance

    @staticmethod
    def _select_knee(front_obj, select_mode='knee', knee_min_rse=2.0,
                     n_min=1, n_max=10, l2_weight=0.3, rse_weight=0.7,
                     select_strategy='knee', front_percentile=0.6):
        """【纯重构抽取】从末代第一前沿目标矩阵中选出最终解，返回其在 front_obj 内的行索引。

        与 attack 内联选择完全等价：可行性过滤（cv = max(0,n-n_max)+max(0,n_min-n)）→
        knee_min_rse 有效性过滤 → 弱前沿回退到攻击最强端点 → 在候选集上按 select_strategy 选点；
        max_rse 模式（select_mode != 'knee'）取 -RSE 最小（RSE 最大）端点。
        front_obj: shape=(k,2) 或 (k,3)；前两列 = [L2, -RSE]；第 3 列（若有）= n_actual 附带属性。
        n_min / n_max: L0 约束边界，用于计算违约度 cv（与 _constraint_violation 口径一致）。
        select_strategy: 候选集选点策略二选一（共用前面的可行性/有效性过滤）：
            'knee'       —— 二维归一化后距加权参考点 (l2_weight, rse_weight) 最近，n_actual / L2 两级 tie-break；
            'percentile' —— 候选解按原始 L2 升序排序，取第 int(k * front_percentile) 个点（k<2 取中间点）。
        l2_weight / rse_weight: knee 策略参考点坐标（均 0~1）：
            (0,0)=距原点最近（偏理想点）/ (0.5,0.5)=正中 / rse_weight 越大越偏攻击效果一侧。
        front_percentile: percentile 策略的分位数位置（0~1，默认 0.6）。
        返回：int，选中行在 front_obj 内的索引（0-based）；空前沿返回 0。
        """
        front_obj = np.asarray(front_obj, dtype=float)
        k = front_obj.shape[0]
        if k == 0:
            return 0
        if select_mode != 'knee':
            return int(np.argmin(front_obj[:, 1]))

        # ---- knee 模式 ----
        # 违约度计算（与 _constraint_violation 口径一致）
        if front_obj.shape[1] >= 3:
            n_vals = front_obj[:, 2]
            cv = np.maximum(0.0, n_vals - n_max) + np.maximum(0.0, n_min - n_vals)
        else:
            cv = np.zeros(k)
        feasible_mask = cv <= 0.0

        if not np.any(feasible_mask):
            # 极端回退（正常配置下不触发）：取 -RSE 最小者
            return 0

        fea_obj = front_obj[feasible_mask]
        if front_obj.shape[1] >= 3:
            fea_n = n_vals[feasible_mask]
        else:
            fea_n = np.zeros(int(feasible_mask.sum()))

        cand_mask = fea_obj[:, 1] <= -knee_min_rse
        if np.any(cand_mask):
            cand_obj = fea_obj[cand_mask]
            cand_n = fea_n[cand_mask]
        else:
            # 弱前沿回退：可行解中攻击最强端点
            end_pos = int(np.argmin(fea_obj[:, 1]))
            cand_obj = fea_obj[[end_pos]]
            cand_n = fea_n[[end_pos]]

        # ---- 在候选集上按 select_strategy 选点（两种策略共用前面的过滤逻辑）----
        if select_strategy == 'percentile':
            # percentile 策略：候选解按原始 L2 升序排序，取第 int(k * front_percentile) 个点
            #   （不需要归一化；k < 2 时直接取中间点即 index 0）
            k_cand = cand_obj.shape[0]
            if k_cand < 2:
                best_in_cand = 0
            else:
                order = np.argsort(cand_obj[:, 0], kind='stable')   # 按 L2 升序
                p_idx = int(k_cand * front_percentile)
                p_idx = min(max(p_idx, 0), k_cand - 1)              # 夹到合法索引范围
                best_in_cand = int(order[p_idx])
        else:
            # knee 策略：二维归一化（按候选集每维最大绝对值缩放到 (0,1]）后距加权参考点最近
            denom = np.max(np.abs(cand_obj), axis=0)
            denom[denom < 1e-12] = 1.0
            normed = np.abs(cand_obj) / denom
            #   只用前 2 列 [L2, -RSE] 归一化值参与距离计算（n_actual 仅作 tie-break）
            _knee_ref = np.array([l2_weight, rse_weight])
            dist = np.sqrt(np.sum((normed[:, :2] - _knee_ref) ** 2, axis=1))
            #   tie-break（容差 1e-8）：距离相等 → n_actual 更小 → L2 更小
            ties = np.where(dist <= dist.min() + 1e-8)[0]
            tie_n = cand_n[ties]
            min_n_pos = np.where(tie_n <= tie_n.min() + 1e-8)[0]
            best_in_cand = ties[min_n_pos[int(np.argmin(cand_obj[ties][min_n_pos, 0]))]]

        # 映射回 front_obj 行索引
        feasible_indices = np.where(feasible_mask)[0]
        if np.any(cand_mask):
            cand_indices = feasible_indices[cand_mask]
        else:
            cand_indices = feasible_indices[[int(np.argmin(fea_obj[:, 1]))]]
        return int(cand_indices[best_in_cand])

    @staticmethod
    def _tournament_selection(population, objectives, rank, crowding, pop_size):
        """二元锦标赛选择
        第一优先级：非支配层级 rank（越小越优）
        第二优先级：拥挤距离（越大越优，维持种群多样性）
        【变长编码适配】仅把种群容器从 ndarray 换成 list（个体长度不一，无法堆成矩形矩阵），
        选择判据只依赖 rank/crowding，与编码长度和目标维度均无关，逻辑完全不变
        返回：selected（list，选中个体副本）
        """
        n = len(population)
        selected = []
        while len(selected) < pop_size:
            i, j = np.random.choice(n, 2, replace=False)
            if rank[i] < rank[j]:
                chosen = i
            elif rank[j] < rank[i]:
                chosen = j
            elif crowding[i] > crowding[j]:
                chosen = i
            else:
                chosen = j
            selected.append(population[chosen].copy())
        return selected

    def _clip_tf_budget(self, c, seq_len, num_features, ranges):
        """统一的合法性修正：t/f 取整截断，p 按动态预算截断
        【变长编码适配】循环上界由固定 self.n 改为 len(c)//3，逻辑不变
        """
        n_points = self._num_points(c)
        for k in range(n_points):
            c[3*k]   = float(np.clip(round(c[3*k]),   0, seq_len - 1))
            c[3*k+1] = float(np.clip(round(c[3*k+1]), 0, num_features - 1))
            f_idx = int(np.clip(round(c[3*k+1]), 0, num_features - 1))
            bud = self.epsilon * ranges[f_idx]
            c[3*k+2] = self._quantize_p(c[3*k+2], bud)
        return c

    def _repair_unique(self, c, seq_len, allowed_features):
        """【变长编码】位置去重：合并重复 (t,f) 位置上的幅值
        固定长度版通过「挪动 t/f」保证 n 个位置互不重复；变长版改为直接合并重复位置的幅值，
        因此去重后个体长度可能缩短（n 减小），随后由 _enforce_length_bounds 拉回 [n_min, n_max]。
        注意：返回值长度可能变化，调用方必须接收返回值而不是原地使用 c。
        allowed_features 形参保留以兼容原调用签名（本实现不再需要挪动 f）。
        """
        c = np.asarray(c, dtype=float)
        n_points = self._num_points(c)
        merged, order = {}, []
        for k in range(n_points):
            t = int(np.clip(round(c[3*k]), 0, seq_len - 1))
            f = int(round(c[3*k+1]))
            key = (t, f)
            if key in merged:
                merged[key] += float(c[3*k+2])   # 重复位置幅值相加
            else:
                merged[key] = float(c[3*k+2])
                order.append(key)                # 保留首次出现顺序，维持基因相对次序
        out = []
        for key in order:
            out.extend([float(key[0]), float(key[1]), merged[key]])
        return np.array(out, dtype=float)

    def _enforce_length_bounds(self, c, seq_len, num_features, ranges, allowed_features):
        """【变长编码 / 方案A】长度合法性修复：仅截断超长到 n_max，不再补齐到 n_min
        1) 长度截断（硬保护）：若 n > n_max，随机删除扰动点直到 n == n_max，个体长度永不超过 n_max
        2)【方案A 移除长度补齐】不再把 n < n_min 的个体补回 n_min：允许 n < n_min 的不可行个体
           （含 n=0 的空个体 shape=(0,)）进入种群，由 Deb 约束支配（可行优先）惩罚，
           使 L0 约束真正在进化中起筛选作用（此前 cv 恒为 0、约束非绑定）。
        返回值可能是空数组 shape=(0,)；截断中 kept 为空时返回空数组而非报错。
        seq_len / num_features / ranges / allowed_features 形参保留以兼容原调用签名（截断不再使用它们）。
        """
        c = np.asarray(c, dtype=float)
        n_points = self._num_points(c)

        # ---- 长度截断（硬保护：n 不得超过 n_max）----
        while n_points > self.n_max:
            drop = int(np.random.randint(0, n_points))
            kept = [c[3*k:3*k+3] for k in range(n_points) if k != drop]
            c = np.concatenate(kept).astype(float) if kept else np.array([], dtype=float)
            n_points -= 1

        return c

    def _crossover(self, parent1, parent2, seq_len, num_features, ranges, allowed_features):
        """【变长编码】变长单点交叉 + 三步合法性修复
        1) 两个父代长度可不同（3n1 / 3n2），分别在各自的三元组边界上随机选一个交叉点，
           交换尾部片段生成两个子代（子代长度因此天然可变）
        2) 位置去重：_repair_unique 合并重复 (t,f) 位置的幅值（n 可能减小）
        3) 长度截断（方案A：仅截断超长到 n_max，不补齐到 n_min）：_enforce_length_bounds
        4) 预算截断：_clip_tf_budget 按最终 f 的动态预算截断幅值
           （先去重再截断：修复过程可能改变 f，须按新 f 的预算重截 p）
        """
        # 【交叉概率门控】随机数 > crossover_prob 则不交叉，直接返回父代副本
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        return self._crossover_single(parent1, parent2, seq_len, num_features, ranges, allowed_features)

    def _crossover_single(self, parent1, parent2, seq_len, num_features, ranges, allowed_features):
        """【变长单点交叉】两个父代在各自三元组边界随机选交叉点、交换尾部片段生成两个变长子代，
        随后统一执行位置去重 + 长度截断 + 预算截断。
        """
        parent1 = np.asarray(parent1, dtype=float)
        parent2 = np.asarray(parent2, dtype=float)
        n1 = self._num_points(parent1)
        n2 = self._num_points(parent2)

        # 交叉点只能落在三元组边界 3k 上，k ∈ [0, n]：k=0 表示整条交给对方，k=n 表示保留整条
        k1 = int(np.random.randint(0, n1 + 1))
        k2 = int(np.random.randint(0, n2 + 1))
        head1, tail1 = parent1[:3*k1].copy(), parent1[3*k1:].copy()
        head2, tail2 = parent2[:3*k2].copy(), parent2[3*k2:].copy()
        # 交换尾部片段 → 两个变长子代
        c1 = np.concatenate([head1, tail2]).astype(float)
        c2 = np.concatenate([head2, tail1]).astype(float)

        repaired = []
        for c in (c1, c2):
            c = self._repair_unique(c, seq_len, allowed_features)                                   # 位置去重
            c = self._enforce_length_bounds(c, seq_len, num_features, ranges, allowed_features)     # 长度截断（方案A：仅截断，不补齐）
            c = self._clip_tf_budget(c, seq_len, num_features, ranges)                              # 预算截断
            repaired.append(c)
        return repaired[0], repaired[1]

    def _mutation(self, individual, seq_len, num_features, ranges, allowed_features, gen=0):
        """【变长编码】变异协同（幅值变异 + SA-MOO 位置替换 + 插入/删除，各概率相互独立）
        1) 幅值变异（逐三元组独立触发，触发率 1/n，完全保留原离散档位逻辑）：
           50% 随机重置到任意离散档，50% 邻域 ±1 移动
        2) SA-MOO 位置替换变异（概率 0.25，替代原 t 邻域±1 + f 随机）：
           每次替换 n_replace 个已有 (t,f) 位置，n_replace 随代数 gen 衰减（前期探索、后期精修）；
           新位置由敏感度引导退火接管（_sample_new_position：候选池引导 + 随机回退）
        3) 插入变异（概率 self.insert_prob，默认 0.1）：新增一个不重复 (t,f,p) 三元组，n+1 且不超过 n_max；
           新位置同样由敏感度引导退火接管
        4) 删除变异（概率 self.delete_prob，默认 0.2）：随机删除一个三元组，n-1；
           【方案A】不再限制 n_min 下限，允许删到 0（产生不可行个体，交由 Deb 约束支配惩罚）
        变异后统一执行 _repair_unique + _enforce_length_bounds + _clip_tf_budget 合法性修复
        gen: 当前进化代数（用于 SA-MOO 替换比例衰减），默认 0
        """
        mutant = np.asarray(individual, dtype=float).copy()
        n_points = self._num_points(mutant)

        # ---------- 1) 幅值变异：逐三元组独立触发（t/f 位置变异已移至下方 SA-MOO 位置替换）----------
        for k in range(n_points):
            b = 3 * k
            # 扰动值 p：离散值变异（20 档位整数编码）
            # 触发率沿用原逻辑 1/n；50% 随机重置到任意离散档，50% 邻域 ±1 移动
            if np.random.random() < 1.0 / max(n_points, 1):
                f_idx = int(np.clip(round(mutant[b+1]), 0, num_features - 1))
                bud = self.epsilon * ranges[f_idx]
                if bud < 1e-12:
                    mutant[b+2] = 0.0
                else:
                    values = np.linspace(-bud, bud, P_DISCRETE_LEVELS)
                    cur_idx = int(np.clip(
                        np.round((mutant[b+2] + bud) / (2 * bud) * (P_DISCRETE_LEVELS - 1)),
                        0, P_DISCRETE_LEVELS - 1))
                    if np.random.random() < 0.5:
                        mutant[b+2] = float(np.random.choice(values))  # 随机重置
                    else:
                        delta = int(np.random.choice([-1, 1]))           # 邻域 ±1
                        new_idx = int(np.clip(cur_idx + delta, 0, P_DISCRETE_LEVELS - 1))
                        mutant[b+2] = float(values[new_idx])

        # ---------- 2) SA-MOO 位置替换变异 + 敏感度引导退火（替代原 t 邻域±1 + f 随机）----------
        if np.random.random() < 0.25 and n_points >= 1:
            # SA-MOO 动态替换比例：随代数 gen 衰减（前期大步探索，后期小步精修）
            p_m_ratio = max(0.1, 1.0 - 0.9 * (gen / max(self.maxiter, 1)))
            n_replace = max(1, int(n_points * 0.3 * p_m_ratio))
            n_replace = min(n_replace, n_points)

            existing_pos = set()
            for k in range(n_points):
                existing_pos.add((int(round(mutant[3*k])), int(round(mutant[3*k+1]))))

            for _ in range(n_replace):
                # SA-MOO：随机选一个要被替换的已有位置 A
                replace_idx = int(np.random.randint(0, n_points))
                t_old = int(round(mutant[3*replace_idx]))
                f_old = int(round(mutant[3*replace_idx+1]))
                # 【敏感度引导退火】新位置 B 由 _sample_new_position 接管：
                # seen 排除现有占用位置，但 (t_old, f_old) 作为被替换位置允许重选
                seen = set(existing_pos)
                seen.discard((t_old, f_old))
                t_new, f_new = self._sample_new_position(seen, allowed_features)
                t_new, f_new = int(t_new), int(f_new)
                # SA-MOO：A 移除、B 加入（仅当 B 不与现有位置重复）
                if (t_new, f_new) not in existing_pos:
                    mutant[3*replace_idx] = float(t_new)
                    mutant[3*replace_idx+1] = float(f_new)
                    # SA-MOO 扰动值迁移：50% 保留原值，50% 按新特征预算重采样离散档
                    if np.random.random() < 0.5:
                        bud = self.epsilon * ranges[f_new]
                        values = np.linspace(-bud, bud, P_DISCRETE_LEVELS)
                        mutant[3*replace_idx+2] = float(np.random.choice(values))
                    existing_pos.discard((t_old, f_old))
                    existing_pos.add((t_new, f_new))

        # ---------- 3) 插入变异：n+1（不超过 n_max），新位置由敏感度引导退火接管 ----------
        if np.random.random() < self.insert_prob and n_points < self.n_max:
            existing_pos = set()
            for k in range(n_points):
                existing_pos.add((int(round(mutant[3*k])), int(round(mutant[3*k+1]))))
            # 【敏感度引导退火】新位置由 _sample_new_position 接管（seen 排除现有占用位置）
            t_new, f_new = self._sample_new_position(existing_pos, allowed_features)
            t_new, f_new = int(t_new), int(f_new)
            if (t_new, f_new) not in existing_pos:
                bud = self.epsilon * ranges[f_new]
                values = np.linspace(-bud, bud, P_DISCRETE_LEVELS)
                p_new = float(np.random.choice(values))
                mutant = np.concatenate([mutant, [float(t_new), float(f_new), p_new]])
                n_points += 1
        # ---------- 4) 删除变异：n-1（【方案A】不再限制 n_min 下限，允许删到 0）----------
        elif np.random.random() < self.delete_prob and n_points > 0:
            drop = int(np.random.randint(0, n_points))
            kept = [mutant[3*k:3*k+3] for k in range(n_points) if k != drop]
            # 【方案A】kept 可能为空（n_points==1 删到 0），返回空数组而非 np.concatenate([]) 报错
            mutant = np.concatenate(kept).astype(float) if kept else np.array([], dtype=float)
            n_points -= 1

        # ---------- 统一合法性修复 ----------
        mutant = self._repair_unique(mutant, seq_len, allowed_features)
        # 【方案A】_enforce_length_bounds 现仅截断超长到 n_max，不再补齐到 n_min；
        #          点级变异/去重合并可能使 n 跌破 n_min，此类不可行个体被保留，交由 Deb 约束支配惩罚
        mutant = self._enforce_length_bounds(mutant, seq_len, num_features, ranges, allowed_features)
        mutant = self._clip_tf_budget(mutant, seq_len, num_features, ranges)
        return mutant

    def attack(self, X, y, seed=None):
        """
        对单个样本执行 NSGA-II 双目标变长攻击（L0 作为约束）

        返回值:
            X_adv: 按 select_mode 选出的对抗样本（'knee'=折中膝点，'max_rse'=攻击上界端点）
            best_mse: 对应的攻击后 MSE
            pareto_solutions: 最终第一 Pareto 前沿解集（【变长编码】list of 一维向量，长度可各不相同）
            pareto_objectives: 最终前沿对应的目标值矩阵，shape=(k, 2)【双目标 [L2, -RSE]】
            generation_pareto: 每一代第一 Pareto 前沿记录列表，
                               每条 shape=(m, 3) = [L2, -RSE, n_actual(附带属性，不参与支配)]
        """
        if seed is not None:
            np.random.seed(seed)

        device = X.device
        x_np = X.detach().cpu().numpy()
        y_val = y.detach().cpu().item()
        seq_len = x_np.shape[1]
        num_features = x_np.shape[2]

        # 【敏感度引导退火】为当前样本注入上下文，供 _compute_sensitivity / _sample_new_position 使用
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.device = device
        self.current_gen = 0
        self.candidate_pool = []

        # 【查询计数 / 过程诊断】每样本开头重置累计查询数、首次成功查询数与诊断计数器
        self.query_count = 0
        self.first_success_query = -1
        self.diag = {'trunc_len_cnt': 0, 'trunc_len_tot': 0,      # 长度截断（n>n_max）触发次数 / 总修复个体数
                     'budget_clip_cnt': 0, 'budget_clip_tot': 0,  # 预算截断改变 p 的点数 / 总点数
                     'empty_after_cross': 0, 'empty_after_mut': 0, 'cross_mut_tot': 0}  # 空个体率

        allowed_features = self._sample_features(num_features)
        ranges = self._get_window_range(x_np)

        # 没有允许扰动的特征，直接返回原始输入
        if len(allowed_features) == 0:
            empty_pareto = np.array([]).reshape(0, 2)   # 双目标：空前沿列数为 2
            self._clean_pred_cache = None
            return X, 0.0, [], empty_pareto, [], {'query_count': 0, 'first_success_query': -1}

        # 【clean 缓存】对当前样本 x_np 只查一次 clean 预测并缓存（eval 确定性，_evaluate_objectives 复用；
        #               model 前向不消耗 np.random 流，种群初始化随机流与原实现逐位一致）
        with torch.no_grad():
            self._clean_pred_cache = self.model(torch.FloatTensor(x_np).to(device)).item()
        self.query_count += 1

        # ========== 【敏感度引导退火】攻击前探测：seq_len*num_features 次前向查询 → top-K 候选池 ==========
        # 额外查询成本 = seq_len*num_features（默认 20×4=80），
        # 相对进化阶段 pop_size*maxiter*~1.5 次查询仅增加 ~1.3%。
        # 预算口径与 NSGA-II 一致（ranges = self._get_window_range(x_np)，use_window_range=False 时为全局 range_f）。
        sensitivity = self._compute_sensitivity(
            x_np[0], self._clean_pred_cache, ranges, allowed_features)
        self.candidate_pool = self._build_candidate_pool(
            sensitivity, self.candidate_pool_size)
        self.current_gen = 0
        # === 新增：输出可视化优化 — 保存敏感度矩阵供后续分析（敏感度 vs 选中频次对比图） ===
        self.last_sensitivity = sensitivity.copy()

        # 【first_success_query 跟踪】封装评估：调用 _evaluate_objectives 后检查是否首次达到 RSE>=knee_min_rse
        def _eval_and_track(ind):
            obj, n_act = self._evaluate_objectives(x_np, y_val, ind, device)
            if self.first_success_query < 0 and (-obj[1]) >= self.knee_min_rse:
                self.first_success_query = self.query_count
            return obj, n_act

        # ========== 1. 种群初始化（【变长编码】每个个体的 n 独立随机采样，种群用 list 承载）==========
        population = [
            self._init_individual(seq_len, num_features, allowed_features, ranges)
            for _ in range(self.pop_size)
        ]

        # ========== 2. 初始种群评估双目标 + 约束变量 ==========
        objectives = np.zeros((self.pop_size, 2))          # 双目标 [L2, -RSE]，形状恒为 (N, 2)
        n_actual_pop = np.zeros(self.pop_size)             # 约束变量 n_actual（不进入 objectives）
        for i in range(self.pop_size):
            objectives[i], n_actual_pop[i] = _eval_and_track(population[i])

        # ========== 3. 每代 Pareto 前沿记录 ==========
        generation_pareto = []

        # ========== 4. NSGA-II 主循环 ==========
        for gen in range(self.maxiter):
            self.current_gen = gen   # 【敏感度引导退火】供 _sample_new_position 计算当前引导比例
            # 非支配排序（Deb 约束支配：可行优先 → 二维目标支配 → 违约度比较）
            cv_pop = self._constraint_violation(n_actual_pop)
            fronts, rank = self._fast_non_dominated_sort(objectives, cv_pop)

            # 记录当前代的第一 Pareto 前沿（前两列为双目标，第三列携带 n_actual 附带属性，仅供下游统计/着色）
            if len(fronts) > 0:
                gen_front = np.column_stack([objectives[fronts[0]], n_actual_pop[fronts[0]]])
                generation_pareto.append(gen_front.copy())

            # 拥挤距离（按 2 个目标的值域归一化，避免量级差异导致选择失衡；n_actual 不参与）
            crowding = self._crowding_distance(objectives, fronts)

            # 二元锦标赛选择
            mating_pool = self._tournament_selection(
                population, objectives, rank, crowding, self.pop_size
            )

            # 生成子代：变长单点交叉 + 点级变异（末尾均含合法性修正、位置去重与长度修复）
            offspring = []
            for i in range(0, self.pop_size, 2):
                j = min(i + 1, self.pop_size - 1)
                p1 = mating_pool[i]
                p2 = mating_pool[j]
                c1, c2 = self._crossover(p1, p2, seq_len, num_features, ranges, allowed_features)
                c1 = self._mutation(c1, seq_len, num_features, ranges, allowed_features, gen)
                c2 = self._mutation(c2, seq_len, num_features, ranges, allowed_features, gen)
                offspring.append(c1)
                offspring.append(c2)
            offspring = offspring[:self.pop_size]   # 【变长编码】保持 list 容器，不堆叠成矩形矩阵

            # 评估子代双目标 + 约束变量
            offspring_obj = np.zeros((len(offspring), 2))
            n_actual_off = np.zeros(len(offspring))
            for i in range(len(offspring)):
                offspring_obj[i], n_actual_off[i] = _eval_and_track(offspring[i])

            # μ+λ 环境选择：合并父代和子代（变长个体无法 vstack，改用 list 拼接）
            combined_pop = population + offspring
            combined_obj = np.vstack([objectives, offspring_obj])
            combined_n_actual = np.concatenate([n_actual_pop, n_actual_off])

            # 对合并后的种群做约束非支配排序 + 拥挤距离（均只按 2 个目标）
            cv_combined = self._constraint_violation(combined_n_actual)
            fronts_combined, rank_combined = self._fast_non_dominated_sort(combined_obj, cv_combined)
            crowding_combined = self._crowding_distance(combined_obj, fronts_combined)

            # 按「约束可行（已由约束支配编码进 rank）→ 非支配层级升序 → 同层多样性降序」选择前 pop_size 个
            selected_indices = []
            for front in fronts_combined:
                if len(selected_indices) + len(front) <= self.pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = self.pop_size - len(selected_indices)
                    front_crowding = crowding_combined[front]
                    front_arr = np.array(front)
                    sorted_front = front_arr[np.argsort(-front_crowding)]
                    selected_indices.extend(sorted_front[:remaining].tolist())
                    break

            population = [combined_pop[i] for i in selected_indices]
            objectives = combined_obj[selected_indices]
            n_actual_pop = combined_n_actual[selected_indices]

        # 【修复】不再重复追加 generation_pareto：for gen in range(maxiter) 的最后一次迭代
        # 开头已记录了该代前沿，再追加会导致多 1 条（绘图出现超出 maxiter 的代数）。
        # final_fronts 仍需计算，供后续膝点选择与最终前沿输出使用。
        cv_pop = self._constraint_violation(n_actual_pop)
        final_fronts, _ = self._fast_non_dominated_sort(objectives, cv_pop)

        # ========== 5. 选择最终解（按 select_mode 决定折中膝点或攻击上界端点）==========
        # 'knee'（默认）：从最终第一前沿的可行解（cv=0）中选择二维归一化后距理想点最近的折中解，
        #   【双目标】obj[1] = -RSE 为负值，归一化改为「按候选集每维最大绝对值缩放」，
        #   把 [L2, -RSE] 映射到 (0,1] 后计算距加权参考点 (l2_weight, rse_weight) 的欧氏距离，兼顾攻击效果与隐蔽性；
        #   稀疏性不再作为目标维，而是在距离打平时作为 tie-break（等优更稀疏）。
        #   有效性下界 knee_min_rse：仅 RSE>=knee_min_rse（即 -RSE<=-knee_min_rse）的前沿点参与膝点评选；
        #   弱前沿上若无有效点，回退到可行且攻击最强端点，保证结果不是近零扰动
        # 'max_rse'：取 RSE 最大端点（攻击强度上界，完全忽略 L2 隐蔽性与稀疏性），保留用于消融对比
        if len(final_fronts) > 0:
            front_idx = np.array(final_fronts[0])
            front_obj = objectives[front_idx]
            front_n = n_actual_pop[front_idx]
            # 构造含 n_actual 附带属性的前沿矩阵，供 _select_knee 内部计算违约度与 tie-break
            front_obj_with_n = np.column_stack([front_obj, front_n])

            # 膝点 / max_rse 选择：统一委托给 _select_knee 静态方法
            knee_pos = self._select_knee(front_obj_with_n, self.select_mode,
                                         self.knee_min_rse, self.n_min, self.n_max,
                                         l2_weight=self.knee_l2_weight,
                                         rse_weight=self.knee_rse_weight,
                                         select_strategy=self.select_strategy,
                                         front_percentile=self.front_percentile)
            best_idx = front_idx[knee_pos]
        else:
            best_idx = np.argmin(objectives[:, 1])  # 上界模式/回退：-RSE 最小即 RSE 最大
        best_ind = population[int(best_idx)].copy()
        x_adv_final = self._apply_perturbation(x_np, best_ind)
        with torch.no_grad():
            pred_best = self.model(torch.FloatTensor(x_adv_final).to(device)).item()
        self.query_count += 1   # 【查询计数】最终选中解复算 best_mse 的一次前向如实计入
        best_mse = (pred_best - y_val) ** 2

        # 最终第一 Pareto 前沿（【变长编码】解集用 list 承载；目标值矩阵恒为 (k, 2)）
        if len(final_fronts) > 0:
            pareto_solutions = [population[i].copy() for i in final_fronts[0]]
            pareto_objectives = objectives[final_fronts[0]].copy()
        else:
            pareto_solutions = []
            pareto_objectives = np.array([]).reshape(0, 2)

        # 【查询计数】返回本样本查询统计：总查询数 + 首次达标(RSE>=knee_min_rse)累计查询数（未达标 -1）
        query_info = {'query_count': int(self.query_count),
                      'first_success_query': int(self.first_success_query)}
        return (
            torch.FloatTensor(x_adv_final).to(device),
            float(best_mse),
            pareto_solutions,
            pareto_objectives,
            generation_pareto,
            query_info,
        )


# =============================================================================
# SECTION 5: NSGA-II 批量攻击与 Pareto 进化可视化
# =============================================================================


def extract_first_front(objs):
    """【双目标适配】提取二维最小化问题的第一 Pareto 前沿（O(N log N) 扫描法）
    objs 前两列为双目标 [L2, -RSE]；若带第 3 列（n_actual 附带属性），
    该列不参与支配判定，仅随前沿点一并携带返回。
    算法：按 obj0 升序排序后扫描，保留 obj1 严格优于当前最小值的点。
    """
    if objs.shape[0] == 0:
        n_col = objs.shape[1] if objs.ndim == 2 else 2
        return np.empty((0, n_col))
    o2 = objs[:, :2]
    order = np.argsort(o2[:, 0], kind='stable')
    keep = []
    best_y = np.inf
    for i in order:
        if o2[i, 1] < best_y:            # obj1 严格更优才保留（obj0 相等时只留首个）
            keep.append(i)
            best_y = o2[i, 1]
    if not keep:
        return np.empty((0, objs.shape[1]))
    return objs[np.array(keep, dtype=int)]


def run_nsga2_standalone_attack(model, X_test, Y_test, beta, n_min, n_max, maxiter, pop_size, device,
                                feature_ranges, feature_constraint=None, perturb_features=None,
                                print_info=False, select_mode='knee', knee_min_rse=2.0,
                                insert_prob=0.1, delete_prob=0.1, use_window_range=False,
                                knee_l2_weight=0.3, knee_rse_weight=0.7,
                                select_strategy='knee', front_percentile=0.6):
    """
    批量运行 NSGA-II 双目标变长攻击（L0 作为约束）

    对每个测试样本运行 NVITA_NSGA2 攻击，收集对抗样本、攻击指标和 Pareto 进化数据。
    【变长编码】原固定参数 n 拆分为 n_min / n_max，扰动点数在区间内逐个体动态变化
    【双目标】目标向量为 (L2 范数, -RSE)，形状恒为 (N, 2)；
             n_actual 为约束变量（要求 ∈ [n_min, n_max]），随前沿记录作为附带属性携带
    select_mode: 最终解选择策略（'knee'=折中膝点，默认 / 'max_rse'=攻击上界端点）
    knee_min_rse: 膝点有效性下界（仅 RSE>=该值的前沿点可作膝点候选）

    返回:
        X_adv_total: 所有对抗样本拼接结果
        metrics: 汇总指标字典（含双目标与约束变量 n_actual 的均值统计）
        all_generation_pareto: 每个样本的 Pareto 进化记录列表
    """
    model.to(device)
    model.eval()

    X_adv_total = torch.empty(0).to(device)
    all_clean_mse = []
    all_adv_mse = []
    all_l2_norms = []
    all_sel_l2 = []        # 选中解的目标1：L2 范数
    all_sel_rse = []       # 选中解的目标2还原量：RSE = -obj[1]
    all_sel_n_actual = []  # 选中解的约束变量：实际扰动点数（L0，要求 ∈ [n_min, n_max]）
    all_generation_pareto = []
    # ====== 【查询效率 / 开销】（第8点）逐样本收集总查询数、首次达标查询数、墙钟耗时 ======
    all_query_count = []
    all_first_success_query = []
    all_wall_time = []
    # === 新增：输出可视化优化 — 收集每样本敏感度矩阵，供“敏感度 vs 选中频次”对比图 ===
    all_sensitivities = []
    _attack_phase_t0 = time.perf_counter()   # 攻击阶段总计时

    attacker = NVITA_NSGA2(
        n_min=n_min, n_max=n_max, epsilon=beta, model=model,   # 变长编码：n_min / n_max
        feature_ranges=feature_ranges,
        maxiter=maxiter, pop_size=pop_size,
        feature_constraint=feature_constraint,
        perturb_features=perturb_features,
        select_mode=select_mode,
        knee_min_rse=knee_min_rse,
        knee_l2_weight=knee_l2_weight,
        knee_rse_weight=knee_rse_weight,
        select_strategy=select_strategy,
        front_percentile=front_percentile,
        insert_prob=insert_prob,
        delete_prob=delete_prob,
        use_window_range=use_window_range,
        # ===== 【敏感度引导退火】从 NSGA2_CONFIG 读取 3 个退火参数并传入构造函数 =====
        candidate_pool_size=NSGA2_CONFIG.get('candidate_pool_size', 20),
        guided_ratio_start=NSGA2_CONFIG.get('guided_ratio_start', 0.6),
        guided_ratio_end=NSGA2_CONFIG.get('guided_ratio_end', 0.3),
    )

    total = X_test.shape[0]
    for test_ind in range(total):
        X_current = X_test[test_ind].unsqueeze(0).to(device)
        y_current = Y_test[test_ind].unsqueeze(0).to(device)

        _t0 = time.perf_counter()   # 【开销】单样本墙钟计时
        X_adv, best_mse, pareto_solutions, pareto_objectives, generation_pareto, query_info = \
            attacker.attack(X_current, y_current, seed=test_ind)
        all_wall_time.append(time.perf_counter() - _t0)
        all_query_count.append(int(query_info.get('query_count', 0)))
        all_first_success_query.append(int(query_info.get('first_success_query', -1)))
        # === 新增：输出可视化优化 — 收集敏感度矩阵 ===
        if getattr(attacker, 'last_sensitivity', None) is not None:
            all_sensitivities.append(attacker.last_sensitivity.copy())

        X_adv_total = torch.cat((X_adv_total, X_adv), dim=0)

        # 计算干净预测 MSE
        with torch.no_grad():
            pred_clean = model(X_current).item()
        y_val = y_current.item()
        clean_mse = (pred_clean - y_val) ** 2

        all_clean_mse.append(clean_mse)
        all_adv_mse.append(best_mse)

        # 平均扰动 L2 范数（取 Pareto 前沿上所有解的平均值）
        if len(pareto_objectives) > 0:
            avg_l2 = np.mean(pareto_objectives[:, 0])
        else:
            avg_l2 = 0.0
        all_l2_norms.append(avg_l2)

        # 选中解的双目标与约束变量实测值（从对抗样本与原始样本的差值反推，口径与目标函数一致）
        diff = (X_adv - X_current).detach().cpu().numpy().reshape(-1)
        all_sel_l2.append(float(np.sqrt(np.sum(diff ** 2))))
        all_sel_n_actual.append(int(np.count_nonzero(np.abs(diff) > 1e-12)))
        all_sel_rse.append(float(np.sqrt(best_mse / clean_mse)) if clean_mse > 1e-10 else 0.0)

        # 保存 Pareto 进化数据（final_pareto_solutions 与 final_pareto_obj 同序，
        # 供高级分析模块解码全局前沿解的 (t,f) 扰动位置，用于位置频次热力图）
        all_generation_pareto.append({
            'sample_idx': test_ind,
            'generation_pareto': generation_pareto,
            'final_pareto_obj': pareto_objectives,
            'final_pareto_solutions': pareto_solutions,
        })

        if print_info and (test_ind + 1) % 20 == 0:
            print(f"  NSGA2 progress: {test_ind + 1}/{total}")

    metrics = {
        'mean_clean_mse': float(np.mean(all_clean_mse)),
        'mean_adv_mse': float(np.mean(all_adv_mse)),
        'mean_l2_norm': float(np.mean(all_l2_norms)),
        # ====== 双目标选中解统计 + 约束变量 n_actual 统计 ======
        'mean_sel_l2_norm': float(np.mean(all_sel_l2)) if all_sel_l2 else 0.0,              # 目标1
        'mean_sel_rse': float(np.mean(all_sel_rse)) if all_sel_rse else 0.0,                # 目标2（还原为 RSE）
        'mean_sel_n_actual': float(np.mean(all_sel_n_actual)) if all_sel_n_actual else 0.0,  # 约束变量（L0）
        # ====== 【L0 约束】约束满足度指标（三目标版无此项，L0 改为约束后新增并持久化）======
        # 口径与 NVITA_NSGA2._constraint_violation 完全一致：n_min <= n_actual <= n_max 双边界同时校验
        'constraint_feasible_cnt': int(np.sum((np.asarray(all_sel_n_actual) >= n_min)
                                             & (np.asarray(all_sel_n_actual) <= n_max))) if all_sel_n_actual else 0,
        'constraint_feasible_rate': float(np.mean((np.asarray(all_sel_n_actual) >= n_min)
                                                 & (np.asarray(all_sel_n_actual) <= n_max))) if all_sel_n_actual else 0.0,
        'clean_mses': all_clean_mse,
        'adv_mses': all_adv_mse,
        'l2_norms': all_l2_norms,
        'sel_l2_norms': all_sel_l2,
        'sel_rses': all_sel_rse,
        'sel_n_actuals': all_sel_n_actual,
        'total_samples': total,
        # ====== 【查询效率 / 开销】（第8点）first_success_query 均值/中位数、未达标占比、平均总查询数、墙钟 ======
        'query_counts': all_query_count,
        'first_success_queries': all_first_success_query,
        'wall_times': all_wall_time,
        'mean_query_count': float(np.mean(all_query_count)) if all_query_count else 0.0,
        'mean_first_success_query': float(np.mean([q for q in all_first_success_query if q >= 0]))
                                    if any(q >= 0 for q in all_first_success_query) else -1.0,
        'median_first_success_query': float(np.median([q for q in all_first_success_query if q >= 0]))
                                      if any(q >= 0 for q in all_first_success_query) else -1.0,
        'first_success_miss_rate': float(np.mean([q < 0 for q in all_first_success_query]))
                                   if all_first_success_query else 0.0,
        'mean_wall_time': float(np.mean(all_wall_time)) if all_wall_time else 0.0,
        # === 新增：输出可视化优化 — 攻击阶段总耗时 + 平均敏感度矩阵 ===
        'total_attack_time': time.perf_counter() - _attack_phase_t0,
        'avg_sensitivity': np.mean(all_sensitivities, axis=0) if all_sensitivities else None,
        'select_mode': select_mode,
    }

    # 汇总所有样本的最终前沿，计算全局最终 Pareto 前沿（跨样本二维非支配筛选，O(N log N)）
    # 记录格式 = [L2, -RSE, n_actual(附带属性)]：支配只按前两列，n_actual 随前沿点一并携带
    final_objs_list = []
    for rec in all_generation_pareto:
        obj = rec['final_pareto_obj']
        if not (isinstance(obj, np.ndarray) and obj.shape[0] > 0):
            continue
        gp = rec.get('generation_pareto') or []
        if gp and gp[-1].shape[0] == obj.shape[0] and gp[-1].shape[1] >= 3:
            n_attr = gp[-1][:, 2]        # 末代前沿记录与 final_pareto_obj 同序，直接取其 n_actual 列
        else:
            # 回退：从变长解编码反解点数（口径与记录列一致，仅兼容性兜底）
            n_attr = np.array([len(s) // 3 for s in rec.get('final_pareto_solutions', [])],
                              dtype=float)
            if n_attr.shape[0] != obj.shape[0]:
                n_attr = np.full(obj.shape[0], np.nan)
        final_objs_list.append(np.column_stack([obj, n_attr]))
    if final_objs_list:
        combined_objs = np.vstack(final_objs_list)
        metrics['global_final_pareto_obj'] = extract_first_front(combined_objs)
    else:
        metrics['global_final_pareto_obj'] = np.empty((0, 3))

    return X_adv_total, metrics, all_generation_pareto


# =============================================================================
# SECTION 5.5: 攻击收敛性与前沿结构高级分析 + 可视化输出优化
# 保留：二维超体积(HV) / 双目标均值与最优值 / 每代 n_actual 等收敛统计（compute_generation_metrics，
#       控制台打印 + analysis_data.pkl 持久化）
# 可视化图（5 张）：
#   ① plot_hv_convergence_curves          — 5 行收敛曲线（HV / max RSE / min L2 / 前沿点数 / n_actual 均值）
#   ② plot_pareto_front_evolution_grid    — 随机 10 样本 PF 进化网格图（2×5 子图，y 轴 = -RSE 向左下收敛）
#   ③ plot_perturb_position_heatmap       — 全局 Pareto 前沿扰动位置频次热力图
#   ④ plot_rse_boxplot                    — 逐样本 RSE 分布箱线图
#   ⑤ plot_sensitivity_vs_selection       — 敏感度排序 vs 实际选中频次对比图
# 全部基于攻击结果数据做后处理（不改动攻击核心逻辑），统计数据持久化到 analysis_data.pkl。
# 【按需求删除】全局聚合 Pareto 前沿图（plot_global_pareto_front）：PF 前沿统一以 -RSE 为纵轴
# （NSGA-II 最小化目标向量原值），进化过程向左下（L2↓、-RSE↓ 即 RSE↑）收敛，不再输出全局聚合图。
# =============================================================================


def extract_first_front_indices(objs):
    """提取二维最小化问题第一 Pareto 前沿的行索引（O(N log N) 扫描法）
    与 extract_first_front 判定口径完全一致，但保留行索引，
    便于从全局前沿目标值回溯对应样本的解编码（决策变量）。
    objs 前两列为双目标 [L2, -RSE]；第 3 列（若有）为 n_actual 附带属性，不参与支配。
    """
    if objs.shape[0] == 0:
        return np.array([], dtype=int)
    o2 = objs[:, :2]
    order = np.argsort(o2[:, 0], kind='stable')
    keep = []
    best_y = np.inf
    for i in order:
        if o2[i, 1] < best_y:
            keep.append(int(i))
            best_y = o2[i, 1]
    return np.array(keep, dtype=int)


def hypervolume_2d(points_2d, ref_2d):
    """二维超体积精确计算（最小化目标，ref_2d 为两目标上界参考点）
    扫描法：按目标1升序遍历，累加 (区间宽 × 目标2当前最优宽度)。
    """
    pts = points_2d[np.all(points_2d < ref_2d, axis=1)]   # 任一维达到/超过参考点的点不贡献体积
    if pts.shape[0] == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    area = 0.0
    prev_x = pts[0, 0]
    best_y = pts[0, 1]
    for i in range(1, pts.shape[0]):
        area += (pts[i, 0] - prev_x) * (ref_2d[1] - best_y)
        if pts[i, 1] < best_y:
            best_y = pts[i, 1]
        prev_x = pts[i, 0]
    area += (ref_2d[0] - prev_x) * (ref_2d[1] - best_y)
    return float(area)


def compute_generation_metrics(all_generation_pareto):
    """从攻击记录后处理每代收敛指标（二维超体积 / 双目标均值与最优值 / 前沿解数量 /
    约束变量 n_actual 每代统计）

    【双目标改造】删除三目标版遗留的 n_max 形参：原三目标参考点第三维取 n_max，
    改为「L0 作约束」后参考点只剩 [L2, -RSE] 两维，n_max 在本函数内已无用途。

    统计口径：逐样本计算后跨样本平均（不同样本代数不齐时以 NaN 对齐），
    参考点跨全部样本统一取值，保证逐样本 HV 可比、平均曲线有意义。
    参考点（nadir）：obj1/obj2 取全部样本全部代的最大值向外扩张 10%（max + 0.1×|max|，
    兼容 -RSE 负值；仅二维 [L2, -RSE]）。
    记录格式：generation_pareto 每条 shape=(m,3) = [L2, -RSE, n_actual(附带属性)]，
    HV 与支配统计只用前两列，第三列仅作约束变量统计。

    返回 None（无有效数据）或字典：
        ref_point:         统一参考点 shape=(2,)
        hv_mean / hv_std:  每代二维 HV 的跨样本均值 / 标准差（曲线数据）
        hv_per_sample:     逐样本每代 HV 明细 shape=(n_samples, n_generations)
        obj1_mean / obj1_best / obj2_mean / obj2_best: 双目标每代均值 / 最小值曲线
        n_actual_mean / n_actual_best: 每代前沿 n_actual 均值 / 最小值（constraint variable，非目标）
        obj_mean_per_sample / obj_best_per_sample: 双目标逐样本明细 shape=(n,gen,2)
        front_size_mean:   每代第一前沿解数量的跨样本均值（仅作为统计量随 analysis_data.pkl
                           持久化；原对应的「每代前沿解数量曲线」属三目标版诊断图，已删除）
        n_samples / n_generations: 参与统计的样本数与代数
    """
    valid_recs = [rec for rec in all_generation_pareto
                  if len(rec.get('generation_pareto', [])) > 0]
    if not valid_recs:
        return None
    n_samp = len(valid_recs)
    n_gen = max(len(rec['generation_pareto']) for rec in valid_recs)

    # 统一参考点：obj1/obj2 取全部样本全部代的最大值向外扩张 10%（max + 0.1×|max|）。
    # 注意不能用 max×1.1：obj2 = -RSE 恒为负，×1.1 会向 0 收缩导致参考点反被前沿支配、HV 恒为 0；
    # max + 0.1×|max| 对正负值都保证参考点严格劣于所有前沿点（n_actual 不进入 HV）
    all_pts = np.vstack([front for rec in valid_recs
                         for front in rec['generation_pareto'] if front.shape[0] > 0])
    ref_point = np.array([all_pts[:, 0].max() + 0.1 * abs(all_pts[:, 0].max()),
                          all_pts[:, 1].max() + 0.1 * abs(all_pts[:, 1].max())])

    # NaN 对齐的逐样本逐代统计矩阵（空前沿的代保持 NaN，汇总时用 nanmean 剔除）
    hv_mat = np.full((n_samp, n_gen), np.nan)
    obj_mean_mat = np.full((n_samp, n_gen, 2), np.nan)
    obj_best_mat = np.full((n_samp, n_gen, 2), np.nan)
    size_mat = np.full((n_samp, n_gen), np.nan)
    n_act_mean_mat = np.full((n_samp, n_gen), np.nan)
    n_act_best_mat = np.full((n_samp, n_gen), np.nan)
    for s, rec in enumerate(valid_recs):
        for g, front in enumerate(rec['generation_pareto']):
            if front.shape[0] == 0:
                continue
            obj2d = front[:, :2]                       # 双目标 [L2, -RSE]
            hv_mat[s, g] = hypervolume_2d(obj2d, ref_point)
            obj_mean_mat[s, g, :] = obj2d.mean(axis=0)
            obj_best_mat[s, g, :] = obj2d.min(axis=0)
            size_mat[s, g] = front.shape[0]
            if front.shape[1] >= 3:                    # n_actual 附带属性 → 约束变量统计（非目标）
                n_act_mean_mat[s, g] = front[:, 2].mean()
                n_act_best_mat[s, g] = front[:, 2].min()

    generation_metrics = {
        'ref_point': ref_point,
        'hv_mean': np.nanmean(hv_mat, axis=0),
        'hv_std': np.nanstd(hv_mat, axis=0),
        'hv_per_sample': hv_mat,
        'obj_mean_per_sample': obj_mean_mat,
        'obj_best_per_sample': obj_best_mat,
        'obj1_mean': np.nanmean(obj_mean_mat[:, :, 0], axis=0),
        'obj1_best': np.nanmean(obj_best_mat[:, :, 0], axis=0),
        'obj2_mean': np.nanmean(obj_mean_mat[:, :, 1], axis=0),
        'obj2_best': np.nanmean(obj_best_mat[:, :, 1], axis=0),
        'n_actual_mean': np.nanmean(n_act_mean_mat, axis=0),   # constraint variable（非目标）
        'n_actual_best': np.nanmean(n_act_best_mat, axis=0),   # constraint variable（非目标）
        'front_size_mean': np.nanmean(size_mat, axis=0),
        'n_samples': n_samp,
        'n_generations': n_gen,
    }
    return generation_metrics


def collect_global_front_solutions(all_generation_pareto):
    """复现全局最终 Pareto 前沿筛选（二维非支配），返回前沿解对应的变长决策向量列表

    依赖 run_nsga2_standalone_attack 写入的 final_pareto_solutions 字段
    （与 final_pareto_obj 行序一一对应），筛选口径与全局前沿完全一致：
    支配只按前两列 [L2, -RSE]，n_actual 附带属性不参与。
    """
    objs_list, solutions_list = [], []
    for rec in all_generation_pareto:
        obj = rec.get('final_pareto_obj')
        sols = rec.get('final_pareto_solutions')
        if not (isinstance(obj, np.ndarray) and obj.shape[0] > 0 and sols):
            continue
        for j in range(min(obj.shape[0], len(sols))):
            objs_list.append(obj[j])
            solutions_list.append(sols[j])
    if not objs_list:
        return []
    combined = np.vstack(objs_list)
    keep = extract_first_front_indices(combined[:, :2])   # 仅按双目标筛选，附带属性不参与支配
    return [solutions_list[i] for i in keep]


def decode_perturb_positions(solution, seq_len, num_features):
    """解码变长个体编码的扰动位置集合（(t,f) 去重，与 _apply_perturbation 的合并口径一致）"""
    positions = set()
    n_points = len(solution) // 3
    for i in range(n_points):
        t = int(np.clip(round(solution[3 * i]), 0, seq_len - 1))
        f = int(np.clip(round(solution[3 * i + 1]), 0, num_features - 1))
        positions.add((t, f))
    return positions


def plot_perturb_position_heatmap(all_generation_pareto, seq_len, num_features,
                                  feature_names, save_path=None):
    """绘制全局 Pareto 前沿扰动位置频次热力图（YlOrRd + 格内数字标注版）

    统计全局前沿所有解中每个 (时间步 t, 特征 f) 位置被选为扰动点的总频次；
    横轴 = 时间步索引，纵轴 = 特征名称，颜色越深（YlOrRd 色阶）代表被选中频次越高，
    每个格子中央标注频次整数（0 频次不标注，避免视觉噪声）。
    返回频次矩阵 shape=(num_features, seq_len)。
    """
    solutions = collect_global_front_solutions(all_generation_pareto)
    freq = np.zeros((num_features, seq_len), dtype=int)
    for sol in solutions:
        for (t, f) in decode_perturb_positions(sol, seq_len, num_features):
            freq[f, t] += 1
    if freq.sum() == 0:
        print("警告: 全局 Pareto 前沿无扰动位置数据，跳过热力图绘制")
        return freq

    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(freq, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Selection Frequency', fontsize=12)

    # === 新增：格内数字标注（0 频次不标；频次越高字色越浅，保证对比度）===
    freq_max = int(freq.max())
    for f_idx in range(num_features):
        for t_idx in range(seq_len):
            v = int(freq[f_idx, t_idx])
            if v == 0:
                continue
            text_color = 'white' if v > freq_max * 0.6 else 'black'
            ax.text(t_idx, f_idx, str(v), ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    ax.set_yticks(range(num_features))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Frequency of Perturbation Positions (Global Pareto Front)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  扰动位置频次热力图已保存至: {save_path}")
    plt.close(fig)
    return freq


# === 新增：HV 收敛曲线（跨样本聚合 5 行子图：HV / max RSE / min L2 / 前沿点数 / n_actual 均值）===
def plot_hv_convergence_curves(all_generation_pareto, save_path):
    """绘制跨样本聚合的 5 行收敛曲线

    从每样本每代的 Pareto 前沿后处理计算：
      - HV:          二维超体积（复用 hypervolume_2d，参考点从全部前沿点 nadir 外扩 10%）
      - max RSE:     前沿中 -RSE 最小（= RSE 最大）
      - min L2:      前沿中 L2 最小
      - front size:  每代前沿点数（非支配解数量）
      - n_actual:    每代前沿 n_actual 均值
    跨样本对齐（代数不齐用 NaN 填充），逐代取 nanmean 得到平均曲线；
    5 行子图共享 x 轴（Generation），每行末点标注最终值。
    """
    valid_recs = [rec for rec in all_generation_pareto
                  if len(rec.get('generation_pareto', [])) > 0]
    if not valid_recs:
        print("警告: all_generation_pareto 无有效记录，跳过 HV 收敛曲线绘制")
        return None

    n_samp = len(valid_recs)
    n_gen = max(len(rec['generation_pareto']) for rec in valid_recs)

    # 统一参考点（与 compute_generation_metrics 同口径）：nadir + 0.1*|nadir|
    all_pts = np.vstack([front for rec in valid_recs
                         for front in rec['generation_pareto'] if front.shape[0] > 0])
    if all_pts.shape[0] == 0:
        print("警告: 所有前沿均为空，跳过 HV 收敛曲线绘制")
        return None
    ref_point = np.array([all_pts[:, 0].max() + 0.1 * abs(all_pts[:, 0].max()),
                          all_pts[:, 1].max() + 0.1 * abs(all_pts[:, 1].max())])

    hv_mat = np.full((n_samp, n_gen), np.nan)
    max_rse_mat = np.full((n_samp, n_gen), np.nan)
    min_l2_mat = np.full((n_samp, n_gen), np.nan)
    # === 新增：输出可视化优化 — 前沿点数 + n_actual 均值 ===
    front_size_mat = np.full((n_samp, n_gen), np.nan)
    n_actual_mean_mat = np.full((n_samp, n_gen), np.nan)
    for s, rec in enumerate(valid_recs):
        for g, front in enumerate(rec['generation_pareto']):
            if front.shape[0] == 0:
                continue
            obj2d = front[:, :2]
            hv_mat[s, g] = hypervolume_2d(obj2d, ref_point)
            max_rse_mat[s, g] = -obj2d[:, 1].min()
            min_l2_mat[s, g] = obj2d[:, 0].min()
            front_size_mat[s, g] = front.shape[0]
            if front.shape[1] >= 3:
                n_actual_mean_mat[s, g] = front[:, 2].mean()

    hv_mean = np.nanmean(hv_mat, axis=0)
    max_rse_mean = np.nanmean(max_rse_mat, axis=0)
    min_l2_mean = np.nanmean(min_l2_mat, axis=0)
    front_size_mean = np.nanmean(front_size_mat, axis=0)
    n_actual_mean = np.nanmean(n_actual_mean_mat, axis=0)
    gens = np.arange(n_gen)

    # === 新增：输出可视化优化 — 从 3 子图扩展为 5 子图 ===
    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    fig.suptitle('Convergence Curves (5 Panels)', fontsize=14, fontweight='bold')

    def _annotate_final(ax, x_last, y_last, text, color):
        """末点标注最终值"""
        if np.isnan(y_last):
            return
        ax.annotate(text, xy=(x_last, y_last), xycoords='data',
                    xytext=(0.72, 0.15), textcoords='axes fraction',
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.6),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.85))

    # 子图 1: HV（蓝色 + 圆点）
    ax = axes[0]
    ax.plot(gens, hv_mean, 'b-o', markersize=3, linewidth=1.2, label='HV (cross-sample mean)')
    ax.set_ylabel('Hypervolume', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    _annotate_final(ax, gens[-1], hv_mean[-1], f'Final HV = {hv_mean[-1]:.4f}', 'blue')

    # 子图 2: max RSE（红色 + 方块）
    ax = axes[1]
    ax.plot(gens, max_rse_mean, 'r-s', markersize=3, linewidth=1.2, label='Max RSE (mean)')
    ax.set_ylabel('Max RSE', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    _annotate_final(ax, gens[-1], max_rse_mean[-1], f'Final = {max_rse_mean[-1]:.4f}', 'red')

    # 子图 3: min L2（绿色 + 三角）
    ax = axes[2]
    ax.plot(gens, min_l2_mean, 'g-^', markersize=3, linewidth=1.2, label='Min L2 (mean)')
    ax.set_ylabel('Min L2 Norm', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    _annotate_final(ax, gens[-1], min_l2_mean[-1], f'Final = {min_l2_mean[-1]:.6f}', 'green')

    # 子图 4: 前沿点数（蓝色）
    ax = axes[3]
    ax.plot(gens, front_size_mean, 'b-', linewidth=1.5, label='Front size (mean)')
    ax.set_ylabel('Pareto Front Size', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    _annotate_final(ax, gens[-1], front_size_mean[-1], f'Final = {front_size_mean[-1]:.1f}', 'blue')

    # 子图 5: n_actual 均值（紫色）
    ax = axes[4]
    ax.plot(gens, n_actual_mean, 'm-', linewidth=1.5, label='n_actual mean')
    ax.set_ylabel('Mean n_actual', fontsize=10)
    ax.set_xlabel('Generation', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    _annotate_final(ax, gens[-1], n_actual_mean[-1], f'Final = {n_actual_mean[-1]:.2f}', 'purple')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  HV 收敛曲线（5子图）已保存至: {save_path}")
    return save_path


# === 新增：Pareto 前沿进化网格图（随机 10 个样本 × 5 个关键代数，2×5 子图一张图）===
def plot_pareto_front_evolution_grid(all_generation_pareto, save_path, n_show=10, seed=42,
                                      knee_min_rse=2.0, n_min=1, n_max=10,
                                      l2_weight=0.3, rse_weight=0.7,
                                      select_strategy='knee', front_percentile=0.6):
    """随机挑选 n_show 个样本，把各自关键代数 Pareto 前沿画进同一张 2×5 网格图

    替代原「单代表样本叠加图（plot_pareto_front_evolution_overlay）」；
    全局聚合 Pareto 前沿图（plot_global_pareto_front）已按需求删除，不再输出跨样本全局前沿。
    每个子图：单个样本的 5 个关键代数（首 / 1/4 / 1/2 / 3/4 / 末）前沿叠加，
    viridis 色阶区分代数（暗→亮 = 早→晚，该语义对所有子图一致），
    y 轴 = -RSE（直接使用 obj[:,1]，与 NSGA-II 目标向量一致），
    直观展示"前沿随代数向左下（L2↓、-RSE↓ 即 RSE↑）收敛"的趋势
    （NSGA-II 标准最小化约定：两个目标都越小越好，好解在左下角）。

    【标注】在每个子图的末代前沿上标注两个特殊解：
      - 红色星号 ★：折中解（knee），由 _select_knee 选出
      - 蓝色菱形 ◆：攻击效果最优解（max RSE），末代前沿中 -RSE 最小（RSE 最大）的点

    n_show: 展示样本数（默认 10；有效样本不足时取全部）
    seed:   随机抽样种子（固定值保证可复现，设为 None 则每次运行重新随机）
    knee_min_rse: 膝点有效性下界（仅 RSE>=该值的前沿点可作膝点候选）
    n_min / n_max: L0 约束边界，用于 _select_knee 计算违约度
    l2_weight / rse_weight: 折中解参考点坐标（与攻击实际选择保持一致，需从外部传入）
    select_strategy / front_percentile: 折中解选点策略（与攻击实际选择保持一致）
    """
    valid_recs = [rec for rec in all_generation_pareto
                  if len(rec.get('generation_pareto', [])) > 0]
    if not valid_recs:
        print("警告: all_generation_pareto 无有效记录，跳过 Pareto 进化网格图绘制")
        return None

    # 随机挑选样本（无放回；有效样本不足 n_show 时取全部）
    n_pick = min(n_show, len(valid_recs))
    if seed is not None:
        pick_pos = np.random.default_rng(seed).choice(
            len(valid_recs), size=n_pick, replace=False)
    else:
        pick_pos = np.random.choice(len(valid_recs), size=n_pick, replace=False)
    pick_pos = np.sort(np.asarray(pick_pos, dtype=int))

    n_cols = 5
    n_rows = int(np.ceil(n_pick / n_cols))
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    plotted_cnt = 0
    legend_ax = None
    for slot, pos in enumerate(pick_pos):
        ax = axes[slot // n_cols][slot % n_cols]
        rec = valid_recs[int(pos)]
        picked_idx = int(rec.get('sample_idx', int(pos)))
        gen_pareto = rec['generation_pareto']
        n_gen = len(gen_pareto)

        # 关键代数：0, n_gen//4, n_gen//2, 3*n_gen//4, n_gen-1（去重后升序）
        key_gens = sorted(set([0, n_gen // 4, n_gen // 2, (3 * n_gen) // 4, n_gen - 1]))
        plotted_any = False
        for k, g in enumerate(key_gens):
            front = gen_pareto[g]
            if front.shape[0] == 0:
                continue
            # 直接使用 obj[:, :2] = [L2, -RSE]（NSGA-II 最小化约定，好解在左下）；按 L2 升序画连线
            pts = front[:, :2].copy()
            pts = pts[np.argsort(pts[:, 0], kind='stable')]
            color = cmap(k / max(len(key_gens) - 1, 1))
            ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=1.4, alpha=0.9,
                    label=f'Gen {g + 1}/{n_gen}')
            ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=22,
                       edgecolors='k', linewidths=0.3, zorder=3)
            plotted_any = True

        # ===== 标注末代前沿上的折中解与攻击效果最优解 =====
        last_front = gen_pareto[-1]
        if plotted_any and last_front.shape[0] > 0:
            # --- 攻击效果最优解（max RSE）：末代前沿中 -RSE 最小（RSE 最大）的点 ---
            max_rse_idx = int(np.argmin(last_front[:, 1]))
            ax.scatter(last_front[max_rse_idx, 0], last_front[max_rse_idx, 1],
                       marker='D', s=90, c='royalblue', edgecolors='navy', linewidths=0.8,
                       zorder=5, label='Max RSE')

            # --- 折中解（knee）：复用 NVITA_NSGA2._select_knee 选择逻辑 ---
            knee_idx = NVITA_NSGA2._select_knee(
                last_front[:, :3], 'knee', knee_min_rse, n_min, n_max,
                l2_weight=l2_weight, rse_weight=rse_weight,
                select_strategy=select_strategy, front_percentile=front_percentile)
            ax.scatter(last_front[knee_idx, 0], last_front[knee_idx, 1],
                       marker='*', s=160, c='red', edgecolors='darkred', linewidths=0.5,
                       zorder=6, label='Knee')

        if plotted_any:
            plotted_cnt += 1
            if legend_ax is None:
                legend_ax = ax   # 图例只放第一个有内容的子图（颜色深浅=进化早晚，全局语义一致）
        else:
            ax.text(0.5, 0.5, 'empty front', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')

        ax.set_title(f'Sample #{picked_idx} ({n_gen} gens)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('L2 Norm', fontsize=9)
        ax.set_ylabel('-RSE', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏样本数不足时的多余子图
    for slot in range(n_pick, n_rows * n_cols):
        axes[slot // n_cols][slot % n_cols].set_visible(False)

    if plotted_cnt == 0:
        print("警告: 所选样本的所有关键代数前沿均为空，跳过网格图")
        plt.close(fig)
        return None
    if legend_ax is not None:
        legend_ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

    fig.suptitle(f'Pareto Front Evolution — {n_pick} Random Samples (y = -RSE)\n'
                 f'(前沿随代数向左下收敛：L2 小 + -RSE 小，即 RSE 大)  '
                 f'★=Knee(折中解)  ◆=Max RSE(攻击最优)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Pareto 前沿进化网格图（随机 {n_pick} 个样本）已保存至: {save_path}")
    return save_path


# === 新增：输出可视化优化 — 逐样本 RSE 分布箱线图 ===
def plot_rse_boxplot(metrics, save_path):
    """逐样本选中解 RSE 的箱线图 + 散点叠加

    展示中位数、上下四分位、须、异常值；叠加 jitter 散点、均值虚线、中位数实线。
    当前只有 NSGA-II 一种方法，先只画一个箱；后续加基线对比可扩展。
    """
    sel_rses = metrics.get('sel_rses', [])
    if not sel_rses or len(sel_rses) < 2:
        print("警告: 逐样本 RSE 数据不足，跳过箱线图")
        return None
    data = np.asarray(sel_rses, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([data], positions=[1], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=5, alpha=0.5))
    # jitter 散点叠加
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(np.ones(len(data)) + jitter, data, s=15, alpha=0.4, c='steelblue',
               edgecolors='none', zorder=3)
    # 均值虚线 + 中位数实线
    mean_val = data.mean()
    median_val = np.median(data)
    ax.axhline(mean_val, color='green', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_val:.4f}')
    ax.axhline(median_val, color='red', linestyle='-', linewidth=1.5,
               label=f'Median = {median_val:.4f}')

    ax.set_xticks([1])
    ax.set_xticklabels(['NSGA-II'], fontsize=11)
    ax.set_xlabel('Selected Solution', fontsize=12)
    ax.set_ylabel('RSE', fontsize=12)
    ax.set_title(f'RSE Distribution Across Test Samples (n={len(data)})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  RSE 分布箱线图已保存至: {save_path}")
    return save_path


# === 新增：输出可视化优化 — 敏感度 vs 实际选中频次对比图 ===
def plot_sensitivity_vs_selection(avg_sensitivity, all_generation_pareto,
                                  seq_len, num_features, feature_names, save_path):
    """敏感度排序 vs 实际选中频次对比图

    x 轴：位置编号（seq_len*num_features 个位置，按敏感度从高到低排序）
    左 y 轴：敏感度值（归一化到 0~1）—— 蓝色实线
    右 y 轴：实际选中频次（归一化到 0~1）—— 橙色柱状
    如果两者趋势一致（正相关），说明敏感度引导有效。
    """
    if avg_sensitivity is None:
        print("警告: 无平均敏感度数据，跳过敏感度 vs 选中频次图")
        return None

    # 计算实际选中频次（从全局前沿解解码位置，与 plot_perturb_position_heatmap 同口径）
    global_solutions = collect_global_front_solutions(all_generation_pareto)
    freq_mat = np.zeros((seq_len, num_features), dtype=int)
    for sol in global_solutions:
        for (t, f) in decode_perturb_positions(sol, seq_len, num_features):
            if 0 <= t < seq_len and 0 <= f < num_features:
                freq_mat[t, f] += 1

    # 展平为一维
    sens_flat = avg_sensitivity.flatten()       # shape (seq_len*num_features,)
    freq_flat = freq_mat.flatten().astype(float)
    n_pos = len(sens_flat)
    if n_pos == 0:
        print("警告: 位置数为 0，跳过敏感度 vs 选中频次图")
        return None

    # 按敏感度降序排序
    sort_idx = np.argsort(-sens_flat, kind='stable')
    sens_sorted = sens_flat[sort_idx]
    freq_sorted = freq_flat[sort_idx]

    # 归一化到 0~1
    sens_max = sens_sorted.max()
    sens_norm = sens_sorted / sens_max if sens_max > 1e-12 else sens_sorted
    freq_max = freq_sorted.max()
    freq_norm = freq_sorted / freq_max if freq_max > 1e-12 else freq_sorted

    # 计算 Pearson 相关系数
    if sens_norm.std() > 1e-12 and freq_norm.std() > 1e-12:
        corr = np.corrcoef(sens_norm, freq_norm)[0, 1]
    else:
        corr = 0.0

    fig, ax1 = plt.subplots(figsize=(12, 5))
    positions = np.arange(n_pos)

    # 橙色柱状：选中频次（右 y 轴）
    ax2 = ax1.twinx()
    ax2.bar(positions, freq_norm, width=0.7, color='orange', alpha=0.5,
            label='Selection frequency (normalized)')
    ax2.set_ylabel('Selection Frequency (normalized)', fontsize=11, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1.05)

    # 蓝色实线：敏感度（左 y 轴）
    ax1.plot(positions, sens_norm, 'b-', linewidth=1.8,
             label='Sensitivity (normalized)')
    ax1.set_xlabel('Position Index (sorted by sensitivity, desc)', fontsize=11)
    ax1.set_ylabel('Sensitivity (normalized)', fontsize=11, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax1.set_title(f'Sensitivity Ranking vs. Actual Selection Frequency\n'
                  f'(Pearson r = {corr:.3f} — {"strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.3 else "weak"} correlation)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  敏感度 vs 选中频次对比图已保存至: {save_path}")
    return save_path


# =============================================================================
# SECTION 6: 攻击效果评估
# =============================================================================

def evaluate_attack(model, X_test, Y_test, X_adv, min_speed, max_speed, attack_name, device):
    """
    评估攻击性能，返回指标字典。

    在原有 MAE / RMSE / MAPE / RSE 基础上补充 R2（决定系数），并统一给出每个指标
    「攻击前(clean) → 攻击后(adv)」的绝对变化量 Δ 与相对变化率(%)，用于量化对抗攻击
    对风速预测精度的破坏程度：
      - 误差类指标(MAE / RMSE / MAPE)：变化率 > 0 表示攻击后误差上升（性能恶化）；
      - 拟合优度 R2：变化率 < 0 表示攻击后拟合变差，其下降率取正值表示下降幅度。
    """
    model.to(device)
    model.eval()

    Y_test_denorm = Y_test.cpu().numpy() * (max_speed - min_speed) + min_speed

    with torch.no_grad():
        Y_pred_clean = model(X_test).cpu().numpy()
        Y_pred_adv = model(X_adv).cpu().numpy()

    Y_pred_clean_denorm = Y_pred_clean * (max_speed - min_speed) + min_speed
    Y_pred_adv_denorm = Y_pred_adv * (max_speed - min_speed) + min_speed

    # ---------- 攻击前（clean）指标 ----------
    mae_clean = mean_absolute_error(Y_test_denorm, Y_pred_clean_denorm)
    rmse_clean = np.sqrt(mean_squared_error(Y_test_denorm, Y_pred_clean_denorm))
    mape_clean = np.mean(np.abs((Y_test_denorm - Y_pred_clean_denorm) / Y_test_denorm)) * 100
    r2_clean = r2_score(Y_test_denorm, Y_pred_clean_denorm)

    # ---------- 攻击后（adv）指标 ----------
    mae_adv = mean_absolute_error(Y_test_denorm, Y_pred_adv_denorm)
    rmse_adv = np.sqrt(mean_squared_error(Y_test_denorm, Y_pred_adv_denorm))
    mape_adv = np.mean(np.abs((Y_test_denorm - Y_pred_adv_denorm) / Y_test_denorm)) * 100
    r2_adv = r2_score(Y_test_denorm, Y_pred_adv_denorm)

    # RSE = rmse_adv / rmse_clean（和 nVITA 原论文一致，统一口径）
    rse_clean = 1.0  # 干净预测的 RSE 恒为 1（rmse_clean / rmse_clean）
    rse_adv = rmse_adv / rmse_clean if rmse_clean > 0 else 0

    def _rel_change(clean, adv):
        """相对变化率(%) = (adv - clean) / |clean| * 100；clean≈0 时返回 0 以避免除零。"""
        return (adv - clean) / abs(clean) * 100 if abs(clean) > 1e-12 else 0.0

    # ---------- 攻击前后变化量 Δ(=adv-clean) 与相对变化率(%) ----------
    delta_mae = mae_adv - mae_clean
    delta_rmse = rmse_adv - rmse_clean
    delta_mape = mape_adv - mape_clean
    delta_r2 = r2_adv - r2_clean

    chg_mae_pct = _rel_change(mae_clean, mae_adv)      # >0 表示误差恶化
    chg_rmse_pct = _rel_change(rmse_clean, rmse_adv)   # >0 表示误差恶化
    chg_mape_pct = _rel_change(mape_clean, mape_adv)   # >0 表示误差恶化
    chg_r2_pct = _rel_change(r2_clean, r2_adv)         # <0 表示拟合变差
    drop_r2_pct = -chg_r2_pct                          # R2 下降率（正值=下降幅度）

    print(f"\n{'=' * 70}")
    print(f"攻击评估: {attack_name}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<10}{'Clean':>14}{'Adv':>14}{'Delta':>14}{'Change%':>14}")
    print(f"{'-' * 70}")
    print(f"{'MAE':<10}{mae_clean:>14.4f}{mae_adv:>14.4f}{delta_mae:>+14.4f}{chg_mae_pct:>+13.2f}%")
    print(f"{'RMSE':<10}{rmse_clean:>14.4f}{rmse_adv:>14.4f}{delta_rmse:>+14.4f}{chg_rmse_pct:>+13.2f}%")
    print(f"{'MAPE(%)':<10}{mape_clean:>14.2f}{mape_adv:>14.2f}{delta_mape:>+14.2f}{chg_mape_pct:>+13.2f}%")
    print(f"{'R2':<10}{r2_clean:>14.4f}{r2_adv:>14.4f}{delta_r2:>+14.4f}{chg_r2_pct:>+13.2f}%")
    print(f"{'RSE':<10}{rse_clean:>14.4f}{rse_adv:>14.4f}{rse_adv - rse_clean:>+14.4f}{'--':>14}")
    print(f"{'-' * 70}")
    print(f"注: MAE/RMSE/MAPE 的 Change%>0 表示攻击后误差上升(性能恶化);")
    print(f"    R2 的 Change%<0 表示拟合优度下降, 下降率 = {drop_r2_pct:.2f}% (正值=下降幅度)")
    print(f"{'=' * 70}")

    return {
        'attack_name': attack_name,
        # 攻击前(clean)
        'mae_clean': mae_clean, 'rmse_clean': rmse_clean, 'mape_clean': mape_clean,
        'r2_clean': r2_clean, 'rse_clean': rse_clean,
        # 攻击后(adv)
        'mae_adv': mae_adv, 'rmse_adv': rmse_adv, 'mape_adv': mape_adv,
        'r2_adv': r2_adv, 'rse_adv': rse_adv,
        # 变化量 Δ(=adv-clean)
        'delta_mae': delta_mae, 'delta_rmse': delta_rmse,
        'delta_mape': delta_mape, 'delta_r2': delta_r2,
        # 相对变化率(%)：误差类>0=恶化；R2<0=拟合变差
        'chg_mae_pct': chg_mae_pct, 'chg_rmse_pct': chg_rmse_pct,
        'chg_mape_pct': chg_mape_pct, 'chg_r2_pct': chg_r2_pct,
        # R2 下降率(%) 与兼容旧字段 drop_rmse_pct
        'drop_r2_pct': drop_r2_pct, 'drop_rmse_pct': chg_rmse_pct,
    }



# =============================================================================
# SECTION 7: REPORT GENERATION
# =============================================================================

def generate_comprehensive_report(model, test_loader, val_loader, device, min_speed, max_speed,
                                  feature_columns, topo, cnn_params, lstm_params,
                                  setting, test_performance, r_test, save_prefix=''):
    print('\n========== 生成综合预测报告 ==========')
    model.eval()

    all_val_targets, all_val_outputs = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_val_targets.extend(targets.cpu().numpy())
            all_val_outputs.extend(outputs.cpu().numpy())

    all_val_targets = np.array(all_val_targets).flatten() * (max_speed - min_speed) + min_speed
    all_val_outputs = np.array(all_val_outputs).flatten() * (max_speed - min_speed) + min_speed

    all_test_targets, all_test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_test_targets.extend(targets.cpu().numpy())
            all_test_outputs.extend(outputs.cpu().numpy())

    all_test_targets = np.array(all_test_targets).flatten() * (max_speed - min_speed) + min_speed
    all_test_outputs = np.array(all_test_outputs).flatten() * (max_speed - min_speed) + min_speed

    val_residuals = all_val_targets - all_val_outputs
    test_residuals = all_test_targets - all_test_outputs

    mae_val = mean_absolute_error(all_val_targets, all_val_outputs)
    rmse_val = np.sqrt(mean_squared_error(all_val_targets, all_val_outputs))
    mape_val = np.mean(np.abs(val_residuals / all_val_targets)) * 100
    r_val = pearsonr(all_val_targets, all_val_outputs)[0]

    mae_test = test_performance['mae']
    rmse_test = test_performance['rmse']
    mape_test = test_performance['mape']

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('MoACB-WSF (Fixed Encoding) - 综合预测报告', fontsize=20, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    sample_idx = np.arange(0, min(500, len(all_val_targets)))
    ax1.plot(sample_idx, all_val_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax1.plot(sample_idx, all_val_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('风速 (m/s)', fontsize=12)
    ax1.set_title(f'验证集预测效果\n(RMSE={rmse_val:.3f}, MAE={mae_val:.3f})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    sample_idx = np.arange(0, min(500, len(all_test_targets)))
    ax2.plot(sample_idx, all_test_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax2.plot(sample_idx, all_test_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('风速 (m/s)', fontsize=12)
    ax2.set_title(f'测试集预测效果\n(RMSE={rmse_test:.3f}, MAE={mae_test:.3f})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(test_residuals, bins=50, color='darkcyan', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(test_residuals), color='red', linestyle='--', linewidth=2,
                label=f'均值: {np.mean(test_residuals):.3f}')
    ax3.set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('测试集残差分布', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_residuals, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax4.set_xlabel('预测误差 (m/s)', fontsize=12)
    ax4.set_ylabel('概率密度', fontsize=12)
    ax4.set_title('测试集误差概率密度', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    info_text = (
        f"固定编码配置\n"
        f"topo: {topo}\n"
        f"CNN: {cnn_params}\n"
        f"BiLSTM: {lstm_params}\n"
        f"Setting: {setting}"
    )
    ax5.text(0.5, 0.5, info_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax5.set_title('固定编码配置', fontsize=14)

    ax6 = plt.subplot(2, 3, 6)
    metrics = ['RMSE', 'MAE', 'MAPE', 'R']
    test_metrics = [rmse_test, mae_test, mape_test, r_test]
    x = np.arange(len(metrics))
    width = 0.35
    bars2 = ax6.bar(x + width / 2, test_metrics, width, label='测试集', color='darkcyan', alpha=0.8)
    ax6.set_xlabel('评估指标', fontsize=12)
    ax6.set_ylabel('指标值', fontsize=12)
    ax6.set_title('性能指标对比', fontsize=14)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    add_value_labels(ax6, bars2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{save_prefix}comprehensive_report.png', dpi=300, bbox_inches='tight')
    plt.show()

    if isinstance(model, nn.DataParallel):
        feature_weights = model.module.feature_weights.detach().cpu().numpy()
    else:
        feature_weights = model.feature_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    weights = feature_weights.flatten()
    plt.bar(range(len(feature_columns)), weights, color=[0.2, 0.5, 0.8])
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=15)
    plt.xlabel('输入特征')
    plt.ylabel('学习到的权重（越大越重要）')
    plt.title('HybridCNNBiLSTM 中各输入特征的权重')
    plt.grid(True, axis='y')
    for i, w in enumerate(weights):
        plt.text(i, w + 0.02, f'{w:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}feature_weights.png', dpi=300)
    plt.show()

    print('\n========== 详细性能报告 ==========')
    print(f'验证集性能: MAE={mae_val:.4f}, RMSE={rmse_val:.4f}, MAPE={mape_val:.2f}%, R={r_val:.4f}')
    print(f'测试集性能: MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, MAPE={mape_test:.2f}%, R={r_test:.4f}')

    return {
        'val_metrics': {'mae': mae_val, 'rmse': rmse_val, 'mape': mape_val, 'r': r_val},
        'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test, 'r': r_test}
    }


# =============================================================================
# SECTION 8: MAIN PROGRAM
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MoACB-WSF (Fixed Encoding) with NSGA-II Bi-objective Variable-length Sparse Attack (L0 as constraint)')
    # 数据集文件名以全局常量 FILENAME (L95) 为唯一默认值来源，避免与硬编码字符串不同步；
    # 命令行传参仍可临时覆盖，不传参时一律生效 FILENAME 中的设置
    parser.add_argument('--data_file', type=str, default=FILENAME, help='Input data file')
    parser.add_argument('--no_attack', action='store_true', help='Skip NSGA-II attack evaluation')
    # 以下 NSGA-II 攻击超参数均以 NSGA2_CONFIG 为唯一默认值来源，避免出现与配置不同步的"死配置"；
    # 命令行传参仍可临时覆盖，不传参时一律生效 NSGA2_CONFIG 中的设置
    parser.add_argument('--beta', type=float, default=NSGA2_CONFIG['beta'],
                        help='nVITA perturbation budget (原论文 β)')
    # ====== 【变长编码】扰动点数上下界，取代原固定 n ======
    parser.add_argument('--n_min', type=int, default=NSGA2_CONFIG['n_min'],
                        help='变长编码：每个个体扰动点数下界 (default: 1)')
    parser.add_argument('--n_max', type=int, default=NSGA2_CONFIG['n_max'],
                        help='变长编码：每个个体扰动点数上界 (default: 6)')
    parser.add_argument('--pop_size', type=int, default=NSGA2_CONFIG['pop_size'],
                        help='NSGA-II population size (增大可提升搜索覆盖度但耗时增加)')
    parser.add_argument('--maxiter', type=int, default=NSGA2_CONFIG['maxiter'],
                        help='Maximum generations for NSGA-II attack')
    parser.add_argument('--select_mode', type=str, default=NSGA2_CONFIG['select_mode'],
                        choices=['knee', 'max_rse'],
                        help="最终解选择策略: 'knee'=双目标归一化折中膝点（n_actual 作 tie-break） / 'max_rse'=攻击强度上界端点")
    parser.add_argument('--knee_min_rse', type=float, default=NSGA2_CONFIG['knee_min_rse'],
                        help='膝点有效性下界：仅 RSE>=该值的前沿点可作膝点候选')
    parser.add_argument('--knee_l2_weight', type=float, default=NSGA2_CONFIG['knee_l2_weight'],
                        help='折中解参考点 L2 维分量（归一化 0~1）：越小越接近理想点(左下)')
    parser.add_argument('--knee_rse_weight', type=float, default=NSGA2_CONFIG['knee_rse_weight'],
                        help='折中解参考点 -RSE 维分量（归一化 0~1）：越大越偏攻击效果一侧')
    parser.add_argument('--select_strategy', type=str, default=NSGA2_CONFIG['select_strategy'],
                        choices=['knee', 'percentile'],
                        help="折中解选点策略（在过滤后的候选集上二选一）: 'knee'=距参考点最近 / 'percentile'=按L2分位数取点")
    parser.add_argument('--front_percentile', type=float, default=NSGA2_CONFIG['front_percentile'],
                        help='percentile策略：候选解按 L2 升序后取第 int(k*该值) 个位置（0~1）')
    parser.add_argument('--num_eval_samples', type=int, default=NSGA2_CONFIG['num_eval_samples'],
                        help='参加 NSGA-II 评估的样本数量（默认 None=全部，正整数则从数据池头部截取）')
    # ====== 模型冻结与复用参数 ======
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model checkpoint (model weights + architecture params)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model checkpoint (skip training)')
    parser.add_argument('--encoding_config', type=str, default=None,
                        help='自定义编码配置文件(JSON)，包含 topo/cnn_params/lstm_params/setting；不指定则使用论文固定编码')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 【变长编码】区间合法性校验：n_min 至少为 1，且不得大于 n_max
    if args.n_min < 1:
        raise ValueError(f"--n_min={args.n_min} 非法：扰动点数下界至少为 1")
    if args.n_min > args.n_max:
        raise ValueError(f"--n_min={args.n_min} 不能大于 --n_max={args.n_max}")

    # 加载自定义编码配置（二选一：直接输入编码 or 使用论文固定编码）
    def load_encoding_config(path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        required = {'topo', 'cnn_params', 'lstm_params', 'setting'}
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"编码配置缺少字段: {missing}")
        return cfg

    if args.encoding_config is not None:
        encoding = load_encoding_config(args.encoding_config)
        print(f"\n>>> 已加载自定义编码配置: {args.encoding_config}")
    else:
        encoding = FIXED_ENCODING

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'=' * 80}")
    print("固定编码 MoACB-WSF: 直接构建论文最佳折中模型并执行 NSGA-II 双目标变长稀疏攻击（L0 作为约束）")
    print(f"{'=' * 80}")
    print(f"Device: {DEVICE}")
    print(f"Data file: {args.data_file}")
    print(f"\n当前使用的编码向量:")
    print(f"  topo:    {encoding['topo']}")
    print(f"  CNN:     {encoding['cnn_params']}")
    print(f"  BiLSTM:  {encoding['lstm_params']}")
    print(f"  Setting: {encoding['setting']}")

    # Load data
    data_result = load_and_preprocess_data(args.data_file)
    train_dataset = data_result['train_dataset']
    val_dataset = data_result['val_dataset']
    test_dataset = data_result['test_dataset']
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']
    num_features = data_result['num_features']
    feature_columns = data_result['feature_columns']

    # Extract test data for attack evaluation
    # 攻击评估数据池 = 测试集（与「测试集性能」同口径，仅用测试集，不再拼接验证集）
    # （LBA 管线已移除，不再需要为 LBA 训练预留前 adv_cnt 个样本，整个数据池直接用于攻击评估）
    test_loader_for_attack = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test_full, Y_test_full = next(iter(test_loader_for_attack))
    X_test_full = X_test_full.to(DEVICE)
    Y_test_full = Y_test_full.to(DEVICE)

    X_attack_pool = X_test_full
    Y_attack_pool = Y_test_full
    print(f"\n[攻击数据池] 测试集 {len(X_attack_pool)} 个样本（与「测试集性能」同口径，仅测试集）")

    # 【扰动预算对齐 FGSM】按训练集逐特征计算全局 range_f = max - min（归一化空间），
    # 与 baseline_gradient_attacks.py 的 compute_feature_ranges 完全同口径；
    # 配合 NVITA_NSGA2(use_window_range=False)，使 NSGA-II 每点预算 = beta*range_f，与 FGSM 一致。
    _train_X = np.stack([train_dataset[i][0].numpy() for i in range(len(train_dataset))], axis=0)  # (N_train, T, F)
    train_feature_ranges = (_train_X.max(axis=(0, 1)) - _train_X.min(axis=(0, 1))).astype('float32')
    print(f"[扰动预算] 训练集逐特征 range_f = {np.round(train_feature_ranges, 4).tolist()}（与 FGSM 同口径）")

    # ====== 判断是否触发 load_model 模式 ======
    load_model_mode = args.load_model is not None and os.path.isfile(args.load_model)
    if load_model_mode:
        print(f"\n>>> 检测到 --load_model 路径: {args.load_model}")
        print(">>> 将跳过模型训练，直接加载已训练模型进行攻击评估。")
        # 加载模型模式下，NUM_RUNS 循环只执行 1 次
        actual_num_runs = 1
    else:
        if args.load_model is not None and not os.path.isfile(args.load_model):
            print(f"\n>>> 警告: --load_model 指定的文件不存在: {args.load_model}")
            print(">>> 将按正常流程使用固定编码训练模型。")
        actual_num_runs = NUM_RUNS

    for run_id in range(actual_num_runs):
        print(f"\n{'*' * 80}")
        print(f"开始第 {run_id + 1}/{actual_num_runs} 次独立运行...")
        print(f"{'*' * 80}\n")

        prefix = f"{OUTPUT_DIR}/run{run_id + 1}_"

        print(f'使用设备: {DEVICE}')

        # ====== 加载模型或使用固定编码训练 ======
        if load_model_mode:
            # ---------- 从 checkpoint 加载已训练模型 ----------
            print(f"\n>>> 正在从 checkpoint 加载模型: {args.load_model}")
            checkpoint = torch.load(args.load_model, map_location=DEVICE)
            topo = checkpoint['topo']
            cnn_params = checkpoint['cnn_params']
            lstm_params = checkpoint['lstm_params']
            setting = checkpoint['setting']
            batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

            model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(">>> 模型加载成功！已跳过模型训练阶段。")
            print(f'\n  已加载模型的混合编码向量:')
            print(f'    topo:    {topo}')
            print(f'    CNN:     {cnn_params}')
            print(f'    BiLSTM:  {lstm_params}')
            print(f'    Setting: {setting}')
            print(f'    batch_size:  {batch_size}')
            print(f'    learn_rate:  {learn_rate:.6f}')
            print(f'    optimizer:   {opt_type}')
            print(f'    regularizer: {reg_type}')

            # 为后续兼容：初始化占位变量（加载模式下不使用）
            train_losses, val_losses = [], []
            test_performance = {}
        else:
            # ---------- 固定编码流程: 直接构建模型并训练 ----------
            print('\n========== 使用固定编码构建并训练模型 ==========')

            topo = encoding['topo']
            cnn_params = encoding['cnn_params']
            lstm_params = encoding['lstm_params']
            setting = encoding['setting']
            best_individual = encode_individual(topo, cnn_params, lstm_params, setting)

            if not is_valid_individual(best_individual):
                raise ValueError("固定编码的拓扑结构不合法: 存在没有输入连接的模块")

            batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

            print(f'\n第 {run_id + 1} 次运行使用的固定编码向量:')
            print(f'  topo:    {topo}')
            print(f'  CNN:     {cnn_params}')
            print(f'  BiLSTM:  {lstm_params}')
            print(f'  Setting: {setting}')

            print(f'\n第 {run_id + 1} 次运行解码后的超参数数值:')
            print(f' batch_size: {batch_size}')
            print(f' learning_rate: {learn_rate:.6f}')
            print(f' optimizer: {opt_type}')
            print(f' regularizer: {reg_type}')

            print('\n训练最终模型...')
            model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(DEVICE)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            criterion = nn.MSELoss()

            if opt_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learn_rate)
            elif opt_type == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
            elif opt_type == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
            else:
                optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

            train_losses = []
            val_losses = []
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
                        batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                        val_loss += batch_loss
                        val_samples += inputs.size(0)
                if val_samples > 0:
                    val_loss /= val_samples
                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss)

                # === 新增：输出可视化优化 — 训练过程精简（每10epoch打印一次） ===
                if (epoch + 1) % 10 == 0:
                    print(f'  Epoch {epoch + 1}/{FINAL_EPOCHS}, Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= FINAL_PATIENCE:
                    print(f'  训练完成: 早停于 epoch {epoch + 1}, 最终验证损失 {best_val_loss:.6f}')
                    break

            # 绘制训练损失曲线
            print('\n正在绘制训练损失曲线...')
            plt.figure(figsize=(12, 8))
            epochs_range = range(1, len(train_losses) + 1)
            plt.plot(epochs_range, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
            plt.plot(epochs_range, val_losses, 'r--', linewidth=2, label='验证损失', alpha=0.8)
            plt.xlabel('训练轮次 (Epoch)', fontsize=14)
            plt.ylabel('损失值 (MSE)', fontsize=14)
            plt.title('模型训练过程损失曲线', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{prefix}training_loss_curve.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 评估最终模型
            model.eval()
            all_test_targets = []
            all_test_outputs = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    all_test_targets.extend(targets.cpu().numpy())
                    all_test_outputs.extend(outputs.cpu().numpy())
            all_test_targets = np.array(all_test_targets) * (max_speed - min_speed) + min_speed
            all_test_outputs = np.array(all_test_outputs) * (max_speed - min_speed) + min_speed
            all_test_targets = all_test_targets.flatten()
            all_test_outputs = all_test_outputs.flatten()

            mae_test = mean_absolute_error(all_test_targets, all_test_outputs)
            rmse_test = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
            mape_test = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
            r2_test = r2_score(all_test_targets, all_test_outputs)
            r_test = pearsonr(all_test_targets, all_test_outputs)[0]

            print(f'\n第 {run_id + 1} 次运行测试集最终性能:')
            print(f'  MAE: {mae_test:.4f} m/s')
            print(f'  RMSE: {rmse_test:.4f} m/s')
            print(f'  MAPE: {mape_test:.2f}%')
            print(f'  R²: {r2_test:.4f}')
            print(f'  相关系数: {r_test:.4f}')

            test_performance = {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test}

            # 本文件已删除结构搜索代码，无模型架构层面的 Pareto 前沿图可绘制

            # 保存结果
            with open(f'{prefix}modeo_cnn_optimization_continuous_lr.pkl', 'wb') as f:
                pickle.dump({
                    'best_individual': best_individual,
                    'test_performance': test_performance,
                    'training_losses': train_losses,
                    'validation_losses': val_losses,
                    'model_params': {
                        'topo': topo,
                        'cnn_params': cnn_params,
                        'lstm_params': lstm_params,
                        'setting': setting,
                        'batch_size': batch_size,
                        'learn_rate': learn_rate,
                        'opt_type': opt_type
                    },
                    'fixed_encoding': encoding,
                    'run_id': run_id + 1,
                }, f)

            # ====== 保存模型 checkpoint ======
            if args.save_model is not None:
                save_path = args.save_model
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'topo': topo,
                    'cnn_params': cnn_params,
                    'lstm_params': lstm_params,
                    'setting': setting,
                }, save_path)
                print(f"\n>>> 模型 checkpoint 已保存至: {save_path}")

            # 生成综合报告
            report_metrics = generate_comprehensive_report(
                model, test_loader, val_loader, DEVICE, min_speed, max_speed,
                feature_columns, topo, cnn_params, lstm_params, setting,
                test_performance, r_test, save_prefix=prefix
            )

            # 保存测试结果CSV
            print('\n========== 代码运行结束后保存文件 ==========')
            test_results_df = pd.DataFrame({
                '真实风速 (m/s)': all_test_targets,
                '预测风速 (m/s)': all_test_outputs
            })
            test_results_df.to_csv(f'{prefix}test_set_wind_speed_predictions.csv', index=False, encoding='utf-8-sig')

            # 本文件已删除结构搜索代码，无架构 Pareto 前沿，无需保存每代 CSV
        # ====== if/else 块结束 ======

        # ==================== Attack Phase ====================
        # 只保留 NSGA-II 变长编码双目标（L0 约束）攻击流程（LBA 分支已整体移除）
        if not args.no_attack:
            print(f"\n{'=' * 80}")
            print(f"开始 NSGA-II 双目标变长稀疏攻击评估（L0 作为约束） (第 {run_id + 1} 次运行)")
            print(f"{'=' * 80}")

            # 攻击评估数据 = 测试集（与「测试集性能」同口径）
            # 【扰动预算对齐 FGSM】使用训练集逐特征全局 range_f（替代原 np.ones），
            # 并配合下方 use_window_range=False，使每点预算 = beta*range_f，与 FGSM 严格一致。
            feature_ranges = train_feature_ranges
            # 按配置截取评估样本子集（None / <=0 表示使用全部）
            if args.num_eval_samples is not None and args.num_eval_samples > 0:
                n_take = min(args.num_eval_samples, X_attack_pool.shape[0])
                X_eval = X_attack_pool[:n_take]
                Y_eval = Y_attack_pool[:n_take]
                print(f"[评估样本截取] 从数据池 {X_attack_pool.shape[0]} 个样本中截取前 {n_take} 个参与 NSGA-II 评估")
            else:
                X_eval = X_attack_pool
                Y_eval = Y_attack_pool

            print(f"\nNSGA-II 攻击配置: 变长编码 n∈[{args.n_min},{args.n_max}]（L0 约束）, "
                  f"pop_size={args.pop_size}, maxiter={args.maxiter}, beta={args.beta}, "
                  f"select_mode={args.select_mode}")
            print(f"双目标（全部最小化）: obj1=扰动L2范数, obj2=-RSE；"
                  f"n_actual(L0) 为约束变量，要求 ∈[{args.n_min},{args.n_max}]（Deb 约束支配）")
            print(f"评估样本数: {X_eval.shape[0]}")

            X_adv_nsga2, metrics, all_generation_pareto = run_nsga2_standalone_attack(
                model, X_eval, Y_eval, beta=args.beta,
                n_min=args.n_min, n_max=args.n_max,   # 变长编码：扰动点数上下界
                maxiter=args.maxiter, pop_size=args.pop_size,
                device=DEVICE, feature_ranges=feature_ranges,
                print_info=True,
                select_mode=args.select_mode,
                knee_min_rse=args.knee_min_rse,
                knee_l2_weight=args.knee_l2_weight,
                knee_rse_weight=args.knee_rse_weight,
                select_strategy=args.select_strategy,
                front_percentile=args.front_percentile,
                insert_prob=NSGA2_CONFIG['insert_prob'],
                delete_prob=NSGA2_CONFIG['delete_prob'],
                use_window_range=False,   # 【扰动预算对齐 FGSM】用全局 range_f 而非逐窗口极差
            )

            # 评估攻击效果
            attack_results = evaluate_attack(
                model, X_eval, Y_eval, X_adv_nsga2,
                min_speed, max_speed, f"NSGA2 (n in [{args.n_min},{args.n_max}])", DEVICE
            )

            # 计算 RSE 和平均扰动 L2
            clean_rmse = attack_results['rmse_clean']
            adv_rmse = attack_results['rmse_adv']
            rse = adv_rmse / clean_rmse if clean_rmse > 0 else 0

            n_tot = len(metrics.get('sel_n_actuals', []))
            print(f"\n{'=' * 70}")
            print(f"NSGA-II 双目标变长攻击汇总（L0 作为约束）· 测试样本攻击效果")
            print(f"{'=' * 70}")
            print(f"评估(测试)样本数: {X_eval.shape[0]}")
            print(f"--- 预测精度破坏程度（clean → adv，反归一化到 m/s 口径） ---")
            print(f"  MAE :  {attack_results['mae_clean']:.4f} → {attack_results['mae_adv']:.4f} m/s "
                  f"(Δ {attack_results['delta_mae']:+.4f}, {attack_results['chg_mae_pct']:+.2f}%)")
            print(f"  RMSE:  {clean_rmse:.4f} → {adv_rmse:.4f} m/s "
                  f"(Δ {attack_results['delta_rmse']:+.4f}, {attack_results['chg_rmse_pct']:+.2f}%)")
            print(f"  MAPE:  {attack_results['mape_clean']:.2f}% → {attack_results['mape_adv']:.2f}% "
                  f"(Δ {attack_results['delta_mape']:+.2f}, {attack_results['chg_mape_pct']:+.2f}%)")
            print(f"  R2  :  {attack_results['r2_clean']:.4f} → {attack_results['r2_adv']:.4f} "
                  f"(下降率 {attack_results['drop_r2_pct']:.2f}%)")
            print(f"  RSE :  {rse:.4f}  (= 攻击后RMSE / 干净RMSE)")
            print(f"--- 扰动规模（跨全部测试样本平均） ---")
            print(f"  平均 L2 范数（选中解实测）: {metrics['mean_sel_l2_norm']:.6f}")
            print(f"  平均 L0 范数（每样本扰动点数 n_actual 均值）: {metrics['mean_sel_n_actual']:.3f}")
            print(f"  前沿平均扰动 L2（前沿全部解平均，仅作参考）: {metrics['mean_l2_norm']:.6f}")
            print(f"  选中解平均 RSE（逐样本 RSE 均值）: {metrics['mean_sel_rse']:.4f}")
            print(f"  L0 约束 n∈[{args.n_min},{args.n_max}] 满足率: "
                  f"{metrics['constraint_feasible_cnt']}/{n_tot} "
                  f"= {metrics['constraint_feasible_rate'] * 100:.1f}%")
            n_pareto = int(metrics['global_final_pareto_obj'].shape[0]) \
                if isinstance(metrics.get('global_final_pareto_obj'), np.ndarray) else 0
            print(f"  全局最终 Pareto 前沿解数量: {n_pareto}")

            # === 新增：输出可视化优化 — RSE/L2/L0 统计分布 + 收敛分析 + 选中解信息 ===
            sel_rses = np.asarray(metrics.get('sel_rses', []), dtype=float)
            sel_l2s = np.asarray(metrics.get('sel_l2_norms', []), dtype=float)
            sel_ns = np.asarray(metrics.get('sel_n_actuals', []), dtype=float)
            n_samples = len(sel_rses)
            if n_samples > 0:
                print(f"--- RSE 统计（逐样本，共 {n_samples} 个样本）---")
                print(f"  全局聚合 RSE:      {rse:.4f}  (= sqrt(sum(adv_mse)/sum(clean_mse)))")
                print(f"  逐样本 RSE 均值:    {sel_rses.mean():.4f} \u00b1 {sel_rses.std():.3f}")
                print(f"  逐样本 RSE 中位数:  {np.median(sel_rses):.4f}")
                print(f"  RSE > 1.1 样本占比: {(sel_rses > 1.1).sum() / n_samples * 100:.1f}%")
                print(f"  RSE > 1.5 样本占比: {(sel_rses > 1.5).sum() / n_samples * 100:.1f}%")
                print(f"  RSE > 2.0 样本占比: {(sel_rses > 2.0).sum() / n_samples * 100:.1f}%")
                print(f"--- L2 统计（逐样本选中解）---")
                print(f"  平均 L2:    {sel_l2s.mean():.5f} \u00b1 {sel_l2s.std():.5f}")
                print(f"  中位数 L2:  {np.median(sel_l2s):.5f}")
                print(f"--- L0 统计（逐样本选中解）---")
                print(f"  平均 n_actual:  {sel_ns.mean():.3f} \u00b1 {sel_ns.std():.3f}")
                print(f"  中位数 n_actual: {np.median(sel_ns):.3f}")

            # --- 收敛分析 ---
            total_evals = int(sum(metrics.get('query_counts', [])))
            total_time = metrics.get('total_attack_time', sum(metrics.get('wall_times', [])))
            # 最终 HV（从 generation_metrics 或重新计算）
            gen_metrics_tmp = compute_generation_metrics(all_generation_pareto)
            final_hv = 0.0
            hv_converge_gen = '未收敛'
            if gen_metrics_tmp is not None:
                hv_arr = np.asarray(gen_metrics_tmp['hv_mean'], dtype=float)
                if len(hv_arr) > 0:
                    final_hv = float(hv_arr[-1])
                    # HV 收敛代数：最后连续10代HV增长<1%的起始代数
                    if len(hv_arr) >= 11:
                        for gi in range(len(hv_arr) - 10):
                            window = hv_arr[gi:gi + 11]
                            if window[0] > 0 and (window[-1] - window[0]) / window[0] < 0.01:
                                hv_converge_gen = f'Gen {gi + 1}'
                                break
            print(f"--- 收敛分析 ---")
            print(f"  总评估次数: {total_evals} (含敏感度探测 + 进化)")
            print(f"  最终 HV:      {final_hv:.4f}")
            print(f"  HV 收敛代数:  {hv_converge_gen}")
            print(f"  运行时间:     {total_time:.1f} 秒")

            # --- 选中解信息 ---
            cur_select_mode = metrics.get('select_mode', args.select_mode)
            if n_samples > 0:
                # 取中位数样本的选中解作为代表
                med_idx = int(np.argsort(sel_rses)[n_samples // 2])
                print(f"--- 选中解（select_mode={cur_select_mode}，中位数样本 #{med_idx}）---")
                print(f"  L2:    {sel_l2s[med_idx]:.5f}")
                print(f"  L0(n):  {sel_ns[med_idx]:.0f}")
                print(f"  RSE:   {sel_rses[med_idx]:.4f}")

            print(f"{'=' * 70}")

            # 保存 Pareto 进化数据与可视化（随机 10 个样本的 PF 进化网格图，y 轴 = -RSE 向左下收敛）
            os.makedirs(f'{OUTPUT_DIR}/run{run_id + 1}', exist_ok=True)
            pareto_save_dir = f'{OUTPUT_DIR}/run{run_id + 1}'

            if len(all_generation_pareto) > 0:
                # 保存所有样本的 Pareto 数据（非图片，供复现与后处理）
                with open(f'{pareto_save_dir}/pareto_evolution_data.pkl', 'wb') as f:
                    pickle.dump(all_generation_pareto, f)
                print(f"Pareto 进化数据已保存至: {pareto_save_dir}/pareto_evolution_data.pkl")

                # === 本项目保留的五张攻击可视化图（HV 曲线 / PF 进化网格 / 位置热力图 / 箱线图 / 敏感度对比）===
                # ① HV / max RSE / min L2 / 前沿点数 / n_actual 均值 五行收敛曲线（跨样本聚合）
                plot_hv_convergence_curves(
                    all_generation_pareto,
                    save_path=f'{pareto_save_dir}/convergence_curves.png'
                )
                # ② 随机 10 个样本的 Pareto 前沿进化网格图（2×5 子图，y 轴 = -RSE，前沿向左下收敛）
                #    末代前沿标注 ★ 折中解（knee）与 ◆ 攻击效果最优解（max RSE）
                plot_pareto_front_evolution_grid(
                    all_generation_pareto,
                    save_path=f'{pareto_save_dir}/pareto_evolution_grid.png',
                    n_show=10, seed=42,
                    knee_min_rse=args.knee_min_rse, n_min=args.n_min, n_max=args.n_max,
                    l2_weight=args.knee_l2_weight, rse_weight=args.knee_rse_weight,
                    select_strategy=args.select_strategy, front_percentile=args.front_percentile
                )
                # ③ 全局 Pareto 前沿扰动位置 (t,f) 频次热力图（YlOrRd + 格内数字标注）
                plot_perturb_position_heatmap(
                    all_generation_pareto, SEQUENCE_LENGTH, num_features, feature_columns,
                    save_path=f'{pareto_save_dir}/perturb_position_heatmap.png'
                )
                # ④ 逐样本 RSE 分布箱线图
                plot_rse_boxplot(
                    metrics,
                    save_path=f'{pareto_save_dir}/rse_distribution_boxplot.png'
                )
                # ⑤ 敏感度 vs 实际选中频次对比图
                plot_sensitivity_vs_selection(
                    metrics.get('avg_sensitivity'), all_generation_pareto,
                    SEQUENCE_LENGTH, num_features, feature_columns,
                    save_path=f'{pareto_save_dir}/sensitivity_vs_selection.png'
                )

            # ==================== 高级攻击分析：仅保留控制台统计与 pkl 持久化（不再输出图片） ====================
            analysis_data = {
                'config': {
                    'n_min': args.n_min, 'n_max': args.n_max,
                    'pop_size': args.pop_size, 'maxiter': args.maxiter,
                    'beta': args.beta, 'select_mode': args.select_mode,
                },
            }

            # 收敛性量化统计（二维 HV / 双目标均值与最优值 / n_actual 约束统计），仅打印与持久化
            generation_metrics = compute_generation_metrics(all_generation_pareto)
            if generation_metrics is not None:
                analysis_data['generation_metrics'] = generation_metrics
                # 每代 n_actual 均值（constraint variable，非 objective）控制台打印
                print(f"  每代前沿 n_actual 均值（constraint variable，非 objective）: "
                      f"{np.round(np.asarray(generation_metrics['n_actual_mean']), 3).tolist()}")

            global_front = metrics.get('global_final_pareto_obj')
            if isinstance(global_front, np.ndarray) and global_front.shape[0] > 0:
                analysis_data['global_final_pareto_obj'] = global_front

            # 统一持久化统计数据，方便后续复用
            with open(f'{pareto_save_dir}/analysis_data.pkl', 'wb') as f:
                pickle.dump(analysis_data, f)
            print(f"高级攻击分析数据已保存至: {pareto_save_dir}/analysis_data.pkl")

            # 保存攻击结果
            with open(f'{prefix}nsga2_attack_results.pkl', 'wb') as f:
                pickle.dump({
                    'attack_results': attack_results,
                    'metrics': metrics,
                    'generation_pareto': all_generation_pareto,
                    'config': {
                        'n_min': args.n_min,
                        'n_max': args.n_max,
                        'pop_size': args.pop_size,
                        'maxiter': args.maxiter,
                        'beta': args.beta,
                        'select_mode': args.select_mode,
                        'knee_min_rse': args.knee_min_rse,
                        'insert_prob': NSGA2_CONFIG['insert_prob'],
                        'delete_prob': NSGA2_CONFIG['delete_prob'],
                        'objectives': ['l2_norm', 'neg_rse'],
                        'constraint': {'type': 'L0_bounds', 'var': 'n_actual',
                                       'n_min': args.n_min, 'n_max': args.n_max},
                    },
                    'run_id': run_id + 1,
                }, f)
            print(f"NSGA2 攻击结果已保存至: {prefix}nsga2_attack_results.pkl")

            # === 逐样本明细 CSV：每个测试样本一行，保存选中解全部指标 ===
            per_sample_df = pd.DataFrame({
                'sample_idx': list(range(len(metrics['sel_rses']))),
                'clean_mse': metrics['clean_mses'],
                'adv_mse': metrics['adv_mses'],
                'rse': metrics['sel_rses'],
                'l2_norm': metrics['sel_l2_norms'],
                'n_actual': metrics['sel_n_actuals'],
                'pareto_size': [len(rec.get('final_pareto_obj', []))
                                for rec in all_generation_pareto],
                'query_count': metrics['query_counts'],
                'first_success_query': metrics['first_success_queries'],
                'wall_time_s': metrics['wall_times'],
            })
            per_sample_csv = f'{pareto_save_dir}/per_sample_results.csv'
            per_sample_df.to_csv(per_sample_csv, index=False, encoding='utf-8-sig')
            print(f"  逐样本明细 CSV: {per_sample_csv} ({len(per_sample_df)} 行)")

        print(f"\n{'=' * 80}")
        print(f"第 {run_id + 1} 次运行全部完成！")
        print(f"输出文件保存至目录: {OUTPUT_DIR}/")
        if not load_model_mode:
            print(f"  - 模型优化结果: {prefix}modeo_cnn_optimization_continuous_lr.pkl")
            print(f"  - 综合报告: {prefix}comprehensive_report.png")
            print(f"  - 测试集预测: {prefix}test_set_wind_speed_predictions.csv")
        if not args.no_attack:
            print(f"  - NSGA2攻击结果: {prefix}nsga2_attack_results.pkl")
            print(f"  - HV 收敛曲线(5子图): {OUTPUT_DIR}/run{run_id + 1}/convergence_curves.png")
            print(f"  - Pareto 前沿进化网格图(随机10样本, y=-RSE): {OUTPUT_DIR}/run{run_id + 1}/pareto_evolution_grid.png")
            print(f"  - 扰动位置频次热力图: {OUTPUT_DIR}/run{run_id + 1}/perturb_position_heatmap.png")
            print(f"  - RSE 分布箱线图: {OUTPUT_DIR}/run{run_id + 1}/rse_distribution_boxplot.png")
            print(f"  - 敏感度 vs 选中频次: {OUTPUT_DIR}/run{run_id + 1}/sensitivity_vs_selection.png")
            print(f"  - Pareto进化数据: {OUTPUT_DIR}/run{run_id + 1}/pareto_evolution_data.pkl")
            print(f"  - 高级分析数据: {OUTPUT_DIR}/run{run_id + 1}/analysis_data.pkl")
            print(f"  - 逐样本明细CSV: {OUTPUT_DIR}/run{run_id + 1}/per_sample_results.csv")
        print(f"{'=' * 80}")

    print(f"\n{'#' * 80}")
    print(f"所有 {actual_num_runs} 次独立运行已完成！")
    print(f"{'#' * 80}")


if __name__ == '__main__':
    main()
