"""
MoACB-WSF with Fixed Manual Encoding (NSGA-II 变长编码双目标稀疏攻击版，L0 作为约束)
=======================================================================
基于论文给出的最佳折中混合编码向量直接构建 CNN-BiLSTM 风速预测模型，
跳过 NSGA-II 结构搜索，直接训练目标模型并执行 nVITA/NSGA-II 稀疏对抗攻击评估。

本版本相对 LBA 版的三点核心改动：
1.【删除 LBA】完整移除 LBA 学习式攻击管线：LBA_Dataset / BayesianConv1d / BayesianLinear /
   SEBlock / CNN_LBA_Model 等类，get_sensitive_point_and_value / get_LBA_Dataset /
   build_sensitive_labels / print_sign_distribution / train_lba_model / calc_sensitive_point_ar /
   calc_perturb_rmse / evaluate_lba_fitting_quality / run_lba_attack / run_lba_pipeline 等函数，
   全部 LBA 可视化函数与 LBA_CONFIG 全局配置，以及 main() 中的 LBA 分支与 LBA 命令行参数。
   保留：数据加载/归一化/划分、HybridCNNBiLSTM 固定编码训练与评估、evaluate_attack、
        run_nsga2_standalone_attack、plot_pareto_evolution。
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
4.【分析扩展】攻击高级分析模块（SECTION 7.5）：二维超体积 HV 逐代收敛曲线、双目标均值/最优值
   进化曲线（每代 n_actual 均值作为约束变量统计一并输出）、每代前沿解数量曲线、
   全局最终前沿主 Pareto 图（L2 vs -RSE，按 n_actual 着色 + 颜色条）、
   RSE 中位典型样本攻击前后预测对比、扰动位置 (t,f) 频次热力图；图片输出到 output/run{id}/，
   统计数据统一持久化到 output/run{id}/analysis_data.pkl。

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
import traceback
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
SEQUENCE_LENGTH =100
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES = 100

NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_MODULES = 5
NUM_CNN_MODULES = 3
NUM_LSTM_MODULES = 2
TOPO_BITS_LENGTH = NUM_MODULES * (NUM_MODULES - 1) // 2
INDIVIDUAL_LENGTH = TOPO_BITS_LENGTH + 5 * NUM_CNN_MODULES + 5 * NUM_LSTM_MODULES + 4

POP_SIZE = 2
MAX_GEN = 1
NUM_RUNS = 1
MUTATION_PROB = 0.6
CROSSOVER_PROB = 0.8
LAMBDA_REG = 1e-4

EVAL_EPOCHS = 20
EVAL_PATIENCE = 5
FINAL_EPOCHS = 200
FINAL_PATIENCE = 20
GRADIENT_CLIP = 1.0
MAX_MODEL_SIZE = 1e7

KNUM_RANGE = (0, 7)
KSIZE_RANGE = (0, 3)
KACT_RANGE = (0, 7)
PT_RANGE = (0, 2)
PS_RANGE = (0, 3)
BS_RANGE = (0, 3)
OPT_RANGE = (0, 3)
LR_RANGE = (0.0001, 0.01)
REG_RANGE = (0, 3)

ACTIVATION_MAP = {
    0: 'Softplus', 1: 'Softsign', 2: 'ELU', 3: 'Softmax',
    4: 'Sigmoid', 5: 'Tanh', 6: 'ReLU', 7: 'Identity'
}

OPTIMIZER_MAP = {0: 'SGD', 1: 'Adam', 2: 'AdaDelta', 3: 'RMSprop'}
REGULARIZER_MAP = {0: None, 1: 'L1', 2: 'L2', 3: 'L1L2'}
BATCH_SIZE_MAP = {0: 32, 1: 64, 2: 96, 3: 128}

# ==================== NSGA-II 变长双目标稀疏攻击配置（L0 作为约束） ====================
# 命名说明（对齐原论文）：
#   beta: nVITA 的扰动预算系数（原论文中的 β）
# 【变长编码】n_min / n_max 取代原固定 n：每个个体独立采样扰动点数 n ∈ [n_min, n_max]
NSGA2_CONFIG = {
    'n_min': 1,      # 变长编码：扰动点数下界（时序稀疏攻击场景默认 1）
    'n_max': 6,      # 变长编码：扰动点数上界（时序稀疏攻击场景默认 6）
    'beta': 0.3,     # nVITA perturbation budget factor (原论文 β)
    'maxiter': 20,   # NSGA-II max generations
    'pop_size': 40,  # NSGA-II population size（增大可提升搜索覆盖度，但每代计算量线性增加）
    'insert_prob': 0.1,  # 变长编码：插入变异概率（n+1）
    'delete_prob': 0.1,  # 变长编码：删除变异概率（n-1）
    'select_mode': 'knee',   # 最终解选择策略：'knee'=归一化折中膝点 / 'max_rse'=攻击强度上界端点
    'knee_min_rse': 2.0,     # 膝点有效性下界：仅 RSE>=该值的前沿点可作膝点候选
}

# ==================== FIXED MANUAL ENCODING FROM PAPER ====================
# 论文表 "BEST TRADE-OFF HYBRID ENCODING VECTORS OBTAINED BY THE PROPOSED MoACB-WSF"
# Dataset: Sotavento 10-min Dataset
FIXED_ENCODING = {
    'topo': [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    'cnn_params': [
        [0, 0, 2, 0, 3],   # CNN module 0
        [0, 3, 7, 0, 2],   # CNN module 1
        [2, 0, 7, 2, 2],   # CNN module 2
    ],
    'lstm_params': [
        [0, 0, 4, 1, 2],   # BiLSTM module 0 (module index 3)
        [0, 0, 3, 1, 3],   # BiLSTM module 1 (module index 4)
    ],
    'setting': [0, 1, 0.0005761369, 0],  # batch_size=32, SGD, lr=0.0072, regularizer=L1L2
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
    try:
        if filename.endswith('.csv'):
            data = pd.read_csv(filename)
        else:
            data = pd.read_excel(filename)
        for col in FEATURE_COLUMNS:
            if col not in data.columns:
                raise ValueError(f'未找到特征列: {col}')
        feature_data = data[FEATURE_COLUMNS].values
    except Exception as e:
        print(f'无法读取数据文件: {e}')
        # Generate synthetic data for demonstration if file not found
        print("生成示例风速数据用于演示...")
        np.random.seed(42)
        n_samples = 1200
        t = np.arange(n_samples)
        wind_speed = 5 + 3 * np.sin(2 * np.pi * t / 144) + 2 * np.random.randn(n_samples)
        wind_direction = 180 + 90 * np.sin(2 * np.pi * t / 288) + 30 * np.random.randn(n_samples)
        theoretical_power = np.maximum(0, 0.5 * wind_speed ** 3 + np.random.randn(n_samples) * 10)
        active_power = np.maximum(0, theoretical_power * 0.9 + np.random.randn(n_samples) * 5)

        feature_data = np.column_stack([
            wind_direction, theoretical_power, active_power, wind_speed
        ])
        # Save for future use
        df = pd.DataFrame(feature_data, columns=FEATURE_COLUMNS)
        df.to_csv('winddata_synthetic.csv', index=False)

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
# SECTION 3: NSGA-II ENCODING AND DECODING
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


def initialize_individual():
    topo = [np.random.randint(0, 2) for _ in range(TOPO_BITS_LENGTH)]
    cnn_params = [[np.random.randint(b[0], b[1] + 1) for b in [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE]]
                  for _ in range(NUM_CNN_MODULES)]
    lstm_params = [
        [np.random.randint(b[0], b[1] + 1) for b in [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE]]
        for _ in range(NUM_LSTM_MODULES)]
    setting = [
        np.random.randint(BS_RANGE[0], BS_RANGE[1] + 1),
        np.random.randint(OPT_RANGE[0], OPT_RANGE[1] + 1),
        np.random.uniform(LR_RANGE[0], LR_RANGE[1]),
        np.random.randint(REG_RANGE[0], REG_RANGE[1] + 1)
    ]
    return encode_individual(topo, cnn_params, lstm_params, setting)


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


def initialize_population(pop_size, train_dataset):
    population = []
    attempts = 0
    max_attempts = pop_size * 100
    while len(population) < pop_size and attempts < max_attempts:
        ind = initialize_individual()
        if is_valid_individual(ind):
            try:
                _, _, _, setting = decode_individual(ind)
                batch_size, _, _, _ = decode_hyperparams(setting)
                if batch_size <= len(train_dataset):
                    population.append(ind)
            except:
                pass
        attempts += 1
    if len(population) < pop_size:
        raise ValueError(f"初始化失败:仅生成{len(population)}/{pop_size}个有效个体")
    return population


def generate_valid_topology():
    while True:
        topo = [np.random.randint(0, 2) for _ in range(TOPO_BITS_LENGTH)]
        valid = True
        for i in range(1, NUM_MODULES):
            num_incoming = 0
            k = 0
            for ii in range(NUM_MODULES):
                for jj in range(ii + 1, NUM_MODULES):
                    if jj == i and topo[k] == 1:
                        num_incoming += 1
                    k += 1
            if num_incoming == 0:
                valid = False
                break
        if valid:
            return topo


def variable_length_mutation(individual):
    topo, cnn_params, lstm_params, setting = decode_individual(individual)
    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        mutation_region = np.random.choice([0, 1, 2])
        new_topo = topo.copy()
        new_cnn_params = [p.copy() for p in cnn_params]
        new_lstm_params = [p.copy() for p in lstm_params]
        new_setting = setting.copy()
        mutated = False

        if mutation_region == 0:
            bits_length = len(topo)
            while True:
                candidate_topo = [np.random.randint(0, 2) for _ in range(bits_length)]
                if candidate_topo != topo:
                    temp_ind = encode_individual(candidate_topo, new_cnn_params, new_lstm_params, new_setting)
                    if is_valid_individual(temp_ind):
                        new_topo = candidate_topo
                        mutated = True
                        break

        elif mutation_region == 1:
            module_type = np.random.choice([0, 1])
            if module_type == 0:
                module_idx = np.random.randint(0, NUM_CNN_MODULES)
                param_idx = np.random.randint(0, 5)
                bounds = [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE][param_idx]
                original_val = new_cnn_params[module_idx][param_idx]
                candidate_vals = [v for v in range(bounds[0], bounds[1] + 1) if v != original_val]
                if candidate_vals:
                    new_cnn_params[module_idx][param_idx] = np.random.choice(candidate_vals)
                    mutated = True
            else:
                module_idx = np.random.randint(0, NUM_LSTM_MODULES)
                param_idx = np.random.randint(0, 5)
                bounds = [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE][param_idx]
                original_val = new_lstm_params[module_idx][param_idx]
                candidate_vals = [v for v in range(bounds[0], bounds[1] + 1) if v != original_val]
                if candidate_vals:
                    new_lstm_params[module_idx][param_idx] = np.random.choice(candidate_vals)
                    mutated = True

        elif mutation_region == 2:
            param_idx = np.random.randint(0, 4)
            original_val = new_setting[param_idx]
            if param_idx == 2:
                min_val, max_val = LR_RANGE
                while True:
                    new_val = np.random.uniform(min_val, max_val)
                    if abs(new_val - original_val) > 0.00001:
                        new_setting[param_idx] = new_val
                        mutated = True
                        break
            else:
                bounds = [BS_RANGE, OPT_RANGE, LR_RANGE, REG_RANGE][param_idx]
                candidate_vals = [v for v in range(int(bounds[0]), int(bounds[1]) + 1) if v != original_val]
                if candidate_vals:
                    new_setting[param_idx] = np.random.choice(candidate_vals)
                    mutated = True

        if mutated:
            new_individual = encode_individual(new_topo, new_cnn_params, new_lstm_params, new_setting)
            if not np.array_equal(new_individual, individual) and is_valid_individual(new_individual):
                return new_individual
        attempts += 1
    return individual.copy()


def crossover_population(mating_pool):
    offspring = []
    pop_size = len(mating_pool)
    L = INDIVIDUAL_LENGTH
    L_t = TOPO_BITS_LENGTH

    for i in range(0, pop_size, 2):
        if i + 1 < pop_size:
            par1 = mating_pool[i]
            par2 = mating_pool[i + 1]
            if np.random.random() < CROSSOVER_PROB:
                k = np.random.randint(0, L)
                if k < L_t:
                    child1 = par1.copy()
                    child2 = par2.copy()
                    child1[:L_t] = generate_valid_topology()
                    child2[:L_t] = generate_valid_topology()
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    child1 = np.concatenate([par1[:k], par2[k:]])
                    child2 = np.concatenate([par2[:k], par1[k:]])
                    offspring.append(child1)
                    offspring.append(child2)
            else:
                offspring.append(par1.copy())
                offspring.append(par2.copy())
        else:
            offspring.append(mating_pool[i].copy())
    return offspring


# =============================================================================
# SECTION 4: MODEL EVALUATION
# =============================================================================

def evaluate_individual(individual, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length,
                        device, epochs=EVAL_EPOCHS):
    try:
        topo, cnn_params, lstm_params, setting = decode_individual(individual)
        batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

        model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, sequence_length).to(device)
        model_size = sum(p.numel() for p in model.parameters())
        if model_size > MAX_MODEL_SIZE:
            raise ValueError(f"模型过大: {model_size} 参数")

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

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if reg_type is not None:
                    l1_penalty = torch.tensor(0., device=device)
                    l2_penalty = torch.tensor(0., device=device)
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

            model.eval()
            val_loss = 0
            val_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                    val_loss += batch_loss
                    val_samples += inputs.size(0)
            if val_samples > 0:
                val_loss /= val_samples

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= EVAL_PATIENCE:
                break

        model.eval()
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        if not all_targets or not all_outputs:
            raise ValueError("没有有效的预测结果")

        all_targets = np.array(all_targets) * (max_speed - min_speed) + min_speed
        all_outputs = np.array(all_outputs) * (max_speed - min_speed) + min_speed
        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        if np.isnan(rmse) or np.isinf(rmse):
            raise ValueError(f"无效的RMSE值: {rmse}")

        complexity = sum(p.numel() for p in model.parameters())
        return rmse, complexity

    except Exception as e:
        print(f"\n评估失败 - 个体ID: {id(individual)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误消息: {e}")
        traceback.print_exc()
        return float('inf'), float('inf')


def evaluate_population(population, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length,
                        device):
    performance, complexity = [], []
    for i, ind in enumerate(population):
        print(f"  评估个体 {i + 1}/{len(population)}...")
        rmse, comp = evaluate_individual(ind, train_dataset, val_dataset, min_speed, max_speed,
                                         num_features, sequence_length, device)
        performance.append(rmse)
        complexity.append(comp)
    return np.array(performance), np.array(complexity)


# =============================================================================
# SECTION 5: NSGA-II ALGORITHM
# =============================================================================

def fast_non_dominated_sort(performance, complexity):
    pop_size = len(performance)
    fronts = []
    rank = np.zeros(pop_size, dtype=int)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]
    valid_mask = np.isfinite(performance) & np.isfinite(complexity)

    for i in range(pop_size):
        if not valid_mask[i]:
            continue
        for j in range(pop_size):
            if i == j or not valid_mask[j]:
                continue
            if (performance[i] <= performance[j] and complexity[i] <= complexity[j]) and \
                    (performance[i] < performance[j] or complexity[i] < complexity[j]):
                dominated_solutions[i].append(j)
            elif (performance[j] <= performance[i] and complexity[j] <= complexity[i]) and \
                    (performance[j] < performance[i] or complexity[j] < complexity[i]):
                domination_count[i] += 1

    current_front = np.where((domination_count == 0) & valid_mask)[0]
    if len(current_front) == 0:
        valid_indices = np.where(valid_mask)[0]
        current_front = valid_indices[:1] if len(valid_indices) > 0 else np.array([0])

    fronts.append(current_front)
    rank[current_front] = 0
    front_idx = 1

    while len(fronts[-1]) > 0:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    rank[j] = front_idx
        if next_front:
            fronts.append(np.array(next_front))
        else:
            break
        front_idx += 1

    rank[~valid_mask] = front_idx + 1
    return fronts, rank


def crowding_distance(performance, complexity, fronts):
    pop_size = len(performance)
    distance = np.zeros(pop_size)

    for front in fronts:
        if len(front) <= 2:
            distance[front] = np.inf if len(front) == 2 else 0
            continue

        valid_mask = np.isfinite(performance[front]) & np.isfinite(complexity[front])
        if not valid_mask.any():
            continue

        front_valid = front[valid_mask]
        sorted_perf_idx = np.argsort(performance[front_valid])
        distance[front_valid[sorted_perf_idx[0]]] = np.inf
        distance[front_valid[sorted_perf_idx[-1]]] = np.inf

        sorted_comp_idx = np.argsort(complexity[front_valid])
        distance[front_valid[sorted_comp_idx[0]]] = np.inf
        distance[front_valid[sorted_comp_idx[-1]]] = np.inf

        perf_range = performance[front_valid[sorted_perf_idx[-1]]] - performance[front_valid[sorted_perf_idx[0]]]
        comp_range = complexity[front_valid[sorted_comp_idx[-1]]] - complexity[front_valid[sorted_comp_idx[0]]]

        if perf_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_perf_idx[i]
                prev_idx = sorted_perf_idx[i - 1]
                next_idx = sorted_perf_idx[i + 1]
                distance[front_valid[idx]] += (performance[front_valid[next_idx]] - performance[
                    front_valid[prev_idx]]) / perf_range

        if comp_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_comp_idx[i]
                prev_idx = sorted_comp_idx[i - 1]
                next_idx = sorted_comp_idx[i + 1]
                distance[front_valid[idx]] += (complexity[front_valid[next_idx]] - complexity[
                    front_valid[prev_idx]]) / comp_range

    return distance


def tournament_selection(population, rank, distance, pop_size):
    mating_pool = []
    pop_size_actual = len(population)
    for _ in range(pop_size):
        idx1, idx2 = np.random.randint(0, pop_size_actual, 2)
        if not np.isfinite(rank[idx1]): rank[idx1] = 1e6
        if not np.isfinite(rank[idx2]): rank[idx2] = 1e6
        if not np.isfinite(distance[idx1]): distance[idx1] = 0
        if not np.isfinite(distance[idx2]): distance[idx2] = 0

        if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and distance[idx1] > distance[idx2]):
            mating_pool.append(population[idx1].copy())
        else:
            mating_pool.append(population[idx2].copy())
    return mating_pool


def environmental_selection(combined_pop, combined_perf, combined_complex, combined_rank, combined_dist, pop_size):
    sorted_idx = np.argsort(combined_rank)
    combined_pop = [combined_pop[i] for i in sorted_idx]
    combined_perf = combined_perf[sorted_idx]
    combined_complex = combined_complex[sorted_idx]
    combined_rank = combined_rank[sorted_idx]
    combined_dist = combined_dist[sorted_idx]

    new_pop, new_perf, new_complex = [], [], []
    current_size = 0

    for rank_val in np.unique(combined_rank):
        if current_size >= pop_size:
            break
        mask = combined_rank == rank_val
        front_pop = [combined_pop[i] for i in range(len(combined_pop)) if mask[i]]
        front_perf = combined_perf[mask]
        front_complex = combined_complex[mask]
        front_dist = combined_dist[mask]

        if current_size + len(front_pop) <= pop_size:
            new_pop.extend(front_pop)
            new_perf.extend(front_perf)
            new_complex.extend(front_complex)
            current_size += len(front_pop)
        else:
            remaining = pop_size - current_size
            valid_dist = np.isfinite(front_dist)
            if not valid_dist.any():
                selected = np.random.choice(len(front_pop), remaining, replace=False)
            else:
                sorted_dist_idx = np.argsort(front_dist[valid_dist])[::-1]
                selected = np.where(valid_dist)[0][sorted_dist_idx[:remaining]]
            new_pop.extend([front_pop[i] for i in selected])
            new_perf.extend(front_perf[selected])
            new_complex.extend(front_complex[selected])
            current_size = pop_size

    return new_pop, np.array(new_perf), np.array(new_complex)



# =============================================================================
# SECTION 6: NVITA_NSGA2 - NSGA-II 变长编码双目标稀疏对抗攻击（L0 作为约束）
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
                 knee_min_rse=2.0, insert_prob=0.1, delete_prob=0.1):
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
        # ====== 【变长编码】长度变异的两个独立概率 ======
        self.insert_prob = insert_prob      # 插入变异概率：随机新增一个三元组，n+1
        self.delete_prob = delete_prob      # 删除变异概率：随机删除一个三元组，n-1

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
    def _num_points(individual):
        """【变长编码】由向量长度反解当前个体的扰动点数 n = len(individual) // 3"""
        return len(individual) // 3

    def _random_new_point(self, seq_len, allowed_features, ranges, existing):
        """【变长编码】随机生成一个与 existing 中 (t,f) 不重复的新三元组
        供「插入变异」与「长度补齐」复用，幅值按该特征的动态预算 epsilon*range[f] 采样
        返回: ([t, f, p], (t, f))
        """
        if len(allowed_features) == 0:
            return [0.0, 0.0, 0.0], (0, 0)
        for _ in range(200):
            t = int(np.random.randint(0, seq_len))
            f = int(np.random.choice(allowed_features))
            if (t, f) not in existing:
                budget = self.epsilon * ranges[f]
                p = float(np.random.uniform(-budget, budget)) if budget > 0 else 0.0
                return [float(t), float(f), p], (t, f)
        # 兜底：位置空间被占满时允许重复，交由 _repair_unique 合并幅值
        t = int(np.random.randint(0, seq_len))
        f = int(np.random.choice(allowed_features))
        budget = self.epsilon * ranges[f]
        p = float(np.random.uniform(-budget, budget)) if budget > 0 else 0.0
        return [float(t), float(f), p], (t, f)

    def _init_individual(self, seq_len, num_features, allowed_features, ranges):
        """【变长编码】初始化一个个体，输出长度为 3n 的一维向量
        步骤:
          1) 先从 [n_min, n_max] 中均匀随机采样一个整数 n
          2) 再生成 n 个互不重复的 (t,f) 位置，每个位置按动态预算 epsilon*range[f] 采样幅值 p
          3) 拼接为 [t1,f1,p1, ..., tn,fn,pn]
        """
        n = int(np.random.randint(self.n_min, self.n_max + 1))   # 变长：逐个体随机采样扰动点数
        positions = set()
        ind = []
        max_attempts = n * 100
        attempts = 0
        while len(positions) < n and attempts < max_attempts:
            t = np.random.randint(0, seq_len)
            f = int(np.random.choice(allowed_features))
            pos_key = (t, f)
            if pos_key not in positions:
                positions.add(pos_key)
                budget = self.epsilon * ranges[f]
                p = np.random.uniform(-budget, budget) if budget > 0 else 0.0
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
        with torch.no_grad():
            pred_adv = self.model(torch.FloatTensor(x_adv).to(device)).item()
            pred_clean = self.model(torch.FloatTensor(x_np).to(device)).item()

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

        def _constr_dominates(i, j):
            """判定个体 i 是否约束支配个体 j（规则 a/b/c）"""
            fi, fj = cv[i] <= 0.0, cv[j] <= 0.0
            if fi and not fj:
                return True                                    # (a) 可行支配不可行
            if fi and fj:
                # (b) 都可行：二维目标上的标准 Pareto 支配
                return bool(np.all(objectives[i] <= objectives[j])
                            and np.any(objectives[i] < objectives[j]))
            if not fi and not fj:
                return cv[i] < cv[j]                             # (c) 都不可行：违约度小者胜
            return False                                         # i 不可行而 j 可行：必不支配

        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                if _constr_dominates(i, j):
                    dominated_set[i].append(j)
                elif _constr_dominates(j, i):
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
    def _tournament_selection(population, objectives, rank, crowding, pop_size):
        """二元锦标赛选择
        第一优先级：非支配层级 rank（越小越优）
        第二优先级：拥挤距离（越大越优，维持种群多样性）
        【变长编码适配】仅把种群容器从 ndarray 换成 list（个体长度不一，无法堆成矩形矩阵），
        选择判据只依赖 rank/crowding，与编码长度和目标维度均无关，逻辑完全不变
        """
        n = len(population)
        selected = []
        while len(selected) < pop_size:
            i, j = np.random.choice(n, 2, replace=False)
            if rank[i] < rank[j]:
                selected.append(population[i].copy())
            elif rank[j] < rank[i]:
                selected.append(population[j].copy())
            elif crowding[i] > crowding[j]:
                selected.append(population[i].copy())
            else:
                selected.append(population[j].copy())
        return selected

    def _clip_tf_budget(self, c, seq_len, num_features, ranges):
        """统一的合法性修正：t/f 取整截断，p 按动态预算截断
        【变长编码适配】循环上界由固定 self.n 改为 len(c)//3，逻辑不变
        """
        n_points = self._num_points(c)
        for k in range(n_points):
            c[3*k]   = float(np.clip(round(c[3*k]),   0, seq_len - 1))
            c[3*k+1] = float(np.clip(round(c[3*k+1]), 0, num_features - 1))
            bud = self.epsilon * ranges[int(c[3*k+1])]
            c[3*k+2] = float(np.clip(c[3*k+2], -bud, bud))
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
        """【变长编码】长度合法性修复：把扰动点数 n 拉回 [n_min, n_max]
        1) 长度截断：若 n > n_max，随机删除扰动点直到 n == n_max
        2) 长度补齐：若 n < n_min，随机新增不重复扰动点直到 n == n_min
        （去重合并、交叉尾部交换都可能让 n 越界，故交叉/变异末尾统一调用）
        """
        c = np.asarray(c, dtype=float)
        n_points = self._num_points(c)

        # ---- 1) 长度截断 ----
        while n_points > self.n_max:
            drop = int(np.random.randint(0, n_points))
            kept = [c[3*k:3*k+3] for k in range(n_points) if k != drop]
            c = np.concatenate(kept).astype(float)
            n_points -= 1

        # ---- 2) 长度补齐 ----
        while n_points < self.n_min:
            existing = {(int(round(c[3*k])), int(round(c[3*k+1]))) for k in range(n_points)}
            new_point = np.array(self._random_new_point(seq_len, allowed_features, ranges, existing)[0],
                                 dtype=float)
            c = np.concatenate([c, new_point]) if n_points > 0 else new_point
            n_points += 1

        return c

    def _crossover(self, parent1, parent2, seq_len, num_features, ranges, allowed_features):
        """【变长编码】变长单点交叉 + 三步合法性修复
        1) 两个父代长度可不同（3n1 / 3n2），分别在各自的三元组边界上随机选一个交叉点，
           交换尾部片段生成两个子代（子代长度因此天然可变）
        2) 位置去重：_repair_unique 合并重复 (t,f) 位置的幅值（n 可能减小）
        3) 长度截断/补齐：_enforce_length_bounds 把 n 拉回 [n_min, n_max]
        4) 预算截断：_clip_tf_budget 按最终 f 的动态预算截断幅值
           （先去重再截断：修复过程可能改变 f，须按新 f 的预算重截 p）
        """
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

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
            c = self._enforce_length_bounds(c, seq_len, num_features, ranges, allowed_features)     # 长度截断/补齐
            c = self._clip_tf_budget(c, seq_len, num_features, ranges)                              # 预算截断
            repaired.append(c)
        return repaired[0], repaired[1]

    def _mutation(self, individual, seq_len, num_features, ranges, allowed_features):
        """【变长编码】三类变异协同（三类变异概率相互独立）
        1) 点级变异（概率最高，逐三元组独立触发，与固定长度版逻辑一致）：
           分别以一定概率修改 t（70% 邻域 / 30% 全局）、f（allowed_features 均匀重采样）
           或 p（归一化到 [0,1] 做标准多项式变异 PM，再线性映回 [-budget, budget]）
        2) 插入变异（概率 self.insert_prob，默认 0.1）：随机新增一个不重复的 (t,f,p) 三元组，
           n+1，且插入后不超过 n_max
        3) 删除变异（概率 self.delete_prob，默认 0.1）：随机删除一个三元组，
           n-1，且删除后不低于 n_min
        变异后统一执行 _repair_unique + _enforce_length_bounds + _clip_tf_budget 合法性修复
        """
        mutant = np.asarray(individual, dtype=float).copy()
        n_points = self._num_points(mutant)

        # ---------- 1) 点级变异 ----------
        for k in range(n_points):
            b = 3 * k
            # 时间步 t：保留原触发率与「70% 邻域 / 30% 全局」策略
            if np.random.random() < 0.15:
                t_cur = int(round(mutant[b]))
                t_new = t_cur + np.random.choice([-1, 1]) if np.random.random() < 0.7 \
                    else np.random.randint(0, seq_len)
                mutant[b] = float(np.clip(t_new, 0, seq_len - 1))
            # 特征 f：从 allowed_features 均匀重采样（保留原逻辑）
            if np.random.random() < 0.15 and len(allowed_features) > 0:
                mutant[b+1] = float(np.random.choice(allowed_features))
            # 扰动值 p：先把 p 归一化到 [0,1] 做标准 PM，再映回 [-budget, budget]
            #            触发率沿用原逻辑 1/n（变长下 n 由当前个体长度反解）
            if np.random.random() < 1.0 / max(n_points, 1):
                f_idx = int(np.clip(round(mutant[b+1]), 0, num_features - 1))
                bud = self.epsilon * ranges[f_idx]
                x = float(np.clip((mutant[b+2] + bud) / (2 * bud + 1e-12), 0.0, 1.0))
                u = np.random.random()
                if u < 0.5:
                    dq = (2 * u) ** (1.0 / (self.eta_m + 1.0)) - 1.0
                else:
                    dq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_m + 1.0))
                mutant[b+2] = float(np.clip(x + dq, 0.0, 1.0)) * 2 * bud - bud

        # ---------- 2) 插入变异：n+1（不超过 n_max）----------
        if np.random.random() < self.insert_prob and n_points < self.n_max:
            existing = {(int(round(mutant[3*k])), int(round(mutant[3*k+1]))) for k in range(n_points)}
            new_point = np.array(self._random_new_point(seq_len, allowed_features, ranges, existing)[0],
                                 dtype=float)
            mutant = np.concatenate([mutant, new_point])
            n_points += 1
        # ---------- 3) 删除变异：n-1（不低于 n_min）----------
        elif np.random.random() < self.delete_prob and n_points > self.n_min:
            drop = int(np.random.randint(0, n_points))
            kept = [mutant[3*k:3*k+3] for k in range(n_points) if k != drop]
            mutant = np.concatenate(kept).astype(float)
            n_points -= 1

        # ---------- 统一合法性修复 ----------
        mutant = self._repair_unique(mutant, seq_len, allowed_features)
        # 变长编码额外保障：点级变异挪动 t/f 后可能触发去重合并，使 n 跌破 n_min，此处统一拉回区间
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
        allowed_features = self._sample_features(num_features)
        ranges = self._get_window_range(x_np)

        # 没有允许扰动的特征，直接返回原始输入
        if len(allowed_features) == 0:
            empty_pareto = np.array([]).reshape(0, 2)   # 双目标：空前沿列数为 2
            return X, 0.0, [], empty_pareto, []

        # ========== 1. 种群初始化（【变长编码】每个个体的 n 独立随机采样，种群用 list 承载）==========
        population = [
            self._init_individual(seq_len, num_features, allowed_features, ranges)
            for _ in range(self.pop_size)
        ]

        # ========== 2. 初始种群评估双目标 + 约束变量 ==========
        objectives = np.zeros((self.pop_size, 2))          # 双目标 [L2, -RSE]，形状恒为 (N, 2)
        n_actual_pop = np.zeros(self.pop_size)             # 约束变量 n_actual（不进入 objectives）
        for i in range(self.pop_size):
            objectives[i], n_actual_pop[i] = self._evaluate_objectives(x_np, y_val, population[i], device)

        # ========== 3. 每代 Pareto 前沿记录 ==========
        generation_pareto = []

        # ========== 4. NSGA-II 主循环 ==========
        for gen in range(self.maxiter):
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

            # 生成子代：变长单点交叉 + 三类变异（末尾均含合法性修正、位置去重与长度修复）
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1 = mating_pool[i]
                p2 = mating_pool[min(i + 1, self.pop_size - 1)]
                c1, c2 = self._crossover(p1, p2, seq_len, num_features, ranges, allowed_features)
                c1 = self._mutation(c1, seq_len, num_features, ranges, allowed_features)
                c2 = self._mutation(c2, seq_len, num_features, ranges, allowed_features)
                offspring.append(c1)
                offspring.append(c2)
            offspring = offspring[:self.pop_size]   # 【变长编码】保持 list 容器，不堆叠成矩形矩阵

            # 评估子代双目标 + 约束变量
            offspring_obj = np.zeros((len(offspring), 2))
            n_actual_off = np.zeros(len(offspring))
            for i in range(len(offspring)):
                offspring_obj[i], n_actual_off[i] = self._evaluate_objectives(
                    x_np, y_val, offspring[i], device
                )

            # μ+λ 环境选择：合并父代和子代（变长个体无法 vstack，改用 list 拼接）
            combined_pop = population + offspring
            combined_obj = np.vstack([objectives, offspring_obj])
            combined_n_actual = np.concatenate([n_actual_pop, n_actual_off])

            # 对合并后的种群做约束非支配排序 + 拥挤距离（均只按 2 个目标）
            cv_combined = self._constraint_violation(combined_n_actual)
            fronts_combined, rank_combined = self._fast_non_dominated_sort(combined_obj, cv_combined)
            crowding_combined = self._crowding_distance(combined_obj, fronts_combined)

            # 按「约束可行（已由约束支配编码进 rank）→ 非支配层级升序 → 同层拥挤距离降序」选择前 pop_size 个
            selected_indices = []
            for front in fronts_combined:
                if len(selected_indices) + len(front) <= self.pop_size:
                    selected_indices.extend(front)
                else:
                    front_crowding = crowding_combined[front]
                    front_arr = np.array(front)
                    sorted_front = front_arr[np.argsort(-front_crowding)]
                    remaining = self.pop_size - len(selected_indices)
                    selected_indices.extend(sorted_front[:remaining].tolist())
                    break

            population = [combined_pop[i] for i in selected_indices]
            objectives = combined_obj[selected_indices]
            n_actual_pop = combined_n_actual[selected_indices]

        # 记录最后一代的第一 Pareto 前沿（同样携带 n_actual 附带属性）
        cv_pop = self._constraint_violation(n_actual_pop)
        final_fronts, _ = self._fast_non_dominated_sort(objectives, cv_pop)
        if len(final_fronts) > 0:
            generation_pareto.append(
                np.column_stack([objectives[final_fronts[0]], n_actual_pop[final_fronts[0]]]).copy())

        # ========== 5. 选择最终解（按 select_mode 决定折中膝点或攻击上界端点）==========
        # 'knee'（默认）：从最终第一前沿的可行解（cv=0）中选择二维归一化后距理想点最近的折中解，
        #   【双目标】obj[1] = -RSE 为负值，归一化改为「按候选集每维最大绝对值缩放」，
        #   把 [L2, -RSE] 映射到 (0,1] 后计算距理想点 (0,0) 的欧氏距离，兼顾攻击效果与隐蔽性；
        #   稀疏性不再作为目标维，而是在距离打平时作为 tie-break（等优更稀疏）。
        #   有效性下界 knee_min_rse：仅 RSE>=knee_min_rse（即 -RSE<=-knee_min_rse）的前沿点参与膝点评选；
        #   弱前沿上若无有效点，回退到可行且攻击最强端点，保证结果不是近零扰动
        # 'max_rse'：取 RSE 最大端点（攻击强度上界，完全忽略 L2 隐蔽性与稀疏性），保留用于消融对比
        if self.select_mode == 'knee' and len(final_fronts) > 0:
            front_idx = np.array(final_fronts[0])
            front_obj = objectives[front_idx]
            front_n = n_actual_pop[front_idx]
            front_cv = self._constraint_violation(front_n)
            feasible_mask = front_cv <= 0.0            # 膝点只在可行解（n_actual ∈ [n_min, n_max]）中评选
            if feasible_mask.any():
                fea_idx = front_idx[feasible_mask]
                fea_obj = front_obj[feasible_mask]
                cand_mask = fea_obj[:, 1] <= -self.knee_min_rse   # -RSE <= -knee_min_rse 即 RSE >= knee_min_rse
                if cand_mask.any():
                    cand_idx = fea_idx[cand_mask]
                    cand_obj = fea_obj[cand_mask]
                    cand_n = front_n[feasible_mask][cand_mask]
                else:
                    end_pos = int(np.argmin(fea_obj[:, 1]))       # 弱前沿回退：可行解中攻击最强端点
                    cand_idx = fea_idx[[end_pos]]
                    cand_obj = fea_obj[[end_pos]]
                    cand_n = front_n[feasible_mask][[end_pos]]
            else:
                # 极端回退：前沿内无可行解时取违约度最小且攻击最强者（正常配置下不会触发）
                order = np.lexsort((front_obj[:, 1], front_cv))
                cand_idx = front_idx[order[:1]]
                cand_obj = front_obj[order[:1]]
                cand_n = front_n[order[:1]]
            # 【二维归一化】按候选集每维最大绝对值缩放到 (0,1]（-RSE 恒为负，|−RSE|/max|−RSE| ∈ (0,1]）
            denom = np.max(np.abs(cand_obj), axis=0)
            denom[denom < 1e-12] = 1.0                 # 防除零（候选集在该维全为 0 时）
            normed = np.abs(cand_obj) / denom
            dist = np.sqrt(np.sum(normed ** 2, axis=1))   # 距理想点 (0,0) 欧氏距离
            # tie-break（容差 1e-8）：距离相等时取 n_actual 更小者（等优更稀疏），仍相等取 L2 更小者
            ties = np.where(dist <= dist.min() + 1e-8)[0]
            tie_n = cand_n[ties]
            min_n_pos = np.where(tie_n <= tie_n.min() + 1e-8)[0]
            best_pos = ties[min_n_pos[int(np.argmin(cand_obj[ties][min_n_pos, 0]))]]
            best_idx = cand_idx[best_pos]
        else:
            best_idx = np.argmin(objectives[:, 1])  # 上界模式/回退：-RSE 最小即 RSE 最大
        best_ind = population[int(best_idx)].copy()
        x_adv_final = self._apply_perturbation(x_np, best_ind)
        with torch.no_grad():
            pred_best = self.model(torch.FloatTensor(x_adv_final).to(device)).item()
        best_mse = (pred_best - y_val) ** 2

        # 最终第一 Pareto 前沿（【变长编码】解集用 list 承载；目标值矩阵恒为 (k, 2)）
        if len(final_fronts) > 0:
            pareto_solutions = [population[i].copy() for i in final_fronts[0]]
            pareto_objectives = objectives[final_fronts[0]].copy()
        else:
            pareto_solutions = []
            pareto_objectives = np.array([]).reshape(0, 2)

        return (
            torch.FloatTensor(x_adv_final).to(device),
            float(best_mse),
            pareto_solutions,
            pareto_objectives,
            generation_pareto,
        )



# =============================================================================
# SECTION 7: NSGA-II 批量攻击与 Pareto 进化可视化
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
                                insert_prob=0.1, delete_prob=0.1):
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

    attacker = NVITA_NSGA2(
        n_min=n_min, n_max=n_max, epsilon=beta, model=model,   # 变长编码：n_min / n_max
        feature_ranges=feature_ranges,
        maxiter=maxiter, pop_size=pop_size,
        feature_constraint=feature_constraint,
        perturb_features=perturb_features,
        select_mode=select_mode,
        knee_min_rse=knee_min_rse,
        insert_prob=insert_prob,
        delete_prob=delete_prob
    )

    total = X_test.shape[0]
    for test_ind in range(total):
        X_current = X_test[test_ind].unsqueeze(0).to(device)
        y_current = Y_test[test_ind].unsqueeze(0).to(device)

        X_adv, best_mse, pareto_solutions, pareto_objectives, generation_pareto = \
            attacker.attack(X_current, y_current, seed=test_ind)

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


def plot_pareto_evolution(generation_pareto, n_min, n_max, pop_size, save_path=None):
    """
    绘制双目标 Pareto 前沿进化二维散点图（L0 作为约束，n in [n_min, n_max]）
    横轴：扰动 L2 范数（目标1，越小越隐蔽）
    纵轴：-RSE（目标2，越小代表攻击效果越强）
    点颜色：实际扰动点数 n_actual（约束变量，颜色条标注，体现稀疏度从前沿涌现）
    代数：透明度随代数递增区分（越晚越实），最后一代加黑色描边突出
    记录格式：generation_pareto 每条 shape=(m,3) = [L2, -RSE, n_actual(附带属性)]
    """
    n_generations = len(generation_pareto)
    if n_generations == 0:
        print("警告: 无 Pareto 进化数据可绘图")
        return

    fig, ax = plt.subplots(figsize=(11, 8))

    # 收集所有代的点（颜色统一映射 n_actual，透明度编码代数）
    all_l2, all_nrse, all_n, all_alpha, last_gen_mask = [], [], [], [], []
    for gen_idx, front_data in enumerate(generation_pareto):
        if front_data.shape[0] == 0:
            continue
        l2_vals = front_data[:, 0]       # 目标1: L2 范数
        neg_rse_vals = front_data[:, 1]  # 目标2: -RSE（已是取负号后的最小化形式）
        # n_actual: 约束变量（记录第 3 列附带属性，不参与支配）
        if front_data.shape[1] >= 3:
            n_actual_vals = front_data[:, 2]
        else:
            n_actual_vals = np.full(front_data.shape[0], np.nan)
        is_last = (gen_idx == n_generations - 1)
        alpha = 0.35 + 0.65 * (gen_idx / max(n_generations - 1, 1))   # 代数越晚越实
        all_l2.append(l2_vals)
        all_nrse.append(neg_rse_vals)
        all_n.append(n_actual_vals)
        all_alpha.append(np.full(front_data.shape[0], alpha))
        last_gen_mask.append(np.full(front_data.shape[0], is_last))
    if not all_l2:
        print("警告: Pareto 进化数据全部为空，跳过绘图")
        plt.close(fig)
        return
    xs = np.concatenate(all_l2)
    ys = np.concatenate(all_nrse)
    cs = np.concatenate(all_n)
    alphas = np.concatenate(all_alpha)
    lasts = np.concatenate(last_gen_mask)

    # 颜色统一映射 n_actual（viridis），逐点 alpha 写入 RGBA 第 4 通道（matplotlib 的
    # scatter alpha 参数只接受标量，透明度随代数递增须通过 RGBA 实现）
    c_arr = np.asarray(cs, dtype=float)
    c_valid = c_arr[np.isfinite(c_arr)]
    vmin = c_valid.min() if c_valid.size else 0.0
    vmax = c_valid.max() if c_valid.size else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    rgba = plt.cm.viridis(norm(np.nan_to_num(c_arr, nan=vmin)))
    rgba[:, 3] = alphas

    sc = ax.scatter(xs[~lasts], ys[~lasts], c=rgba[~lasts], s=45, edgecolors='none')
    ax.scatter(xs[lasts], ys[lasts], c=rgba[lasts], s=60,
               edgecolors='k', linewidths=0.6,
               label=f'Final gen ({n_generations - 1})')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label('n_actual (constraint variable, sparsity)', fontsize=12)

    ax.set_xlabel('Perturbation L2 Norm (obj1, smaller = more imperceptible)', fontsize=12)
    ax.set_ylabel('-RSE (obj2, smaller = stronger attack)', fontsize=12)
    ax.set_title(f'Bi-objective Pareto Front Evolution (L0 as constraint, n in [{n_min},{n_max}], '
                 f'pop_size={pop_size})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Generation', loc='upper right', fontsize=9)
    ax.text(0.02, -0.13, '点颜色 = n_actual（约束变量，稀疏度从前沿涌现）；透明度随代数递增（越晚越实），'
            '最后一代加黑色描边',
            transform=ax.transAxes, fontsize=9, color='dimgray')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Pareto 进化图已保存至: {save_path}")
    plt.show()


# =============================================================================
# SECTION 7.5: 攻击收敛性与前沿结构高级分析
# 包含：二维超体积(HV)逐代收敛曲线 / 双目标均值与最优值进化曲线（n_actual 作为约束变量统计）/
#       每代前沿解数量曲线 / 全局最终前沿主 Pareto 图（按 n_actual 着色）/
#       RSE中位典型样本攻击效果对比 / 扰动位置频次热力图
# 全部基于攻击结果数据做后处理（不改动攻击核心逻辑），统计数据持久化到 analysis_data.pkl
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
        front_size_mean:   每代第一前沿解数量的跨样本均值
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


def plot_hv_convergence(generation_metrics, save_path=None):
    """绘制超体积收敛曲线（横轴 = 进化代数，纵轴 = 跨样本平均 HV）"""
    hv_mean = np.asarray(generation_metrics['hv_mean'])
    gens = np.arange(len(hv_mean))
    plt.figure(figsize=(10, 6))
    plt.plot(gens, hv_mean, '-o', color='tab:blue', linewidth=2.5, markersize=5,
             alpha=0.9, label='Mean Hypervolume')
    plt.xlabel('Generation', fontsize=13)
    plt.ylabel('Hypervolume Value', fontsize=13)
    plt.title('Convergence of Hypervolume (Bi-objective: L2 vs -RSE)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  HV 收敛曲线已保存至: {save_path}")
    plt.show()


def plot_objective_evolution(generation_metrics, save_path=None):
    """绘制双目标逐代均值/最优值曲线（上下排列 2 张子图，每图含 Mean 与 Best 两条曲线）
    n_actual 不再是目标维（已改为约束变量），其每代均值以 n_actual_mean 键
    保留在 generation_metrics 中供控制台打印/持久化，不再绘制目标子图。
    """
    gens = np.arange(len(generation_metrics['obj1_mean']))
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    specs = [
        ('obj1_mean', 'obj1_best', 'tab:blue', 'Perturbation L2 Norm (obj1)\nsmaller = more imperceptible'),
        ('obj2_mean', 'obj2_best', 'tab:orange', '-RSE (obj2)\nsmaller = stronger attack'),
    ]
    for ax, (mean_key, best_key, color, ylabel) in zip(axes, specs):
        ax.plot(gens, generation_metrics[mean_key], '-', color=color, linewidth=2.2,
                label='Mean')
        ax.plot(gens, generation_metrics[best_key], '--', color=color, linewidth=2.0,
                alpha=0.85, label='Best (min)')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    axes[-1].set_xlabel('Generation', fontsize=13)
    fig.suptitle('Evolution of Two Objectives (L0 as constraint)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  双目标进化曲线已保存至: {save_path}")
    plt.show()


def plot_pareto_count_evolution(generation_metrics, save_path=None):
    """绘制每代第一 Pareto 前沿解数量曲线（横轴 = 代数，纵轴 = 平均解数量）"""
    sizes = np.asarray(generation_metrics['front_size_mean'])
    gens = np.arange(len(sizes))
    plt.figure(figsize=(10, 6))
    plt.plot(gens, sizes, '-s', color='tab:purple', linewidth=2.5, markersize=5,
             alpha=0.9, label='Mean Front Size')
    plt.xlabel('Generation', fontsize=13)
    plt.ylabel('Number of Pareto Solutions', fontsize=13)
    plt.title('Number of Pareto Front Solutions per Generation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  前沿解数量曲线已保存至: {save_path}")
    plt.show()


def plot_global_pareto_front(global_front, save_dir):
    """绘制最终全局 Pareto 前沿的主图（L2 vs -RSE，按 n_actual 着色 + 颜色条）

    【双目标改造】原 plot_pareto_2d_projections 的三张两两组合投影合并为一张主 Pareto 图，
    函数随之更名：双目标下前沿本身就是二维平面曲线，n-RSE / n-L2 不再是 Pareto 投影，直接删除。
    global_front: shape=(k,3)，列顺序 (L2, -RSE, n_actual 附带属性，不参与支配)
    返回实际保存的图片路径列表。
    """
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(global_front[:, 0], global_front[:, 1], c=global_front[:, 2],
                     cmap='viridis', s=70, alpha=0.9, edgecolors='k', linewidths=0.5)
    cb = plt.colorbar(sc)
    cb.set_label('n_actual (constraint variable, sparsity)', fontsize=12)
    plt.xlabel('Perturbation L2 Norm (obj1, smaller = more imperceptible)', fontsize=12)
    plt.ylabel('-RSE (obj2, smaller = stronger attack)', fontsize=12)
    plt.title('Global Pareto Front: L2 Norm vs -RSE (colored by sparsity n, L0 as constraint)',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pareto_l2_rse.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  主 Pareto 图已保存至: {save_path}")
    plt.show()
    return [save_path]


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
    """绘制全局 Pareto 前沿扰动位置频次热力图

    统计全局前沿所有解中每个 (时间步 t, 特征 f) 位置被选为扰动点的总频次；
    横轴 = 时间步索引，纵轴 = 特征名称，颜色越深代表被选中频次越高。
    返回频次矩阵 shape=(num_features, seq_len)。
    """
    solutions = collect_global_front_solutions(all_generation_pareto)
    freq = np.zeros((num_features, seq_len))
    for sol in solutions:
        for (t, f) in decode_perturb_positions(sol, seq_len, num_features):
            freq[f, t] += 1
    if freq.sum() == 0:
        print("警告: 全局 Pareto 前沿无扰动位置数据，跳过热力图绘制")
        return freq
    plt.figure(figsize=(14, 3.5))
    im = plt.imshow(freq, aspect='auto', cmap='Blues', interpolation='nearest')
    cb = plt.colorbar(im)
    cb.set_label('Selection Frequency', fontsize=12)
    plt.yticks(range(num_features), feature_names, fontsize=10)
    plt.xlabel('Time Step Index', fontsize=12)
    plt.title('Frequency of Perturbation Positions (Global Pareto Front)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  扰动位置频次热力图已保存至: {save_path}")
    plt.show()
    return freq


def plot_sample_attack_compare(model, X_eval, Y_eval, X_adv_total, metrics,
                               min_speed, max_speed, device, save_path=None):
    """绘制 RSE 中位典型样本的攻击前后预测对比图

    样本选择：按选中解 RSE 排序，取中位附近的 2 个典型样本。
    每个子图内容（单步预测模型只输出下一时刻单个预测值，故按输入序列 + 预测点呈现）：
      - 真实风速输入序列（蓝色实线）与攻击后输入序列（红色虚线）
      - 下一时刻的真实值 / 干净预测 / 攻击后预测三个标记点对比
    图内标注该样本的 L2 范数、RSE、实际扰动点数 n。
    返回选取样本的双目标与约束变量信息列表（供 analysis_data.pkl 持久化）。
    """
    sel_rses = np.asarray(metrics.get('sel_rses', []))
    n_total = len(sel_rses)
    if n_total == 0:
        print("警告: 无样本级攻击数据可绘图")
        return []
    order = np.argsort(sel_rses)
    if n_total >= 2:
        picks = [int(order[n_total // 2 - 1]), int(order[n_total // 2])]   # RSE 中位附近 2 个
    else:
        picks = [int(order[0])]

    model.to(device)
    model.eval()
    wind_col = X_eval.shape[2] - 1        # 风速特征为 FEATURE_COLUMNS 最后一列
    scale = max_speed - min_speed
    typical_info = []

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 5 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, picks):
        x_clean = X_eval[idx].detach().cpu().numpy()          # (seq_len, num_features) 归一化输入
        x_adv = X_adv_total[idx].detach().cpu().numpy()       # 攻击后输入
        y_true = Y_eval[idx].detach().cpu().item()
        with torch.no_grad():
            pred_clean = model(X_eval[idx].unsqueeze(0).to(device)).item()
            pred_adv = model(X_adv_total[idx].unsqueeze(0).to(device)).item()

        # 反归一化到真实风速尺度 (m/s)
        seq_true = x_clean[:, wind_col] * scale + min_speed
        seq_adv = x_adv[:, wind_col] * scale + min_speed
        y_t = y_true * scale + min_speed
        p_clean = pred_clean * scale + min_speed
        p_adv = pred_adv * scale + min_speed
        t_axis = np.arange(len(seq_true) + 1)

        ax.plot(t_axis[:-1], seq_true, color='tab:blue', linewidth=1.8,
                label='真实风速值 (输入窗口)')
        ax.plot(t_axis[:-1], seq_adv, color='tab:red', linewidth=1.4, linestyle='--',
                alpha=0.85, label='攻击后输入序列')
        ax.plot([t_axis[-1]], [y_t], 'o', color='tab:blue', markersize=9,
                label='真实值 (下一时刻)')
        ax.plot([t_axis[-1]], [p_clean], '^', color='tab:green', markersize=10,
                label='干净模型预测值')
        ax.plot([t_axis[-1]], [p_adv], 'X', color='tab:red', markersize=11,
                label='攻击后预测值')
        # 标注风速特征上被扰动的位置（其他特征的扰动不改变风速曲线，仅在 L2/n 统计中体现）
        diff_mask = np.abs(x_adv[:, wind_col] - x_clean[:, wind_col]) > 1e-12
        if np.any(diff_mask):
            ax.scatter(np.where(diff_mask)[0], seq_adv[diff_mask], s=90, facecolors='none',
                       edgecolors='red', linewidths=1.6, zorder=5, label='扰动位置 (风速特征)')

        l2 = metrics['sel_l2_norms'][idx]
        rse = metrics['sel_rses'][idx]
        n_act = metrics['sel_n_actuals'][idx]
        ax.set_title(f'Attack Effect Comparison on Sample #{idx}\n'
                     f'L2 = {l2:.4f}, RSE = {rse:.3f}, n_actual = {n_act}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('风速 (m/s)', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        typical_info.append({'sample_idx': idx, 'l2_norm': float(l2),
                             'rse': float(rse), 'n_actual': int(n_act)})

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  典型样本攻击对比图已保存至: {save_path}")
    plt.show()
    return typical_info


# =============================================================================
# SECTION 8: 攻击效果评估
# =============================================================================

def evaluate_attack(model, X_test, Y_test, X_adv, min_speed, max_speed, attack_name, device):
    """
    评估攻击性能，返回指标字典
    """
    model.to(device)
    model.eval()

    Y_test_denorm = Y_test.cpu().numpy() * (max_speed - min_speed) + min_speed

    with torch.no_grad():
        Y_pred_clean = model(X_test).cpu().numpy()
        Y_pred_adv = model(X_adv).cpu().numpy()

    Y_pred_clean_denorm = Y_pred_clean * (max_speed - min_speed) + min_speed
    Y_pred_adv_denorm = Y_pred_adv * (max_speed - min_speed) + min_speed

    mae_clean = mean_absolute_error(Y_test_denorm, Y_pred_clean_denorm)
    rmse_clean = np.sqrt(mean_squared_error(Y_test_denorm, Y_pred_clean_denorm))
    mape_clean = np.mean(np.abs((Y_test_denorm - Y_pred_clean_denorm) / Y_test_denorm)) * 100

    mae_adv = mean_absolute_error(Y_test_denorm, Y_pred_adv_denorm)
    rmse_adv = np.sqrt(mean_squared_error(Y_test_denorm, Y_pred_adv_denorm))
    mape_adv = np.mean(np.abs((Y_test_denorm - Y_pred_adv_denorm) / Y_test_denorm)) * 100

    # RSE = rmse_adv / rmse_clean（和 nVITA 原论文一致，统一口径）
    rse_clean = 1.0  # 干净预测的 RSE 恒为 1（rmse_clean / rmse_clean）
    rse_adv = rmse_adv / rmse_clean if rmse_clean > 0 else 0

    drop_rmse = (rmse_adv - rmse_clean) / rmse_clean * 100

    print(f"\n{'=' * 50}")
    print(f"攻击评估: {attack_name}")
    print(f"{'=' * 50}")
    print(f"清洁预测  - MAE: {mae_clean:.4f}, RMSE: {rmse_clean:.4f}, MAPE: {mape_clean:.2f}%, RSE: {rse_clean:.4f}")
    print(f"对抗预测  - MAE: {mae_adv:.4f}, RMSE: {rmse_adv:.4f}, MAPE: {mape_adv:.2f}%, RSE: {rse_adv:.4f}")
    print(f"RMSE 下降率: {drop_rmse:.2f}%")
    print(f"{'=' * 50}")

    return {
        'attack_name': attack_name,
        'mae_clean': mae_clean, 'rmse_clean': rmse_clean, 'mape_clean': mape_clean, 'rse_clean': rse_clean,
        'mae_adv': mae_adv, 'rmse_adv': rmse_adv, 'mape_adv': mape_adv, 'rse_adv': rse_adv,
        'drop_rmse_pct': drop_rmse
    }



# =============================================================================
# SECTION 9: REPORT GENERATION
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
# SECTION 10: MAIN PROGRAM
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MoACB-WSF (Fixed Encoding) with NSGA-II Bi-objective Variable-length Sparse Attack (L0 as constraint)')
    parser.add_argument('--data_file', type=str, default='winddata.xlsx', help='Input data file')
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
    # 攻击评估数据池 = 验证集(15%) + 测试集(15%)，按时序拼接
    # （LBA 管线已移除，不再需要为 LBA 训练预留前 adv_cnt 个样本，整个数据池直接用于攻击评估）
    test_loader_for_attack = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test_full, Y_test_full = next(iter(test_loader_for_attack))
    X_test_full = X_test_full.to(DEVICE)
    Y_test_full = Y_test_full.to(DEVICE)

    val_loader_for_attack = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    X_val_full, Y_val_full = next(iter(val_loader_for_attack))
    X_val_full = X_val_full.to(DEVICE)
    Y_val_full = Y_val_full.to(DEVICE)

    X_attack_pool = torch.cat([X_val_full, X_test_full], dim=0)
    Y_attack_pool = torch.cat([Y_val_full, Y_test_full], dim=0)
    print(f"\n[攻击数据池] 验证集 {len(X_val_full)} + 测试集 {len(X_test_full)} = 总计 {len(X_attack_pool)} 个样本")

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
            print(">>> 模型加载成功！已跳过 NSGA-II 搜索和最终训练阶段。")
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

                if (epoch + 1) % 5 == 0:
                    print(f'  Epoch {epoch + 1}/{FINAL_EPOCHS}, 训练损失: {train_loss / len(train_loader):.6f}, 验证损失: {val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= FINAL_PATIENCE:
                    print(f'早停于 epoch {epoch + 1}')
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

            # 固定编码模式不绘制结构搜索 Pareto 前沿相关图像

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

            # 固定编码模式不存在结构搜索 Pareto 前沿，无需保存每代 CSV
        # ====== if/else 块结束 ======

        # ==================== Attack Phase ====================
        # 只保留 NSGA-II 变长编码双目标（L0 约束）攻击流程（LBA 分支已整体移除）
        if not args.no_attack:
            print(f"\n{'=' * 80}")
            print(f"开始 NSGA-II 双目标变长稀疏攻击评估（L0 作为约束） (第 {run_id + 1} 次运行)")
            print(f"{'=' * 80}")

            # 攻击评估数据 = 验证集 + 测试集拼接的完整数据池
            feature_ranges = np.ones(num_features)
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
                insert_prob=NSGA2_CONFIG['insert_prob'],
                delete_prob=NSGA2_CONFIG['delete_prob']
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

            print(f"\n{'=' * 50}")
            print(f"NSGA-II 双目标变长攻击汇总（L0 作为约束）")
            print(f"{'=' * 50}")
            print(f"干净预测 RMSE: {clean_rmse:.4f} m/s")
            print(f"攻击后 RMSE:   {adv_rmse:.4f} m/s")
            print(f"RSE:            {rse:.4f}")
            print(f"前沿平均扰动 L2:  {metrics['mean_l2_norm']:.6f}")
            print(f"--- 选中解统计（双目标 + 约束变量） ---")
            print(f"  obj1 扰动 L2 范数:     {metrics['mean_sel_l2_norm']:.6f}")
            print(f"  obj2 RSE (-obj2 还原): {metrics['mean_sel_rse']:.4f}")
            n_tot = len(metrics.get('sel_n_actuals', []))
            print(f"  n_actual 均值（约束变量，非目标）: {metrics['mean_sel_n_actual']:.3f} "
                  f"(平均稀疏度即每样本扰动点数)")
            print(f"  L0 约束 n∈[{args.n_min},{args.n_max}] 满足率: "
                  f"{metrics['constraint_feasible_cnt']}/{n_tot} "
                  f"= {metrics['constraint_feasible_rate'] * 100:.1f}%")
            n_pareto = int(metrics['global_final_pareto_obj'].shape[0]) \
                if isinstance(metrics.get('global_final_pareto_obj'), np.ndarray) else 0
            print(f"全局最终 Pareto 前沿解数量 (跨样本二维非支配筛选，n_actual 仅作附带属性): {n_pareto}")
            print(f"{'=' * 50}")

            # 保存 Pareto 进化图片和数据
            os.makedirs(f'{OUTPUT_DIR}/run{run_id + 1}', exist_ok=True)
            pareto_save_dir = f'{OUTPUT_DIR}/run{run_id + 1}'

            if len(all_generation_pareto) > 0:
                last_sample_pareto = all_generation_pareto[-1]['generation_pareto']
                if len(last_sample_pareto) > 0:
                    plot_pareto_evolution(
                        last_sample_pareto,
                        n_min=args.n_min,
                        n_max=args.n_max,
                        pop_size=args.pop_size,
                        save_path=f'{pareto_save_dir}/pareto_evolution.png'
                    )
                # 保存所有样本的 Pareto 数据
                with open(f'{pareto_save_dir}/pareto_evolution_data.pkl', 'wb') as f:
                    pickle.dump(all_generation_pareto, f)
                print(f"Pareto 进化数据已保存至: {pareto_save_dir}/pareto_evolution_data.pkl")

            # ==================== 高级攻击分析：收敛性 / 二维投影 / 样本级可视化 ====================
            # 全部基于攻击结果数据做后处理（不改动攻击核心逻辑），默认自动执行
            print(f"\n正在生成高级攻击分析图表（收敛曲线 / 二维投影 / 样本级可视化）...")
            analysis_data = {
                'config': {
                    'n_min': args.n_min, 'n_max': args.n_max,
                    'pop_size': args.pop_size, 'maxiter': args.maxiter,
                    'beta': args.beta, 'select_mode': args.select_mode,
                },
            }

            # 一、收敛性量化分析（二维 HV / 双目标均值与最优值 / 前沿解数量 / n_actual 约束统计，逐样本计算后跨样本平均）
            generation_metrics = compute_generation_metrics(all_generation_pareto)
            if generation_metrics is not None:
                analysis_data['generation_metrics'] = generation_metrics
                # 每代 n_actual 均值（constraint variable，非 objective）控制台打印
                print(f"  每代前沿 n_actual 均值（constraint variable，非 objective）: "
                      f"{np.round(np.asarray(generation_metrics['n_actual_mean']), 3).tolist()}")
                plot_hv_convergence(generation_metrics,
                                    save_path=f'{pareto_save_dir}/hv_convergence.png')
                plot_objective_evolution(generation_metrics,
                                         save_path=f'{pareto_save_dir}/objective_evolution.png')
                plot_pareto_count_evolution(generation_metrics,
                                            save_path=f'{pareto_save_dir}/pareto_count_evolution.png')

            # 二、最终全局 Pareto 前沿主图（L2 vs -RSE，按 n_actual 着色 + 颜色条；原三张两两投影合并为一张）
            #     双目标下 n_actual 仅为约束变量，图上以颜色条呈现，不再占据坐标轴
            global_front = metrics.get('global_final_pareto_obj')
            if isinstance(global_front, np.ndarray) and global_front.shape[0] > 0:
                analysis_data['global_final_pareto_obj'] = global_front
                plot_global_pareto_front(global_front, pareto_save_dir)

            # 三、样本级攻击效果可视化（RSE 中位典型样本对比 + 扰动位置频次热力图）
            typical_info = plot_sample_attack_compare(
                model, X_eval, Y_eval, X_adv_nsga2, metrics,
                min_speed, max_speed, DEVICE,
                save_path=f'{pareto_save_dir}/sample_attack_compare.png')
            if typical_info:
                analysis_data['typical_samples'] = typical_info

            seq_len_eval = X_eval.shape[1]
            num_features_eval = X_eval.shape[2]
            analysis_data['perturb_position_freq'] = plot_perturb_position_heatmap(
                all_generation_pareto, seq_len_eval, num_features_eval, feature_columns,
                save_path=f'{pareto_save_dir}/perturb_position_heatmap.png')

            # 统一持久化新增统计数据，方便后续复用
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

        print(f"\n{'=' * 80}")
        print(f"第 {run_id + 1} 次运行全部完成！")
        print(f"输出文件保存至目录: {OUTPUT_DIR}/")
        if not load_model_mode:
            print(f"  - 模型优化结果: {prefix}modeo_cnn_optimization_continuous_lr.pkl")
            print(f"  - 综合报告: {prefix}comprehensive_report.png")
            print(f"  - 测试集预测: {prefix}test_set_wind_speed_predictions.csv")
        if not args.no_attack:
            print(f"  - NSGA2攻击结果: {prefix}nsga2_attack_results.pkl")
            print(f"  - Pareto进化图: {OUTPUT_DIR}/run{run_id + 1}/pareto_evolution.png")
            print(f"  - Pareto进化数据: {OUTPUT_DIR}/run{run_id + 1}/pareto_evolution_data.pkl")
            print(f"  - 高级分析图(HV收敛/双目标进化/前沿数量/主Pareto图/样本对比/热力图): {OUTPUT_DIR}/run{run_id + 1}/")
            print(f"  - 高级分析数据: {OUTPUT_DIR}/run{run_id + 1}/analysis_data.pkl")
        print(f"{'=' * 80}")

    print(f"\n{'#' * 80}")
    print(f"所有 {actual_num_runs} 次独立运行已完成！")
    print(f"{'#' * 80}")


if __name__ == '__main__':
    main()
