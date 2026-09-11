"""
MoACB-WSF with LBA Adversarial Attack Integration
=================================================
Multi-Objective Automated CNN-BiLSTM for Wind Speed Forecasting
with Learning-Based Adversarial Attack Evaluation

说明：MoACB-WSF 主体代码完全保留，仅修正 LBA 对抗攻击模块
- nVITA：标准 DE/rand/1/bin 差分进化稀疏黑盒攻击
- LBA：双分支CNN学习攻击模式，支持扰动特征数量约束
"""
""
## 第一次运行：训练并保存模型
#python LBA-MOACB-WSF.py --save_model output/best_model.pt

# 后续消融实验：加载模型，仅跑攻击评估（跳过搜索和训练）
#python LBA-MOACB-WSF.py --load_model output/best_model.pt --perturb_mask "0010"
#python LBA-MOACB-WSF.py --load_model output/best_model.pt --perturb_mask "1111" --feat_constraint 2

import os
import sys
import ast
import random
import time
import warnings
import traceback
import math
import pickle
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
os.makedirs(os.path.join(OUTPUT_DIR, 'LBA_models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'attack_results'), exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== MoACB-WSF Configuration ====================
FILENAME = 'winddata.xlsx'
FEATURE_COLUMNS = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
TARGET_COLUMN = 'Wind Speed (m/s)'
SEQUENCE_LENGTH =24
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

POP_SIZE =20
MAX_GEN = 40
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

# ==================== LBA Attack Configuration ====================
# 命名说明（对齐原论文）：
#   beta: nVITA 的扰动预算系数（原论文中的 β）
#   delta: LBA 生成扰动时的缩放系数（原论文中的 δ）
LBA_CONFIG = {
    'n': 1,  # number of perturbations per sample
    'beta': 0.01,  # nVITA perturbation budget factor (原论文 β)
    'maxiter': 60,  # DE max iterations for nVITA baseline
    'tol': 0.01,  # tolerance for nVITA
    'adv_cnt': 100,  # number of adv examples for LBA training
    'lba_epochs': 50,  # LBA model training epochs
    'lba_lr': 0.005,  # LBA model learning rate
    'lba_batch_size': 8,  # LBA model batch size
    'delta_list': [0.75, 1.0, 1.5, 1.75],  # LBA attack scaling factors (原论文 δ)
    'use_bayesian': True,   # 启用贝叶斯卷积层，提升泛化性和不确定性估计
    'perturb_mask': '1111',  # 二进制掩码表示是否扰动特征，1表示扰动，0表示不扰动
                             # 例如'0010'表示只扰动第3个特征(0-based索引2)
                             # '1111'表示扰动所有4个特征
    'feature_constraint': None,  # 扰动特征数量约束 (None=不限制, 整数=限定特征数)
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
# SECTION 6: LBA ADVERSARIAL ATTACK MODULES (修正版)
# =============================================================================

class LBA_Dataset(Dataset):
    """
    LBA 训练数据集
    每个样本包含: 原始时序输入、扰动位置掩码(0/1)、扰动值矩阵
    采用首次赋值、后续拼接的方式，避免初始化时维度不匹配
    """
    def __init__(self, device='cpu'):
        self.data = None
        self.mask = None
        self.perturb = None
        self.device = device

    def add_sample(self, x, mask, perturb):
        x = x.to(self.device)
        mask = mask.to(self.device)
        perturb = perturb.to(self.device)
        if self.data is None:
            self.data = x
            self.mask = mask
            self.perturb = perturb
        else:
            self.data = torch.cat((self.data, x), dim=0)
            self.mask = torch.cat((self.mask, mask), dim=0)
            self.perturb = torch.cat((self.perturb, perturb), dim=0)

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.perturb[idx]


class BayesianConv1d(nn.Module):
    """
    贝叶斯卷积层：使用权重不确定性建模，提升模型泛化性和不确定性估计
    采用局部重参数化技巧 (Local Reparameterization Trick) 提高效率
    无需外部库 blitz，完全自包含实现
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding

        # 后验参数: 权重均值 和 log方差
        self.weight_mu = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.weight_log_sigma = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_channels))

        # 先验: 标准正态 N(0, 1)
        self.prior_mu = 0.0
        self.prior_sigma = 1.0

        self.reset_parameters()

    def reset_parameters(self):
        # 用 Glorot 初始化均值，log_sigma 初始为较小负值
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, x):
        # 局部重参数化技巧：直接采样输出，而非采样权重
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # 输出的均值和方差
        act_mu = F.conv1d(x, self.weight_mu, self.bias_mu,
                         stride=self.stride, padding=self.padding)
        act_var = F.conv1d(x ** 2, weight_sigma ** 2, bias_sigma ** 2,
                          stride=self.stride, padding=self.padding)
        act_std = torch.sqrt(act_var + 1e-8)

        # 重参数化采样: output = mu + sigma * eps
        eps = torch.randn_like(act_mu)
        return act_mu + act_std * eps

    def kl_divergence(self):
        """
        计算后验与先验之间的 KL 散度
        KL(q(w|θ) || p(w))，其中 q 为高斯后验，p 为标准高斯先验
        """
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # KL for weights: log(σ_prior/σ_q) + (σ_q^2 + (μ_q - μ_prior)^2)/(2σ_prior^2) - 0.5
        kl_w = (
            torch.log(torch.tensor(self.prior_sigma) / weight_sigma)
            + (weight_sigma ** 2 + (self.weight_mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        # KL for bias
        kl_b = (
            torch.log(torch.tensor(self.prior_sigma) / bias_sigma)
            + (bias_sigma ** 2 + (self.bias_mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()

        return kl_w + kl_b


class BayesianLinear(nn.Module):
    """
    贝叶斯全连接层：同样使用权重不确定性建模
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))

        self.prior_mu = 0.0
        self.prior_sigma = 1.0

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        act_mu = F.linear(x, self.weight_mu, self.bias_mu)
        act_var = F.linear(x ** 2, weight_sigma ** 2, bias_sigma ** 2)
        act_std = torch.sqrt(act_var + 1e-8)

        eps = torch.randn_like(act_mu)
        return act_mu + act_std * eps

    def kl_divergence(self):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        kl_w = (
            torch.log(torch.tensor(self.prior_sigma) / weight_sigma)
            + (weight_sigma ** 2 + (self.weight_mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()
        kl_b = (
            torch.log(torch.tensor(self.prior_sigma) / bias_sigma)
            + (bias_sigma ** 2 + (self.bias_mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2)
            - 0.5
        ).sum()
        return kl_w + kl_b


class CNN_LBA_Model(nn.Module):
    """
    LBA 学习模型：学习输入时序 -> 敏感位置 + 扰动值 的映射
    卷积在时间维度滑动，双分支输出全位置的分类 logits 与回归值
    use_bayesian=True 时启用贝叶斯卷积层和全连接层，提升泛化性和不确定性估计
    """
    def __init__(self, num_features, seq_len, n, use_bayesian=False):
        super(CNN_LBA_Model, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = n
        self.total_positions = seq_len * num_features
        self.use_bayesian = use_bayesian

        if use_bayesian:
            # 贝叶斯卷积层 + 全连接层
            self.conv1 = BayesianConv1d(num_features, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = BayesianConv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc_shared = BayesianLinear(64 * seq_len, 128)
            self.fc_cls = BayesianLinear(128, self.total_positions)
            self.fc_reg = BayesianLinear(128, self.total_positions)
            # 贝叶斯模式下仍保留 BN（用于稳定训练）
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn_shared = nn.BatchNorm1d(128)
        else:
            # 普通确定性层
            self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc_shared = nn.Linear(64 * seq_len, 128)
            self.bn_shared = nn.BatchNorm1d(128)
            self.fc_cls = nn.Linear(128, self.total_positions)
            self.fc_reg = nn.Linear(128, self.total_positions)

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> 转置为 (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.bn_shared(self.fc_shared(x)))

        cls_logits = self.fc_cls(x)    # (batch, total_positions)
        reg_values = self.fc_reg(x)    # (batch, total_positions)
        return cls_logits, reg_values

    def kl_divergence(self):
        """汇总所有贝叶斯层的 KL 散度（仅贝叶斯模式下有效）"""
        if not self.use_bayesian:
            return torch.tensor(0.0)
        kl = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, (BayesianConv1d, BayesianLinear)):
                kl = kl + module.kl_divergence()
        return kl

    def __str__(self):
        mode = "Bayesian" if self.use_bayesian else "Deterministic"
        return f"CNN_LBA_Model({mode})"


def build_lba_labels(X_adv, X_clean, device):
    """
    从对抗样本与干净样本构建 LBA 训练标签
    返回: mask(0/1), perturb(扰动值)，形状均为 (batch, seq_len*features)
    """
    batch_size = X_clean.shape[0]
    seq_len = X_clean.shape[1]
    num_features = X_clean.shape[2]
    total = seq_len * num_features

    eta = X_adv - X_clean  # (batch, seq_len, features)
    mask = (eta.abs() > 1e-8).float().reshape(batch_size, total)
    perturb = eta.reshape(batch_size, total)
    return mask.to(device), perturb.to(device)


def train_lba_model(train_data, model, batch_size=25, learning_rate=0.001, epochs=50,
                    device='cpu', print_info=False, n=1, use_bayesian=False):
    """
    训练 LBA 模型
    分类损失: 
      - n=1 时使用 CrossEntropyLoss (单标签多分类，与原论文一致)
      - n>1 时使用 BCEWithLogitsLoss (多标签二分类)
    回归损失: MSELoss (仅在真实扰动位置计算)
    贝叶斯模式: 额外添加 KL 散度正则项，约束后验权重接近先验
    """
    if n == 1:
        criterion_cls = nn.CrossEntropyLoss()
    else:
        criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 贝叶斯模式下使用 KL 散度权重系数：1 / (batch_size * num_batches)
    # 保证 KL 项与数据似然项量级相当
    num_batches = max(1, len(train_data) // batch_size)
    kl_weight = 1.0 / (batch_size * num_batches) if use_bayesian else 0.0

    loss_cls_list = []
    loss_reg_list = []

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss_cls = 0.0
        total_loss_reg = 0.0

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        for inputs, mask_true, perturb_true in train_dataloader:
            inputs = inputs.to(device)
            mask_true = mask_true.to(device)
            perturb_true = perturb_true.to(device)

            outputs_cls, outputs_reg = model(inputs)

            optimizer.zero_grad()
            # 分类损失: 根据 n 的值选择损失函数
            if n == 1:
                # 单标签多分类: 将 mask 转换为类别索引
                # mask_true shape: (batch, total_positions) -> 找到唯一的 1 的位置
                cls_targets = mask_true.argmax(dim=1)  # (batch,)
                loss_cls = criterion_cls(outputs_cls, cls_targets)
            else:
                loss_cls = criterion_cls(outputs_cls, mask_true)

            # 回归损失: 仅计算有扰动的位置
            mask_bool = mask_true.bool()
            if mask_bool.any():
                loss_reg = criterion_reg(outputs_reg[mask_bool], perturb_true[mask_bool])
            else:
                loss_reg = torch.tensor(0.0, device=device)

            # 联合损失
            loss = loss_cls + 0.5 * loss_reg

            # 贝叶斯模式：添加 KL 散度正则项
            if use_bayesian:
                kl_loss = model.kl_divergence() * kl_weight
                loss = loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss_cls += loss_cls.item()
            total_loss_reg += loss_reg.item()

        avg_cls = total_loss_cls / len(train_dataloader)
        avg_reg = total_loss_reg / len(train_dataloader)
        loss_cls_list.append(avg_cls)
        loss_reg_list.append(avg_reg)

        if print_info and (epoch + 1) % 10 == 0:
            kl_str = f", KL: {model.kl_divergence().item() * kl_weight:.6f}" if use_bayesian else ""
            print(f"  LBA Epoch {epoch + 1}/{epochs}, "
                  f"Cls Loss: {avg_cls:.6f}, Reg Loss: {avg_reg:.6f}{kl_str}")

    return loss_cls_list, loss_reg_list


# =============================================================================
# SECTION 6b: LBA 拟合能力评估指标
# =============================================================================

def calc_sensitive_point_ar(mask_true, cls_logits, n):
    """
    计算敏感点预测准确率 AR（Accuracy Rate）
    衡量 LBA 预测的 Top-n 敏感点与 nVITA 真实敏感点的重合比例
    对应原论文公式 (10)

    mask_true:  (batch, total_positions) 真实扰动掩码 (0/1)
    cls_logits: (batch, total_positions) LBA 输出的分类 logits
    n:          每个样本的扰动点数
    """
    batch_size = mask_true.shape[0]
    _, top_pred = torch.topk(cls_logits, n, dim=1)
    ar_total = 0.0
    valid_cnt = 0
    for i in range(batch_size):
        true_pos = set(torch.where(mask_true[i] > 0.5)[0].cpu().numpy())
        pred_pos = set(top_pred[i].cpu().numpy())
        if len(true_pos) == 0:
            continue
        overlap = len(true_pos & pred_pos)
        ar_total += overlap / len(true_pos)
        valid_cnt += 1
    return ar_total / valid_cnt if valid_cnt > 0 else 0.0


def calc_perturb_rmse(mask_true, perturb_true, perturb_pred):
    """
    计算扰动值预测 RMSE
    在真实扰动位置上，衡量 LBA 预测扰动值与 nVITA 真实扰动值的均方根误差
    对应原论文的扰动拟合能力评估

    mask_true:    (batch, total_positions) 真实扰动掩码 (0/1)
    perturb_true: (batch, total_positions) nVITA 真实扰动值
    perturb_pred: (batch, total_positions) LBA 预测扰动值
    """
    mask_bool = mask_true.bool()
    if not mask_bool.any():
        return 0.0
    rmse = torch.sqrt(F.mse_loss(perturb_pred[mask_bool], perturb_true[mask_bool]))
    return rmse.item()


def evaluate_lba_fitting_quality(lba_model, X_eval, mask_true, perturb_true, n, device, batch_size=64):
    """
    批量评估 LBA 模型的拟合质量，返回 AR 和扰动 RMSE
    lba_model: 已训练的 LBA 模型
    X_eval:    (N, seq_len, features) 评估集原始输入
    mask_true: (N, total_positions)   nVITA 真实扰动掩码
    perturb_true: (N, total_positions) nVITA 真实扰动值
    """
    lba_model.to(device)
    lba_model.eval()
    all_cls_logits = []
    all_reg_values = []
    N = X_eval.shape[0]
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_eval[start:end].to(device)
            cls_logits, reg_values = lba_model(X_batch)
            all_cls_logits.append(cls_logits)
            all_reg_values.append(reg_values)
    all_cls_logits = torch.cat(all_cls_logits, dim=0)
    all_reg_values = torch.cat(all_reg_values, dim=0)

    ar = calc_sensitive_point_ar(mask_true, all_cls_logits, n)
    perturb_rmse = calc_perturb_rmse(mask_true, perturb_true, all_reg_values)
    return ar, perturb_rmse


# =============================================================================
# SECTION 7: nVITA BASELINE ATTACK (差分进化实现)
# =============================================================================

class NVITA:
    """
    n-Values Time Series Attack (nVITA)
    基于差分进化 DE/rand/1/bin 的稀疏黑盒攻击
    支持特征数量约束: feature_constraint=None 表示无约束，整数表示限定扰动的特征数
    """

    def __init__(self, n, epsilon, model, feature_ranges, maxiter=60, pop_size=15,
                 F=0.8, CR=0.9, targeted=False, feature_constraint=None, perturb_features=None):
        self.n = n                      # 总扰动点数
        self.epsilon = epsilon          # 扰动预算系数
        self.model = model
        self.feature_ranges = feature_ranges  # 每个特征的全局取值范围 (numpy array)
        self.maxiter = maxiter
        self.pop_size = pop_size
        self.F = F                      # DE 变异因子
        self.CR = CR                    # DE 交叉概率
        self.targeted = targeted
        self.feature_constraint = feature_constraint  # 限定扰动特征数量
        self.perturb_features = perturb_features  # 精确指定要扰动的特征索引

    def _sample_features(self, num_features):
        """根据约束采样允许扰动的特征索引"""
        # 如果指定了精确的扰动特征列表
        if self.perturb_features is not None:
            # 验证特征索引是否有效
            valid_features = [f for f in self.perturb_features if 0 <= f < num_features]
            if valid_features:
                return np.array(valid_features)
            # 如果没有有效特征，回退到允许所有特征
        # 原有的数量约束逻辑
        if self.feature_constraint is None or self.feature_constraint >= num_features:
            return np.arange(num_features)
        return np.random.choice(num_features, self.feature_constraint, replace=False)

    def _init_individual(self, seq_len, num_features, allowed_features):
        """初始化一个个体: n组 (时间步, 特征, 扰动值) 展平为 3n 维向量
        确保 n 个扰动点的 (时间步, 特征) 位置唯一，避免重复扰动
        """
        positions = set()
        ind = []
        max_attempts = self.n * 100
        attempts = 0
        while len(positions) < self.n and attempts < max_attempts:
            t = np.random.randint(0, seq_len)
            f = int(np.random.choice(allowed_features))
            pos_key = (t, f)
            if pos_key not in positions:
                positions.add(pos_key)
                budget = self.epsilon * self.feature_ranges[f]
                p = np.random.uniform(-budget, budget)
                ind.extend([float(t), float(f), p])
            attempts += 1
        # 如果无法生成足够的唯一位置（极端情况），回退填充
        while len(positions) < self.n:
            t = np.random.randint(0, seq_len)
            f = int(np.random.choice(allowed_features))
            budget = self.epsilon * self.feature_ranges[f]
            p = np.random.uniform(-budget, budget)
            ind.extend([float(t), float(f), p])
            positions.add((t, f))
        return np.array(ind)

    def _apply_perturbation(self, x_np, individual):
        """将个体编码的扰动应用到输入样本上"""
        x_adv = x_np.copy()
        seq_len = x_np.shape[1]
        num_features = x_np.shape[2]
        for i in range(self.n):
            t = int(np.clip(round(individual[3 * i]), 0, seq_len - 1))
            f = int(np.clip(round(individual[3 * i + 1]), 0, num_features - 1))
            p = individual[3 * i + 2]
            x_adv[0, t, f] += p
        # 截断到归一化范围 [0, 1]
        x_adv = np.clip(x_adv, 0.0, 1.0)
        return x_adv

    def attack(self, X, y, seed=None):
        """
        对单个样本执行 nVITA 攻击
        X: (1, seq_len, features)
        y: 真实标签
        """
        if seed is not None:
            np.random.seed(seed)

        device = X.device
        x_np = X.detach().cpu().numpy()
        y_val = y.detach().cpu().item()
        seq_len = x_np.shape[1]
        num_features = x_np.shape[2]
        allowed_features = self._sample_features(num_features)

        # 初始化种群
        population = []
        for _ in range(self.pop_size):
            population.append(self._init_individual(seq_len, num_features, allowed_features))
        population = np.array(population)

        # 计算初始适应度
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            x_adv = self._apply_perturbation(x_np, population[i])
            with torch.no_grad():
                pred = self.model(torch.FloatTensor(x_adv).to(device)).item()
            # 非目标攻击: 使用 MSE 作为适应度（与原论文一致）
            fitness[i] = (pred - y_val) ** 2

        best_idx = np.argmax(fitness)
        best_ind = population[best_idx].copy()
        best_fit = fitness[best_idx]

        # DE 主循环
        for gen in range(self.maxiter):
            for i in range(self.pop_size):
                # 1. 变异 DE/rand/1
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])

                # 边界处理
                for k in range(self.n):
                    # 时间步取整截断
                    mutant[3 * k] = np.clip(round(mutant[3 * k]), 0, seq_len - 1)
                    # 特征索引取整截断
                    mutant[3 * k + 1] = np.clip(round(mutant[3 * k + 1]), 0, num_features - 1)
                    f_idx = int(mutant[3 * k + 1])
                    # 扰动值截断到预算内
                    budget = self.epsilon * self.feature_ranges[f_idx]
                    mutant[3 * k + 2] = np.clip(mutant[3 * k + 2], -budget, budget)

                # 2. 二项式交叉
                trial = population[i].copy()
                j_rand = np.random.randint(0, 3 * self.n)
                for j in range(3 * self.n):
                    if np.random.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # 去重校验: 确保 n 个扰动点位置唯一
                seen_positions = set()
                for k in range(self.n):
                    t_val = int(np.clip(round(trial[3 * k]), 0, seq_len - 1))
                    f_val = int(np.clip(round(trial[3 * k + 1]), 0, num_features - 1))
                    trial[3 * k] = float(t_val)
                    trial[3 * k + 1] = float(f_val)
                    pos_key = (t_val, f_val)
                    retry = 0
                    while pos_key in seen_positions and retry < 20:
                        t_val = np.random.randint(0, seq_len)
                        f_val = int(np.random.choice(allowed_features))
                        trial[3 * k] = float(t_val)
                        trial[3 * k + 1] = float(f_val)
                        pos_key = (t_val, f_val)
                        retry += 1
                    seen_positions.add(pos_key)

                # 3. 贪婪选择
                x_adv_trial = self._apply_perturbation(x_np, trial)
                with torch.no_grad():
                    pred_trial = self.model(torch.FloatTensor(x_adv_trial).to(device)).item()
                fit_trial = (pred_trial - y_val) ** 2

                if fit_trial > fitness[i]:
                    population[i] = trial
                    fitness[i] = fit_trial
                    if fit_trial > best_fit:
                        best_fit = fit_trial
                        best_ind = trial.copy()

        x_adv_final = self._apply_perturbation(x_np, best_ind)
        return torch.FloatTensor(x_adv_final).to(device), best_fit


# =============================================================================
# SECTION 8: LBA ATTACK EXECUTION (修正版)
# =============================================================================

def run_nvita_attack(model, X_test, Y_test, beta, n, maxiter, tol, device,
                     feature_ranges, feature_constraint=None, perturb_features=None, print_info=False):
    """
    批量运行 nVITA 攻击，生成对抗样本
    beta: nVITA 扰动预算系数（原论文 β）
    """
    model.to(device)
    model.eval()

    X_adv_total = torch.empty(0).to(device)
    Y_adv_total = torch.empty(0).to(device)
    Y_pred_total = torch.empty(0).to(device)

    nvita = NVITA(
        n=n, epsilon=beta, model=model,
        feature_ranges=feature_ranges,
        maxiter=maxiter, targeted=False,
        feature_constraint=feature_constraint,
        perturb_features=perturb_features
    )

    total = X_test.shape[0]
    for test_ind in range(total):
        X_current = X_test[test_ind].unsqueeze(0).to(device)
        y_current = Y_test[test_ind].unsqueeze(0).to(device)

        X_adv, _ = nvita.attack(X_current, y_current, seed=test_ind)

        with torch.no_grad():
            original_pred = model(X_current).item()
            adv_pred = model(X_adv).item()

        X_adv_total = torch.cat((X_adv_total, X_adv), dim=0)
        Y_adv_total = torch.cat((Y_adv_total, torch.tensor([[adv_pred]]).to(device)), dim=0)
        Y_pred_total = torch.cat((Y_pred_total, torch.tensor([[original_pred]]).to(device)), dim=0)

        if print_info and (test_ind + 1) % 20 == 0:
            print(f"  nVITA progress: {test_ind + 1}/{total}")

    return X_adv_total, Y_adv_total, Y_pred_total


def run_lba_attack(model, lba_model, X_test, Y_test, delta, n, device,
                   feature_constraint=None, perturb_features=None, print_info=False):
    """
    使用训练好的 LBA 模型执行批量攻击
    delta: 扰动值缩放系数（原论文 δ）
    n: 选取的敏感点数量
    feature_constraint: 限定扰动特征数量
    perturb_features: 精确指定要扰动的特征索引
    """
    model.to(device)
    lba_model.to(device)
    model.eval()
    lba_model.eval()

    X_adv_total = torch.empty(0).to(device)
    Y_adv_total = torch.empty(0).to(device)
    Y_pred_total = torch.empty(0).to(device)

    seq_len = X_test.shape[1]
    num_features = X_test.shape[2]

    # 预处理精确扰动特征列表
    valid_perturb_features = None
    if perturb_features is not None:
        valid_perturb_features = [f for f in perturb_features if 0 <= f < num_features]

    for test_ind in range(X_test.shape[0]):
        X_current = X_test[test_ind].unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logits, reg_values = lba_model(X_current)

        # 如果指定了精确的扰动特征列表
        if valid_perturb_features:
            # 构造掩码: 只保留选中特征的位置
            mask = torch.zeros_like(cls_logits)
            mask_reshaped = mask.reshape(seq_len, num_features)
            mask_reshaped[:, valid_perturb_features] = 1.0
            # 非选中特征设为 -inf，不会被 topk 选中
            cls_logits = cls_logits + (mask - 1) * 1e9
        # 如果限定特征数量，则只在允许的特征内选点
        elif feature_constraint is not None and feature_constraint < num_features:
            # 选敏感度最高的 k 个特征
            cls_reshaped = cls_logits.reshape(seq_len, num_features)
            feat_sensitivity = cls_reshaped.sum(dim=0)
            top_feats = torch.topk(feat_sensitivity, feature_constraint).indices
            # 构造掩码: 只保留选中特征的位置
            mask = torch.zeros_like(cls_logits)
            mask_reshaped = mask.reshape(seq_len, num_features)
            mask_reshaped[:, top_feats] = 1.0
            # 非选中特征设为 -inf，不会被 topk 选中
            cls_logits = cls_logits + (mask - 1) * 1e9

        # 选取 Top-n 敏感点
        _, top_indices = torch.topk(cls_logits, n, dim=1)

        X_adv = X_current.clone()
        for idx in top_indices[0]:
            idx = idx.item()
            t_idx = idx // num_features
            f_idx = idx % num_features
            perturb = delta * reg_values[0, idx].item()
            X_adv[0, t_idx, f_idx] += perturb

        # 截断到 [0, 1]
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        with torch.no_grad():
            adv_pred = model(X_adv).item()
            original_pred = model(X_current).item()

        X_adv_total = torch.cat((X_adv_total, X_adv), dim=0)
        Y_adv_total = torch.cat((Y_adv_total, torch.tensor([[adv_pred]]).to(device)), dim=0)
        Y_pred_total = torch.cat((Y_pred_total, torch.tensor([[original_pred]]).to(device)), dim=0)

    return X_adv_total, Y_adv_total, Y_pred_total


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

    sd = np.sqrt(np.mean((Y_test_denorm - np.mean(Y_test_denorm)) ** 2))
    rse_clean = rmse_clean / sd if sd > 0 else 0
    rse_adv = rmse_adv / sd if sd > 0 else 0

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
# SECTION 9: INTEGRATED LBA PIPELINE FOR MoACB-WSF (修正版)
# =============================================================================

def run_lba_pipeline(model, X_test, Y_test, min_speed, max_speed, device,
                     beta=0.1, n=1, maxiter=60, tol=0.01,
                     adv_cnt=100, lba_epochs=50, lba_lr=0.001, lba_batch_size=25,
                     delta_list=None, use_bayesian=False,
                     feature_constraint=None, perturb_features=None, print_info=True):
    """
    完整 LBA 攻击管线（攻击目标为传入的 MoACB 模型）
    1. 拆分测试集为 LBA 训练子集 和 评估子集，严格隔离避免效果虚高
    2. 在训练子集上跑 nVITA 生成对抗样本，训练 LBA 模型
    3. 在评估子集上同时测试 nVITA 基线 和 LBA 攻击，保证对比公平

    beta: nVITA 扰动预算系数（原论文 β）
    delta_list: LBA 攻击缩放系数列表（原论文 δ）
    feature_constraint: 扰动特征数量约束 (None=不限制, 整数=限定特征数)
    perturb_features: 精确指定要扰动的特征索引 (None=不限制)
    """
    if delta_list is None:
        delta_list = [0.75, 1.0, 1.5, 1.75]

    results = {}
    seq_len = X_test.shape[1]
    num_features = X_test.shape[2]

    # 归一化后特征取值范围均为 [0,1]，极差=1
    feature_ranges = np.ones(num_features)

    print("\n" + "=" * 60)
    print("启动 LBA 对抗攻击评估流程 (目标模型: MoACB-WSF)")
    if perturb_features is not None:
        # 显示二进制掩码和对应的特征
        feature_names = ['风向', '理论功率', '实际功率', '风速']
        mask_str = ''.join(['1' if i in perturb_features else '0' for i in range(4)])
        feature_desc = []
        for i in perturb_features:
            if i < len(feature_names):
                feature_desc.append(f"{i}({feature_names[i]})")
        print(f"扰动特征掩码: {mask_str}")
        print(f"精确扰动特征: {', '.join(feature_desc)}")
    elif feature_constraint is not None:
        print(f"扰动特征数量约束: {feature_constraint} 个特征")
    else:
        print("扰动特征: 无限制（所有特征）")
    print("=" * 60)

    X_test_att = X_test.to(device)
    Y_test_att = Y_test.to(device)

    # ========== 关键修正: 拆分测试集为 LBA 训练子集 和 评估子集 ==========
    total_num = X_test_att.shape[0]
    indices = torch.randperm(total_num, device=device)
    train_idx = indices[:adv_cnt]   # LBA 训练用样本
    eval_idx = indices[adv_cnt:]    # 攻击效果评估用样本

    X_lba_train = X_test_att[train_idx]
    Y_lba_train = Y_test_att[train_idx]
    X_eval = X_test_att[eval_idx]
    Y_eval = Y_test_att[eval_idx]

    print(f"\n数据集拆分: 总计 {total_num} 个样本")
    print(f"  LBA 训练子集: {len(train_idx)} 个样本")
    print(f"  攻击评估子集: {len(eval_idx)} 个样本")

    # Step 1: 在训练子集上跑 nVITA，生成 LBA 训练数据
    print(f"\n[Step 1/4] nVITA 在 LBA 训练子集上生成对抗样本 (beta={beta}, n={n})...")
    X_adv_train, _, _ = run_nvita_attack(
        model, X_lba_train, Y_lba_train, beta, n, maxiter, tol, device,
        feature_ranges=feature_ranges,
        feature_constraint=feature_constraint,
        perturb_features=perturb_features,
        print_info=print_info
    )

    # Step 2: 构建 LBA 训练数据集
    print(f"\n[Step 2/4] 构建 LBA 训练集...")
    lba_data = LBA_Dataset(device=device)
    mask, perturb = build_lba_labels(X_adv_train, X_lba_train, device)
    lba_data.add_sample(X_lba_train, mask, perturb)
    print(f"  训练集大小: {len(lba_data)} 个样本")

    # Step 3: 训练 LBA 模型
    print(f"\n[Step 3/4] 训练 LBA 模型 (epochs={lba_epochs}, lr={lba_lr}, bayesian={use_bayesian})...")
    lba_model = CNN_LBA_Model(num_features, seq_len, n, use_bayesian=use_bayesian)
    train_lba_model(
        lba_data, lba_model,
        batch_size=lba_batch_size, learning_rate=lba_lr, epochs=lba_epochs,
        device=device, print_info=print_info, n=n, use_bayesian=use_bayesian
    )

    lba_save_path = os.path.join(OUTPUT_DIR, 'LBA_models', 'lba_model_moacb_wsf.pt')
    torch.save(lba_model.state_dict(), lba_save_path)
    print(f"  LBA 模型已保存至: {lba_save_path}")

    # Step 4: 在评估子集上同时测试 nVITA 基线 和 LBA 攻击
    print(f"\n[Step 4/4] 在评估子集上对比攻击性能...")

    # nVITA 基线攻击（在评估子集上）
    print(f"\n  --- nVITA Baseline (beta={beta}) on eval set ---")
    X_adv_nvita_eval, _, _ = run_nvita_attack(
        model, X_eval, Y_eval, beta, n, maxiter, tol, device,
        feature_ranges=feature_ranges,
        feature_constraint=feature_constraint,
        perturb_features=perturb_features,
        print_info=print_info
    )
    nvita_results = evaluate_attack(
        model, X_eval, Y_eval, X_adv_nvita_eval,
        min_speed, max_speed, "nVITA (Baseline)", device
    )
    results['nVITA'] = nvita_results

    # 构建 nVITA 真实扰动标签（用于评估 LBA 拟合能力）
    eval_mask_true, eval_perturb_true = build_lba_labels(X_adv_nvita_eval, X_eval, device)

    # LBA 攻击（在评估子集上）+ 拟合质量指标
    for delta in delta_list:
        print(f"\n  --- LBA Attack (delta={delta}) on eval set ---")
        X_adv_lba_eval, _, _ = run_lba_attack(
            model, lba_model, X_eval, Y_eval, delta, n, device,
            feature_constraint=feature_constraint, perturb_features=perturb_features, print_info=False
        )
        lba_results = evaluate_attack(
            model, X_eval, Y_eval, X_adv_lba_eval,
            min_speed, max_speed, f"LBA (delta={delta})", device
        )

        # 计算 LBA 拟合质量指标：AR 和扰动 RMSE
        ar, perturb_rmse = evaluate_lba_fitting_quality(
            lba_model, X_eval, eval_mask_true, eval_perturb_true, n, device
        )
        lba_results['sensitive_ar'] = ar
        lba_results['perturb_rmse'] = perturb_rmse
        print(f"  [LBA 拟合指标] 敏感点准确率 AR: {ar:.4f}, 扰动值 RMSE: {perturb_rmse:.6f}")

        results[f'LBA_delta_{delta}'] = lba_results

    # 总结对比
    print("\n" + "=" * 60)
    print("LBA 攻击评估总结 (目标模型: MoACB-WSF)")
    print("=" * 60)
    print(f"{'Attack Method':<20} {'RMSE_adv':>10} {'MAPE(%)':>10} {'Drop%':>10} {'AR':>8} {'Pert_RMSE':>10}")
    print("-" * 78)
    for key, val in results.items():
        ar_str = f"{val['sensitive_ar']:>8.4f}" if 'sensitive_ar' in val else f"{'—':>8}"
        pr_str = f"{val['perturb_rmse']:>10.6f}" if 'perturb_rmse' in val else f"{'—':>10}"
        print(f"{val['attack_name']:<20} {val['rmse_adv']:>10.4f} {val['mape_adv']:>10.2f} "
              f"{val['drop_rmse_pct']:>10.2f} {ar_str} {pr_str}")

    return results, lba_model, X_adv_nvita_eval


# =============================================================================
# SECTION 10: REPORT GENERATION
# =============================================================================

def generate_comprehensive_report(model, test_loader, val_loader, device, min_speed, max_speed,
                                  all_pareto_fronts, feature_columns, topo, cnn_params, lstm_params,
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
    fig.suptitle('MoACB-WSF with LBA Attack - 综合预测报告', fontsize=20, fontweight='bold', y=0.98)

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
    final_pareto = all_pareto_fronts[-1]
    if final_pareto['num_solutions'] > 0:
        perf_vals = final_pareto['performance']
        comp_vals = final_pareto['complexity']
        valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
        if np.any(valid_mask):
            ax5.scatter(comp_vals[valid_mask], perf_vals[valid_mask], c='darkgreen', s=100,
                        edgecolors='black', linewidth=1, alpha=0.8)
    ax5.set_xlabel('模型复杂度 (参数数量)', fontsize=12)
    ax5.set_ylabel('验证集 RMSE (m/s)', fontsize=12)
    ax5.set_title('最后一代Pareto前沿', fontsize=14)
    ax5.grid(True, alpha=0.3)

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
# SECTION 11: MAIN PROGRAM
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MoACB-WSF with LBA Adversarial Attack')
    parser.add_argument('--data_file', type=str, default='winddata.xlsx', help='Input data file')
    parser.add_argument('--no_attack', action='store_true', help='Skip LBA attack evaluation')
    parser.add_argument('--beta', type=float, default=0.1, help='nVITA perturbation budget (原论文 β)')
    parser.add_argument('--n_perturb', type=int, default=1, help='Number of perturbations (nVITA)')
    parser.add_argument('--lba_epochs', type=int, default=50, help='LBA training epochs')
    parser.add_argument('--lba_lr', type=float, default=0.001, help='LBA learning rate')
    parser.add_argument('--adv_cnt', type=int, default=100, help='Adv examples for LBA training')
    parser.add_argument('--delta_list', type=float, nargs='+', default=[0.75, 1.0, 1.5, 1.75],
                        help='LBA delta values (原论文 δ)')
    parser.add_argument('--use_bayesian', action='store_true', default=LBA_CONFIG['use_bayesian'], help='Use BayesianConv1d (requires blitz)')
    parser.add_argument('--feat_constraint', type=int, default=LBA_CONFIG['feature_constraint'],
                        help='Number of features allowed to perturb (None = all features)')
    parser.add_argument('--perturb_mask', type=str, default=LBA_CONFIG['perturb_mask'],
                        help='Binary mask for feature perturbation. E.g., "0010" means only perturb feature 2')
    # ====== 新增: 模型冻结与复用参数 ======
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model checkpoint (model weights + architecture params)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model checkpoint (skip NSGA-II search and training)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # 使用配置文件中的参数作为默认值
    if args.perturb_mask is None:
        args.perturb_mask = LBA_CONFIG['perturb_mask']
    if args.feat_constraint is None:
        args.feat_constraint = LBA_CONFIG['feature_constraint']
    
    # 将二进制掩码转换为特征索引列表
    def mask_to_features(mask):
        if not mask or mask == '1111':
            return None  # 不限制，所有特征都可以扰动
        features = []
        # 从左到右对应特征0到特征3
        for i, c in enumerate(mask):
            if c == '1':
                features.append(i)
        return features if features else None
    
    args.perturb_features = mask_to_features(args.perturb_mask)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'=' * 80}")
    print("MoACB-WSF: Multi-Objective Automated CNN-BiLSTM for Wind Speed Forecasting")
    print("with Learning-Based Adversarial Attack (LBA) Integration")
    print(f"{'=' * 80}")
    print(f"Device: {DEVICE}")
    print(f"Data file: {args.data_file}")

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
    test_loader_for_attack = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test_full, Y_test_full = next(iter(test_loader_for_attack))
    X_test_full = X_test_full.to(DEVICE)
    Y_test_full = Y_test_full.to(DEVICE)

    # ====== 新增: 判断是否触发 load_model 模式 ======
    load_model_mode = args.load_model is not None and os.path.isfile(args.load_model)
    if load_model_mode:
        print(f"\n>>> 检测到 --load_model 路径: {args.load_model}")
        print(">>> 将跳过 NSGA-II 搜索和最终训练，直接加载已训练模型进行攻击评估。")
        # 加载模型模式下，NUM_RUNS 循环只执行 1 次
        actual_num_runs = 1
    else:
        if args.load_model is not None and not os.path.isfile(args.load_model):
            print(f"\n>>> 警告: --load_model 指定的文件不存在: {args.load_model}")
            print(">>> 将按正常流程执行 NSGA-II 搜索和训练。")
        actual_num_runs = NUM_RUNS

    for run_id in range(actual_num_runs):
        print(f"\n{'*' * 80}")
        print(f"开始第 {run_id + 1}/{actual_num_runs} 次独立运行...")
        print(f"{'*' * 80}\n")

        prefix = f"{OUTPUT_DIR}/run{run_id + 1}_"

        print(f'使用设备: {DEVICE}')

        # ====== 新增: 加载模型或执行 NSGA-II 搜索 + 训练 ======
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
            all_pareto_fronts = []
            train_losses, val_losses = [], []
            test_performance = {}
            best_individual = None
        else:
            # ---------- 正常流程: NSGA-II 搜索 + 最终训练 ----------
            print('\n========== 开始NSGA-II优化自动化深度学习模型 ==========')
            start_time = time.time()

            population = initialize_population(POP_SIZE, train_dataset)
            print(f'初始化完成,种群大小: {len(population)}')

            best_rmse_history = []
            best_complexity_history = []
            all_pareto_fronts = []

            for gen in range(MAX_GEN):
                print(f'\n第 {gen + 1}/{MAX_GEN} 代...')
                performance, complexity = evaluate_population(
                    population, train_dataset, val_dataset, min_speed, max_speed,
                    num_features, SEQUENCE_LENGTH, DEVICE
                )
                fronts, rank = fast_non_dominated_sort(performance, complexity)
                distance = crowding_distance(performance, complexity, fronts)
                mating_pool = tournament_selection(population, rank, distance, POP_SIZE)
                offspring = crossover_population(mating_pool)
                mutated_offspring = []
                for child in offspring:
                    if np.random.random() < MUTATION_PROB:
                        mutated_child = variable_length_mutation(child)
                        mutated_offspring.append(mutated_child)
                    else:
                        mutated_offspring.append(child.copy())
                offspring_perf, offspring_comp = evaluate_population(
                    mutated_offspring, train_dataset, val_dataset, min_speed, max_speed,
                    num_features, SEQUENCE_LENGTH, DEVICE
                )
                combined_pop = population + mutated_offspring
                combined_perf = np.concatenate([performance, offspring_perf])
                combined_comp = np.concatenate([complexity, offspring_comp])
                combined_fronts, combined_rank = fast_non_dominated_sort(combined_perf, combined_comp)
                combined_dist = crowding_distance(combined_perf, combined_comp, combined_fronts)
                population, performance, complexity = environmental_selection(
                    combined_pop, combined_perf, combined_comp, combined_rank, combined_dist, POP_SIZE
                )
                pareto_front = {
                    'params': [combined_pop[i] for i in combined_fronts[0]],
                    'performance': combined_perf[combined_fronts[0]],
                    'complexity': combined_comp[combined_fronts[0]],
                    'num_solutions': len(combined_fronts[0]),
                    'generation': gen + 1
                }
                all_pareto_fronts.append(pareto_front)

                valid_combined_perf = combined_perf[np.isfinite(combined_perf)]
                if len(valid_combined_perf) > 0:
                    best_idx_combined = np.argmin(valid_combined_perf)
                    best_rmse_history.append(valid_combined_perf[best_idx_combined])
                    best_complexity_history.append(combined_comp[best_idx_combined])
                else:
                    best_rmse_history.append(float('inf'))
                    best_complexity_history.append(float('inf'))

            end_time = time.time()
            print(f'\n第 {run_id + 1} 次优化完成!总耗时: {end_time - start_time:.2f} 秒')

            # 选择最优个体
            final_pareto = all_pareto_fronts[-1]
            if final_pareto['num_solutions'] > 0:
                perf_vals = final_pareto['performance']
                comp_vals = final_pareto['complexity']
                valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
                if valid_mask.any():
                    perf_vals = perf_vals[valid_mask]
                    comp_vals = comp_vals[valid_mask]
                    params = [final_pareto['params'][i] for i in range(len(valid_mask)) if valid_mask[i]]
                    normalized_perf = (perf_vals - perf_vals.min()) / (perf_vals.max() - perf_vals.min() + 1e-10)
                    normalized_comp = (comp_vals - comp_vals.min()) / (comp_vals.max() - comp_vals.min() + 1e-10)
                    trade_off_scores = np.sqrt(normalized_perf ** 2 + normalized_comp ** 2)
                    best_idx = np.argmin(trade_off_scores)
                    best_individual = params[best_idx]
                else:
                    raise ValueError("最后一代没有有效解")
            else:
                all_perf = np.concatenate([pf['performance'] for pf in all_pareto_fronts])
                all_params = [p for pf in all_pareto_fronts for p in pf['params']]
                valid_mask = np.isfinite(all_perf)
                if valid_mask.any():
                    best_idx = np.argmin(all_perf[valid_mask])
                    best_individual = np.array(all_params)[valid_mask][best_idx]
                else:
                    raise ValueError("所有个体评估均失败!")

            topo, cnn_params, lstm_params, setting = decode_individual(best_individual)
            batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)

            print(f'\n第 {run_id + 1} 次运行最终优化的混合编码向量:')
            print(f'  topo:    {topo}')
            print(f'  CNN:     {cnn_params}')
            print(f'  BiLSTM:  {lstm_params}')
            print(f'  Setting: {setting}')

            print(f'\n第 {run_id + 1} 次运行最终优化的超参数数值:')
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

            # Pareto前沿演化图
            print('\n正在绘制 Pareto 前沿演化图...')
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(all_pareto_fronts)))
            for idx, pf in enumerate(all_pareto_fronts):
                if pf['num_solutions'] > 0:
                    perf = pf['performance']
                    comp = pf['complexity']
                    valid_mask = np.isfinite(perf) & np.isfinite(comp)
                    if np.any(valid_mask):
                        plt.scatter(comp[valid_mask], perf[valid_mask], c=[colors[idx]], s=60, alpha=0.6,
                                    label=f'第 {pf["generation"]} 代', edgecolors='k', linewidth=0.5)
            plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
            plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
            plt.title('NSGA-II Pareto 前沿演化过程', fontsize=16)
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{prefix}pareto_evolution_continuous_lr.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 关键代数Pareto前沿对比图（带连接线）
            print('\n正在绘制关键代数Pareto前沿对比图（带连接线）...')
            target_generations = [1, 5, 10, 20, 25, 30]
            plt.figure(figsize=(14, 10))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            markers = ['o', 's', '^', 'D', 'v', 'p']
            plotted_any = False
            for idx, target_gen in enumerate(target_generations):
                pf = None
                for front in all_pareto_fronts:
                    if front['generation'] == target_gen:
                        pf = front
                        break
                if pf is None or pf['num_solutions'] == 0:
                    continue
                perf = pf['performance']
                comp = pf['complexity']
                valid_mask = np.isfinite(perf) & np.isfinite(comp)
                if not np.any(valid_mask):
                    continue
                perf = perf[valid_mask]
                comp = comp[valid_mask]
                sort_idx = np.argsort(comp)
                comp_sorted = comp[sort_idx]
                perf_sorted = perf[sort_idx]
                plt.plot(comp_sorted, perf_sorted, color=colors[idx], linewidth=2, alpha=0.7,
                         label=f'第 {target_gen} 代 (n={len(perf)})')
                plt.scatter(comp, perf, c=colors[idx], marker=markers[idx], s=80, alpha=0.9,
                            edgecolors='black', linewidth=0.5)
                plotted_any = True
            if plotted_any:
                plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
                plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
                plt.title('NSGA-II Pareto前沿进化过程对比（带连接线）', fontsize=16, fontweight='bold')
                plt.legend(fontsize=11, loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.figtext(0.5, 0.02,
                            '注：第1代为初始种群，第30代为最终进化结果\n每条实线连接该代的所有Pareto最优解，展示前沿面形状',
                            ha='center', fontsize=10, style='italic')
                plt.tight_layout(rect=[0, 0.05, 1, 0.96])
                plt.savefig(f'{prefix}pareto_evolution_comparison_connected.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 最终代Pareto前沿详细图
            print('\n正在绘制最终代Pareto前沿详细图（第30代）...')
            final_pf = None
            for pf in all_pareto_fronts:
                if pf['generation'] == 30:
                    final_pf = pf
                    break
            if final_pf and final_pf['num_solutions'] > 0:
                perf_final = final_pf['performance']
                comp_final = final_pf['complexity']
                valid_final = np.isfinite(perf_final) & np.isfinite(comp_final)
                if np.any(valid_final):
                    perf_final = perf_final[valid_final]
                    comp_final = comp_final[valid_final]
                    sort_idx = np.argsort(comp_final)
                    comp_sorted = comp_final[sort_idx]
                    perf_sorted = perf_final[sort_idx]
                    plt.figure(figsize=(12, 8))
                    plt.plot(comp_sorted, perf_sorted, color='darkred', linewidth=3, alpha=0.8,
                             marker='o', markersize=10, markerfacecolor='red',
                             markeredgecolor='black', markeredgewidth=1.5)
                    plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
                    plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
                    plt.title('最终Pareto前沿（第30代）', fontsize=16, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    for i, (comp_val, perf_val) in enumerate(zip(comp_sorted, perf_sorted)):
                        plt.annotate(f'({comp_val:.0f}, {perf_val:.3f})', xy=(comp_val, perf_val),
                                     xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(f'{prefix}pareto_front_final_gen30.png', dpi=300, bbox_inches='tight')
                    plt.show()

            # 保存结果
            with open(f'{prefix}modeo_cnn_optimization_continuous_lr.pkl', 'wb') as f:
                pickle.dump({
                    'best_individual': best_individual,
                    'best_rmse_history': best_rmse_history,
                    'pareto_fronts': all_pareto_fronts,
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
                    'run_id': run_id + 1,
                }, f)

            # ====== 新增: 保存模型 checkpoint ======
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
                all_pareto_fronts, feature_columns, topo, cnn_params, lstm_params, setting,
                test_performance, r_test, save_prefix=prefix
            )

            # 保存测试结果CSV
            print('\n========== 代码运行结束后保存文件 ==========')
            test_results_df = pd.DataFrame({
                '真实风速 (m/s)': all_test_targets,
                '预测风速 (m/s)': all_test_outputs
            })
            test_results_df.to_csv(f'{prefix}test_set_wind_speed_predictions.csv', index=False, encoding='utf-8-sig')

            # 保存每代Pareto前沿CSV
            for pf in all_pareto_fronts:
                gen = pf['generation']
                perf = pf['performance']
                comp = pf['complexity']
                valid_mask = np.isfinite(perf) & np.isfinite(comp)
                if np.any(valid_mask):
                    pareto_df = pd.DataFrame({
                        '模型复杂度 (参数数量)': comp[valid_mask],
                        '验证集RMSE (m/s)': perf[valid_mask]
                    })
                    pareto_df = pareto_df.sort_values(by='模型复杂度 (参数数量)')
                    filename = f'{prefix}pareto_front_generation_{gen}.csv'
                    pareto_df.to_csv(filename, index=False, encoding='utf-8-sig')
        # ====== 新增: if/else 块结束 ======

        # ==================== LBA Attack Phase ====================
        if not args.no_attack:
            print(f"\n{'=' * 80}")
            print(f"开始 LBA 对抗攻击评估 (第 {run_id + 1} 次运行)")
            print(f"{'=' * 80}")

            lba_results, lba_model, X_adv_nvita = run_lba_pipeline(
                model, X_test_full, Y_test_full, min_speed, max_speed, DEVICE,
                beta=args.beta, n=args.n_perturb,
                maxiter=LBA_CONFIG['maxiter'], tol=LBA_CONFIG['tol'],
                adv_cnt=args.adv_cnt, lba_epochs=args.lba_epochs,
                lba_lr=args.lba_lr, lba_batch_size=LBA_CONFIG['lba_batch_size'],
                delta_list=args.delta_list, use_bayesian=args.use_bayesian,
                feature_constraint=args.feat_constraint,
                perturb_features=args.perturb_features,
                print_info=True
            )

            # Save LBA results
            with open(f'{prefix}lba_attack_results.pkl', 'wb') as f:
                pickle.dump({
                    'lba_results': lba_results,
                    'config': LBA_CONFIG,
                    'feat_constraint': args.feat_constraint,
                    'run_id': run_id + 1,
                }, f)
            print(f"\nLBA 攻击结果已保存至: {prefix}lba_attack_results.pkl")

        print(f"\n{'=' * 80}")
        print(f"第 {run_id + 1} 次运行全部完成！")
        print(f"输出文件保存至目录: {OUTPUT_DIR}/")
        print(f"  - 模型优化结果: {prefix}modeo_cnn_optimization_continuous_lr.pkl")
        print(f"  - 综合报告: {prefix}comprehensive_report.png")
        print(f"  - 测试集预测: {prefix}test_set_wind_speed_predictions.csv")
        if not args.no_attack:
            print(f"  - LBA攻击结果: {prefix}lba_attack_results.pkl")
        print(f"{'=' * 80}")

    print(f"\n{'#' * 80}")
    print(f"所有 {NUM_RUNS} 次独立运行已完成！")
    print(f"{'#' * 80}")


if __name__ == '__main__':
    main()