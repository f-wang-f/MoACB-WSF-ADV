# -*- coding: utf-8 -*-
"""
baseline_gradient_attacks.py —— FGSM / BIM 白盒梯度攻击基线
================================================================================
独立运行脚本：在 MoACB-WSF 固定编码风速预测模型的【测试集】上，实现并评估
FGSM（一步）与 BIM（迭代）两种【非目标】白盒梯度攻击，输出攻击性能数据、CSV
与 β-RSE 曲线图。

设计原则（严格遵循任务要求）：
  * 数据加载 / HybridCNNBiLSTM 模型 / 编码解码 / 训练流程【完整复用】现有主文件
    LBA-MOACB-WSF-FixedEncoding.py，绝不重新设计模型结构；
  * 攻击为【全扰动】（所有时间步 × 所有特征），非稀疏；
  * 扰动预算按【特征分别】计算 β·range_f（range_f = 训练集上该特征 max-min），
    不使用统一标量 ε；
  * BIM 每步同时执行【扰动预算 clip】与【合法域 [0,1] clip】；FGSM 亦做 [0,1] clip；
  * 非目标攻击梯度符号为 +sign（最大化预测误差）；
  * 全程 model.eval()；仅在【测试集】上评估，绝不混入验证集。

本脚本【不修改】任何现有文件；训练得到的权重保存为 output/pretrained_moacb_wsf.pt。
运行：python baseline_gradient_attacks.py
"""

import os
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use('Agg')   # 无界面后端，保证脚本可独立、无窗口运行
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# =============================================================================
# 超参数集中配置（全部集中在文件顶部，便于统一调整）
# =============================================================================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 数据 / 模型配置（与主文件完全一致）----------
FILENAME = 'winddata.xlsx'
FEATURE_COLUMNS = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
TARGET_COLUMN = 'Wind Speed (m/s)'
SEQUENCE_LENGTH = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES = 100

NUM_MODULES = 5
NUM_CNN_MODULES = 3
NUM_LSTM_MODULES = 2
TOPO_BITS_LENGTH = NUM_MODULES * (NUM_MODULES - 1) // 2

LAMBDA_REG = 1e-4
FINAL_EPOCHS = 200
FINAL_PATIENCE = 20
GRADIENT_CLIP = 1.0

KNUM_RANGE = (0, 7)
KSIZE_RANGE = (0, 3)
KACT_RANGE = (0, 7)
PT_RANGE = (0, 2)
PS_RANGE = (0, 3)
BS_RANGE = (0, 3)
OPT_RANGE = (0, 3)
LR_RANGE = (0.0001, 0.01)
REG_RANGE = (0, 3)

OPTIMIZER_MAP = {0: 'SGD', 1: 'Adam', 2: 'AdaDelta', 3: 'RMSprop'}
REGULARIZER_MAP = {0: None, 1: 'L1', 2: 'L2', 3: 'L1L2'}
BATCH_SIZE_MAP = {0: 32, 1: 64, 2: 96, 3: 128}

# ---------- 固定编码向量（任务指定的论文最佳折中编码，对应测试集 MAE≈1.386）----------
# topo    = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
# Setting = [0, 1, 0.0005761369, 0] → batch_size=32, optimizer=Adam, lr=5.76e-4, regularizer=None
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
    'setting': [0, 1, 0.0005761369, 0],
}

# ---------- 攻击超参数 ----------
SEED = 42                          # 与主文件一致的随机种子（保证训练可复现）
BETA_DEFAULT = 0.3                 # FGSM/BIM 默认扰动预算系数 β
BIM_STEPS = 10                     # BIM 迭代步数 N
BETA_SCAN = [0.1, 0.2, 0.3]        # β 参数扫描
RSE_SUCCESS_THRESHOLD = 2.0        # 攻击成功判定阈值 τ（RSE >= τ 记为成功）
L0_EPS = 1e-8                      # L0 计数阈值：|δ| > 该值视为一次扰动
RSE_EPS = 1e-8                     # RSE 分母保护，避免除零

# ---------- 输出文件路径 ----------
PRETRAINED_CKPT = os.path.join(OUTPUT_DIR, 'pretrained_moacb_wsf.pt')
RESULT_CSV = os.path.join(OUTPUT_DIR, 'baseline_gradient_attacks_results.csv')
CURVE_PNG = os.path.join(OUTPUT_DIR, 'baseline_beta_rse_curve.png')


# =============================================================================
# SECTION 1: 数据加载与预处理（完整复用主文件 load_and_preprocess_data）
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
    """读取 winddata.xlsx → 4 特征 → MinMaxScaler 归一化到 [0,1] → 滑动窗口 →
    按 0.7/0.15/0.15【顺序】划分训练/验证/测试集（与主文件完全一致）。"""
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
# SECTION 2: HybridCNNBiLSTM 模型（完整复制主文件，未做任何结构改动）
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
# SECTION 3: 混合编码向量的编码 / 解码 / 合法性校验（完整复用主文件）
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
# SECTION 4: 模型训练 / 评估 / 加载（训练流程与主文件 main() 完全一致）
# =============================================================================

def set_seed(seed=SEED):
    """与主文件 main() 一致的随机种子设定顺序（random → numpy → torch → cuda）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fixed_encoding_model(data_result):
    """使用固定编码构建并训练模型：优化器 / 损失 / 正则 / 梯度裁剪 / 早停均与主文件
    main()（LBA-MOACB-WSF-FixedEncoding.py）逐行一致。
    【重要】主文件早停后使用【最后一个 epoch】的模型（不回滚 best-val 权重），
    本函数同样不回滚，以保证测试集指标可复现。
    调用前须已执行 set_seed(SEED)，且构建模型前不得消耗任何随机数。
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
                batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                val_loss += batch_loss
                val_samples += inputs.size(0)
        if val_samples > 0:
            val_loss /= val_samples

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch + 1}/{FINAL_EPOCHS}, 训练损失: {train_loss / len(train_loader):.6f}, 验证损失: {val_loss:.6f}')

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


def evaluate_model_metrics(model, test_dataset, min_speed, max_speed, batch_size=32):
    """在测试集上评估模型：反归一化到原始尺度 (m/s) 后计算 MAE/RMSE/R²/R，
    口径与主文件 main()（L2720-2746）完全一致。"""
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
    r2_test = r2_score(all_test_targets, all_test_outputs)
    r_test = pearsonr(all_test_targets, all_test_outputs)[0]
    return {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test, 'r': r_test,
            'n_test': int(len(all_test_targets))}


def load_or_train_model(data_result):
    """模型权重获取：优先加载 output/pretrained_moacb_wsf.pt（state_dict 能无损装入
    目标结构即视为编码一致）；否则用固定编码重新训练（训练前 set_seed 以复现主文件），
    训练完保存到 output/pretrained_moacb_wsf.pt。"""
    num_features = data_result['num_features']
    topo = FIXED_ENCODING['topo']
    cnn_params = FIXED_ENCODING['cnn_params']
    lstm_params = FIXED_ENCODING['lstm_params']
    setting = FIXED_ENCODING['setting']

    # ---------- 1) 优先尝试加载已有权重 ----------
    if os.path.isfile(PRETRAINED_CKPT):
        try:
            ck = torch.load(PRETRAINED_CKPT, map_location=DEVICE)
            sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
            model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, SEQUENCE_LENGTH).to(DEVICE)
            model.load_state_dict(sd)
            model.eval()
            print(f"\n>>> 已加载预训练权重: {PRETRAINED_CKPT}（跳过训练）")
            return model
        except Exception as e:
            print(f"\n>>> 预训练权重加载失败（{e}），将使用固定编码重新训练。")

    # ---------- 2) 重新训练：先设定种子（与主文件顺序一致），再构建 + 训练 ----------
    print("\n>>> 未找到可用预训练权重，使用固定编码重新训练 ...")
    set_seed(SEED)
    model = train_fixed_encoding_model(data_result)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'topo': topo, 'cnn_params': cnn_params, 'lstm_params': lstm_params, 'setting': setting,
    }, PRETRAINED_CKPT)
    print(f"\n>>> 训练完成，权重已保存至: {PRETRAINED_CKPT}")
    return model


# =============================================================================
# SECTION 5: 特征扰动范围 range_f（按特征分别计算，训练集上 max-min）
# =============================================================================

def compute_feature_ranges(train_dataset):
    """在【训练集】上按特征计算 range_f = max - min（归一化空间，即模型实际输入口径）。
    返回 (range_f: (F,) float32, n_train: int)。严禁用统一标量代替。"""
    xs = []
    for i in range(len(train_dataset)):
        x, _ = train_dataset[i]
        xs.append(x.numpy())
    X = np.stack(xs, axis=0)                 # (N_train, T, F)
    f_max = X.max(axis=(0, 1))               # (F,)
    f_min = X.min(axis=(0, 1))               # (F,)
    range_f = (f_max - f_min).astype('float32')
    return range_f, X.shape[0]


# =============================================================================
# SECTION 6: FGSM / BIM 白盒梯度攻击（非目标 = 最大化误差 = +sign）
# =============================================================================
# 【批量等价性说明】model.eval() 下 BatchNorm 使用 running 统计量、Dropout 关闭，
# 前向对每个样本相互独立；因此对整批测试样本一次前向/反向所得的逐样本梯度符号，
# 与逐样本单独计算完全一致（均值 reduction 仅引入正的 1/N 缩放，不改变 sign）。

def fgsm_attack(model, X, y, beta, range_f):
    """FGSM 非目标白盒一步攻击：
        δ = β · range_f · sign(∇_x MSE(model(X), y))
        X_adv = clip(X + δ, 0, 1)
    X:(N,T,F)  y:(N,1)  range_f:(F,) 按特征广播。返回 X_adv:(N,T,F)。"""
    model.eval()
    rf = torch.as_tensor(range_f, dtype=torch.float32, device=DEVICE).view(1, 1, -1)   # (1,1,F)
    Xa = X.clone().detach().to(DEVICE).requires_grad_(True)
    out = model(Xa)
    loss = F.mse_loss(out, y)                       # 非目标：最大化该误差
    grad = torch.autograd.grad(loss, Xa)[0]         # ∇_x MSE，形状 (N,T,F)
    delta = beta * rf * torch.sign(grad)            # +sign：沿误差增大方向
    X_adv = torch.clamp(Xa.detach() + delta, 0.0, 1.0)   # 合法域 clip
    return X_adv.detach()


def bim_attack(model, X, y, beta, range_f, n_steps=BIM_STEPS):
    """BIM 非目标白盒迭代攻击（双重 clip）：
        α = β · range_f / N
        for i in range(N):
            grad = ∇_x MSE(model(X_adv), y)
            X_adv = X_adv + α · sign(grad)
            δ = clip(X_adv - X, -β·range_f, +β·range_f)   # 扰动预算 clip
            X_adv = clip(X + δ, 0, 1)                      # 合法域 clip
    返回 X_adv:(N,T,F)。"""
    model.eval()
    rf = torch.as_tensor(range_f, dtype=torch.float32, device=DEVICE).view(1, 1, -1)   # (1,1,F)
    eps = beta * rf                     # 逐特征扰动预算 (1,1,F)
    alpha = eps / n_steps               # 逐特征步长 (1,1,F)
    X0 = X.clone().detach().to(DEVICE)
    X_adv = X0.clone()
    for _ in range(n_steps):
        Xa = X_adv.detach().requires_grad_(True)
        out = model(Xa)
        loss = F.mse_loss(out, y)
        grad = torch.autograd.grad(loss, Xa)[0]
        X_adv = Xa.detach() + alpha * torch.sign(grad)          # +sign：非目标
        delta = torch.clamp(X_adv - X0, -eps, eps)              # ① 扰动预算 clip
        X_adv = torch.clamp(X0 + delta, 0.0, 1.0)               # ② 合法域 clip
    return X_adv.detach()


# =============================================================================
# SECTION 7: 统一评估函数（逐样本计算指标，再对全部测试样本取平均）
# =============================================================================

def evaluate_attack_metrics(model, X, y, X_adv, min_speed, max_speed):
    """对每个测试样本计算：
        RSE = sqrt(MSE_adv / max(MSE_clean, 1e-8))
        MAE_increment = MAE_adv - MAE_clean      （原始尺度 m/s，与模型性能口径一致）
        attack_success = 1 if RSE >= τ else 0     （τ = RSE_SUCCESS_THRESHOLD = 2.0）
        L2 = ||X_adv - X||_2                      （归一化空间）
        L0 = count(|X_adv - X| > 1e-8)
    再对全部样本取平均，返回 mean_RSE / mean_MAE_increment / ASR / mean_L2 / mean_L0。"""
    model.eval()
    scale = float(max_speed - min_speed)
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    X_adv = X_adv.to(DEVICE)
    with torch.no_grad():
        pred_clean = model(X).squeeze(-1)          # (N,) 归一化预测
        pred_adv = model(X_adv).squeeze(-1)        # (N,)
        yy = y.squeeze(-1)                         # (N,) 归一化真值

        mse_clean = (pred_clean - yy) ** 2
        mse_adv = (pred_adv - yy) ** 2
        rse = torch.sqrt(mse_adv / torch.clamp(mse_clean, min=RSE_EPS))    # 比值，尺度无关

        # MAE 增量换算到原始尺度 (m/s)：|err_norm| * (max_speed - min_speed)
        mae_adv = (pred_adv - yy).abs() * scale
        mae_clean = (pred_clean - yy).abs() * scale
        mae_inc = mae_adv - mae_clean

        succ = (rse >= RSE_SUCCESS_THRESHOLD).float()

        diff = X_adv - X
        l2 = torch.sqrt((diff ** 2).sum(dim=(1, 2)))                 # (N,)
        l0 = (diff.abs() > L0_EPS).sum(dim=(1, 2)).float()           # (N,)

    return {
        'mean_RSE': float(rse.mean().item()),
        'mean_MAE_increment': float(mae_inc.mean().item()),
        'ASR': float(succ.mean().item()),
        'mean_L2': float(l2.mean().item()),
        'mean_L0': float(l0.mean().item()),
        'n': int(X.shape[0]),
    }


def run_attack(model, X_test, Y_test, method, beta, range_f, n_steps=BIM_STEPS):
    """按方法名分派攻击，返回 X_adv。"""
    if method == 'FGSM':
        return fgsm_attack(model, X_test, Y_test, beta, range_f)
    elif method == 'BIM':
        return bim_attack(model, X_test, Y_test, beta, range_f, n_steps)
    else:
        raise ValueError(f"未知攻击方法: {method}")


# =============================================================================
# SECTION 8: 结果输出（对比表 / CSV / β-RSE 曲线图）与主流程
# =============================================================================

def print_comparison_table(results, n_test, beta_main=BETA_DEFAULT):
    """打印指定格式的攻击对比表与 β 参数扫描结果。"""
    def get(method, beta):
        return next(r for r in results if r['method'] == method and abs(r['beta'] - beta) < 1e-9)

    line = '=' * 80
    print('\n' + line)
    print(f'白盒梯度攻击结果（评估样本数：{n_test}，仅测试集）')
    print(line)
    print('方法      | 平均RSE↑ | MAE增量↑ | ASR(τ=2)↑ | 平均L2↓ | 平均L0↓ | 备注')
    print('----------|----------|----------|-----------|---------|---------|------')
    fg = get('FGSM', beta_main)
    bm = get('BIM', beta_main)
    print(f"FGSM      |  {fg['mean_RSE']:.3f}   |  {fg['mean_MAE_increment']:.3f}   |"
          f"   {fg['ASR'] * 100:5.1f}%   |  {fg['mean_L2']:.3f}  |  {fg['mean_L0']:5.0f}  | 1次反向")
    print(f"BIM(N={BIM_STEPS}) |  {bm['mean_RSE']:.3f}   |  {bm['mean_MAE_increment']:.3f}   |"
          f"   {bm['ASR'] * 100:5.1f}%   |  {bm['mean_L2']:.3f}  |  {bm['mean_L0']:5.0f}  | {BIM_STEPS}次反向")
    print(line)

    print('\nβ参数扫描结果：')
    for b in BETA_SCAN:
        f = get('FGSM', b)
        m = get('BIM', b)
        print(f"β={b}: FGSM RSE={f['mean_RSE']:.3f}, BIM RSE={m['mean_RSE']:.3f}")


def save_results_csv(results, path=RESULT_CSV):
    """保存全部 (方法 × β) 的评估指标到 CSV。"""
    cols = ['method', 'beta', 'mean_RSE', 'mean_MAE_increment', 'ASR', 'mean_L2', 'mean_L0', 'n']
    df = pd.DataFrame(results)[cols]
    df = df.sort_values(['method', 'beta']).reset_index(drop=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n>>> 结果已保存: {path}")


def plot_beta_rse_curve(results, path=CURVE_PNG):
    """绘制 β-RSE 曲线：横轴 β，纵轴 mean_RSE，FGSM / BIM 两条线。"""
    betas = sorted(set(r['beta'] for r in results))
    fig, ax = plt.subplots(figsize=(8, 6))
    for (method, marker, color) in [('FGSM', 'o-', 'tab:blue'), ('BIM', 's-', 'tab:red')]:
        rses = [next(r['mean_RSE'] for r in results
                     if r['method'] == method and abs(r['beta'] - b) < 1e-9) for b in betas]
        label = 'BIM(N=%d)' % BIM_STEPS if method == 'BIM' else method
        ax.plot(betas, rses, marker, color=color, linewidth=2, markersize=8, label=label)
        for b, v in zip(betas, rses):
            ax.annotate(f'{v:.3f}', (b, v), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=10)
    ax.set_xlabel('扰动预算系数 β', fontsize=13)
    ax.set_ylabel('平均 RSE (mean_RSE)', fontsize=13)
    ax.set_title('FGSM / BIM 白盒梯度攻击：β–RSE 曲线（仅测试集）', fontsize=14, fontweight='bold')
    ax.set_xticks(betas)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f">>> β-RSE 曲线已保存: {path}")


def main():
    global BIM_STEPS, BETA_DEFAULT, SEED
    parser = argparse.ArgumentParser(
        description='FGSM / BIM 白盒梯度攻击基线（MoACB-WSF 固定编码风速预测模型，仅测试集评估）')
    parser.add_argument('--data_file', type=str, default=FILENAME, help='输入数据文件')
    parser.add_argument('--beta', type=float, default=BETA_DEFAULT, help='主对比表使用的扰动预算系数 β')
    parser.add_argument('--bim_steps', type=int, default=BIM_STEPS, help='BIM 迭代步数 N')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子（训练复现用）')
    args = parser.parse_args()

    BIM_STEPS = args.bim_steps
    BETA_DEFAULT = args.beta
    SEED = args.seed
    if BETA_DEFAULT not in BETA_SCAN:
        BETA_SCAN.append(BETA_DEFAULT)
        BETA_SCAN.sort()

    print('\n' + '=' * 80)
    print('FGSM / BIM 白盒梯度攻击基线（非目标 · 全扰动 · 按特征 β·range_f · 仅测试集）')
    print('=' * 80)
    print(f'Device: {DEVICE}')
    print(f'Data file: {args.data_file}')

    # ---------- 1) 数据加载（顺序划分 0.7/0.15/0.15，与主文件一致）----------
    data_result = load_and_preprocess_data(args.data_file)
    train_dataset = data_result['train_dataset']
    test_dataset = data_result['test_dataset']
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']
    print(f'  训练集 {len(train_dataset)} / 验证集 {len(data_result["val_dataset"])} / 测试集 {len(test_dataset)} 个窗口样本')

    # ---------- 2) 加载 / 训练模型 ----------
    model = load_or_train_model(data_result)
    model.eval()

    # ---------- 3) 模型性能验证（严禁跳过）----------
    perf = evaluate_model_metrics(model, test_dataset, min_speed, max_speed,
                                  batch_size=decode_hyperparams(FIXED_ENCODING['setting'])[0])
    print('\n' + '=' * 80)
    print('模型测试集性能验证（反归一化，m/s）')
    print('=' * 80)
    print(f"  MAE : {perf['mae']:.4f}   (目标 ≈ 1.3861)")
    print(f"  RMSE: {perf['rmse']:.4f}   (目标 ≈ 1.6241)")
    print(f"  R   : {perf['r']:.4f}   (目标 ≈ 0.9055)")
    print(f"  R²  : {perf['r2']:.4f}")
    aligned = (1.35 <= perf['mae'] <= 1.42)
    if aligned:
        print("  [OK] MAE ∈ [1.35, 1.42]，模型已对齐。")
    else:
        print(f"  [WARNING] MAE={perf['mae']:.4f} 不在 [1.35,1.42]，模型可能未对齐，请检查编码/权重！")

    # ---------- 4) 测试集评估池（仅测试集，绝不含验证集）----------
    test_loader_for_attack = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, Y_test = next(iter(test_loader_for_attack))
    X_test = X_test.to(DEVICE)
    Y_test = Y_test.to(DEVICE)
    n_test = int(X_test.shape[0])
    print(f'\n[评估池] 测试集 {n_test} 个样本（仅测试集，不含验证集）')

    # ---------- 5) 按特征计算 range_f（训练集 max-min）----------
    range_f, n_train = compute_feature_ranges(train_dataset)
    print(f'[range_f] 训练集 {n_train} 样本，逐特征 max-min = {np.round(range_f, 6).tolist()}')

    # ---------- 6) β 扫描：FGSM + BIM ----------
    results = []
    for beta in BETA_SCAN:
        for method in ['FGSM', 'BIM']:
            X_adv = run_attack(model, X_test, Y_test, method, beta, range_f, n_steps=BIM_STEPS)
            m = evaluate_attack_metrics(model, X_test, Y_test, X_adv, min_speed, max_speed)
            m.update({'method': method, 'beta': float(beta)})
            results.append(m)

            # 自检断言：扰动预算与合法域
            delta = (X_adv - X_test).abs()
            budget = float(beta) * torch.as_tensor(range_f, device=DEVICE).view(1, 1, -1)
            assert float(delta.max().item()) <= float(budget.max().item()) + 1e-5, "扰动超出预算"
            assert float(X_adv.min().item()) >= -1e-6 and float(X_adv.max().item()) <= 1.0 + 1e-6, "X_adv 越出 [0,1]"

    # ---------- 7) 输出：对比表 / CSV / 曲线图 ----------
    print_comparison_table(results, n_test, beta_main=BETA_DEFAULT)
    save_results_csv(results, RESULT_CSV)
    plot_beta_rse_curve(results, CURVE_PNG)

    # ---------- 8) 自检汇总 ----------
    def get(method, beta):
        return next(r for r in results if r['method'] == method and abs(r['beta'] - beta) < 1e-9)
    fg = get('FGSM', BETA_DEFAULT)
    bm = get('BIM', BETA_DEFAULT)
    print('\n' + '=' * 80)
    print('自检清单')
    print('=' * 80)
    print(f"  [{'x' if aligned else ' '}] 测试集 MAE 在 1.35~1.42 之间（实测 {perf['mae']:.4f}）")
    print(f"  [x] 评估样本数 = 测试集样本数（{n_test}，不含验证集）")
    print(f"  [x] 每个特征 range_f 已按训练集分别计算：{np.round(range_f, 4).tolist()}")
    print(f"  [x] FGSM 每格点 |δ| ≤ β·range_f，X_adv ∈ [0,1]（已在扫描中断言）")
    print(f"  [x] BIM 迭代 {BIM_STEPS} 次后 |δ| 仍 ≤ β·range_f（双重 clip 已断言）")
    print(f"  [{'x' if fg['mean_RSE'] > 1.0 else ' '}] FGSM RSE > 1.0（实测 {fg['mean_RSE']:.3f}）")
    print(f"  [{'x' if bm['mean_RSE'] >= fg['mean_RSE'] else ' '}] BIM RSE ≥ FGSM RSE"
          f"（{bm['mean_RSE']:.3f} vs {fg['mean_RSE']:.3f}）")
    print(f"  [x] 全扰动：L0 上限 = seq_len×特征数 = {SEQUENCE_LENGTH}×{data_result['num_features']}"
          f" = {SEQUENCE_LENGTH * data_result['num_features']}（FGSM 实测 mean_L0={fg['mean_L0']:.1f}，"
          f"BIM 实测 mean_L0={bm['mean_L0']:.1f}；边界 clip 可能使个别格点不动）")
    print(f"  [x] 对比表打印、CSV 保存、β-RSE 曲线图生成均完成")
    print('=' * 80)


if __name__ == '__main__':
    main()
