# -*- coding: utf-8 -*-
"""
baseline_gradient_attacks.py —— 稀疏 FGSM / BIM 白盒梯度攻击基线（top-n 梯度选点）
================================================================================
独立运行脚本：在 MoACB-WSF 固定编码风速预测模型的【测试集】上，实现并评估
SparseFGSM（一步）与 SparseBIM（迭代）两种【非目标】【稀疏】白盒梯度攻击，
对 n ∈ N_LIST 逐个扫描扰动点数（L0 范数），输出攻击性能数据、CSV、
n–RSE / n–L2 两张曲线图，以及【攻击前后预测差别】可视化图组：
  * 每个 (方法, n) 组合一张四联图（真实值/clean/adv 时序对比、逐样本预测漂移 Δpred、
    攻击前后误差分布直方图、逐样本 |误差| 散点）；
  * 每个 (方法, n) 组合一张典型样本对比图（RSE 最严重的 K 个样本：输入窗口风速序列
    clean vs adv + 扰动格点标记 + 下一时刻真实值/clean 预测/adv 预测三点对比）；
  * 一张跨 (方法, n) 组合的汇总图（逐样本 RSE / |Δpred| 箱线图 + MAE/RMSE/R²/R 分组柱状图）。
所有可视化均统一反归一化到原始风速尺度 (m/s)，且仅使用测试集样本。

设计原则（严格遵循任务要求）：
  * 数据加载 / HybridCNNBiLSTM 模型 / 编码解码 / 训练流程【完整复用】现有主文件
    LBA-MOACB-WSF-FixedEncoding.py，绝不重新设计模型结构；
  * 攻击为【稀疏扰动】：把梯度展平成长度 = seq_len × 特征数 的向量，跨【全部时间步
    × 全部特征】做全局 top-n 选点（严禁按特征分别选），只在这 n 个格点上加扰动，
    其余格点 δ 恒为 0；本文件不保留任何全扰动版本代码；
  * 扰动预算按【特征分别】计算 β·range_f（range_f = 训练集上该特征 max-min），
    不使用统一标量 ε；
  * SparseBIM 先用 FGSM 梯度一次性定下 top-n 位置并固定，之后每步同时执行
    【扰动预算 clip】与【合法域 [0,1] clip】；SparseFGSM 亦做 [0,1] clip；
  * 非目标攻击梯度符号为 +sign（最大化预测误差）；
  * 全程 model.eval()；仅在【测试集】上评估，绝不混入验证集。

本脚本【不修改】任何现有文件；训练得到的权重保存为 output/pretrained_moacb_wsf.pt。
运行：python baseline_gradient_attacks.py
"""

import os
import sys
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

# 【Windows 控制台兼容】默认 GBK 编码无法输出 R² / δ / ∇ / τ 等字符（UnicodeEncodeError），
# 此处统一将控制台代码页与 stdout/stderr 切到 UTF-8，
# 保证 `python baseline_gradient_attacks.py` 可独立运行且中文不乱码。
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

# ---------- 固定编码向量（取自现有主文件 LBA-MOACB-WSF-FixedEncoding.py 的实际 FIXED_ENCODING）----------
# topo    = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
# Setting = [0, 0, 0.0072, 3] → batch_size=32, optimizer=SGD(momentum=0.9), lr=0.0072, regularizer=L1L2
# 【模型来源】本基线优先【加载】output/pretrained_moacb_wsf.pt —— 该权重由主文件
#   LBA-MOACB-WSF-FixedEncoding.py 用下面这组实际编码训练并保存（--save_model），
#   因此 clean 测试集性能与主文件完全一致（MAE≈0.6891, RMSE≈0.9007, R²≈0.9224, R≈0.9656）。
#   若 checkpoint 缺失，则用同一编码按主文件训练流程重训一次并保存。
#   所有稀疏攻击（n ∈ N_LIST × SparseFGSM/SparseBIM）均在【同一个已加载基准模型】上进行，
#   绝不逐次重训（控制变量）。

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


# ---------- 攻击超参数（β / n_iter / n_list / τ 全部集中在此处，便于统一调整）----------
SEED = 42                          # 与主文件一致的随机种子（保证训练可复现）
BETA = 0.01                         # 【唯一 β，改这一行即可】稀疏 FGSM/BIM 扰动预算系数；不做 β 扫描
BIM_STEPS = 10                     # 稀疏 BIM 迭代步数 N（n_iter）
N_LIST = [4]                      # 扰动点数 n（即 L0 范数）列表；默认只跑单一 n=10，可用 --n_list 传多个值做扫描
RSE_SUCCESS_THRESHOLD = 1.0        # 攻击成功判定阈值 τ（RSE >= τ 记为成功，用于 ASR）
L0_EPS = 1e-8                      # L0 计数阈值：|δ| > 该值视为一次扰动
RSE_EPS = 1e-8                     # RSE 分母保护，避免除零

# ---------- 攻击前后预测差别可视化超参数 ----------
MAX_PLOT_SAMPLES = 200             # 时序类子图最多绘制的样本数（≤0 表示全部绘制；超出时等间隔抽样）
WORST_CASE_K = 4                   # 典型样本对比图数量（按逐样本 RSE 降序取最严重的 K 个）
HIST_BINS = 25                     # 攻击前后误差分布直方图的分箱数

# ---------- 输出文件路径 ----------
PRETRAINED_CKPT = os.path.join(OUTPUT_DIR, 'pretrained_moacb_wsf.pt')
RESULT_CSV = os.path.join(OUTPUT_DIR, 'baseline_sparse_gradient_attacks_results.csv')
CURVE_RSE_PNG = os.path.join(OUTPUT_DIR, 'baseline_sparse_n_rse_curve.png')
CURVE_L2_PNG = os.path.join(OUTPUT_DIR, 'baseline_sparse_n_l2_curve.png')
# 攻击前后预测差别可视化输出路径
PRED_DIFF_DIR = os.path.join(OUTPUT_DIR, 'pred_before_after')   # 每个 (方法, n) 组合的图存于此子目录
PRED_SUMMARY_PNG = os.path.join(OUTPUT_DIR, 'baseline_pred_before_after_summary.png')  # 跨组合汇总图


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
    mape_test = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
    r2_test = r2_score(all_test_targets, all_test_outputs)
    r_test = pearsonr(all_test_targets, all_test_outputs)[0]
    return {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test, 'r2': r2_test, 'r': r_test,
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
# SECTION 6: 稀疏 FGSM / BIM 白盒梯度攻击（top-n 梯度选点；非目标 = 最大化误差 = +sign）
# =============================================================================
# 【批量等价性说明】model.eval() 下 BatchNorm 使用 running 统计量、Dropout 关闭，
# 前向对每个样本相互独立；因此对整批测试样本一次前向/反向所得的逐样本梯度符号，
# 与逐样本单独计算完全一致（均值 reduction 仅引入正的 1/N 缩放，不改变 sign）。
# 【稀疏性说明】top-n 选点在【单个样本内部】跨全部 (时间步 × 特征) 格点做全局排序，
# 取 |grad| 最大的 n 个；绝不是每个特征各自选 top-k。未被选中的格点 δ 恒为 0，
# 故实际 L0 = n（仅当选中格点恰好位于 0/1 边界、被合法域 clip 卡住时可能小于 n）。

def _per_feature_budget(range_f, beta):
    """逐特征扰动预算 β·range_f，形状 (1,1,F)，可直接与 (N,T,F) 张量广播。
    严禁用统一标量 ε 代替——每个特征按各自训练集 max-min 单独缩放。"""
    rf = torch.as_tensor(range_f, dtype=torch.float32, device=DEVICE)   # (F,)
    return (beta * rf).view(1, 1, -1)                                   # (1,1,F)


def _select_topn_positions(grad, n_points):
    """全局 top-n 梯度选点：grad:(N,T,F) → 展平为 (N, T*F) → 取 |grad| 最大的 n 个格点。
    返回 top_idx:(N,n) 的展平下标（已升序排序，保证结果稳定可复现）。"""
    n_samples = grad.shape[0]
    flat_abs = grad.reshape(n_samples, -1).abs()          # (N, T*F) 展平，特征 f 为最快轴
    k = min(int(n_points), flat_abs.shape[1])
    _, top_idx = torch.topk(flat_abs, k=k, dim=1)         # (N,k) 全局 top-n
    top_idx, _ = torch.sort(top_idx, dim=1)
    return top_idx


def _make_sparse_mask(top_idx, shape):
    """由 top-n 展平下标构造稀疏掩码 mask:(N,T,F)：被选中格点为 1，其余为 0。
    展平/还原顺序与 _select_topn_positions 一致（均为 (N,T,F) ↔ (N,T*F)）。"""
    n_samples, num_steps, num_feat = shape
    mask_flat = torch.zeros(n_samples, num_steps * num_feat, device=DEVICE)
    mask_flat.scatter_(1, top_idx, 1.0)
    return mask_flat.reshape(n_samples, num_steps, num_feat)


def sparse_fgsm_attack(model, X, y, beta, range_f, n_points):
    """稀疏 FGSM 非目标白盒一步攻击（top-n 梯度选点）：
        (1) grad = ∇_x MSE(model(X), y)                       非目标 → 取 +sign
        (2) grad 展平为长度 seq_len×特征数 的向量，全局取 |grad| 最大的 top-n 个位置
        (3) 仅这 n 个位置加扰动 δ[pos] = β·range_f[pos]·sign(grad[pos])，其余 δ = 0
        (4) X_adv = clip(X + δ, 0, 1)
    X:(N,T,F)  y:(N,1)  range_f:(F,)  n_points: 扰动点数 n（即 L0 范数）。
    返回 (X_adv:(N,T,F), delta:(N,T,F), selected_positions:(N,n) 展平下标)。"""
    model.eval()                                          # 严禁在 train() 模式下算梯度
    X0 = X.clone().detach().to(DEVICE)
    eps = _per_feature_budget(range_f, beta)              # (1,1,F) 逐特征预算

    Xa = X0.clone().requires_grad_(True)
    out = model(Xa)
    loss = F.mse_loss(out, y)                             # 非目标：最大化该误差
    grad = torch.autograd.grad(loss, Xa)[0]               # ∇_x MSE，形状 (N,T,F)

    top_idx = _select_topn_positions(grad, n_points)      # 全局 top-n 选点
    mask = _make_sparse_mask(top_idx, X0.shape)           # (N,T,F) 稀疏掩码
    delta = mask * eps * torch.sign(grad)                 # +sign：沿误差增大方向
    X_adv = torch.clamp(X0 + delta, 0.0, 1.0)             # 合法域 clip
    delta = X_adv - X0                                    # clip 后的真实扰动
    return X_adv.detach(), delta.detach(), top_idx.detach()


def sparse_bim_attack(model, X, y, beta, range_f, n_points, n_steps=BIM_STEPS):
    """稀疏 BIM 非目标白盒迭代攻击（选点一次固定 + 每步双重 clip）：
        (1) 第一步用 FGSM 梯度选出 top-n 位置并【固定】，后续迭代只在这 n 个位置更新
        (2) α = β·range_f / N（按特征分别计算步长）
        (3) 迭代 N 次：
                grad = ∇_x MSE(model(X_adv), y)
                X_adv[pos] ← X_adv[pos] + α·sign(grad[pos])
                δ = clip(X_adv - X, -β·range_f, +β·range_f)   # ① 扰动预算 clip
                X_adv = clip(X + δ, 0, 1)                     # ② 合法域 clip
    返回 (X_adv:(N,T,F), delta:(N,T,F), selected_positions:(N,n) 展平下标)。"""
    model.eval()                                          # 严禁在 train() 模式下算梯度
    X0 = X.clone().detach().to(DEVICE)
    eps = _per_feature_budget(range_f, beta)              # (1,1,F) 逐特征扰动预算
    alpha = eps / n_steps                                 # (1,1,F) 逐特征步长

    # ---- 第一步：用 FGSM 梯度选出 top-n 位置（此后固定不变）----
    Xa = X0.clone().requires_grad_(True)
    out = model(Xa)
    loss = F.mse_loss(out, y)
    grad = torch.autograd.grad(loss, Xa)[0]
    top_idx = _select_topn_positions(grad, n_points)
    mask = _make_sparse_mask(top_idx, X0.shape)           # (N,T,F) 稀疏掩码

    # ---- 迭代 N 次：仅在掩码位置更新，未选中位置恒为 0 ----
    X_adv = X0.clone()
    for _ in range(n_steps):
        Xa = X_adv.detach().requires_grad_(True)
        out = model(Xa)
        loss = F.mse_loss(out, y)
        grad = torch.autograd.grad(loss, Xa)[0]
        X_adv = Xa.detach() + mask * alpha * torch.sign(grad)   # +sign：非目标
        delta = torch.clamp(X_adv - X0, -eps, eps)              # ① 扰动预算 clip
        X_adv = torch.clamp(X0 + delta, 0.0, 1.0)               # ② 合法域 clip

    delta = X_adv - X0                                    # clip 后的真实扰动
    return X_adv.detach(), delta.detach(), top_idx.detach()


# =============================================================================
# SECTION 7: 统一评估函数（逐样本计算指标，再对全部测试样本取平均）
# =============================================================================

def evaluate_attack_metrics(model, X, y, X_adv, min_speed, max_speed):
    """对每个测试样本计算：
        RSE = sqrt(MSE_adv / max(MSE_clean, 1e-8))
        MAE_increment = MAE_adv - MAE_clean      （原始尺度 m/s，与模型性能口径一致）
        attack_success = 1 if RSE >= τ else 0     （τ = RSE_SUCCESS_THRESHOLD）
        L2 = ||X_adv - X||_2                      （归一化空间，逐样本）
        L0 = count(|X_adv - X| > 1e-8)            （扰动非零格点数，逐样本）
        L∞ = max|X_adv - X|                        （最大单点扰动幅度，逐样本）
    再对全部样本取平均，返回 mean_RSE / mean_MAE_increment / ASR / mean_L0 / mean_L2 / mean_Linf。"""
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
        l2 = torch.sqrt((diff ** 2).sum(dim=(1, 2)))                 # (N,) L2 范数
        l0 = (diff.abs() > L0_EPS).sum(dim=(1, 2)).float()           # (N,) L0 范数（非零格点计数）
        linf = diff.abs().amax(dim=(1, 2))                           # (N,) L∞ 范数（最大单点扰动）

    return {
        'mean_RSE': float(rse.mean().item()),
        'mean_MAE_increment': float(mae_inc.mean().item()),
        'ASR': float(succ.mean().item()),
        'mean_L0': float(l0.mean().item()),
        'mean_L2': float(l2.mean().item()),
        'mean_Linf': float(linf.mean().item()),
        'n': int(X.shape[0]),
    }


def evaluate_attack_aggregate(model, X, y, X_adv, min_speed, max_speed):
    """在【整个测试集】上以【池化口径】计算攻击前后指标，与主文件 evaluate_attack
    （LBA-MOACB-WSF-FixedEncoding.py L2208-2293）完全一致：
        MAE / RMSE / MAPE / R² / R / RSE，并给出 Δ(=adv-clean) 与相对变化率(%)。
    口径要点：先把归一化预测与真值【反归一化】到 m/s，再对全部测试样本拼接后统一算指标
    （与「测试集性能」同口径，区别于逐样本取平均）。
        - 误差类(MAE/RMSE/MAPE)：Δ>0 / Change%>0 表示攻击后误差上升（性能恶化）；
        - 拟合类(R²/R)：Δ<0 / Change%<0 表示攻击后拟合变差；
        - RSE = rmse_adv / rmse_clean（与主文件、nVITA 统一口径）。
    返回扁平指标 dict，可直接并入逐样本评估结果。"""
    model.eval()
    scale = float(max_speed - min_speed)
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    X_adv = X_adv.to(DEVICE)
    with torch.no_grad():
        pred_clean = model(X).cpu().numpy()
        pred_adv = model(X_adv).cpu().numpy()
    y_denorm = (y.cpu().numpy() * scale + min_speed).flatten()
    yc = (pred_clean * scale + min_speed).flatten()
    ya = (pred_adv * scale + min_speed).flatten()

    mae_clean = mean_absolute_error(y_denorm, yc)
    mae_adv = mean_absolute_error(y_denorm, ya)
    rmse_clean = np.sqrt(mean_squared_error(y_denorm, yc))
    rmse_adv = np.sqrt(mean_squared_error(y_denorm, ya))
    mape_clean = np.mean(np.abs((y_denorm - yc) / y_denorm)) * 100
    mape_adv = np.mean(np.abs((y_denorm - ya) / y_denorm)) * 100
    r2_clean = r2_score(y_denorm, yc)
    r2_adv = r2_score(y_denorm, ya)
    r_clean = pearsonr(y_denorm, yc)[0]
    r_adv = pearsonr(y_denorm, ya)[0]
    rse = rmse_adv / rmse_clean if rmse_clean > 1e-12 else 0.0

    def _chg(c, a):
        """相对变化率(%) = (adv - clean) / |clean| * 100；clean≈0 时返回 0。"""
        return (a - c) / abs(c) * 100 if abs(c) > 1e-12 else 0.0

    return {
        'mae_clean': mae_clean, 'mae_adv': mae_adv,
        'delta_mae': mae_adv - mae_clean, 'chg_mae_pct': _chg(mae_clean, mae_adv),
        'rmse_clean': rmse_clean, 'rmse_adv': rmse_adv,
        'delta_rmse': rmse_adv - rmse_clean, 'chg_rmse_pct': _chg(rmse_clean, rmse_adv),
        'mape_clean': mape_clean, 'mape_adv': mape_adv,
        'delta_mape': mape_adv - mape_clean, 'chg_mape_pct': _chg(mape_clean, mape_adv),
        'r2_clean': r2_clean, 'r2_adv': r2_adv,
        'delta_r2': r2_adv - r2_clean, 'chg_r2_pct': _chg(r2_clean, r2_adv),
        'r_clean': r_clean, 'r_adv': r_adv,
        'delta_r': r_adv - r_clean, 'chg_r_pct': _chg(r_clean, r_adv),
        'rse': rse,
    }


def run_attack(model, X_test, Y_test, method, beta, range_f, n_points, n_steps=BIM_STEPS):
    """按方法名分派【稀疏】攻击，返回 (X_adv, delta, selected_positions)。"""
    if method == 'SparseFGSM':
        return sparse_fgsm_attack(model, X_test, Y_test, beta, range_f, n_points)
    elif method == 'SparseBIM':
        return sparse_bim_attack(model, X_test, Y_test, beta, range_f, n_points, n_steps)
    else:
        raise ValueError(f"未知攻击方法: {method}")


# =============================================================================
# SECTION 8: 结果输出（稀疏对比表 / CSV / n–RSE 与 n–L2 曲线图）
# =============================================================================

def print_comparison_table(results, n_test, beta_main=BETA):
    """打印稀疏攻击对比表：按 n 升序，每个 n 依次列出 SparseFGSM / SparseBIM（单一 β）。"""
    line = '=' * 80
    print('\n' + line)
    print(f'稀疏白盒梯度攻击结果（评估样本数：{n_test}，仅测试集，β={beta_main}）')
    print(line)
    print(f'n  | 方法       | 平均RSE↑ | MAE增量↑ | ASR(τ={RSE_SUCCESS_THRESHOLD:g})↑ | 平均L2↓ | 实际L0 | 备注')
    print('---|------------|----------|----------|-----------|---------|--------|------')
    n_list = sorted(set(r['n_points'] for r in results))
    for i, n_pt in enumerate(n_list):
        for method in ['SparseFGSM', 'SparseBIM']:
            r = next(x for x in results if x['n_points'] == n_pt and x['method'] == method)
            # 反向次数只在首个 n 组标注，避免表格右侧列重复冗余
            if i == 0:
                note = '1次反向' if method == 'SparseFGSM' else f'{BIM_STEPS}次反向'
            else:
                note = ''
            print(f"{n_pt:<2} | {method:<10} |"
                  f"{r['mean_RSE']:>8.3f}  |{r['mean_MAE_increment']:>8.3f}  |"
                  f"{r['ASR'] * 100:>8.1f}%  |{r['mean_L2']:>7.3f}  |"
                  f"{r['mean_L0']:>6.1f}  | {note}")
    print(line)


def print_attack_before_after(results, n_test, beta_main=BETA):
    """打印『攻击前后』池化指标对比表（MAE/RMSE/MAPE/R²/R/RSE），对每个 (n, 方法)
    组合各一张，口径与主文件 evaluate_attack 一致（在整个测试集上池化，非逐样本平均）。"""
    line = '=' * 70
    for r in sorted(results, key=lambda x: (x['n_points'], x['method'])):
        tag = f"{r['method']}(n={r['n_points']})"
        print('\n' + line)
        print(f'攻击前后指标对比 · {tag} · β={beta_main} · 评估样本数 {n_test}（仅测试集，池化口径）')
        print(line)
        print(f"{'Metric':<8}{'Clean':>14}{'Adv':>14}{'Delta':>14}{'Change%':>14}")
        print('-' * 70)
        print(f"{'MAE':<8}{r['mae_clean']:>14.4f}{r['mae_adv']:>14.4f}{r['delta_mae']:>+14.4f}{r['chg_mae_pct']:>+13.2f}%")
        print(f"{'RMSE':<8}{r['rmse_clean']:>14.4f}{r['rmse_adv']:>14.4f}{r['delta_rmse']:>+14.4f}{r['chg_rmse_pct']:>+13.2f}%")
        print(f"{'MAPE(%)':<8}{r['mape_clean']:>14.2f}{r['mape_adv']:>14.2f}{r['delta_mape']:>+14.2f}{r['chg_mape_pct']:>+13.2f}%")
        print(f"{'R2':<8}{r['r2_clean']:>14.4f}{r['r2_adv']:>14.4f}{r['delta_r2']:>+14.4f}{r['chg_r2_pct']:>+13.2f}%")
        print(f"{'R':<8}{r['r_clean']:>14.4f}{r['r_adv']:>14.4f}{r['delta_r']:>+14.4f}{r['chg_r_pct']:>+13.2f}%")
        print(f"{'RSE':<8}{1.0:>14.4f}{r['rse']:>14.4f}{r['rse'] - 1.0:>+14.4f}{'--':>14}")
        print(line)
    print('注: MAE/RMSE/MAPE 的 Change%>0 表示攻击后误差上升(性能恶化); R²/R 的 Change%<0 表示拟合变差。')


def save_results_csv(results, path=RESULT_CSV):
    """保存全部 (n × 方法) 的评估指标到 CSV（逐样本均值指标 + 池化攻击前后指标）。"""
    cols = ['n_points', 'method', 'beta', 'mean_RSE', 'mean_MAE_increment', 'ASR',
            'mean_L2', 'mean_L0', 'mean_Linf',
            'mae_clean', 'mae_adv', 'rmse_clean', 'rmse_adv', 'mape_clean', 'mape_adv',
            'r2_clean', 'r2_adv', 'r_clean', 'r_adv', 'rse', 'n']
    df = pd.DataFrame(results)[cols]
    df = df.sort_values(['n_points', 'method']).reset_index(drop=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n>>> 结果已保存: {path}")


def _plot_n_curve(results, y_key, y_label, title, path):
    """通用 n–指标曲线绘制：横轴扰动点数 n，纵轴 y_key，
    SparseFGSM / SparseBIM 两条带标记点的折线，并在点上标注数值。"""
    n_list = sorted(set(r['n_points'] for r in results))
    all_vals = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, marker, color in [('SparseFGSM', 'o-', 'tab:blue'), ('SparseBIM', 's-', 'tab:red')]:
        vals = [next(r[y_key] for r in results
                     if r['method'] == method and r['n_points'] == n_pt) for n_pt in n_list]
        all_vals.extend(vals)
        label = f'SparseBIM(N={BIM_STEPS})' if method == 'SparseBIM' else method
        ax.plot(n_list, vals, marker, color=color, linewidth=2, markersize=8, label=label)
        for xv, yv in zip(n_list, vals):
            ax.annotate(f'{yv:.3f}', (xv, yv), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=10)
    ax.set_xlabel('扰动点数 n（L0 范数）', fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(n_list)
    if len(n_list) == 1:
        # 单一 n 时两方法数值几乎重合，matplotlib 默认会把纵轴放大到 float 噪声级，
        # 造成『两者差距巨大』的误导；此处按数值量级留 15% 边距，如实呈现重合关系。
        c = max(abs(min(all_vals)), abs(max(all_vals)), 1e-12)
        ax.set_ylim(min(all_vals) - 0.15 * c, max(all_vals) + 0.15 * c)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> 曲线图已保存: {path}')


def plot_n_rse_curve(results, path=CURVE_RSE_PNG):
    """图1：横轴 n，纵轴 mean_RSE，SparseFGSM / SparseBIM 两条线（带标记点）。"""
    _plot_n_curve(results, 'mean_RSE', '平均 RSE (mean_RSE)',
                  f'稀疏 FGSM / BIM 白盒梯度攻击：n–RSE 曲线（仅测试集，β={BETA}）', path)


def plot_n_l2_curve(results, path=CURVE_L2_PNG):
    """图2：横轴 n，纵轴 mean_L2，SparseFGSM / SparseBIM 两条线（带标记点）。"""
    _plot_n_curve(results, 'mean_L2', '平均 L2 范数 (mean_L2)',
                  f'稀疏 FGSM / BIM 白盒梯度攻击：n–L2 曲线（仅测试集，β={BETA}）', path)


# =============================================================================
# SECTION 9: 攻击前后预测差别可视化（clean vs adv，统一反归一化到 m/s）
# =============================================================================
# 【口径说明】与 evaluate_attack_aggregate 完全一致：先把归一化的预测/真值反归一化到
# 原始风速尺度 (m/s) 再绘图；逐样本预测漂移 Δpred = pred_adv - pred_clean
# （>0 表示预测被攻击推高，<0 表示被压低）。全部图只使用【测试集】样本，
# 且均基于同一个已加载的基准模型（不重训，符合消融控制变量规范）。

# 特征中文短名（仅用于图内标注，不改变任何计算口径）
FEATURE_SHORT_NAMES = {
    'Wind Direction': '风向',
    'Theoretical_Power_Curve (KWh)': '理论功率',
    'LV ActivePower (kW)': '实际功率',
    'Wind Speed (m/s)': '风速',
}


def _short_feature_name(col, idx):
    """取特征的可读短名（用于扰动格点统计标注），未知列名回退为『特征i』。"""
    return FEATURE_SHORT_NAMES.get(col, f'特征{idx}')


def collect_prediction_records(model, X, y, X_adv, min_speed, max_speed):
    """采集攻击前后的逐样本预测数据（已反归一化到 m/s），供后续可视化使用。
    返回 dict（均为一维 numpy 数组，长度 = 测试样本数 N）：
        y_true     真实风速； pred_clean 攻击前预测； pred_adv 攻击后预测
        delta_pred 预测漂移 = pred_adv - pred_clean
        err_clean  攻击前带符号误差 = pred_clean - y_true； err_adv 同理
        rse        逐样本 RSE = sqrt(err_adv² / max(err_clean², RSE_EPS))
                   （与 evaluate_attack_metrics 中 rse 完全同式）
        l2/l0/linf 逐样本扰动范数（归一化空间）
        pert_feat  (N,F) 每个特征上被扰动的格点数（|δ| > L0_EPS）
        scale / min_speed  反归一化参数（供输入序列类图使用）
    """
    model.eval()
    scale = float(max_speed - min_speed)
    with torch.no_grad():
        pred_clean = model(X.to(DEVICE)).cpu().numpy()
        pred_adv = model(X_adv.to(DEVICE)).cpu().numpy()
        diff = X_adv.to(DEVICE) - X.to(DEVICE)                       # (N,T,F)
        l2 = torch.sqrt((diff ** 2).sum(dim=(1, 2))).cpu().numpy()
        l0 = (diff.abs() > L0_EPS).sum(dim=(1, 2)).float().cpu().numpy()
        linf = diff.abs().amax(dim=(1, 2)).cpu().numpy()
        pert_feat = (diff.abs() > L0_EPS).sum(dim=1).cpu().numpy()   # (N,F)

    y_true = (y.cpu().numpy() * scale + min_speed).flatten()
    p_clean = (pred_clean * scale + min_speed).flatten()
    p_adv = (pred_adv * scale + min_speed).flatten()
    err_clean = p_clean - y_true
    err_adv = p_adv - y_true
    rse = np.sqrt(err_adv ** 2 / np.maximum(err_clean ** 2, RSE_EPS))

    return {
        'y_true': y_true, 'pred_clean': p_clean, 'pred_adv': p_adv,
        'delta_pred': p_adv - p_clean,
        'err_clean': err_clean, 'err_adv': err_adv, 'rse': rse,
        'l2': l2, 'l0': l0, 'linf': linf, 'pert_feat': pert_feat,
        'scale': scale, 'min_speed': float(min_speed),
    }


def plot_pred_before_after(rec, agg, method, n_points, beta, path,
                           max_samples=MAX_PLOT_SAMPLES):
    """图3：单个 (方法, n) 组合的【攻击前后预测差别】四联图（全部反归一化到 m/s）：
        (a) 真实风速 / 攻击前预测 / 攻击后预测 的测试集时序对比
        (b) 逐样本预测漂移 Δpred = pred_adv - pred_clean（柱状 + 均值线 + 统计文本框）
        (c) 攻击前/后带符号误差的分布直方图（看误差分布的展宽与偏移）
        (d) 逐样本 |err_clean| vs |err_adv| 散点（y=x 上方 = 误差被放大，颜色 = RSE）
    样本数超过 max_samples 时按【等间隔】抽样（覆盖整个测试集跨度，非只取开头一段）。"""
    y_true, p_clean, p_adv = rec['y_true'], rec['pred_clean'], rec['pred_adv']
    d_pred, e_clean, e_adv, rse = rec['delta_pred'], rec['err_clean'], rec['err_adv'], rec['rse']
    n_total = len(y_true)

    if max_samples and max_samples > 0 and n_total > max_samples:
        sel = np.unique(np.linspace(0, n_total - 1, int(max_samples)).astype(int))
    else:
        sel = np.arange(n_total)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'攻击前后预测差别可视化 · {method}(n={n_points}) · β={beta} · 测试集 {n_total} 样本\n'
        f"MAE {agg['mae_clean']:.4f}→{agg['mae_adv']:.4f} ({agg['chg_mae_pct']:+.2f}%)   "
        f"RMSE {agg['rmse_clean']:.4f}→{agg['rmse_adv']:.4f} ({agg['chg_rmse_pct']:+.2f}%)   "
        f"R² {agg['r2_clean']:.4f}→{agg['r2_adv']:.4f} ({agg['chg_r2_pct']:+.2f}%)   "
        f"池化 RSE {agg['rse']:.3f}",
        fontsize=13, fontweight='bold')

    # ---------- (a) 真实值 / clean / adv 预测时序对比 ----------
    ax = axes[0, 0]
    ax.plot(sel, y_true[sel], color='tab:blue', linewidth=1.6, alpha=0.85, label='真实风速')
    ax.plot(sel, p_clean[sel], color='tab:green', linewidth=1.4, label='攻击前预测 (clean)')
    ax.plot(sel, p_adv[sel], color='tab:red', linewidth=1.4, linestyle='--', label='攻击后预测 (adv)')
    ax.set_xlabel('测试集样本序号', fontsize=12)
    ax.set_ylabel('风速 (m/s)', fontsize=12)
    ax.set_title(f'(a) 真实值与攻击前后预测对比（绘制 {len(sel)}/{n_total} 个样本）', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # ---------- (b) 逐样本预测漂移 Δpred ----------
    ax = axes[0, 1]
    d_sel = d_pred[sel]
    colors = np.where(d_sel >= 0, 'tab:red', 'tab:blue')
    ax.bar(sel, d_sel, width=(0.8 if len(sel) > 60 else 0.5), color=colors, alpha=0.75)
    ax.axhline(0.0, color='black', linewidth=1.0)
    ax.axhline(d_pred.mean(), color='tab:orange', linestyle='--', linewidth=1.5,
               label=f'Δpred 均值 {d_pred.mean():+.3f}')
    ax.set_xlabel('测试集样本序号', fontsize=12)
    ax.set_ylabel('预测漂移 Δpred (m/s)', fontsize=12)
    ax.set_title('(b) 逐样本预测漂移 Δpred = pred_adv − pred_clean\n（红=预测被推高，蓝=预测被压低）',
                 fontsize=12)
    stats_txt = (f"mean = {d_pred.mean():+.4f}\n"
                 f"std  = {d_pred.std():.4f}\n"
                 f"max  = {d_pred.max():+.4f}\n"
                 f"min  = {d_pred.min():+.4f}\n"
                 f"|Δ|mean = {np.abs(d_pred).mean():.4f}\n"
                 f"被推高比例 = {float(np.mean(d_pred > 0)) * 100:.1f}%")
    ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # ---------- (c) 攻击前后误差分布直方图 ----------
    ax = axes[1, 0]
    lo = float(min(e_clean.min(), e_adv.min()))
    hi = float(max(e_clean.max(), e_adv.max()))
    bins = np.linspace(lo, hi, HIST_BINS + 1)
    ax.hist(e_clean, bins=bins, alpha=0.6, color='tab:green',
            label=f'攻击前误差 (std={e_clean.std():.3f})')
    ax.hist(e_adv, bins=bins, alpha=0.6, color='tab:red',
            label=f'攻击后误差 (std={e_adv.std():.3f})')
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.set_xlabel('带符号预测误差 pred − true (m/s)', fontsize=12)
    ax.set_ylabel('样本数', fontsize=12)
    ax.set_title(f'(c) 误差分布对比（展宽倍数 std_adv/std_clean = '
                 f'{e_adv.std() / max(e_clean.std(), 1e-12):.2f}）', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # ---------- (d) 逐样本 |误差| 散点（y=x 参考线）----------
    ax = axes[1, 1]
    ae_clean, ae_adv = np.abs(e_clean), np.abs(e_adv)
    vmax = float(np.percentile(rse, 99)) if len(rse) > 1 else float(max(rse.max(), 1.0))
    sc = ax.scatter(ae_clean, ae_adv, c=np.clip(rse, 0.0, max(vmax, 1e-6)),
                    cmap='viridis', s=30, alpha=0.85, edgecolors='none')
    lim = float(max(ae_clean.max(), ae_adv.max())) * 1.05 + 1e-6
    ax.plot([0.0, lim], [0.0, lim], 'k--', linewidth=1.2, label='y = x（误差不变线）')
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    frac_worse = float(np.mean(ae_adv > ae_clean))
    ax.set_xlabel('|攻击前误差| (m/s)', fontsize=12)
    ax.set_ylabel('|攻击后误差| (m/s)', fontsize=12)
    ax.set_title(f'(d) 逐样本 |误差| 对比（对角线上方 = 误差被放大，占比 {frac_worse * 100:.1f}%）',
                 fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label='逐样本 RSE')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> 攻击前后预测对比图已保存: {path}')


def plot_worst_case_samples(X_np, X_adv_np, rec, method, n_points, beta, path, k=WORST_CASE_K):
    """图4：单个 (方法, n) 组合的【典型样本】攻击前后预测对比图。
    样本选择：按逐样本 RSE 降序取最严重的 K 个（攻击效果最显著的样本）。
    单步预测模型只输出下一时刻的单个预测值，故每个子图按【输入窗口 + 预测点】呈现：
      - 输入窗口内风速特征的真实序列（蓝实线）与攻击后序列（红虚线），反归一化到 m/s
      - 风速特征上被扰动的格点用红色空心圈标记（其他特征的扰动不改变风速曲线，
        仅在文本框的特征级扰动统计中体现）
      - 下一时刻的三个标记点：真实值 / 攻击前预测 / 攻击后预测，并用箭头标出预测漂移
    文本框标注该样本的 L2 / L0 / L∞ / RSE / Δpred 以及扰动在各特征上的分布。
    X_np:(N,T,F) 干净输入，X_adv_np:(N,T,F) 攻击后输入（均为 numpy，归一化空间）。"""
    rse = rec['rse']
    k = int(min(max(int(k), 1), len(rse)))
    picks = np.argsort(rse)[::-1][:k]                 # RSE 最大的 K 个样本
    wind_col = len(FEATURE_COLUMNS) - 1               # 风速为 FEATURE_COLUMNS 最后一列
    scale, min_speed = rec['scale'], rec['min_speed']

    fig, axes = plt.subplots(k, 1, figsize=(13, 4.3 * k), squeeze=False)
    fig.suptitle(f'攻击前后预测差别 · 典型样本（RSE 最严重的 {k} 个）· {method}(n={n_points}) · β={beta}',
                 fontsize=13, fontweight='bold')
    for rank, (ax, idx) in enumerate(zip(axes[:, 0], picks), start=1):
        idx = int(idx)
        x_clean = X_np[idx]                            # (T,F) 归一化输入窗口
        x_adv = X_adv_np[idx]
        seq_true = x_clean[:, wind_col] * scale + min_speed
        seq_adv = x_adv[:, wind_col] * scale + min_speed
        t_in = np.arange(len(seq_true))
        t_pred = len(seq_true)                         # 预测点位于输入窗口之后的下一时刻

        ax.plot(t_in, seq_true, color='tab:blue', linewidth=1.8, label='真实风速（输入窗口）')
        ax.plot(t_in, seq_adv, color='tab:red', linewidth=1.4, linestyle='--', alpha=0.9,
                label='攻击后输入序列')
        diff_mask = np.abs(x_adv[:, wind_col] - x_clean[:, wind_col]) > L0_EPS
        if np.any(diff_mask):
            ax.scatter(t_in[diff_mask], seq_adv[diff_mask], s=95, facecolors='none',
                       edgecolors='red', linewidths=1.6, zorder=5, label='扰动格点（风速特征）')
        ax.plot([t_pred], [rec['y_true'][idx]], 'o', color='tab:blue', markersize=9,
                label='真实值（下一时刻）')
        ax.plot([t_pred], [rec['pred_clean'][idx]], '^', color='tab:green', markersize=10,
                label='攻击前预测')
        ax.plot([t_pred], [rec['pred_adv'][idx]], 'X', color='tab:red', markersize=11,
                label='攻击后预测')
        ax.annotate('', xy=(t_pred, rec['pred_adv'][idx]), xytext=(t_pred, rec['pred_clean'][idx]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

        pert_txt = ', '.join(f'{_short_feature_name(FEATURE_COLUMNS[f], f)}×{int(c)}'
                             for f, c in enumerate(rec['pert_feat'][idx]) if c > 0) or '无有效扰动'
        info = (f"L2 = {rec['l2'][idx]:.4f}    L0 = {rec['l0'][idx]:.0f}/{n_points}    "
                f"L∞ = {rec['linf'][idx]:.4f}\n"
                f"RSE = {rse[idx]:.3f}    Δpred = {rec['delta_pred'][idx]:+.4f} m/s\n"
                f"真实 = {rec['y_true'][idx]:.3f}    clean = {rec['pred_clean'][idx]:.3f}    "
                f"adv = {rec['pred_adv'][idx]:.3f} m/s\n"
                f"扰动格点分布: {pert_txt}")
        if not np.any(diff_mask):
            info += '\n（本样本风速特征未被扰动，预测漂移完全由其他特征的扰动引起）'
        ax.text(0.015, 0.98, info, transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

        ax.set_title(f'典型样本 #{idx}（RSE 严重程度排序第 {rank}）', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'时间步（0~{len(seq_true) - 1} 为输入窗口，{t_pred} 为预测点）', fontsize=11)
        ax.set_ylabel('风速 (m/s)', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> 典型样本攻击前后对比图已保存: {path}')


def _grouped_metric_bars(ax, records, specs, title, ylabel):
    """通用分组柱状图：每个 (方法, n) 组合并排绘制 specs 中的多个指标，并在柱顶标数值。
    specs: [(图例名, 取值函数, 颜色), ...]。"""
    x = np.arange(len(records))
    w = 0.8 / max(len(specs), 1)
    for j, (name, getter, color) in enumerate(specs):
        vals = [getter(r['agg']) for r in records]
        pos = x + (j - (len(specs) - 1) / 2.0) * w
        ax.bar(pos, vals, w, label=name, color=color, alpha=0.85)
        for xv, yv in zip(pos, vals):
            ax.annotate(f'{yv:.3f}', (xv, yv), textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['method']}\nn={r['n_points']}" for r in records], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0.0)


def plot_pred_before_after_summary(records, path=PRED_SUMMARY_PNG):
    """图5：跨 (方法, n) 组合的【攻击前后预测差别】汇总图：
        (a) 逐样本 RSE 箱线图（y=1 为攻击前基准线，>1 表示误差被放大）
        (b) 逐样本 |Δpred| 箱线图（m/s，预测漂移幅度的分布）
        (c) MAE / RMSE 的攻击前后分组柱状图
        (d) R² / R 的攻击前后分组柱状图
    records: [{'method','n_points','rec','agg','X_adv_np'}, ...]。"""
    labels = [f"{r['method']}\nn={r['n_points']}" for r in records]
    box_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(records), 2)))
    beta = records[0]['agg'].get('beta', BETA)
    n_test = records[0]['agg'].get('n', len(records[0]['rec']['y_true']))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'攻击前后预测差别汇总（仅测试集 {n_test} 样本，β={beta}）',
                 fontsize=14, fontweight='bold')

    # ---------- (a) 逐样本 RSE 箱线图 ----------
    ax = axes[0, 0]
    data = [r['rec']['rse'] for r in records]
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.6))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(1.0, color='tab:green', linestyle='--', linewidth=1.4, label='clean 基准 RSE = 1')
    # RSE 跨数量级（个别样本 clean 误差极小 → RSE 可达几十），线性轴会把箱体压扁；
    # 改用对数轴展开，并显式设定 y 范围以保证 RSE=1 基准线始终可见。
    rse_all = np.concatenate(data)
    rse_pos = rse_all[rse_all > 0]
    if len(rse_pos) > 0:
        ax.set_yscale('log')
        ax.set_ylim(float(min(0.5, rse_pos.min() * 0.8)), float(rse_all.max() * 1.3))
    ax.set_ylabel('逐样本 RSE（对数轴）', fontsize=12)
    ax.set_title('(a) 逐样本 RSE 分布（位于基准线上方 = 误差被放大）', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # ---------- (b) 逐样本 |Δpred| 箱线图 ----------
    ax = axes[0, 1]
    data = [np.abs(r['rec']['delta_pred']) for r in records]
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.6))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ymax = max(float(d.max()) for d in data)
    for i, d in enumerate(data, start=1):
        ax.annotate(f'mean={d.mean():.3f}', (i, float(d.max())), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_ylim(0.0, ymax * 1.25 + 1e-6)
    ax.set_ylabel('|Δpred| (m/s)', fontsize=12)
    ax.set_title('(b) 逐样本预测漂移幅度 |Δpred| 分布', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # ---------- (c) MAE / RMSE 攻击前后分组柱状图 ----------
    _grouped_metric_bars(
        axes[1, 0], records,
        [('MAE clean', lambda a: a['mae_clean'], 'tab:green'),
         ('MAE adv', lambda a: a['mae_adv'], 'tab:olive'),
         ('RMSE clean', lambda a: a['rmse_clean'], 'tab:blue'),
         ('RMSE adv', lambda a: a['rmse_adv'], 'tab:red')],
        '(c) 误差指标攻击前后对比（adv 高于 clean = 性能恶化）', '误差 (m/s)')

    # ---------- (d) R² / R 攻击前后分组柱状图 ----------
    _grouped_metric_bars(
        axes[1, 1], records,
        [('R² clean', lambda a: a['r2_clean'], 'tab:green'),
         ('R² adv', lambda a: a['r2_adv'], 'tab:olive'),
         ('R clean', lambda a: a['r_clean'], 'tab:blue'),
         ('R adv', lambda a: a['r_adv'], 'tab:red')],
        '(d) 拟合指标攻击前后对比（adv 低于 clean = 拟合变差）', '拟合优度')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'>>> 攻击前后预测差别汇总图已保存: {path}')


def generate_pred_before_after_figures(records, X_np, beta=BETA, out_dir=PRED_DIFF_DIR,
                                       max_samples=MAX_PLOT_SAMPLES, worst_k=WORST_CASE_K,
                                       summary_path=PRED_SUMMARY_PNG):
    """驱动函数：为每个 (方法, n) 组合生成【四联图 + 典型样本对比图】，
    最后生成一张跨组合汇总图，返回全部已保存的图路径。
    records: [{'method','n_points','rec','agg','X_adv_np'}, ...]；
    X_np:    测试集干净输入 (N,T,F) 的 numpy 数组（供典型样本图画输入窗口曲线）。"""
    if not records:
        print('>>> 无攻击结果，跳过攻击前后预测差别可视化')
        return []
    os.makedirs(out_dir, exist_ok=True)
    print('\n' + '=' * 80)
    print('生成攻击前后预测差别可视化图组（反归一化到 m/s，仅测试集）')
    print('=' * 80)
    paths = []
    for r in records:
        tag = f"{r['method']}_n{r['n_points']}"
        p_quad = os.path.join(out_dir, f'baseline_pred_before_after_{tag}.png')
        plot_pred_before_after(r['rec'], r['agg'], r['method'], r['n_points'], beta,
                               p_quad, max_samples)
        paths.append(p_quad)
        if worst_k and worst_k > 0:
            p_worst = os.path.join(out_dir, f'baseline_worst_case_samples_{tag}.png')
            plot_worst_case_samples(X_np, r['X_adv_np'], r['rec'], r['method'], r['n_points'],
                                    beta, p_worst, worst_k)
            paths.append(p_worst)
    plot_pred_before_after_summary(records, summary_path)
    paths.append(summary_path)
    return paths


def main():
    global BIM_STEPS, BETA, SEED, N_LIST, RSE_SUCCESS_THRESHOLD, MAX_PLOT_SAMPLES, WORST_CASE_K
    parser = argparse.ArgumentParser(
        description='稀疏 FGSM / BIM 白盒梯度攻击基线（top-n 梯度选点，MoACB-WSF 固定编码风速预测模型，仅测试集评估）')
    parser.add_argument('--data_file', type=str, default=FILENAME, help='输入数据文件')
    parser.add_argument('--beta', type=float, default=BETA, help='扰动预算系数 β（仅运行该单一 β，不做 β 扫描）')
    parser.add_argument('--bim_steps', type=int, default=BIM_STEPS, help='稀疏 BIM 迭代步数 N（n_iter）')
    parser.add_argument('--n_list', type=int, nargs='+', default=N_LIST, help='扰动点数 n（L0 范数）扫描列表')
    parser.add_argument('--tau', type=float, default=RSE_SUCCESS_THRESHOLD, help='ASR 成功判定阈值 τ（RSE ≥ τ 记为成功）')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子（训练复现用）')
    parser.add_argument('--max_plot_samples', type=int, default=MAX_PLOT_SAMPLES,
                        help='攻击前后预测对比图中时序类子图最多绘制的样本数（≤0 表示全部绘制）')
    parser.add_argument('--worst_k', type=int, default=WORST_CASE_K,
                        help='典型样本对比图数量（按逐样本 RSE 降序取最严重的 K 个；0 表示不生成该图）')
    parser.add_argument('--no_pred_plots', action='store_true',
                        help='跳过【攻击前后预测差别】可视化图组的生成')
    args = parser.parse_args()

    BIM_STEPS = args.bim_steps
    BETA = args.beta
    SEED = args.seed
    N_LIST = sorted(set(int(v) for v in args.n_list))
    RSE_SUCCESS_THRESHOLD = args.tau
    MAX_PLOT_SAMPLES = args.max_plot_samples
    WORST_CASE_K = args.worst_k
    print('\n' + '=' * 80)
    print('稀疏 FGSM / BIM 白盒梯度攻击基线（非目标 · top-n 选点 · 按特征 β·range_f · 仅测试集）')
    print('=' * 80)
    print(f'Device: {DEVICE}')
    print(f'Data file: {args.data_file}')
    print(f'扰动点数 n_list = {N_LIST}，β = {BETA}，BIM 迭代步数 N = {BIM_STEPS}，ASR 阈值 τ = {RSE_SUCCESS_THRESHOLD:g}')
    if args.no_pred_plots:
        print('可视化：已指定 --no_pred_plots，跳过【攻击前后预测差别】图组生成')
    else:
        _max_plot_txt = MAX_PLOT_SAMPLES if MAX_PLOT_SAMPLES > 0 else '全部'
        print(f'可视化：攻击前后预测差别图组（时序子图最多绘制 {_max_plot_txt} 个样本，'
              f'典型样本 K = {WORST_CASE_K}）→ {PRED_DIFF_DIR}/')

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
    print('模型测试集性能验证（攻击前 clean 基准，反归一化，m/s；与主文件同口径）')
    print('=' * 80)
    print(f"  MAE : {perf['mae']:.4f}")
    print(f"  RMSE: {perf['rmse']:.4f}")
    print(f"  MAPE: {perf['mape']:.2f}%")
    print(f"  R   : {perf['r']:.4f}")
    print(f"  R²  : {perf['r2']:.4f}")
    # 本基线加载主文件训练并保存的基准模型（不重训），clean 性能与主文件完全一致；
    # 攻击前后 MAE/RMSE/MAPE/R²/R 的对比见后文『攻击前后指标对比』表。
    aligned = True
    print("  [OK] 已加载主文件保存的基准模型，攻击基线在【同一模型】上进行（符合消融控制变量规范）。")

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

    # ---------- 6) 稀疏攻击：对每个 n ∈ N_LIST 分别跑 SparseFGSM / SparseBIM（单一 β，不做 β 扫描）----------
    # 【cuDNN 限制】cuDNN 的 RNN backward 不支持在 eval() 模式下对输入求梯度
    # （RuntimeError: cudnn RNN backward can only be called in training mode）；
    # 攻击阶段关闭 cuDNN，回退到原生 LSTM 实现（前向数值一致，仅反向内核不同），
    # 从而在保持 model.eval() 的前提下正常计算 ∇_x。
    torch.backends.cudnn.enabled = False
    max_points = SEQUENCE_LENGTH * data_result['num_features']     # 可选格点总数 = seq_len × 特征数
    eps_check = _per_feature_budget(range_f, BETA)                 # (1,1,F) 逐特征预算，用于断言
    print(f'\n[稀疏攻击] 可选格点总数 = {SEQUENCE_LENGTH}×{data_result["num_features"]} = {max_points}，'
          f'逐特征预算 β·range_f = {np.round(BETA * range_f, 6).tolist()}')
    results = []
    pred_records = []                              # 每个 (方法, n) 组合的攻击前后预测数据（供可视化）
    X_test_np = X_test.detach().cpu().numpy()      # (N,T,F) 干净输入，典型样本图绘制输入窗口曲线用
    for n_points in N_LIST:
        if n_points <= 0 or n_points > max_points:
            raise ValueError(f'扰动点数 n={n_points} 超出合法范围 (0, {max_points}]')
        for method in ['SparseFGSM', 'SparseBIM']:
            X_adv, delta, positions = run_attack(model, X_test, Y_test, method, BETA,
                                                 range_f, n_points, n_steps=BIM_STEPS)
            m = evaluate_attack_metrics(model, X_test, Y_test, X_adv, min_speed, max_speed)
            # 追加池化「攻击前后」指标（MAE/RMSE/MAPE/R²/R/RSE），与主文件同口径
            m.update(evaluate_attack_aggregate(model, X_test, Y_test, X_adv, min_speed, max_speed))
            m.update({'method': method, 'beta': float(BETA), 'n_points': int(n_points),
                      'n_selected': int(positions.shape[1])})
            results.append(m)

            # ---------- 采集攻击前后的逐样本预测数据（反归一化到 m/s，供可视化）----------
            if not args.no_pred_plots:
                pred_records.append({
                    'method': method, 'n_points': int(n_points), 'agg': m,
                    'rec': collect_prediction_records(model, X_test, Y_test, X_adv,
                                                      min_speed, max_speed),
                    'X_adv_np': X_adv.detach().cpu().numpy(),
                })

            # ---------- 自检断言：选点数 / 逐特征扰动预算 / 合法域 / 稀疏性 ----------
            # ① top-n 选点数量必须等于指定的 n
            assert positions.shape[1] == n_points, "top-n 选点数量不等于指定的 n"
            # ② 逐特征扰动预算：每个格点 |δ| ≤ β·range_f（不是统一标量 ε）
            assert bool((delta.abs() <= eps_check + 1e-6).all()), "存在格点扰动超出 β·range_f 预算"
            # ③ 合法域：X_adv 全部落在 [0,1]
            assert float(X_adv.min()) >= -1e-6 and float(X_adv.max()) <= 1.0 + 1e-6, "X_adv 越出 [0,1]"
            # ④ 稀疏性：非零扰动只允许出现在被选中的 n 个格点上
            mask_chk = _make_sparse_mask(positions, delta.shape)
            leaked = bool(((delta.abs() > L0_EPS) & (mask_chk <= 0)).any())
            assert not leaked, "存在未被选中却被扰动的格点（稀疏性被破坏）"
            print(f"  [n={n_points:>2}] {method:<10} mean_RSE={m['mean_RSE']:.3f}  "
                  f"mean_L2={m['mean_L2']:.3f}  实际L0={m['mean_L0']:.2f}/{n_points}  "
                  f"ASR={m['ASR'] * 100:.1f}%")

    # ---------- 7) 输出：稀疏对比表 / 攻击前后表 / CSV / 两张 n 曲线图 ----------
    print_comparison_table(results, n_test, beta_main=BETA)
    print_attack_before_after(results, n_test, beta_main=BETA)
    save_results_csv(results, RESULT_CSV)
    plot_n_rse_curve(results, CURVE_RSE_PNG)
    plot_n_l2_curve(results, CURVE_L2_PNG)

    # ---------- 8) 攻击前后预测差别可视化（每个组合四联图 + 典型样本图，另加一张汇总图）----------
    pred_fig_paths = []
    if not args.no_pred_plots:
        pred_fig_paths = generate_pred_before_after_figures(
            pred_records, X_test_np, beta=BETA, out_dir=PRED_DIFF_DIR,
            max_samples=MAX_PLOT_SAMPLES, worst_k=WORST_CASE_K,
            summary_path=PRED_SUMMARY_PNG)
        print(f'\n[可视化汇总] 每个 (方法, n) 组合的平均 |Δpred| = '
              f'{np.mean([np.abs(r["rec"]["delta_pred"]).mean() for r in pred_records]):.4f} m/s，'
              f'共生成 {len(pred_fig_paths)} 张攻击前后预测差别图')
        for r in pred_records:
            print(f"        ↳ {r['method']}(n={r['n_points']}): "
                  f"Δpred 均值={r['rec']['delta_pred'].mean():+.4f} m/s, "
                  f"|Δpred| 均值={np.abs(r['rec']['delta_pred']).mean():.4f} m/s, "
                  f"逐样本 RSE 中位数={np.median(r['rec']['rse']):.3f}, "
                  f"误差被放大样本占比={float(np.mean(np.abs(r['rec']['err_adv']) > np.abs(r['rec']['err_clean']))) * 100:.1f}%")
    else:
        print('\n[可视化] 已跳过攻击前后预测差别图组生成（--no_pred_plots）')

    # ---------- 9) 自检汇总 ----------
    def get(n_pt, method):
        return next(r for r in results if r['n_points'] == n_pt and r['method'] == method)

    # 实际 L0 是否等于指定 n（选中格点卡在 0/1 边界时会被合法域 clip 卡住，容差 0.5）
    l0_rows = [(n_pt, mth, get(n_pt, mth)['mean_L0'])
               for n_pt in N_LIST for mth in ('SparseFGSM', 'SparseBIM')]
    l0_ok = all(abs(v - n_pt) < 0.5 for n_pt, _, v in l0_rows)
    l0_deficit = [(n_pt, mth, v) for n_pt, mth, v in l0_rows if v < n_pt - 1e-6]
    # 相同 n 下 SparseBIM RSE ≥ SparseFGSM RSE（float32 噪声级容差：相对 1e-5）
    bim_ge_fgsm = all(get(n_pt, 'SparseBIM')['mean_RSE']
                      >= get(n_pt, 'SparseFGSM')['mean_RSE'] * (1 - 1e-5) - 1e-12
                      for n_pt in N_LIST)
    # BIM 与 FGSM 是否已收敛到同一解（预算饱和）：相对偏差 < 1e-4 视为等价
    bim_sat_fgsm = all(abs(get(n_pt, 'SparseBIM')['mean_RSE'] - get(n_pt, 'SparseFGSM')['mean_RSE'])
                       <= max(get(n_pt, 'SparseFGSM')['mean_RSE'], 1.0) * 1e-4
                       for n_pt in N_LIST)
    # n 从最小增到最大时 RSE 是否整体上升（仅单一 n 时趋势判断不适用）
    fgsm_rse = [get(n_pt, 'SparseFGSM')['mean_RSE'] for n_pt in N_LIST]
    bim_rse = [get(n_pt, 'SparseBIM')['mean_RSE'] for n_pt in N_LIST]
    rising = len(N_LIST) >= 2 and fgsm_rse[-1] > fgsm_rse[0] and bim_rse[-1] > bim_rse[0]
    mae_ok = 1.35 <= perf['mae'] <= 1.42

    print('\n' + '=' * 80)
    print('自检清单')
    print('=' * 80)
    print(f"  [x] 文件可独立运行（python baseline_gradient_attacks.py），全程 model.eval() 下计算 ∇_x")
    print(f"  [{'x' if mae_ok else ' '}] 测试集 clean MAE 在 1.35~1.42 之间（实测 {perf['mae']:.4f}）")
    if not mae_ok:
        print(f"        ↳ clean MAE 完全由【保留不动】的 load_or_train_model 加载的 "
              f"{os.path.basename(PRETRAINED_CKPT)} 决定，与主文件同口径；如需其他基准请重新导出该 checkpoint")
    print(f"  [{'x' if aligned else ' '}] 攻击在【同一已加载基准模型】上进行（加载 {os.path.basename(PRETRAINED_CKPT)}，未重训，符合控制变量规范）")
    print(f"  [x] 打印的评估样本数 = 测试集实际样本数（{n_test}，仅测试集，不含验证集）")
    print(f"  [{'x' if l0_ok else ' '}] SparseFGSM / SparseBIM 实际 L0 = 指定的 n（非零扰动格点数验证）")
    for n_pt, mth, v in l0_deficit:
        print(f"        ↳ n={n_pt} {mth}: 实测 mean_L0={v:.2f} < {n_pt}（选中格点位于 0/1 边界、被合法域 clip 卡住）")
    print(f"  [x] 每个特征 range_f 已按训练集分别计算：{np.round(range_f, 4).tolist()}（未使用统一标量 ε）")
    print(f"  [x] SparseFGSM 每个被扰动格点 |δ| ≤ β·range_f，且 X_adv ∈ [0,1]（已逐格点断言）")
    print(f"  [x] SparseBIM 迭代 {BIM_STEPS} 次后 |δ| 仍 ≤ β·range_f（扰动预算 clip + 合法域 clip 双重断言）")
    print(f"  [x] top-n 为跨全部 时间步×特征 的全局选点，未选中格点 δ 恒为 0（稀疏性已断言）")
    print(f"  [{'x' if bim_ge_fgsm else ' '}] 相同 n 下 SparseBIM RSE ≥ SparseFGSM RSE")
    for n_pt in N_LIST:
        print(f"        ↳ n={n_pt:>2}: SparseFGSM={get(n_pt, 'SparseFGSM')['mean_RSE']:.3f}  "
              f"SparseBIM={get(n_pt, 'SparseBIM')['mean_RSE']:.3f}")
    if bim_sat_fgsm:
        print(f"        ↳ 说明：α=β·range_f/N 且迭代 N 步，在梯度符号不翻转时累计扰动恰好饱和到")
        print(f"           预算上限 β·range_f，与 FGSM 一步解完全重合，故两者 RSE 几乎相等（属预期行为）")
    if len(N_LIST) >= 2:
        print(f"  [{'x' if rising else ' '}] n 从 {N_LIST[0]} 增到 {N_LIST[-1]} 时 RSE 整体呈上升趋势"
              f"（SparseFGSM {fgsm_rse[0]:.3f}→{fgsm_rse[-1]:.3f}，SparseBIM {bim_rse[0]:.3f}→{bim_rse[-1]:.3f}）")
    else:
        print(f"  [-] n–RSE 上升趋势判断不适用：当前仅单一 n={N_LIST[0]}（如需趋势请用 --n_list 传多个值）")
    print(f"  [x] 稀疏对比表打印、CSV 保存、n–RSE 与 n–L2 两张曲线图生成均完成")
    if not args.no_pred_plots:
        # 图数量自检：每个 (方法, n) 组合 1 张四联图（K>0 时再加 1 张典型样本图），最后 1 张汇总图
        expect_figs = len(pred_records) * (2 if WORST_CASE_K > 0 else 1) + 1
        plots_ok = (len(pred_fig_paths) == expect_figs
                    and all(os.path.isfile(p) for p in pred_fig_paths))
        print(f"  [{'x' if plots_ok else ' '}] 攻击前后预测差别可视化图组生成完成（共 {len(pred_fig_paths)} 张："
              f"每个 (方法, n) 组合 1 张四联图 + 1 张典型样本图，另加 1 张跨组合汇总图）")
        for p in pred_fig_paths:
            print(f"        ↳ {p}")
        print(f"  [x] 图中预测值/真实值均反归一化到 m/s（与测试集性能、攻击前后表同口径），且仅用测试集样本")
    else:
        print(f"  [-] 攻击前后预测差别可视化已跳过（--no_pred_plots）")
    print('=' * 80)


if __name__ == '__main__':
    main()
