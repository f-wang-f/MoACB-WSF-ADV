# -*- coding: utf-8 -*-
"""
GenAttack.py —— GenAttack（遗传算法 GA，黑盒，回归改造）
=====================================================================
在 MoACB-WSF 固定编码风速预测模型的【测试集】上，实现 Alzantot et al. (2019) 的
GenAttack 的【遗传算法版】，作为黑盒进化类攻击基线。它与本项目方法（NSGA-II/nVITA）
同属黑盒进化类，是本次最关键的对照，故严格同预算、同测试集、同模型。

------------------------------------------------------------------
复用现有约定（严格对齐 nVITA_DE_Attack.py / baseline_gradient_attacks.py）：
  * 用 importlib 动态加载主文件 LBA-MOACB-WSF-FixedEncoding.py（模块名 fe），复用
    SEQUENCE_LENGTH / FEATURE_COLUMNS / DEVICE / WEIGHTS_DIR / FIXED_ENCODING、
    load_and_preprocess_data、HybridCNNBiLSTM（仅 checkpoint 缺失需重训时使用）、
    scripted_ckpt_path / load_scripted_model / export_scripted_model；
  * 模型加载顺序与 baseline_gradient_attacks.py 的 load_or_train_model 完全一致：
      1) 优先 TorchScript（weights/pretrained_moacb_wsf_scripted.pt，BytesIO 中转）
      2) 回退 state_dict（weights/pretrained_moacb_wsf.pt + 固定编码重建结构）
      3) 两者都缺失才用固定编码重训一次并保存
    绝不逐次重训；三个基线脚本跑在【同一基准模型】上（控制变量）；
  * 只用测试集；扰动预算 = beta * range_f[f]（训练集逐特征 max-min，归一化空间），
    beta=0.01；严禁用统一标量 ε 代替逐特征预算；
  * SEED=42，保证可复现。

------------------------------------------------------------------
核心算法（回归改造）：
  * 黑盒：只查询模型输出，不取梯度；
  * 遗传算法流程：种群初始化 → 适应度评估（最大化预测误差）→ 锦标赛选择 → 交叉 →
    【自适应变异】→ 精英保留；
  * 【自适应变异】是 GenAttack 的核心机制：变异强度 σ 随代数指数衰减
        σ_gen = σ0 * (mutation_decay ** gen)
    （前期大步探索、后期小步精修），绝非固定变异率；
  * 目标函数改造为回归版：原文“误分类”换成【最大化预测误差】——
        fitness = MSE_adv = (pred_adv - y)^2（GA 最大化）；
  * 每个基因 = (格点位置 pos, 幅值分数 val∈[-1,1])，实际扰动 δ[pos] = val * β·range_f[f(pos)]，
    与其它方法同预算；
  * --n_points 控制扰动点数：默认 5（稀疏版，与 5VITA / 本项目 n_max 同口径）；
    n_points = 0 表示稠密（全部 80 格点）；
  * 逐样本独立搜索；每代种群评估批量前向加速。

------------------------------------------------------------------
输出（三个基线脚本字段名逐字一致，可直接 pd.concat）：
  * output/baseline_genattack_results.csv
  * output/adv_GenAttack.npz（内含 X_adv，numpy 格式）
  * 控制台打印攻击前后对比表（clean → adv，带 Δ 与 Change%）
运行：python GenAttack.py                        （全测试集，稀疏 n_points=5）
      python GenAttack.py --num_eval_samples 5   （小样本快速验证）
      python GenAttack.py --n_points 0           （稠密：全部 80 格点）
"""

import os
import sys
import io
import time
import random
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 无界面后端，且必须在导入 fe（其顶层会 import matplotlib）之前设定，避免交互式后端阻塞
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')

# 【Windows 控制台兼容】默认 GBK 编码无法输出 R² / δ / β 等字符（UnicodeEncodeError），
# 统一将控制台代码页与 stdout/stderr 切到 UTF-8，保证脚本可独立运行且中文不乱码。
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
# 通过 importlib 复用 attacks/lba/ 下的 LBA-MOACB-WSF-FixedEncoding.py（文件名含连字符无法常规 import）
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
# 本脚本在 attacks/baselines/，主文件在 attacks/lba/，按相对路径定位并规范化为绝对路径
_SRC_PATH = os.path.abspath(os.path.join(_HERE, '..', 'lba', 'LBA-MOACB-WSF-FixedEncoding.py'))
_spec = importlib.util.spec_from_file_location('fe', _SRC_PATH)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)   # 仅执行其顶层定义（不触发 main，__name__ != '__main__'）

SEQUENCE_LENGTH = fe.SEQUENCE_LENGTH
FEATURE_COLUMNS = fe.FEATURE_COLUMNS
FIXED_ENCODING = fe.FIXED_ENCODING
DEVICE = fe.DEVICE
WEIGHTS_DIR = fe.WEIGHTS_DIR
HybridCNNBiLSTM = fe.HybridCNNBiLSTM
load_and_preprocess_data = fe.load_and_preprocess_data
decode_hyperparams = fe.decode_hyperparams
encode_individual = fe.encode_individual
is_valid_individual = fe.is_valid_individual
scripted_ckpt_path = fe.scripted_ckpt_path
load_scripted_model = fe.load_scripted_model
export_scripted_model = fe.export_scripted_model

# 训练相关超参（与主文件完全一致，用于 checkpoint 缺失时重训）
FINAL_EPOCHS = fe.FINAL_EPOCHS
FINAL_PATIENCE = fe.FINAL_PATIENCE
GRADIENT_CLIP = fe.GRADIENT_CLIP
LAMBDA_REG = fe.LAMBDA_REG

OUTPUT_DIR = fe.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
PRETRAINED_CKPT = os.path.join(WEIGHTS_DIR, 'pretrained_moacb_wsf.pt')

# =============================================================================
# 全局约定（三个基线脚本必须逐字一致，否则无法合并成一张表）
# =============================================================================
SEED = 42                       # 随机种子，保证可复现
BETA = 0.0085                     # 扰动预算系数 beta（逐特征 range_f 缩放），与其它方法一致
FILENAME = fe.FILENAME
METHOD_NAME = 'GenAttack'       # CSV method 列 / npz 文件名
RESULT_CSV = os.path.join(OUTPUT_DIR, 'baseline_genattack_results.csv')
ADV_NPZ = os.path.join(OUTPUT_DIR, f'adv_{METHOD_NAME}.npz')

# 指标口径常量（与 baseline_gradient_attacks.py 一致）
RSE_SUCCESS_THRESHOLD = 1.0     # ASR 判定阈值 τ（逐样本 RSE >= τ 记为成功）
L0_EPS = 1e-8                   # L0 计数阈值：|δ| > 该值视为一次扰动
RSE_EPS = 1e-8                  # RSE 分母保护，避免除零

# CSV 列名（三个基线脚本完全一致，便于直接 pd.concat）
CSV_COLUMNS = ['method', 'beta', 'n_points', 'mean_RSE', 'ASR', 'mean_L0', 'mean_L2', 'mean_Linf',
               'mae_clean', 'mae_adv', 'mse_clean', 'mse_adv', 'rmse_clean', 'rmse_adv',
               'mape_clean', 'mape_adv', 'r2_clean', 'r2_adv', 'rse', 'n']

# =============================================================================
# GenAttack 超参（集中配置）
# =============================================================================
GENATTACK_CONFIG = {
    'pop_size': 20,           # 种群大小
    'maxiter': 60,            # 最大代数
    'mutation_rate': 0.3,     # 每个基因被变异的概率
    'mutation_decay': 0.95,   # 自适应变异强度衰减因子（σ_gen = σ0 * decay^gen）
    'n_points': 5,            # 扰动点数（稀疏版，与 5VITA / 本项目 n_max 同口径）；0=稠密（全部 80 格点）
    'num_eval_samples': None,  # 参加评估的测试样本数（None/<=0=全部，正整数则从测试集头部截取）
}
# 变异强度初值 σ0 与锦标赛规模（GenAttack 内部机制参数，不放入上面的核心配置字典）
MUTATION_SIGMA0 = 0.5          # 自适应变异初值 σ0（前期大步探索）
TOURNAMENT_K = 3               # 锦标赛选择的竞争规模
POS_MUTATE_PROB = 0.5          # 稀疏模式下，被变异基因同时改变位置 (t,f) 的概率


# =============================================================================
# SECTION 1: 模型与数据准备（加载顺序与 baseline_gradient_attacks.load_or_train_model 完全一致）
# =============================================================================
def set_seed(seed=SEED):
    """与主文件 main() 一致的随机种子设定顺序（random → numpy → torch → cuda）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fixed_encoding_model(data_result):
    """使用固定编码构建并训练模型（仅 checkpoint 缺失时触发一次）：优化器/损失/正则/
    梯度裁剪/早停均与主文件 main() 逐行一致；早停后用【最后一个 epoch】的模型（不回滚）。
    调用前须已执行 set_seed(SEED)。"""
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


def _jit_disable_fusion():
    """禁用 TorchScript TensorExpr 融合与 profiling executor：本机 CUDA 缺失
    nvrtc-builtins64_118.dll，融合内核 JIT 编译会失败；关闭后回退到普通算子，避免中断。"""
    try:
        torch._C._jit_set_texpr_fuser_enabled(False)
    except Exception:
        pass
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except Exception:
        pass


def _load_scripted_model(path, device):
    """以 TorchScript 方式加载模型（BytesIO 中转兼容中文路径），显式指定 map_location 设备。"""
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    m = torch.jit.load(buf, map_location=device)
    m.eval()
    return m


def _build_eager_model(sd, device, num_features):
    """用固定编码重建 eager HybridCNNBiLSTM 并装入 state_dict（设备可自由指定，无 nvrtc 依赖）。"""
    m = HybridCNNBiLSTM(FIXED_ENCODING['topo'], FIXED_ENCODING['cnn_params'],
                        FIXED_ENCODING['lstm_params'], num_features, SEQUENCE_LENGTH).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def _smoke_test(model, device, num_features, need_grad):
    """在目标设备上跑一次前向（need_grad 时再加一次对输入的反向），验证是否可用。
    本机 CUDA 上 TorchScript 反向缺 nvrtc、eager LSTM 在 eval 下反向受 cudnn 限制，
    故用冒烟测试自动选择可用设备（CUDA 不可用则回退 CPU）。返回 True/False。"""
    try:
        model.eval()
        x = torch.rand(2, SEQUENCE_LENGTH, num_features, device=device, dtype=torch.float32)
        if need_grad:
            x = x.requires_grad_(True)
            out = model(x)
            torch.autograd.grad(out.sum(), x)[0]
        else:
            with torch.no_grad():
                model(x)
        return True
    except Exception as e:
        print(f">>> 设备 {device} 冒烟测试失败（{str(e)[:120]}），尝试回退。")
        return False


def load_or_train_model(data_result, need_grad=False):
    """加载基准模型并自动选择可用设备，返回 (model, device)。
    加载优先级与 baseline_gradient_attacks.py 的 load_or_train_model 一致：
      1) TorchScript（weights/pretrained_moacb_wsf_scripted.pt，BytesIO 中转）
      2) state_dict（weights/pretrained_moacb_wsf.pt + 固定编码重建 HybridCNNBiLSTM）
      3) 两者都缺失才用固定编码重训一次并保存（+ 导出 TorchScript）
    在每个候选设备（fe.DEVICE → CPU）上做冒烟测试，保证前向/反向在本机可跑通；
    三个基线脚本共用同一权重（控制变量，绝不逐次重训）。
    need_grad：白盒方法（C&W）传 True，要求反向也可用；黑盒方法传 False（仅需前向）。"""
    num_features = data_result['num_features']
    topo = FIXED_ENCODING['topo']
    cnn_params = FIXED_ENCODING['cnn_params']
    lstm_params = FIXED_ENCODING['lstm_params']
    setting = FIXED_ENCODING['setting']
    _jit_disable_fusion()

    scripted_path = scripted_ckpt_path(PRETRAINED_CKPT)
    sd = None
    if os.path.isfile(PRETRAINED_CKPT):
        try:
            ck = torch.load(PRETRAINED_CKPT, map_location='cpu')
            sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
        except Exception as e:
            print(f">>> 读取 state_dict checkpoint 失败（{e}）。")

    # ---------- 1) / 2)：按设备候选（CUDA 优先，回退 CPU）依次尝试 TorchScript → state_dict ----------
    seen = set()
    for device in [DEVICE, torch.device('cpu')]:
        if str(device) in seen:
            continue
        seen.add(str(device))
        # 1) TorchScript 优先（无需模型类定义，直接推理）
        if os.path.isfile(scripted_path):
            try:
                m = _load_scripted_model(scripted_path, device)
                if _smoke_test(m, device, num_features, need_grad):
                    print(f"\n>>> 已加载 TorchScript 基准模型（无需模型类定义）: {scripted_path} @ {device}（跳过训练）")
                    return m, device
            except Exception as e:
                print(f">>> TorchScript 加载失败 @ {device}（{str(e)[:120]}）")
        # 2) 回退 state_dict + 固定编码重建 eager
        if sd is not None:
            try:
                m = _build_eager_model(sd, device, num_features)
                if _smoke_test(m, device, num_features, need_grad):
                    print(f"\n>>> 已加载 state_dict 基准模型（eager，固定编码重建）: {PRETRAINED_CKPT} @ {device}（跳过训练）")
                    return m, device
            except Exception as e:
                print(f">>> state_dict 加载失败 @ {device}（{str(e)[:120]}）")

    # ---------- 3) 都缺失/都不可用才重训一次并保存 ----------
    print("\n>>> 未找到可用预训练权重/设备，使用固定编码重新训练 ...")
    set_seed(SEED)
    model = train_fixed_encoding_model(data_result)
    torch.save({
        'model_state_dict': model.state_dict(),
        'topo': topo, 'cnn_params': cnn_params, 'lstm_params': lstm_params, 'setting': setting,
    }, PRETRAINED_CKPT)
    print(f"\n>>> 训练完成，权重已保存至: {PRETRAINED_CKPT}")
    export_scripted_model(model, scripted_ckpt_path(PRETRAINED_CKPT), num_features=num_features)
    return model, DEVICE


def extract_test_data(test_dataset):
    """一次性取出整个测试集为张量：X_test:(N,20,4)、Y_test:(N,1)，均在 DEVICE 上。"""
    loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, Y_test = next(iter(loader))
    return X_test.to(DEVICE), Y_test.to(DEVICE)


def compute_feature_ranges(data_result):
    """在【训练集】上按特征计算 range_f = max - min（归一化空间，即模型实际输入口径）。
    返回 (range_f:(F,) float32, n_train:int)。严禁用统一标量代替。"""
    train_dataset = data_result['train_dataset']
    num_features = data_result['num_features']
    xs = np.concatenate([x.numpy() for x, _ in DataLoader(train_dataset, batch_size=512, shuffle=False)], axis=0)
    flat = xs.reshape(-1, num_features)
    range_f = (flat.max(axis=0) - flat.min(axis=0)).astype('float32')
    return range_f, xs.shape[0]


# =============================================================================
# SECTION 2: GenAttack 遗传算法攻击器（黑盒，回归改造 = 最大化预测误差，自适应变异）
# =============================================================================
class GenAttack:
    """GenAttack（Alzantot et al. 2019）的遗传算法版，回归改造：最大化 MSE_adv。

    基因编码：个体 G 形状 (k,2)，G[:,0]=格点位置 pos∈[0,T*F)，G[:,1]=幅值分数 val∈[-1,1]；
              实际扰动 δ[pos] = val * β·range_f[f(pos)]，f(pos)=pos % F（(T,F) 行主序展平）。
      · 稀疏（n_points>0）：k=n_points，位置在进化中可变（去重保证唯一）；
      · 稠密（n_points=0）：k=T*F，位置固定为全部格点（arange），只进化 val。
    流程：种群初始化 → 批量适应度评估 → 锦标赛选择 → 均匀交叉 →【自适应变异】→ 精英保留。
    【自适应变异】σ_gen = σ0 * (mutation_decay ** gen)，随代数指数衰减（前期大、后期小），
                  这是 GenAttack 的核心机制，绝不退化成固定变异率。
    """

    def __init__(self, model, range_f, beta=BETA, pop_size=GENATTACK_CONFIG['pop_size'],
                 maxiter=GENATTACK_CONFIG['maxiter'], mutation_rate=GENATTACK_CONFIG['mutation_rate'],
                 mutation_decay=GENATTACK_CONFIG['mutation_decay'], n_points=GENATTACK_CONFIG['n_points'],
                 sigma0=MUTATION_SIGMA0, tournament_k=TOURNAMENT_K, device=DEVICE):
        self.model = model
        self.beta = float(beta)
        self.pop_size = max(2, int(pop_size))
        self.maxiter = int(maxiter)
        self.mutation_rate = float(mutation_rate)
        self.mutation_decay = float(mutation_decay)
        self.sigma0 = float(sigma0)
        self.tournament_k = int(tournament_k)
        self.device = device
        self.seq_len = int(SEQUENCE_LENGTH)
        self.num_features = int(len(FEATURE_COLUMNS))
        self.n_grid = self.seq_len * self.num_features
        self.dense = (int(n_points) == 0)
        self.k = self.n_grid if self.dense else max(1, int(n_points))
        # 逐特征预算展平到格点：eps_flat[pos] = beta * range_f[pos % F]（(T,F) 行主序）
        rf = np.asarray(range_f, dtype=float).reshape(-1)
        self.eps_flat = np.tile(self.beta * rf, self.seq_len)     # (n_grid,)
        self.query_count = 0

    # ---------------- 种群初始化 ----------------
    def _init_individual(self):
        G = np.zeros((self.k, 2), dtype=float)
        if self.dense:
            G[:, 0] = np.arange(self.n_grid)                       # 稠密：位置固定为全部格点
        else:
            G[:, 0] = np.random.choice(self.n_grid, size=self.k, replace=False)  # 稀疏：互异位置
        G[:, 1] = np.random.random(self.k) * 2 - 1                 # val ~ U(-1,1)
        return G

    def _repair_unique(self, G):
        """稀疏模式：修复重复位置，保证 k 个基因的 pos 互异（维持 L0 = k）。"""
        seen = set()
        for j in range(self.k):
            p = int(G[j, 0])
            guard = 0
            while p in seen and guard < 200:
                p = int(np.random.randint(0, self.n_grid))
                guard += 1
            seen.add(p)
            G[j, 0] = p
        return G

    # ---------------- 由基因构建扰动 (批量) ----------------
    def _build_delta_batch(self, Gs):
        """Gs:(P,k,2) → δ:(P,T,F)。δ_flat[pos] = val * eps_flat[pos]（重复位置后者覆盖）。"""
        P = Gs.shape[0]
        dflat = np.zeros((P, self.n_grid), dtype=float)
        for i in range(P):
            pos = np.clip(Gs[i, :, 0].astype(int), 0, self.n_grid - 1)
            val = np.clip(Gs[i, :, 1], -1.0, 1.0)
            dflat[i, pos] = val * self.eps_flat[pos]
        return dflat.reshape(P, self.seq_len, self.num_features)

    # ---------------- 批量适应度评估：fitness = MSE_adv（最大化） ----------------
    def _eval_batch(self, x_window, y_true, Gs):
        delta = self._build_delta_batch(Gs)
        P = Gs.shape[0]
        Xa = np.tile(x_window, (P, 1, 1)) + delta
        with torch.no_grad():
            pred = self.model(torch.as_tensor(Xa, dtype=torch.float32, device=self.device)).squeeze(-1)
            pred = pred.detach().cpu().numpy()
        self.query_count += P
        return (pred - y_true) ** 2                                # (P,) MSE_adv，越大越好

    # ---------------- 锦标赛选择 ----------------
    def _tournament_select(self, pop, fits):
        idxs = np.random.randint(0, self.pop_size, size=self.tournament_k)
        best = idxs[int(np.argmax(fits[idxs]))]
        return pop[best].copy()

    # ---------------- 均匀交叉（逐基因行来自父/母） ----------------
    def _crossover(self, pa, pb):
        child = pa.copy()
        take_b = np.random.random(self.k) < 0.5
        child[take_b] = pb[take_b]
        if self.dense:
            child[:, 0] = np.arange(self.n_grid)                   # 稠密：位置列保持固定
        return child

    # ---------------- 【自适应变异】σ 随代数衰减 ----------------
    def _mutate(self, child, sigma):
        for j in range(self.k):
            if np.random.random() < self.mutation_rate:
                # 幅值变异：加高斯噪声（强度 = 当前代 σ），裁剪到 [-1,1]
                child[j, 1] = np.clip(child[j, 1] + np.random.normal(0.0, sigma), -1.0, 1.0)
                # 位置变异（仅稀疏）：以 POS_MUTATE_PROB 概率迁移到新格点
                if not self.dense and np.random.random() < POS_MUTATE_PROB:
                    child[j, 0] = float(np.random.randint(0, self.n_grid))
        return child

    # ---------------- 单样本 GA 攻击 ----------------
    def attack_single_sample(self, x_window, y_true, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x_window = np.asarray(x_window, dtype=float)

        pop = np.stack([self._init_individual() for _ in range(self.pop_size)], axis=0)  # (P,k,2)
        if not self.dense:
            pop = np.stack([self._repair_unique(pop[i].copy()) for i in range(self.pop_size)], axis=0)
        fits = self._eval_batch(x_window, y_true, pop)
        best_i = int(np.argmax(fits))
        best_G, best_fit = pop[best_i].copy(), fits[best_i]

        for gen in range(self.maxiter):
            sigma = self.sigma0 * (self.mutation_decay ** gen)     # 【自适应变异】衰减强度
            new_pop = [pop[int(np.argmax(fits))].copy()]           # 精英保留
            while len(new_pop) < self.pop_size:
                pa = self._tournament_select(pop, fits)
                pb = self._tournament_select(pop, fits)
                child = self._crossover(pa, pb)
                child = self._mutate(child, sigma)
                if not self.dense:
                    child = self._repair_unique(child)
                new_pop.append(child)
            pop = np.stack(new_pop, axis=0)
            fits = self._eval_batch(x_window, y_true, pop)
            cur_best = int(np.argmax(fits))
            if fits[cur_best] > best_fit:
                best_fit = fits[cur_best]
                best_G = pop[cur_best].copy()

        # 由最优基因构建对抗样本（不裁剪到 [0,1]，保持扰动预算精确，与 nVITA 黑盒约定一致）
        delta = self._build_delta_batch(best_G[None])[0]           # (T,F)
        x_adv = x_window + delta
        return x_adv, float(best_fit)

    # ---------------- 批量攻击 ----------------
    def attack_batch(self, X_test, Y_test, print_info=True):
        self.model.to(self.device)
        self.model.eval()
        n_total = X_test.shape[0]
        X_adv_list = []
        for i in range(n_total):
            x_window = X_test[i].detach().cpu().numpy()
            y_true = float(Y_test[i].detach().cpu().numpy().reshape(-1)[0])
            x_adv, _ = self.attack_single_sample(x_window, y_true, seed=int(i))
            X_adv_list.append(x_adv)
            if print_info and (i + 1) % 20 == 0:
                print(f"  GenAttack progress: {i + 1}/{n_total}")
        X_adv = torch.as_tensor(np.stack(X_adv_list, axis=0), dtype=torch.float32, device=self.device)
        return X_adv


# =============================================================================
# SECTION 3: 统一评估（两个口径：A 池化 m/s；B 逐样本 归一化空间）
# =============================================================================
def evaluate_per_sample(model, X, y, X_adv):
    """【B. 逐样本口径】先逐样本计算再取平均，全部在【归一化空间】，
    与 baseline_gradient_attacks.py 的 evaluate_attack_metrics 同口径：
        mean_L0   = mean( count(|X_adv - X| > 1e-8) )
        mean_L2   = mean( ||X_adv - X||_2 )
        mean_Linf = mean( max|X_adv - X| )
        mean_RSE  = mean( sqrt(MSE_adv_norm / max(MSE_clean_norm, 1e-8)) )
        ASR       = mean( RSE_sample >= 1.0 )   （τ=1.0）
    【重要】此处 MSE 为【归一化空间】，与池化口径的 m/s 空间 MSE 是两个不同口径，字段名不混用。"""
    model.eval()
    X = X.to(DEVICE); y = y.to(DEVICE); X_adv = X_adv.to(DEVICE)
    with torch.no_grad():
        pred_clean = model(X).squeeze(-1)
        pred_adv = model(X_adv).squeeze(-1)
        yy = y.squeeze(-1)
        mse_clean_norm = (pred_clean - yy) ** 2
        mse_adv_norm = (pred_adv - yy) ** 2
        rse_sample = torch.sqrt(mse_adv_norm / torch.clamp(mse_clean_norm, min=RSE_EPS))
        succ = (rse_sample >= RSE_SUCCESS_THRESHOLD).float()
        diff = X_adv - X
        l0 = (diff.abs() > L0_EPS).sum(dim=(1, 2)).float()
        l2 = torch.sqrt((diff ** 2).sum(dim=(1, 2)))
        linf = diff.abs().amax(dim=(1, 2))
    return {
        'mean_L0': float(l0.mean().item()),
        'mean_L2': float(l2.mean().item()),
        'mean_Linf': float(linf.mean().item()),
        'mean_RSE': float(rse_sample.mean().item()),
        'ASR': float(succ.mean().item()),
    }


def evaluate_pooled(model, X, y, X_adv, min_speed, max_speed):
    """【A. 池化口径】把全部测试样本拼接后统一计算，与主文件 evaluate_attack 同口径：
    先反归一化到 m/s：v_denorm = v*(max_speed-min_speed)+min_speed，再算
    MAE / MSE / RMSE / MAPE(%) / R² / RSE(=rmse_adv/rmse_clean)，并给出 Δ 与 Change%。
    【重要】此处 MSE 为【m/s 平方口径】（= RMSE²），供报告用，与逐样本归一化 MSE 分开存储。"""
    model.eval()
    scale = float(max_speed - min_speed)
    X = X.to(DEVICE); y = y.to(DEVICE); X_adv = X_adv.to(DEVICE)
    with torch.no_grad():
        pred_clean = model(X).cpu().numpy()
        pred_adv = model(X_adv).cpu().numpy()
    y_denorm = (y.cpu().numpy() * scale + min_speed).flatten()
    yc = (pred_clean * scale + min_speed).flatten()
    ya = (pred_adv * scale + min_speed).flatten()

    mae_clean = mean_absolute_error(y_denorm, yc)
    mae_adv = mean_absolute_error(y_denorm, ya)
    mse_clean = mean_squared_error(y_denorm, yc)      # m/s 平方口径（= rmse_clean²）
    mse_adv = mean_squared_error(y_denorm, ya)        # m/s 平方口径（= rmse_adv²）
    rmse_clean = np.sqrt(mse_clean)
    rmse_adv = np.sqrt(mse_adv)
    mape_clean = np.mean(np.abs((y_denorm - yc) / y_denorm)) * 100
    mape_adv = np.mean(np.abs((y_denorm - ya) / y_denorm)) * 100
    r2_clean = r2_score(y_denorm, yc)
    r2_adv = r2_score(y_denorm, ya)
    rse = rmse_adv / rmse_clean if rmse_clean > 1e-12 else 0.0

    def _chg(c, a):
        """相对变化率(%) = (adv - clean) / |clean| * 100；clean≈0 时返回 0。"""
        return (a - c) / abs(c) * 100 if abs(c) > 1e-12 else 0.0

    return {
        'mae_clean': mae_clean, 'mae_adv': mae_adv,
        'delta_mae': mae_adv - mae_clean, 'chg_mae_pct': _chg(mae_clean, mae_adv),
        'mse_clean': mse_clean, 'mse_adv': mse_adv,
        'delta_mse': mse_adv - mse_clean, 'chg_mse_pct': _chg(mse_clean, mse_adv),
        'rmse_clean': rmse_clean, 'rmse_adv': rmse_adv,
        'delta_rmse': rmse_adv - rmse_clean, 'chg_rmse_pct': _chg(rmse_clean, rmse_adv),
        'mape_clean': mape_clean, 'mape_adv': mape_adv,
        'delta_mape': mape_adv - mape_clean, 'chg_mape_pct': _chg(mape_clean, mape_adv),
        'r2_clean': r2_clean, 'r2_adv': r2_adv,
        'delta_r2': r2_adv - r2_clean, 'chg_r2_pct': _chg(r2_clean, r2_adv),
        'rse': rse,
    }


# =============================================================================
# SECTION 4: 输出（控制台对比表 / CSV / npz）
# =============================================================================
def print_attack_before_after(pooled, method, n_points, beta, n_test):
    """打印『攻击前后』池化指标对比表，格式对齐 baseline_gradient_attacks.print_attack_before_after。"""
    line = '=' * 70
    print('\n' + line)
    print(f'攻击前后指标对比 · {method}(n_points={n_points}) · β={beta} · 评估样本数 {n_test}（仅测试集，池化口径）')
    print(line)
    print(f"{'Metric':<10}{'Clean':>14}{'Adv':>14}{'Delta':>14}{'Change%':>14}")
    print('-' * 70)
    print(f"{'MAE':<10}{pooled['mae_clean']:>14.4f}{pooled['mae_adv']:>14.4f}"
          f"{pooled['delta_mae']:>+14.4f}{pooled['chg_mae_pct']:>+13.2f}%")
    print(f"{'MSE':<10}{pooled['mse_clean']:>14.4f}{pooled['mse_adv']:>14.4f}"
          f"{pooled['delta_mse']:>+14.4f}{pooled['chg_mse_pct']:>+13.2f}%")
    print(f"{'RMSE':<10}{pooled['rmse_clean']:>14.4f}{pooled['rmse_adv']:>14.4f}"
          f"{pooled['delta_rmse']:>+14.4f}{pooled['chg_rmse_pct']:>+13.2f}%")
    print(f"{'MAPE(%)':<10}{pooled['mape_clean']:>14.2f}{pooled['mape_adv']:>14.2f}"
          f"{pooled['delta_mape']:>+14.2f}{pooled['chg_mape_pct']:>+13.2f}%")
    print(f"{'R2':<10}{pooled['r2_clean']:>14.4f}{pooled['r2_adv']:>14.4f}"
          f"{pooled['delta_r2']:>+14.4f}{pooled['chg_r2_pct']:>+13.2f}%")
    print(f"{'RSE':<10}{1.0:>14.4f}{pooled['rse']:>14.4f}{pooled['rse'] - 1.0:>+14.4f}{'--':>14}")
    print(line)
    print('注: MAE/RMSE/MAPE/MSE 的 Change%>0 表示攻击后误差上升(性能恶化); R² 的 Change%<0 表示拟合变差。')


def save_results_csv(row, path=RESULT_CSV):
    """保存评估指标到 CSV（列名固定为 CSV_COLUMNS，编码 utf-8-sig，可直接 pd.concat）。"""
    df = pd.DataFrame([row])[CSV_COLUMNS]
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n>>> 结果已保存: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global DEVICE
    parser = argparse.ArgumentParser(description='GenAttack（遗传算法，黑盒）基线')
    parser.add_argument('--data_file', type=str, default=FILENAME, help='输入数据文件')
    parser.add_argument('--beta', type=float, default=BETA, help='扰动预算系数 β（逐特征 range_f 缩放）')
    parser.add_argument('--pop_size', type=int, default=GENATTACK_CONFIG['pop_size'], help='种群大小')
    parser.add_argument('--maxiter', type=int, default=GENATTACK_CONFIG['maxiter'], help='最大代数')
    parser.add_argument('--mutation_rate', type=float, default=GENATTACK_CONFIG['mutation_rate'],
                        help='每个基因被变异的概率')
    parser.add_argument('--mutation_decay', type=float, default=GENATTACK_CONFIG['mutation_decay'],
                        help='自适应变异强度衰减因子（σ_gen = σ0 * decay^gen）')
    parser.add_argument('--n_points', type=int, default=GENATTACK_CONFIG['n_points'],
                        help='扰动点数：默认 5（稀疏）；0=稠密（全部 80 格点）')
    parser.add_argument('--num_eval_samples', type=int, default=GENATTACK_CONFIG['num_eval_samples'],
                        help='参加评估的测试样本数（默认 None=全部，正整数则从测试集头部截取）')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    args = parser.parse_args()

    set_seed(args.seed)

    n_grid = int(SEQUENCE_LENGTH) * int(len(FEATURE_COLUMNS))
    dense = (args.n_points == 0)
    n_points = n_grid if dense else args.n_points

    print(f"\n{'=' * 80}")
    print("GenAttack（遗传算法，黑盒）—— 回归改造：最大化预测误差 + 自适应变异")
    print(f"{'=' * 80}")
    print(f"首选设备      : {DEVICE}（实际设备由冒烟测试自动确定，CUDA 不可用则回退 CPU）")
    print(f"Data file     : {args.data_file}")
    print(f"模式          : {'稠密（全部 %d 格点）' % n_grid if dense else '稀疏 n_points=%d' % args.n_points}")
    print(f"pop_size      : {args.pop_size}   maxiter: {args.maxiter}")
    print(f"mutation      : rate={args.mutation_rate}, decay={args.mutation_decay}, σ0={MUTATION_SIGMA0}")
    print(f"beta (预算)   : {args.beta}")

    # ---------- 1) 加载数据 ----------
    data_result = load_and_preprocess_data(args.data_file)
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']

    # ---------- 2) 扰动预算基数：训练集逐特征 range_f ----------
    range_f, n_train = compute_feature_ranges(data_result)
    print(f"[扰动预算] 训练集逐特征 range_f = {np.round(range_f, 4).tolist()}（预算=β·range_f，与其它方法同口径）")

    # ---------- 3) 获取基准模型（优先加载权威 checkpoint，控制变量；自动选择可用设备） ----------
    model, DEVICE = load_or_train_model(data_result, need_grad=False)
    print(f"[设备] 实际使用: {DEVICE}")

    # ---------- 4) 提取测试数据 ----------
    X_test, Y_test = extract_test_data(data_result['test_dataset'])
    print(f"\n[测试集] 共 {X_test.shape[0]} 个样本，shape={tuple(X_test.shape)}")

    if args.num_eval_samples is not None and args.num_eval_samples > 0:
        n_take = min(args.num_eval_samples, X_test.shape[0])
        sample_indices = list(range(n_take))
        print(f"[评估样本截取] 取测试集前 {n_take} 个样本参与评估")
    else:
        n_take = X_test.shape[0]
        sample_indices = None

    X_eval = X_test if sample_indices is None else X_test[sample_indices]
    Y_eval = Y_test if sample_indices is None else Y_test[sample_indices]

    # ---------- 5) 运行 GenAttack ----------
    print(f"\n{'=' * 80}")
    print(f"开始 GenAttack（{'稠密 %d 格点' % n_grid if dense else '稀疏 n_points=%d' % args.n_points}，"
          f"最大化 MSE_adv，逐特征 β·range_f 预算，自适应变异）")
    print(f"{'=' * 80}")
    attacker = GenAttack(model, range_f, beta=args.beta, pop_size=args.pop_size, maxiter=args.maxiter,
                         mutation_rate=args.mutation_rate, mutation_decay=args.mutation_decay,
                         n_points=args.n_points, device=DEVICE)
    t0 = time.perf_counter()
    X_adv = attacker.attack_batch(X_eval, Y_eval, print_info=True)
    print(f"\n>>> GenAttack 完成，总耗时 {time.perf_counter() - t0:.2f} s，"
          f"累计模型查询 {attacker.query_count} 次")

    # ---------- 6) 两层指标 ----------
    per_sample = evaluate_per_sample(model, X_eval, Y_eval, X_adv)   # 逐样本（归一化空间）
    pooled = evaluate_pooled(model, X_eval, Y_eval, X_adv, min_speed, max_speed)  # 池化（m/s）

    print(f"\n--- 逐样本口径（归一化空间）---")
    print(f"  mean_L0  : {per_sample['mean_L0']:.3f}   (期望 ≈ {n_points})")
    print(f"  mean_L2  : {per_sample['mean_L2']:.6f}")
    print(f"  mean_Linf: {per_sample['mean_Linf']:.6f}")
    print(f"  mean_RSE : {per_sample['mean_RSE']:.4f}")
    print(f"  ASR(τ={RSE_SUCCESS_THRESHOLD:g}): {per_sample['ASR'] * 100:.1f}%")

    # ---------- 7) 打印攻击前后对比表 ----------
    print_attack_before_after(pooled, METHOD_NAME, n_points, args.beta, n_take)

    # ---------- 8) 组装 CSV 行（列名与另两个基线脚本完全一致） ----------
    row = {
        'method': METHOD_NAME, 'beta': args.beta, 'n_points': n_points,
        'mean_RSE': per_sample['mean_RSE'], 'ASR': per_sample['ASR'],
        'mean_L0': per_sample['mean_L0'], 'mean_L2': per_sample['mean_L2'], 'mean_Linf': per_sample['mean_Linf'],
        'mae_clean': pooled['mae_clean'], 'mae_adv': pooled['mae_adv'],
        'mse_clean': pooled['mse_clean'], 'mse_adv': pooled['mse_adv'],
        'rmse_clean': pooled['rmse_clean'], 'rmse_adv': pooled['rmse_adv'],
        'mape_clean': pooled['mape_clean'], 'mape_adv': pooled['mape_adv'],
        'r2_clean': pooled['r2_clean'], 'r2_adv': pooled['r2_adv'],
        'rse': pooled['rse'], 'n': int(n_take),
    }
    save_results_csv(row, RESULT_CSV)

    # ---------- 9) 保存对抗样本供复用 ----------
    np.savez(ADV_NPZ, X_adv=X_adv.detach().cpu().numpy())
    print(f">>> 对抗样本已保存: {ADV_NPZ}")

    print(f"\n{'=' * 80}")
    print("GenAttack 基线完成！输出文件：")
    print(f"  - 结果 CSV   : {RESULT_CSV}")
    print(f"  - 对抗样本   : {ADV_NPZ}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
