# -*- coding: utf-8 -*-
"""
baseline_dense_gradient_attacks.py —— 标准【全局/稠密】FGSM / BIM 白盒梯度攻击基线
================================================================================
独立运行脚本：在 MoACB-WSF 固定编码风速预测模型的【测试集】上，实现并评估
DenseFGSM（一步）与 DenseBIM（迭代 I-FGSM）两种【非目标】【稠密】白盒梯度攻击，
作为 baseline_gradient_attacks.py 中 SparseFGSM / SparseBIM（top-n 稀疏选点）的
【对照】——本脚本【不做任何稀疏选点】，扰动作用于全部 (seq_len × 特征) = 80 个格点，
输出攻击性能数据、CSV、对抗样本 npz 与控制台 clean→adv 对比表，可与稀疏版直接
用 pd.concat 合并成一张横向对比表。

------------------------------------------------------------------
复用现有约定（严格对齐 CW_L2_Attack.py / baseline_gradient_attacks.py）：
  * 用 importlib 动态加载主文件 LBA-MOACB-WSF-FixedEncoding.py（模块名 fe），复用
    SEQUENCE_LENGTH / FEATURE_COLUMNS / DEVICE / WEIGHTS_DIR / FIXED_ENCODING、
    load_and_preprocess_data、HybridCNNBiLSTM、decode_hyperparams / encode_individual /
    is_valid_individual、scripted_ckpt_path / load_scripted_model / export_scripted_model；
  * 模型加载顺序与 baseline_gradient_attacks.py 的 load_or_train_model 完全一致：
      1) 优先 TorchScript（weights/pretrained_moacb_wsf_scripted.pt，BytesIO 中转）
      2) 回退 state_dict（weights/pretrained_moacb_wsf.pt + 固定编码重建结构）
      3) 两者都缺失才用固定编码重训一次并保存
    绝不逐次重训；各基线脚本跑在【同一基准模型】上（控制变量）；
  * 白盒攻击全程 model.eval()，仅对【输入 X】求梯度（本机 CUDA 反向不可用时经冒烟
    测试自动回退 CPU eager 模型）；
  * 扰动预算 = beta * range_f[f]（训练集逐特征 max-min，归一化空间），beta=0.01；
    严禁用统一标量 ε 代替逐特征预算；
  * 非目标攻击梯度符号为 +sign（最大化预测误差 MSE）；
  * SEED=42，保证可复现；仅用测试集，绝不混入验证集。

------------------------------------------------------------------
与稀疏版（baseline_gradient_attacks.py）的唯一区别 = 稀疏 vs 稠密：
  * SparseFGSM/SparseBIM：把梯度展平后跨全部格点做全局 top-n 选点，仅在 n 个格点扰动；
  * DenseFGSM/DenseBIM ：【不加任何掩码】，全部 80 个格点同时按 ±β·range_f·sign(grad) 扰动
    （DenseBIM 每步在全部格点上按 α=β·range_f/N 累积，并做预算 clip + 合法域 [0,1] clip）。
  因此稠密版实际 L0 ≈ 80（被 [0,1] 边界 clip 卡住的格点可能略小，如实体现，不伪装）。

运行：python baseline_dense_gradient_attacks.py
      python baseline_dense_gradient_attacks.py --num_eval_samples 20   （小样本快速验证）
      python baseline_dense_gradient_attacks.py --beta 0.05 --bim_steps 200
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

# 【Windows 控制台兼容】默认 GBK 编码无法输出 R² / δ / ∇ / β 等字符（UnicodeEncodeError），
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
# 全局约定（各基线脚本必须逐字一致，否则无法 pd.concat 合并成一张表）
# =============================================================================
SEED = 42                       # 随机种子，保证可复现
BETA = 0.0025                     # 扰动预算系数 beta（逐特征 range_f 缩放），与其它方法一致
BIM_STEPS = 10                 # 稠密 BIM 迭代步数 N（n_iter），与稀疏版一致
FILENAME = fe.FILENAME
METHODS = ['DenseFGSM', 'DenseBIM']    # 本脚本对比的两种全局稠密攻击

# 指标口径常量（与 baseline_gradient_attacks.py / CW_L2_Attack.py 一致）
RSE_SUCCESS_THRESHOLD = 1.0     # ASR 判定阈值 τ（逐样本 RSE >= τ 记为成功）
L0_EPS = 1e-8                   # L0 计数阈值：|δ| > 该值视为一次扰动
RSE_EPS = 1e-8                  # RSE 分母保护，避免除零

# CSV 列名（各基线脚本完全一致，便于直接 pd.concat）
CSV_COLUMNS = ['method', 'beta', 'n_points', 'mean_RSE', 'ASR', 'mean_L0', 'mean_L2', 'mean_Linf',
               'mae_clean', 'mae_adv', 'mse_clean', 'mse_adv', 'rmse_clean', 'rmse_adv',
               'mape_clean', 'mape_adv', 'r2_clean', 'r2_adv', 'rse', 'n']

RESULT_CSV = os.path.join(OUTPUT_DIR, 'baseline_dense_gradient_attacks_results.csv')


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


def load_or_train_model(data_result, need_grad=True):
    """加载基准模型并自动选择可用设备，返回 (model, device)。
    加载优先级与 baseline_gradient_attacks.py 的 load_or_train_model 一致：
      1) TorchScript（weights/pretrained_moacb_wsf_scripted.pt，BytesIO 中转；traced 保留 autograd）
      2) state_dict（weights/pretrained_moacb_wsf.pt + 固定编码重建 HybridCNNBiLSTM）
      3) 两者都缺失才用固定编码重训一次并保存（+ 导出 TorchScript）
    在每个候选设备（fe.DEVICE → CPU）上做冒烟测试；FGSM/BIM 为白盒，need_grad=True 要求
    【对输入的反向】也可用（本机 CUDA 反向不可用时自动回退 CPU），保证白盒梯度可算；
    各基线脚本共用同一权重（控制变量，绝不逐次重训）。"""
    num_features = data_result['num_features']
    topo = FIXED_ENCODING['topo']
    cnn_params = FIXED_ENCODING['cnn_params']
    lstm_params = FIXED_ENCODING['lstm_params']
    setting = FIXED_ENCODING['setting']
    _jit_disable_fusion()

    # 白盒需对输入求梯度：cuDNN 的 RNN 反向仅支持 train 模式，eval 模式下会报
    # "cudnn RNN backward can only be called in training mode"；关闭 cuDNN 后 LSTM 走
    # PyTorch 原生实现，即可在 GPU 上对输入做 eval 反向，从而无需回退 CPU。
    if need_grad and torch.backends.cudnn.enabled:
        torch.backends.cudnn.enabled = False
        print(">>> 已关闭 cuDNN（LSTM 走原生实现），以便在 GPU 上进行输入梯度反向。")

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


def extract_test_data(test_dataset, device):
    """一次性取出整个测试集为张量：X_test:(N,20,4)、Y_test:(N,1)，均在指定 device 上。"""
    loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, Y_test = next(iter(loader))
    return X_test.to(device), Y_test.to(device)


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
# SECTION 2: 标准【全局/稠密】FGSM / BIM 白盒梯度攻击（不选点，全部格点扰动）
# =============================================================================
# 【批量等价性说明】model.eval() 下 BatchNorm 使用 running 统计量、Dropout 关闭，
# 前向对每个样本相互独立；因此对整批测试样本一次前向/反向所得的逐样本梯度符号，
# 与逐样本单独计算完全一致（均值 reduction 仅引入正的 1/N 缩放，不改变 sign）。
# 【稠密性说明】与稀疏版唯一区别：此处【不做 top-n 选点、不加掩码】，delta 作用于
# 全部 (seq_len × 特征) 格点，故实际 L0 ≈ 80（仅当格点位于 0/1 边界被合法域 clip 卡住时略小）。

def _per_feature_budget(range_f, beta, device):
    """逐特征扰动预算 β·range_f，形状 (1,1,F)，可直接与 (N,T,F) 张量广播。
    严禁用统一标量 ε 代替——每个特征按各自训练集 max-min 单独缩放。"""
    rf = torch.as_tensor(range_f, dtype=torch.float32, device=device)   # (F,)
    return (beta * rf).view(1, 1, -1)                                   # (1,1,F)


def dense_fgsm_attack(model, X, y, beta, range_f, device):
    """标准【全局】FGSM 非目标白盒一步攻击（稠密，不选点）：
        (1) grad = ∇_x MSE(model(X), y)                       非目标 → 取 +sign
        (2) 对【全部格点】施加 δ = β·range_f·sign(grad)，不做任何 top-n 稀疏
        (3) X_adv = clip(X + δ, 0, 1)
    X:(N,T,F)  y:(N,1)  range_f:(F,)。返回 (X_adv:(N,T,F), delta:(N,T,F))。"""
    model.eval()                                          # 严禁在 train() 模式下算梯度
    X0 = X.clone().detach().to(device)
    eps = _per_feature_budget(range_f, beta, device)      # (1,1,F) 逐特征预算

    Xa = X0.clone().requires_grad_(True)
    out = model(Xa)
    loss = F.mse_loss(out, y)                             # 非目标：最大化该误差
    grad = torch.autograd.grad(loss, Xa)[0]               # ∇_x MSE，形状 (N,T,F)

    delta = eps * torch.sign(grad)                        # +sign：沿误差增大方向，全局稠密
    X_adv = torch.clamp(X0 + delta, 0.0, 1.0)             # 合法域 clip
    delta = X_adv - X0                                    # clip 后的真实扰动
    return X_adv.detach(), delta.detach()


def dense_bim_attack(model, X, y, beta, range_f, device, n_steps=BIM_STEPS):
    """标准【全局】BIM（I-FGSM）非目标白盒迭代攻击（稠密，不选点）：
        (1) α = β·range_f / N（按特征分别计算步长）
        (2) 迭代 N 次，每步在【全部格点】上：
                grad = ∇_x MSE(model(X_adv), y)              每步重估梯度符号
                X_adv ← X_adv + α·sign(grad)
                δ = clip(X_adv - X, -β·range_f, +β·range_f)   # ① 扰动预算 clip
                X_adv = clip(X + δ, 0, 1)                     # ② 合法域 clip
    与稀疏 BIM 的区别：无掩码、不固定 top-n，全部 80 格点每步都更新。
    X:(N,T,F)  y:(N,1)  range_f:(F,)。返回 (X_adv:(N,T,F), delta:(N,T,F))。"""
    model.eval()                                          # 严禁在 train() 模式下算梯度
    X0 = X.clone().detach().to(device)
    eps = _per_feature_budget(range_f, beta, device)      # (1,1,F) 逐特征扰动预算
    alpha = eps / n_steps                                 # (1,1,F) 逐特征步长

    X_adv = X0.clone()
    for _ in range(n_steps):
        Xa = X_adv.detach().requires_grad_(True)
        out = model(Xa)
        loss = F.mse_loss(out, y)
        grad = torch.autograd.grad(loss, Xa)[0]           # 每步在当前点对输入重估梯度
        X_adv = Xa.detach() + alpha * torch.sign(grad)    # +sign：非目标，全局稠密
        delta = torch.clamp(X_adv - X0, -eps, eps)        # ① 扰动预算 clip
        X_adv = torch.clamp(X0 + delta, 0.0, 1.0)         # ② 合法域 clip

    delta = X_adv - X0                                    # clip 后的真实扰动
    return X_adv.detach(), delta.detach()


def run_attack(model, X, y, method, beta, range_f, device, n_steps=BIM_STEPS):
    """按方法名分派到稠密 FGSM / BIM，返回 (X_adv, delta)。"""
    if method == 'DenseFGSM':
        return dense_fgsm_attack(model, X, y, beta, range_f, device)
    elif method == 'DenseBIM':
        return dense_bim_attack(model, X, y, beta, range_f, device, n_steps=n_steps)
    else:
        raise ValueError(f'未知方法: {method}（仅支持 DenseFGSM / DenseBIM）')


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
    """打印『攻击前后』池化指标对比表，格式对齐 baseline_gradient_attacks / CW_L2_Attack。"""
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


def save_results_csv(rows, path=RESULT_CSV):
    """保存评估指标到 CSV（列名固定为 CSV_COLUMNS，编码 utf-8-sig，可直接 pd.concat）。"""
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n>>> 结果已保存: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    global DEVICE, BETA, BIM_STEPS, SEED
    parser = argparse.ArgumentParser(
        description='标准【全局/稠密】FGSM / BIM 白盒梯度攻击基线（非目标 · 全部格点扰动 · 仅测试集评估）')
    parser.add_argument('--data_file', type=str, default=FILENAME, help='输入数据文件')
    parser.add_argument('--beta', type=float, default=BETA, help='扰动预算系数 β（逐特征 range_f 缩放）')
    parser.add_argument('--bim_steps', type=int, default=BIM_STEPS, help='稠密 BIM 迭代步数 N（n_iter）')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='参加评估的测试样本数（默认 None=全部，正整数则从测试集头部截取）')
    args = parser.parse_args()

    BETA = args.beta
    BIM_STEPS = args.bim_steps
    SEED = args.seed
    set_seed(SEED)

    n_grid = int(SEQUENCE_LENGTH) * int(len(FEATURE_COLUMNS))   # 稠密：全部格点数（=80）
    n_points = n_grid                                           # 写入 CSV 的 n_points 列

    print(f"\n{'=' * 80}")
    print("标准【全局/稠密】FGSM / BIM 白盒梯度攻击基线（非目标 · 全部格点扰动 · 仅测试集）")
    print(f"{'=' * 80}")
    print(f"首选设备      : {DEVICE}（实际设备由冒烟测试自动确定；白盒需反向，CUDA 不可用则回退 CPU）")
    print(f"Data file     : {args.data_file}")
    print(f"扰动模式      : 稠密（全部 {n_grid} 个格点，不做 top-n 选点，对照稀疏版）")
    print(f"beta (预算)   : {BETA}   预算 = β·range_f（逐特征）")
    print(f"BIM 迭代步数 N: {BIM_STEPS}   ASR 阈值 τ = {RSE_SUCCESS_THRESHOLD:g}")

    # ---------- 1) 加载数据 ----------
    data_result = load_and_preprocess_data(args.data_file)
    min_speed = data_result['min_speed']
    max_speed = data_result['max_speed']

    # ---------- 2) 扰动预算基数：训练集逐特征 range_f ----------
    range_f, n_train = compute_feature_ranges(data_result)
    print(f"[扰动预算] 训练集逐特征 range_f = {np.round(range_f, 4).tolist()}（预算=β·range_f，与其它方法同口径）")

    # ---------- 3) 获取基准模型（优先加载权威 checkpoint，控制变量；自动选择可用设备） ----------
    model, DEVICE = load_or_train_model(data_result, need_grad=True)
    print(f"[设备] 实际使用: {DEVICE}")

    # ---------- 4) 提取测试数据 ----------
    X_test, Y_test = extract_test_data(data_result['test_dataset'], DEVICE)
    n_test = X_test.shape[0]
    print(f"\n[测试集] 共 {n_test} 个样本，shape={tuple(X_test.shape)}")

    if args.num_eval_samples is not None and args.num_eval_samples > 0:
        n_take = min(args.num_eval_samples, n_test)
        X_eval = X_test[:n_take]
        Y_eval = Y_test[:n_take]
        print(f"[评估样本截取] 取测试集前 {n_take} 个样本参与评估")
    else:
        n_take = n_test
        X_eval, Y_eval = X_test, Y_test

    # ---------- 5) 逐个方法运行稠密攻击并评估 ----------
    results = []
    for method in METHODS:
        print(f"\n{'=' * 80}")
        reverse_note = '1 次反向' if method == 'DenseFGSM' else f'{BIM_STEPS} 次反向'
        print(f"开始 {method}（{reverse_note}，全部 {n_grid} 格点扰动，逐特征 β·range_f 预算，X_adv clip 到 [0,1]）")
        print(f"{'=' * 80}")
        t0 = time.perf_counter()
        X_adv, delta = run_attack(model, X_eval, Y_eval, method, BETA, range_f, DEVICE, n_steps=BIM_STEPS)
        dt = time.perf_counter() - t0
        print(f">>> {method} 攻击完成，耗时 {dt:.2f} s")

        # 逐格点断言：稠密扰动不超过预算，且落在合法域 [0,1]
        eps_full = _per_feature_budget(range_f, BETA, DEVICE).expand_as(delta)
        assert bool((delta.abs() <= eps_full + 1e-6).all()), f'{method}: 存在 |δ| > β·range_f'
        assert bool(((X_eval + delta) >= -1e-6).all() and ((X_eval + delta) <= 1 + 1e-6).all()), \
            f'{method}: X_adv 越出合法域 [0,1]'

        per_sample = evaluate_per_sample(model, X_eval, Y_eval, X_adv)   # 逐样本（归一化空间）
        pooled = evaluate_pooled(model, X_eval, Y_eval, X_adv, min_speed, max_speed)  # 池化（m/s）

        print(f"\n--- 逐样本口径（归一化空间）---")
        print(f"  mean_L0  : {per_sample['mean_L0']:.3f}   (稠密期望 ≈ {n_grid}，被 [0,1] 边界 clip 可能略小)")
        print(f"  mean_L2  : {per_sample['mean_L2']:.6f}")
        print(f"  mean_Linf: {per_sample['mean_Linf']:.6f}")
        print(f"  mean_RSE : {per_sample['mean_RSE']:.4f}")
        print(f"  ASR(τ={RSE_SUCCESS_THRESHOLD:g}): {per_sample['ASR'] * 100:.1f}%")

        print_attack_before_after(pooled, method, n_points, BETA, n_take)

        row = {
            'method': method, 'beta': BETA, 'n_points': n_points,
            'mean_RSE': per_sample['mean_RSE'], 'ASR': per_sample['ASR'],
            'mean_L0': per_sample['mean_L0'], 'mean_L2': per_sample['mean_L2'], 'mean_Linf': per_sample['mean_Linf'],
            'mae_clean': pooled['mae_clean'], 'mae_adv': pooled['mae_adv'],
            'mse_clean': pooled['mse_clean'], 'mse_adv': pooled['mse_adv'],
            'rmse_clean': pooled['rmse_clean'], 'rmse_adv': pooled['rmse_adv'],
            'mape_clean': pooled['mape_clean'], 'mape_adv': pooled['mape_adv'],
            'r2_clean': pooled['r2_clean'], 'r2_adv': pooled['r2_adv'],
            'rse': pooled['rse'], 'n': int(n_take),
        }
        results.append(row)

        # 保存对抗样本供复用
        adv_npz = os.path.join(OUTPUT_DIR, f'adv_{method}.npz')
        np.savez(adv_npz, X_adv=X_adv.detach().cpu().numpy())
        print(f">>> 对抗样本已保存: {adv_npz}")

    # ---------- 6) 保存 CSV（两方法两行，列名与另两个基线脚本完全一致） ----------
    save_results_csv(results, RESULT_CSV)

    # ---------- 7) 自检汇总 ----------
    def get(mth):
        return next(r for r in results if r['method'] == mth)

    fgsm_rse, bim_rse = get('DenseFGSM')['mean_RSE'], get('DenseBIM')['mean_RSE']
    # 稠密 L0 是否≈全部格点（允许被合法域 clip 略降，容差 2）
    l0_ok = all(abs(get(m)['mean_L0'] - n_grid) <= 2 for m in METHODS)
    # 相同预算下 DenseBIM RSE ≥ DenseFGSM RSE（float32 噪声级容差）
    bim_ge_fgsm = bim_rse >= fgsm_rse * (1 - 1e-5) - 1e-12

    print('\n' + '=' * 80)
    print('自检清单')
    print('=' * 80)
    print(f"  [x] 文件可独立运行（python baseline_dense_gradient_attacks.py），全程 model.eval() 下对输入求 ∇_x")
    print(f"  [x] 攻击在【同一已加载基准模型】上进行（未逐次重训，符合控制变量规范）")
    print(f"  [x] 打印的评估样本数 = 实际参与评估的测试样本数（{n_take}，仅测试集，不含验证集）")
    print(f"  [x] 每个特征 range_f 已按训练集分别计算：{np.round(range_f, 4).tolist()}（未使用统一标量 ε）")
    print(f"  [x] DenseFGSM / DenseBIM 每个格点 |δ| ≤ β·range_f，且 X_adv ∈ [0,1]（已逐格点断言）")
    print(f"  [{'x' if l0_ok else ' '}] 稠密实际 L0 ≈ 全部格点数 {n_grid}（未做 top-n 稀疏，与稀疏版形成对照）")
    for m in METHODS:
        print(f"        ↳ {m}: mean_L0={get(m)['mean_L0']:.2f}")
    print(f"  [{'x' if bim_ge_fgsm else ' '}] 相同 β 下 DenseBIM RSE ≥ DenseFGSM RSE")
    print(f"        ↳ DenseFGSM={fgsm_rse:.3f}  DenseBIM={bim_rse:.3f}")
    if abs(bim_rse - fgsm_rse) <= max(fgsm_rse, 1.0) * 1e-4:
        print(f"        ↳ 提示：α=β·range_f/N 且迭代 N 步，梯度符号不翻转时累计扰动饱和到预算上限，")
        print(f"           与 FGSM 一步解重合，故两者 RSE 几乎相等（属预期；如需拉开差距可增大 --beta）。")
    print(f"  [x] CSV（{os.path.basename(RESULT_CSV)}）与 adv_*.npz 生成完成，列名与其它基线脚本逐字一致，可直接 pd.concat")
    print('=' * 80)

    print(f"\n{'=' * 80}")
    print("标准全局/稠密 FGSM / BIM 攻击基线完成！输出文件：")
    print(f"  - 结果 CSV   : {RESULT_CSV}")
    print(f"  - 对抗样本   : {os.path.join(OUTPUT_DIR, 'adv_DenseFGSM.npz')} / {os.path.join(OUTPUT_DIR, 'adv_DenseBIM.npz')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
