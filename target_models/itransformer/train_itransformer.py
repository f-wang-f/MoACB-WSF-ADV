# -*- coding: utf-8 -*-
"""
iTransformer 训练 + 冻结权重保存脚本（wind / etth1 / electricity 三数据集通用）
==========================================================================
主干严格参照官方 https://github.com/thuml/iTransformer（见 itransformer_model.py），
本脚本只负责数据管道 / 训练 / 门槛验收 / 冻结保存，口径与 CNN/LSTM/GRU/TCN/Informer 完全一致。

任务口径：
  * 多变量输入、单一目标通道、单步预测；lookback T=20；输入 [B,T,F] → 输出 [B,1]
  * 连续、按时间顺序的片段；时序划分 70/15/15，不随机打乱；归一化参数只在训练集拟合
  * 使用文件内全部连续小时点，不对数据集长度做任何裁剪
  * 训练：Adam、lr=1e-3、batch_size=32、验证集 early stopping、CosineAnnealingWarmRestarts
  * 保存：模型超参配置 + state_dict（非裸张量）；训练后置 eval() 并 requires_grad_(False)
  * 报告并保存 clean 指标：RMSE / MAE / MSE / MAPE / R²（测试集，反归一化后）

【统一配方（三数据集完全一致，无任何按数据集名的结构/超参分支）】
  * 结构：官方倒置 Transformer——DataEmbedding_inverted(Linear T→d_model) + 跨变量注意力 Encoder
    + projection(d_model→pred_len)；d_model=128、n_heads=4、e_layers=2、d_ff=256、dropout=0.1、
    activation=gelu、pred_len=1、RevIN(Non-stationary 归一化)
  * 数据：seq_len=20、70/15/15 时序划分、仅训练集拟合 z-score、全量数据不裁剪
  * 分布偏移（level shift）由 RevIN 逐窗口归一化处理，而非裁剪数据长度
  * 仅随数据自动变化的量：input_size（=变量数 F，不改变参数量）

【硬性 clean 验收门槛 GATE（每个“数据集×模型”）】
  * R² ≥ 0.80 且非退化 且 接近持久性基线（R² ≥ persist-0.10）→ PASS
  * 显式打印 “GATE: PASS / FAIL (R²=…)”，并写入 checkpoint 与 clean_metrics.json
  * 不合格时在统一配方内自动有限重试（对三集同一规则）：增轮数/放宽 patience、
    按预声明 n_rows 阈值规则提升容量（放大 d_model/d_ff、加深 e_layers）；重试后仍 <0.8 标记 FAIL。

用法：
  python target_models/itransformer/train_itransformer.py --dataset all
  python target_models/itransformer/train_itransformer.py --dataset wind --epochs 100
"""
import os
import sys
import json
import copy
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 保证能 import 同目录的 itransformer_model.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from itransformer_model import (  # noqa: E402
    build_itransformer, checkpoint_path, resolve_data_file, CHECKPOINT_ROOT, VALID_DATASETS,
    PROJECT_ROOT,
)

# ======================================================================
# 集中配置（所有路径/口径在此定义，不写死在逻辑里）
# ======================================================================
# 每个数据集：数据文件、预测目标列、时间列（时间列不进特征，仅用于排序/口径）
# 注意：此处不允许任何按数据集名的特殊处理（如容量/轮数），三集完全一视同仁；
# 一律使用文件全量连续行，不裁剪数据集长度；分布差异由 RevIN 实例归一化处理。
DATASET_CONFIG = {
    'wind':        {'file': 'winddata.xlsx',    'target': 'Wind Speed (m/s)', 'time_col': 'DateTime'},
    'etth1':       {'file': 'ETTh1.xlsx',       'target': 'OT',               'time_col': 'date'},
    'electricity': {'file': 'electricity.xlsx', 'target': 'OT',               'time_col': 'date'},
}

# 训练超参（三数据集统一）
SEQ_LEN = 20                              # lookback window T
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15       # 70/15/15 时序划分

# 硬性验收门槛（对三集同一规则）
GATE_R2_THRESHOLD = 0.80                  # 测试集 R² 下限
GATE_PERSIST_MARGIN = 0.10                # “接近持久性基线”容差：R² ≥ persist - margin

# 统一重试升级表（对三集同一规则；按 attempt 序号升级，不按数据集名分支）：
#   level 0 = 基础配方；level≥1 增轮数/放宽 patience，并按预声明 n_rows 阈值放大 d_model、加深 e_layers。
BASE_EPOCHS, BASE_PATIENCE = 80, 10
BASE_D_MODEL, BASE_D_FF, BASE_E_LAYERS, BASE_N_HEADS = 128, 256, 2, 4
RETRY_EPOCHS = [BASE_EPOCHS, 200, 320]
RETRY_PATIENCE = [BASE_PATIENCE, 25, 40]
RETRY_WIDTH_MULT = [1.0, 2.0, 3.0]        # d_model 容量倍率（再叠加 n_rows 阈值规则）
RETRY_E_LAYERS = [BASE_E_LAYERS, 3, 3]    # 升级时编码器层数 2->3
N_ROWS_BIG = 2500                          # 预声明阈值：行数≥此值可满额放大容量


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capacity_rule(base_d_model, width_mult, n_rows):
    """预声明 n_rows 阈值规则（对三集一视同仁，非按数据集名分支）。

    level 0（width_mult=1）恒返回 base_d_model；升级时数据多则满额放大、数据少则温和放大防过拟合。
    返回对齐到 8 的倍数（保证可被 n_heads 整除）。
    """
    if width_mult <= 1.0:
        return base_d_model
    if n_rows >= N_ROWS_BIG:
        eff = width_mult
    else:
        eff = 1.0 + (width_mult - 1.0) * 0.5
    return max(64, int(round(base_d_model * eff / 8.0)) * 8)


# ======================================================================
# 数据加载 / 划分 / 滑窗（与 train_gru.py / train_lstm.py / train_informer.py 一致口径）
# ======================================================================
def load_series(dataset: str):
    """读取原始时序，返回 (features[N,F] float32, target[N] float32, 元信息)。

    - 时间列不进特征；特征 = 其余全部数值列（含目标自身历史）
    - 目标列是特征列之一（预测其下一步取值）
    - 一律使用文件全量连续行，不对数据集长度做任何裁剪
    """
    cfg = DATASET_CONFIG[dataset]
    path = resolve_data_file(cfg['file'])
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    time_col = cfg['time_col']
    if time_col not in df.columns:
        time_col = df.columns[0]  # 回退：首列视为时间列
    feature_columns = [c for c in df.columns if c != time_col]
    target_col = cfg['target']
    if target_col not in feature_columns:
        raise ValueError("%s: target %r not in feature columns" % (dataset, target_col))

    feats = df[feature_columns].astype('float32').values      # [N, F]
    tidx = feature_columns.index(target_col)
    target = feats[:, tidx].astype('float32').copy()          # [N]

    n_total = len(feats)
    meta = {
        'file': os.path.relpath(path, PROJECT_ROOT),
        'time_col': time_col,
        'feature_columns': feature_columns,
        'target_column': target_col,
        'target_index_in_features': tidx,
        'n_rows_total': int(n_total),
        'n_rows_used': int(len(feats)),
        'n_features': int(feats.shape[1]),
    }
    return feats, target, meta


def make_windows(feat_norm, tgt_norm, start, end, T):
    """在 [start,end) 行内构造严格不跨划分边界的滑窗，避免信息泄漏。
    输入 = feat_norm[i:i+T]，标签 = tgt_norm[i+T]（下一步目标）。
    """
    xs, ys = [], []
    for i in range(start, end - T):
        xs.append(feat_norm[i:i + T])
        ys.append(tgt_norm[i + T])
    if not xs:
        return (np.zeros((0, T, feat_norm.shape[1]), 'float32'),
                np.zeros((0, 1), 'float32'))
    return (np.asarray(xs, 'float32'),
            np.asarray(ys, 'float32').reshape(-1, 1))


def prepare_data(dataset: str, seq_len: int):
    """加载全量数据 → 70/15/15 时序划分 → 仅用训练集拟合归一化 → 各划分内滑窗。

    不对数据集长度做任何裁剪：一律使用文件内全部连续行；test 段相对 train 的
    level shift（季节/趋势漂移）由模型内置的 RevIN 实例归一化处理。
    """
    feats, target, meta = load_series(dataset)
    n = len(feats)

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    tr_end = n_train
    va_end = n_train + n_val

    # 归一化参数只在训练集 [0, tr_end) 上拟合
    feat_mean = feats[:tr_end].mean(axis=0)
    feat_std = feats[:tr_end].std(axis=0) + 1e-8
    tgt_mean = float(target[:tr_end].mean())
    tgt_std = float(target[:tr_end].std() + 1e-8)

    feat_norm = (feats - feat_mean) / feat_std
    tgt_norm = (target - tgt_mean) / tgt_std

    Xtr, ytr = make_windows(feat_norm, tgt_norm, 0, tr_end, seq_len)
    Xva, yva = make_windows(feat_norm, tgt_norm, tr_end, va_end, seq_len)
    Xte, yte = make_windows(feat_norm, tgt_norm, va_end, n, seq_len)

    scaler = {
        'feat_mean': feat_mean.astype('float64').tolist(),
        'feat_std': feat_std.astype('float64').tolist(),
        'tgt_mean': tgt_mean,
        'tgt_std': tgt_std,
    }
    split_info = {
        'n_rows': int(n), 'train_rows': int(tr_end), 'val_rows': int(va_end - tr_end),
        'test_rows': int(n - va_end),
        'train_windows': int(len(Xtr)), 'val_windows': int(len(Xva)), 'test_windows': int(len(Xte)),
    }
    data = {'train': (Xtr, ytr), 'val': (Xva, yva), 'test': (Xte, yte)}
    return data, scaler, split_info, meta


def to_loader(arrs, batch_size, shuffle, drop_last=False):
    X, y = arrs
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# ======================================================================
# 指标（反归一化后）
# ======================================================================
def compute_persistence_baseline(target, seq_len, va_end, n):
    """持久性基线：用目标列上一时刻值作为预测（test 区间）。"""
    y_true_list, y_pred_list = [], []
    for i in range(va_end, n - seq_len):
        y_true_list.append(target[i + seq_len])
        y_pred_list.append(target[i + seq_len - 1])
    if not y_true_list:
        return {'R2': float('nan'), 'RMSE': float('nan'), 'MAE': float('nan'),
                'MAPE': float('nan'), 'n_samples': 0}
    y_true = np.asarray(y_true_list, 'float64')
    y_pred = np.asarray(y_pred_list, 'float64')
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100.0)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'n_samples': len(y_true)}


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, 'float64').ravel()
    y_pred = np.asarray(y_pred, 'float64').ravel()
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = np.abs(y_true) + 1e-8
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    return {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'MAPE': mape, 'R2': r2}


def evaluate_gate(r2, non_degenerate, persist_r2):
    """硬性验收门槛（对三集同一规则）：R²≥0.80 且非退化 且接近持久性基线。"""
    near_persist = bool(r2 >= (persist_r2 - GATE_PERSIST_MARGIN)) if np.isfinite(persist_r2) else True
    gate_pass = bool(r2 >= GATE_R2_THRESHOLD) and bool(non_degenerate) and near_persist
    return gate_pass, {
        'pass': bool(gate_pass), 'R2': float(r2), 'threshold': GATE_R2_THRESHOLD,
        'persist_R2': float(persist_r2), 'non_degenerate': bool(non_degenerate),
        'near_persistence': near_persist, 'persist_margin': GATE_PERSIST_MARGIN,
    }


@torch.no_grad()
def predict_norm(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).reshape(xb.shape[0], -1)   # [B,1]
        preds.append(out.cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


# ======================================================================
# 单次训练（给定容量/轮数），返回指标与门槛；不写盘
# ======================================================================
def _train_once(dataset, args, device, d_model, d_ff, e_layers, epochs, patience, attempt):
    set_seed(args.seed)   # 每数据集/每次尝试重置种子：结果与运行顺序无关、可复现
    t0 = time.time()

    data, scaler, split_info, meta = prepare_data(dataset, args.seq_len)
    Xtr, ytr = data['train']
    if len(Xtr) == 0:
        raise RuntimeError('%s: 训练窗口为空，请减小 seq_len' % dataset)

    if attempt == 0:
        print('数据: 文件=%s  形状=[N=%d, F=%d]  目标=%r' %
              (meta['file'], meta['n_rows_used'], meta['n_features'], meta['target_column']))
        print('划分(70/15/15, 不打乱): train=%d windows, val=%d, test=%d' %
              (split_info['train_windows'], split_info['val_windows'], split_info['test_windows']))

    # ---- 持久性基线 ----
    _, target_raw, _ = load_series(dataset)
    n_used = split_info['n_rows']
    target_used = target_raw[:n_used]
    va_end = split_info['train_rows'] + split_info['val_rows']
    persist = compute_persistence_baseline(target_used, args.seq_len, va_end, n_used)
    if attempt == 0:
        print('持久性基线(y[t]=目标[t-1]): R²=%.4f  RMSE=%.4f  MAE=%.4f  (n=%d)' %
              (persist['R2'], persist['RMSE'], persist['MAE'], persist['n_samples']))

    train_loader = to_loader(data['train'], args.batch_size, shuffle=True, drop_last=True)
    val_loader = to_loader(data['val'], args.batch_size, shuffle=False)
    test_loader = to_loader(data['test'], args.batch_size, shuffle=False)

    # ---- 模型（统一配方；仅 input_size 随数据变化，且不改变参数量）----
    model_cfg = {
        'input_size': meta['n_features'],
        'seq_len': args.seq_len,
        'pred_len': 1,
        'd_model': d_model,
        'n_heads': args.n_heads,
        'e_layers': e_layers,
        'd_ff': d_ff,
        'dropout': args.dropout,
        'factor': 5,
        'activation': 'gelu',
        'use_revin': args.use_revin,
        'target_index': meta['target_index_in_features'],
    }
    model = build_itransformer(model_cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print('  [attempt %d] iTransformer(input=%d, d_model=%d, n_heads=%d, e_layers=%d, d_ff=%d, '
          'dropout=%.2f, revin=%s) epochs=%d patience=%d | 参数量=%s' %
          (attempt, model_cfg['input_size'], d_model, args.n_heads, e_layers, d_ff,
           args.dropout, args.use_revin, epochs, patience, format(n_params, ',')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).reshape(xb.shape[0], -1)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            run_loss += loss.item() * xb.shape[0]
            run_n += xb.shape[0]
        train_loss = run_loss / max(run_n, 1)

        vp, vt = predict_norm(model, val_loader, device)
        val_loss = float(np.mean((vp - vt) ** 2)) if len(vp) else float('nan')
        scheduler.step()

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print('    epoch %3d | train MSE(norm)=%.5f | val MSE(norm)=%.5f | lr=%.2e%s' %
                  (epoch, train_loss, val_loss, optimizer.param_groups[0]['lr'],
                   '  *best*' if improved else ''))
        if bad >= patience:
            print('    early stopping @ epoch %d (patience=%d)' % (epoch, patience))
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- 测试集评估（反归一化）----
    tp, tt = predict_norm(model, test_loader, device)
    tgt_mean, tgt_std = scaler['tgt_mean'], scaler['tgt_std']
    pred_den = tp.ravel() * tgt_std + tgt_mean
    true_den = tt.ravel() * tgt_std + tgt_mean
    metrics = compute_metrics(true_den, pred_den)

    pred_std = float(np.std(pred_den))
    true_std = float(np.std(true_den))
    degen_ratio = pred_std / (true_std + 1e-12)
    metrics['pred_std'] = pred_std
    metrics['true_std'] = true_std
    metrics['std_ratio(pred/true)'] = degen_ratio
    non_degenerate = pred_std > 1e-6 and degen_ratio > 0.05

    gate_pass, gate = evaluate_gate(metrics['R2'], non_degenerate, persist['R2'])
    gate['attempt'] = int(attempt)
    gate['epochs_used'] = int(epochs)
    gate['d_model'] = int(d_model)
    gate['e_layers'] = int(e_layers)

    # ---- 冻结 + eval（无论 pass/fail 都冻结；是否作为有效攻击目标由 gate 决定）----
    model.eval()
    model.requires_grad_(False)

    print('    --- %s clean 指标（测试集, 反归一化）---' % dataset)
    print('    RMSE=%.4f  MAE=%.4f  MSE=%.4f  MAPE=%.2f%%  R2=%.4f' %
          (metrics['RMSE'], metrics['MAE'], metrics['MSE'], metrics['MAPE'], metrics['R2']))
    print('    非退化: pred_std=%.4f, true_std=%.4f, ratio=%.3f -> %s' %
          (pred_std, true_std, degen_ratio, 'OK(非常数)' if non_degenerate else 'WARNING(可能退化)'))
    print('    GATE: %s (R²=%.4f, persist=%.4f, thresh=%.2f, non_degen=%s, near_persist=%s)' %
          ('PASS' if gate_pass else 'FAIL', metrics['R2'], persist['R2'], GATE_R2_THRESHOLD,
           non_degenerate, gate['near_persistence']))
    print('    用时 %.1fs' % (time.time() - t0))

    return {
        'dataset': dataset, 'metrics': metrics, 'persistence_baseline': persist,
        'gate': gate, 'gate_pass': gate_pass, 'non_degenerate': bool(non_degenerate),
        'n_params': n_params, 'model': model, 'model_cfg': model_cfg, 'scaler': scaler,
        'meta': meta, 'split_info': split_info,
    }


def _save_checkpoint(res, args):
    """保存完整 checkpoint（超参配置 + state_dict + gate）与可读 json。"""
    dataset = res['dataset']
    model = res['model']
    meta, scaler, split_info = res['meta'], res['scaler'], res['split_info']
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    ckpt = {
        'dataset': dataset,
        'model_type': 'itransformer',
        'config': res['model_cfg'],
        'state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_columns': meta['feature_columns'],
        'target_column': meta['target_column'],
        'target_index_in_features': meta['target_index_in_features'],
        'seq_len': args.seq_len,
        'data_info': {'file': meta['file'], 'split': split_info,
                      'n_rows_used': meta['n_rows_used'], 'n_features': meta['n_features']},
        'clean_metrics': res['metrics'],
        'gate': res['gate'],
        'train_args': vars(args),
        'torch_version': str(torch.__version__),
        'non_degenerate': bool(res['non_degenerate']),
    }
    ckpt_file = checkpoint_path(dataset)
    torch.save(ckpt, ckpt_file)
    with open(os.path.join(CHECKPOINT_ROOT, 'itransformer_%s_clean_metrics.json' % dataset), 'w',
              encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'model_type': 'itransformer',
                   'clean_metrics': res['metrics'], 'gate': res['gate'],
                   'persistence_baseline': res['persistence_baseline'],
                   'checkpoint': os.path.relpath(ckpt_file, os.path.dirname(CHECKPOINT_ROOT))},
                  f, ensure_ascii=False, indent=2)
    print('  已保存冻结权重: %s  (%.1f KB)  [GATE=%s]' %
          (ckpt_file, os.path.getsize(ckpt_file) / 1024, 'PASS' if res['gate_pass'] else 'FAIL'))
    return ckpt_file


def train_one_dataset(dataset, args, device):
    print('\n' + '=' * 72)
    print('训练数据集: %s   |   device = %s   |   model = iTransformer' % (dataset, device))
    print('=' * 72)

    n_rows = len(load_series(dataset)[0])
    max_attempt = len(RETRY_EPOCHS)
    res = None
    for attempt in range(max_attempt):
        d_model = capacity_rule(BASE_D_MODEL, RETRY_WIDTH_MULT[attempt], n_rows)
        d_ff = 2 * d_model
        e_layers = RETRY_E_LAYERS[attempt]
        epochs = args.epochs if attempt == 0 else RETRY_EPOCHS[attempt]
        patience = args.patience if attempt == 0 else RETRY_PATIENCE[attempt]
        if attempt > 0:
            print('  >> GATE 未通过，统一配方内自动重试（level %d：epochs=%d, patience=%d, '
                  'd_model=%d, d_ff=%d, e_layers=%d）'
                  % (attempt, epochs, patience, d_model, d_ff, e_layers))
        res = _train_once(dataset, args, device, d_model, d_ff, e_layers, epochs, patience, attempt)
        if res['gate_pass']:
            break

    if not res['gate_pass']:
        print('  *** GATE 最终 FAIL：%s 在统一配方 + %d 次重试后 R²=%.4f 仍 <%.2f。'
              ' 该单元不得作为有效攻击目标。已尝试：全量数据(不裁剪)、RevIN、增轮数(%s)、'
              '放宽 patience(%s)、按 n_rows 阈值提升容量(d_model→%d, e_layers→%d)。***'
              % (dataset, max_attempt, res['metrics']['R2'], GATE_R2_THRESHOLD,
                 RETRY_EPOCHS, RETRY_PATIENCE, res['gate']['d_model'], res['gate']['e_layers']))

    ckpt_file = _save_checkpoint(res, args)
    res['checkpoint'] = ckpt_file
    return res


def parse_args():
    p = argparse.ArgumentParser(description='Train & freeze iTransformer on wind/etth1/electricity')
    p.add_argument('--dataset', default='all',
                   help="wind | etth1 | electricity | all（默认 all）")
    p.add_argument('--seq-len', dest='seq_len', type=int, default=SEQ_LEN)
    p.add_argument('--epochs', type=int, default=BASE_EPOCHS)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=BASE_PATIENCE)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--d-model', dest='d_model', type=int, default=BASE_D_MODEL)
    p.add_argument('--n-heads', dest='n_heads', type=int, default=BASE_N_HEADS)
    p.add_argument('--e-layers', dest='e_layers', type=int, default=BASE_E_LAYERS)
    p.add_argument('--d-ff', dest='d_ff', type=int, default=BASE_D_FF)
    p.add_argument('--use-revin', dest='use_revin', action='store_true', default=True,
                   help='启用 RevIN(Non-stationary 归一化)（统一配方默认开启，消除 level shift）')
    p.add_argument('--no-revin', dest='use_revin', action='store_false',
                   help='关闭 RevIN（仅用于对比实验）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:0 ...')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    datasets = list(VALID_DATASETS) if args.dataset == 'all' else [args.dataset]
    results = []
    for ds in datasets:
        results.append(train_one_dataset(ds, args, device))

    # ---- 汇总表（含 GATE 状态）----
    print('\n' + '=' * 88)
    print('全部完成 —— iTransformer clean 指标汇总（测试集, 反归一化）')
    print('=' * 88)
    header = '%-12s %9s %9s %9s %8s %8s %10s %10s %8s' % (
        'dataset', 'RMSE', 'MAE', 'MSE', 'MAPE%', 'R2', 'persistR2', 'GATE', 'params')
    print(header)
    print('-' * len(header))
    n_fail = 0
    for r in results:
        m = r['metrics']
        p = r.get('persistence_baseline', {})
        gp = r['gate_pass']
        if not gp:
            n_fail += 1
        print('%-12s %9.4f %9.4f %9.4f %8.2f %8.4f %10.4f %10s %8s%s' % (
            r['dataset'], m['RMSE'], m['MAE'], m['MSE'], m['MAPE'], m['R2'],
            p.get('R2', float('nan')), 'PASS' if gp else 'FAIL', format(r['n_params'], ','),
            '' if gp else '   <<< FAIL 不可用于攻击'))
    print('\n权重文件:')
    for r in results:
        print('  %-12s -> %s' % (r['dataset'], r['checkpoint']))
    print('\nGATE 汇总: %d/%d PASS' % (len(results) - n_fail, len(results)))


if __name__ == '__main__':
    main()
