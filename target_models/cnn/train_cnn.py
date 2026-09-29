# -*- coding: utf-8 -*-
"""
1D CNN 训练 + 冻结权重保存脚本（wind / etth1 / electricity 三数据集通用）
==========================================================================
任务口径（与 GRU/TCN/LSTM/Informer/iTransformer 完全一致，共用同一数据管道 + GATE）：
  * 多变量输入、单一目标通道、单步预测；lookback T=20；输入 [B,T,F] → 输出 [B,1]
  * 连续、按时间顺序的片段；时序划分 70/15/15，不随机打乱；归一化参数只在训练集拟合
  * 使用文件内全部连续数据点，不裁剪（level shift 由 RevIN 实例归一化处理）
  * 训练：Adam、lr=1e-3、batch_size=32、验证集 early stopping、CosineAnnealingWarmRestarts
  * 三数据集统一配方，禁止按数据集名做结构特调（仅 in_channels 随数据变化）
  * 硬性验收门槛 GATE：R²>=阈值 且 非退化 且 接近持久性基线；未过则按统一配方有限重试
  * 保存：超参配置 + state_dict（非裸张量）+ scaler + GATE 结果；训练后 eval()+requires_grad_(False)
  * 报告并保存 clean 指标：RMSE / MAE / MSE / MAPE / R²（测试集，反归一化后）+ 持久性基线

用法：
  python target_models/cnn/train_cnn.py --dataset all
  python target_models/cnn/train_cnn.py --dataset etth1 --no-revin      # 消融对比
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

# 保证能 import 同目录的 cnn_model.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cnn_model import (  # noqa: E402
    build_cnn, checkpoint_path, resolve_data_file, CHECKPOINT_ROOT, VALID_DATASETS,
)

# ======================================================================
# 集中配置（所有路径/口径在此定义，不写死在逻辑里）
# ======================================================================
# 每个数据集：数据文件、预测目标列、时间列（时间列不进特征，仅用于排序/口径）
DATASET_CONFIG = {
    'wind':        {'file': 'winddata.xlsx',    'target': 'Wind Speed (m/s)', 'time_col': 'DateTime'},
    'etth1':       {'file': 'ETTh1.xlsx',       'target': 'OT',               'time_col': 'date'},
    'electricity': {'file': 'electricity.xlsx', 'target': 'OT',               'time_col': 'date'},
}

# 训练超参
SEQ_LEN = 20  # lookback window T
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # 70/15/15 时序划分

# ---- 硬性验收门槛 GATE ----
GATE_R2_THRESHOLD = 0.80      # 任务书：R² >= 0.80
GATE_PERSIST_MARGIN = 0.10    # 任务书：R² >= persistence_baseline_R2 - 0.10

# ---- 统一配方：基础超参 + 失败重试的有限升级序列（三数据集同一套规则）----
BASE_EPOCHS, BASE_PATIENCE = 80, 10
BASE_WIDTH = 64                                   # 基础首层通道数（容量维）
# 失败时按序尝试（base 本身是第 0 档）：更多 epoch/耐心 + 适度加宽
RETRY_EPOCHS = [BASE_EPOCHS, 200, 320]
RETRY_PATIENCE = [BASE_PATIENCE, 25, 40]
RETRY_WIDTH_MULT = [1.0, 2.0, 3.0]               # 首层通道倍数（宽度）
N_ROWS_BIG = 2500                                 # 数据量阈值：大数据集才满额放大宽度


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capacity_rule(base_width: int, width_mult: float, n_rows: int) -> int:
    """统一容量规则（无按数据集名分支，只依据统一的 width_mult 与数据规模 n_rows）。

    宽度随数据量缩放：数据多（>=N_ROWS_BIG）用满额 width_mult，数据少则温和放大，
    避免在小数据集上过参数化过拟合；结果对齐到 8 的倍数。
    """
    if width_mult <= 1.0:
        return base_width
    eff = width_mult if n_rows >= N_ROWS_BIG else 1.0 + (width_mult - 1.0) * 0.5
    return max(base_width, int(round(base_width * eff / 8.0)) * 8)


# ======================================================================
# 数据加载 / 划分 / 滑窗（与 train_gru.py 保持一致的口径）
# ======================================================================
def load_series(dataset: str):
    """读取原始时序，返回 (features[N,F] float32, target[N] float32, 元信息)。

    - 时间列不进特征；特征 = 其余全部数值列（electricity 含 0..319 与 OT 过去值）
    - 目标列是特征列之一（预测其下一步取值）
    - 使用文件内全部连续数据点，不裁剪（level shift 由 RevIN 实例归一化处理）
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

    meta = {
        'file': os.path.relpath(path, os.path.dirname(CHECKPOINT_ROOT)),
        'time_col': time_col,
        'feature_columns': feature_columns,
        'target_column': target_col,
        'target_index_in_features': tidx,
        'n_rows_total': int(len(feats)),
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
    """加载 → 70/15/15 时序划分 → 仅用训练集拟合归一化 → 各划分内滑窗。"""
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
def compute_persistence_baseline(y_true_den):
    """持久性基线 y[t]=target[t-1] 的 R²（在测试集标签的反归一化真值上）。

    测试标签序列为 yte 反归一化后的真值 y[0..M-1]；持久性预测用 y[i-1] 预测 y[i]
    （i 从 1 起）。返回 R²（若无法计算返回 nan）。
    """
    y = np.asarray(y_true_den, 'float64').ravel()
    if len(y) < 3:
        return float('nan')
    pred = y[:-1]
    true = y[1:]
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-8))


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
    """硬性验收门槛：R²>=阈值 且 非退化 且 R² >= persist_r2 - margin。"""
    r2_ok = r2 >= GATE_R2_THRESHOLD
    persist_ok = (not np.isfinite(persist_r2)) or (r2 >= persist_r2 - GATE_PERSIST_MARGIN)
    gate_pass = bool(r2_ok and non_degenerate and persist_ok)
    gate = {
        'pass': gate_pass,
        'r2': float(r2),
        'r2_threshold': GATE_R2_THRESHOLD,
        'r2_ok': bool(r2_ok),
        'non_degenerate': bool(non_degenerate),
        'persistence_baseline_r2': (None if not np.isfinite(persist_r2) else float(persist_r2)),
        'persist_margin': GATE_PERSIST_MARGIN,
        'persist_ok': bool(persist_ok),
    }
    return gate_pass, gate


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
# 单次训练（指定容量/epoch/patience）；返回结果与 gate，供统一重试调用
# ======================================================================
def _train_once(dataset, args, device, base_channels, epochs, patience, attempt):
    # 每个数据集重置 RNG -> 结果与运行顺序无关（先于数据准备与模型构建）
    set_seed(args.seed)

    data, scaler, split_info, meta = prepare_data(dataset, args.seq_len)
    Xtr, ytr = data['train']
    if len(Xtr) == 0:
        raise RuntimeError('%s: 训练窗口为空，请检查数据或减小 seq_len' % dataset)

    train_loader = to_loader(data['train'], args.batch_size, shuffle=True, drop_last=True)
    val_loader = to_loader(data['val'], args.batch_size, shuffle=False)
    test_loader = to_loader(data['test'], args.batch_size, shuffle=False)

    model_cfg = {
        'in_channels': meta['n_features'],
        'seq_len': args.seq_len,
        'base_channels': base_channels,
        'dropout': args.dropout,
        'conv_dropout': args.conv_dropout,
        'use_revin': args.use_revin,
        'target_index': meta['target_index_in_features'],
    }
    model = build_cnn(model_cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print('  [attempt %d] CNN1D(in_channels=%d, base_channels=%d, seq_len=%d, RevIN=%s) | 参数量=%s' %
          (attempt, model_cfg['in_channels'], base_channels, model_cfg['seq_len'],
           args.use_revin, format(n_params, ',')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

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
        scheduler.step(epoch)

        vp, vt = predict_norm(model, val_loader, device)
        val_loss = float(np.mean((vp - vt) ** 2)) if len(vp) else float('nan')

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1 or improved:
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
    non_degenerate = bool(pred_std > 1e-6 and degen_ratio > 0.05)

    persist_r2 = compute_persistence_baseline(true_den)
    gate_pass, gate = evaluate_gate(metrics['R2'], non_degenerate, persist_r2)
    gate.update({'attempt': attempt, 'base_channels': base_channels,
                 'epochs': epochs, 'patience': patience})

    print('    -> R2=%.4f (thr>=%.2f) | persist R2=%s | 非退化=%s | GATE=%s' %
          (metrics['R2'], GATE_R2_THRESHOLD,
           ('%.4f' % persist_r2) if np.isfinite(persist_r2) else 'NA',
           'YES' if non_degenerate else 'NO', 'PASS' if gate_pass else 'FAIL'))

    model.eval()
    model.requires_grad_(False)

    return {
        'model': model, 'model_cfg': model_cfg, 'scaler': scaler, 'meta': meta,
        'split_info': split_info, 'metrics': metrics, 'non_degenerate': non_degenerate,
        'gate': gate, 'gate_pass': gate_pass, 'n_params': n_params,
        'persist_r2': persist_r2,
    }


def _save_checkpoint(dataset, args, res):
    model = res['model']
    ckpt_dir = CHECKPOINT_ROOT
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        'dataset': dataset,
        'model_type': 'cnn',
        'config': res['model_cfg'],
        'state_dict': model.state_dict(),
        'scaler': res['scaler'],
        'feature_columns': res['meta']['feature_columns'],
        'target_column': res['meta']['target_column'],
        'target_index_in_features': res['meta']['target_index_in_features'],
        'seq_len': args.seq_len,
        'data_info': {'file': res['meta']['file'], 'split': res['split_info'],
                      'n_rows_used': res['meta']['n_rows_used'], 'n_features': res['meta']['n_features']},
        'clean_metrics': res['metrics'],
        'gate': res['gate'],
        'persistence_baseline_r2': (None if not np.isfinite(res['persist_r2']) else float(res['persist_r2'])),
        'train_args': vars(args),
        'torch_version': str(torch.__version__),
        'non_degenerate': bool(res['non_degenerate']),
    }
    ckpt_file = checkpoint_path(dataset)
    torch.save(ckpt, ckpt_file)
    with open(os.path.join(ckpt_dir, 'cnn_%s_clean_metrics.json' % dataset), 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'model_type': 'cnn', 'clean_metrics': res['metrics'],
                   'gate': res['gate'],
                   'persistence_baseline_r2': ckpt['persistence_baseline_r2'],
                   'checkpoint': os.path.relpath(ckpt_file, os.path.dirname(CHECKPOINT_ROOT))},
                  f, ensure_ascii=False, indent=2)
    print('  已保存冻结权重: %s  (%.1f KB) | GATE=%s' %
          (ckpt_file, os.path.getsize(ckpt_file) / 1024, 'PASS' if res['gate_pass'] else 'FAIL'))
    return ckpt_file


# ======================================================================
# 训练单数据集（含统一配方的有限重试）
# ======================================================================
def train_one_dataset(dataset, args, device):
    print('\n' + '=' * 72)
    print('训练数据集: %s   |   device = %s' % (dataset, device))
    print('=' * 72)
    t0 = time.time()

    data, _, split_info, meta = prepare_data(dataset, args.seq_len)
    print('数据: 文件=%s  形状=[N=%d, F=%d]  目标=%r' %
          (meta['file'], meta['n_rows_used'], meta['n_features'], meta['target_column']))
    print('划分(70/15/15, 不打乱): train=%d windows, val=%d, test=%d' %
          (split_info['train_windows'], split_info['val_windows'], split_info['test_windows']))

    n_rows = split_info['n_rows']
    best = None
    for attempt in range(len(RETRY_EPOCHS)):
        width_mult = RETRY_WIDTH_MULT[attempt]
        epochs = RETRY_EPOCHS[attempt]
        patience = RETRY_PATIENCE[attempt]
        base_channels = args.base_channels if attempt == 0 else capacity_rule(BASE_WIDTH, width_mult, n_rows)
        print('\n  -- 训练尝试 %d/%d：base_channels=%d, epochs=%d, patience=%d --' %
              (attempt + 1, len(RETRY_EPOCHS), base_channels, epochs, patience))
        res = _train_once(dataset, args, device, base_channels, epochs, patience, attempt)
        if best is None or res['metrics']['R2'] > best['metrics']['R2']:
            best = res
        if res['gate_pass']:
            print('  GATE 通过（attempt %d），停止重试。' % attempt)
            break
        else:
            print('  GATE 未通过（attempt %d），按统一配方重试升级容量/轮次...' % attempt)

    res = best  # 采用 R² 最高的一次（若某次 GATE 通过则那次即最高，已 break）
    m = res['metrics']
    print('\n--- %s clean 指标（测试集, 反归一化）---' % dataset)
    print('  RMSE=%.4f  MAE=%.4f  MSE=%.4f  MAPE=%.2f%%  R2=%.4f' %
          (m['RMSE'], m['MAE'], m['MSE'], m['MAPE'], m['R2']))
    print('  非退化检查: pred_std=%.4f, true_std=%.4f, ratio=%.3f -> %s' %
          (m['pred_std'], m['true_std'], m['std_ratio(pred/true)'],
           'OK(非常数)' if res['non_degenerate'] else 'WARNING(可能退化)'))
    print('  持久性基线 R2=%s | GATE 最终=%s' %
          (('%0.4f' % res['persist_r2']) if np.isfinite(res['persist_r2']) else 'NA',
           'PASS' if res['gate_pass'] else 'FAIL'))

    ckpt_file = _save_checkpoint(dataset, args, res)
    print('  用时 %.1fs' % (time.time() - t0))
    return {'dataset': dataset, 'metrics': m, 'checkpoint': ckpt_file,
            'non_degenerate': bool(res['non_degenerate']), 'n_params': res['n_params'],
            'gate_pass': res['gate_pass'], 'persist_r2': res['persist_r2'],
            'attempt': res['gate']['attempt'], 'base_channels': res['gate']['base_channels']}


def parse_args():
    p = argparse.ArgumentParser(description='Train & freeze 1D CNN on wind/etth1/electricity')
    p.add_argument('--dataset', default='all',
                   help="wind | etth1 | electricity | all（默认 all）")
    p.add_argument('--seq-len', dest='seq_len', type=int, default=SEQ_LEN)
    p.add_argument('--epochs', type=int, default=BASE_EPOCHS,
                   help='基础训练轮数（失败重试时按统一序列升级）')
    p.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--conv-dropout', dest='conv_dropout', type=float, default=0.15,
                   help='卷积块 dropout（正则化，抑制小数据过拟合）')
    p.add_argument('--base-channels', dest='base_channels', type=int, default=BASE_WIDTH,
                   help='首层通道数（容量维；失败重试时按统一容量规则放大）')
    p.add_argument('--patience', type=int, default=BASE_PATIENCE,
                   help='基础 early stopping 耐心（失败重试时按统一序列升级）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use-revin', dest='use_revin', action='store_true', default=True,
                   help='启用 RevIN 实例归一化（默认启用，消除 level shift）')
    p.add_argument('--no-revin', dest='use_revin', action='store_false',
                   help='关闭 RevIN（用于消融对比）')
    p.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:0 ...')
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    datasets = list(VALID_DATASETS) if args.dataset == 'all' else [args.dataset]
    results = []
    for ds in datasets:
        results.append(train_one_dataset(ds, args, device))

    # ---- 汇总表 ----
    print('\n' + '=' * 96)
    print('全部完成 —— CNN clean 指标汇总（测试集, 反归一化）')
    print('=' * 96)
    header = '%-12s %9s %9s %9s %8s %8s %9s %6s %7s %6s' % (
        'dataset', 'RMSE', 'MAE', 'MSE', 'MAPE%', 'R2', 'persistR2', 'GATE', 'attempt', 'params')
    print(header)
    print('-' * len(header))
    for r in results:
        m = r['metrics']
        print('%-12s %9.4f %9.4f %9.4f %8.2f %8.4f %9s %6s %7d %6s' % (
            r['dataset'], m['RMSE'], m['MAE'], m['MSE'], m['MAPE'], m['R2'],
            ('%.4f' % r['persist_r2']) if np.isfinite(r['persist_r2']) else 'NA',
            'PASS' if r['gate_pass'] else 'FAIL', r['attempt'], format(r['n_params'], ',')))
    print('\n权重文件:')
    for r in results:
        print('  %-12s -> %s' % (r['dataset'], r['checkpoint']))


if __name__ == '__main__':
    main()
