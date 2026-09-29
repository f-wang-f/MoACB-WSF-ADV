# -*- coding: utf-8 -*-
"""
LSTM 目标模型（wind / etth1 / electricity 三数据集通用）
==========================================================
任务口径：
  * 多变量输入、单一目标通道、单步预测；lookback T=20；输入 [B,T,F] → 输出 [B,1]
  * 堆叠 3 层 nn.LSTM，hidden_size=256，dropout=0.2
  * 取最后时刻输出经 2 层全连接输出 1
  * RevIN 可逆实例归一化（use_revin，默认 True）：逐窗口逐特征标准化，末端头输出后
    仅对目标通道用其窗口统计量反归一化，消除 test 段相对 train 的 level shift；
    内置于 forward，对外接口仍为 [B,T,F]->[B,1]。

统一加载接口见文件末尾 load_frozen_lstm()。
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# 集中路径配置（不写死；含中文的路径统一按 UTF-8 处理）
# ======================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../target_models/lstm
_MODEL_DIR = _THIS_DIR                                             # lstm 模型目录
CHECKPOINT_ROOT = os.path.join(_THIS_DIR, 'checkpoints')

# 数据根目录候选：任务约定 data/raw/，实际文件当前在 data/，两者都查
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'data', 'raw'),
    os.path.join(PROJECT_ROOT, 'data'),
]

VALID_DATASETS = ('wind', 'etth1', 'electricity')


def checkpoint_path(dataset: str) -> str:
    """返回某数据集冻结权重的绝对路径：checkpoints/lstm_<dataset>.pt"""
    if dataset not in VALID_DATASETS:
        raise ValueError("dataset must be one of %s, got %r" % (VALID_DATASETS, dataset))
    return os.path.join(CHECKPOINT_ROOT, 'lstm_%s.pt' % dataset)


def resolve_data_file(filename: str) -> str:
    """在 DATA_DIR_CANDIDATES 中按序查找数据文件（支持 .xlsx/.csv 互为回退）。"""
    stem, ext = os.path.splitext(filename)
    tried = []
    for d in DATA_DIR_CANDIDATES:
        for cand in (filename, stem + ('.csv' if ext == '.xlsx' else '.xlsx')):
            p = os.path.join(d, cand)
            tried.append(p)
            if os.path.exists(p):
                return p
    raise FileNotFoundError("data file not found for %r; tried:\n  %s"
                            % (filename, "\n  ".join(tried)))


# ======================================================================
# LSTM 模型定义（参数量做大）
# ======================================================================
class LSTMModel(nn.Module):
    """
    Multi-layer LSTM for time series regression.
    Input: [B, T, F] -> LSTM layers -> last timestep -> FC -> [B, 1]
    
    Architecture:
      - 3 LSTM layers: hidden_size=256, dropout=0.2
      - Take last timestep output
      - 2 FC layers with dropout -> output 1
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, 
                 dropout: float = 0.2, seq_len: int = 20,
                 use_revin: bool = True, target_index: int = 0, revin_eps: float = 1e-5):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.use_revin = use_revin
        self.target_index = int(target_index)
        self.revin_eps = revin_eps
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # FC layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        x: [B, T, F] input sequence
        return: [B, 1] prediction
        """
        # ---- RevIN 正向：逐窗口逐特征实例归一化（消除 level shift）----
        if self.use_revin:
            revin_mean = x.mean(dim=1, keepdim=True)                     # [B,1,F]
            revin_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps)
            x_in = (x - revin_mean) / revin_std
        else:
            x_in = x

        # LSTM forward
        # lstm_out: [B, T, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x_in)
        
        # Take last timestep output
        last_out = lstm_out[:, -1, :]  # [B, hidden_size]
        
        # FC layers
        out = self.relu(self.fc1(last_out))
        out = self.fc_dropout1(out)
        out = self.fc2(out)  # [B, 1]

        # ---- RevIN 反向：仅对目标通道用其窗口统计量反归一化 ----
        if self.use_revin:
            tm = revin_mean[:, :, self.target_index]   # [B, 1]
            ts = revin_std[:, :, self.target_index]    # [B, 1]
            out = out * ts + tm
        
        return out


def build_lstm(cfg: dict, device=None) -> LSTMModel:
    """按超参配置字典构建 LSTM（train 与 load 共用，保证结构一致）。"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(
        input_size=cfg['input_size'],
        hidden_size=cfg.get('hidden_size', 256),
        num_layers=cfg.get('num_layers', 3),
        dropout=cfg.get('dropout', 0.2),
        seq_len=cfg.get('seq_len', 20),
        use_revin=cfg.get('use_revin', False),
        target_index=cfg.get('target_index', 0),
    )
    return model.to(device)


class FrozenLSTM(nn.Module):
    """封装 LSTM 成攻击直接可用的形态：forward(x[B,T,F]) -> [B,1]。"""
    
    def __init__(self, lstm: LSTMModel):
        super(FrozenLSTM, self).__init__()
        self.lstm = lstm
    
    def forward(self, x):
        out = self.lstm(x)  # [B, 1]
        return out.reshape(out.shape[0], -1)  # [B, 1]


def load_frozen_lstm(dataset: str, device='cpu'):
    """统一加载接口，供后续攻击直接调用。
    
    dataset ∈ {'wind','etth1','electricity'}；从 checkpoints 重建模型，
    返回 eval()、参数已冻结的 LSTM 封装，输入 [B,T,F] → 输出 [B,1]。
    """
    ckpt_file = checkpoint_path(dataset)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            "未找到 %s 的冻结权重：%s\n请先运行 train_lstm.py 训练该数据集。"
            % (dataset, ckpt_file))
    device = torch.device(device)
    # weights_only=True：本 checkpoint 仅含张量与基础类型(dict/list/str/int/float/bool/None)，
    #   安全加载并消除 PyTorch 2.x 的 FutureWarning
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
    model = build_lstm(ckpt['config'], device=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.requires_grad_(False)
    frozen = FrozenLSTM(model).to(device)
    frozen.eval()
    frozen.requires_grad_(False)
    # 附带归一化/数据元信息，便于攻击侧对齐输入口径
    frozen.meta = {k: ckpt.get(k) for k in
                   ('dataset', 'feature_columns', 'target_column', 'scaler', 'data_info', 'clean_metrics')}
    return frozen


# ======================================================================
# 快速验证（直接运行本文件时测试前向传播）
# ======================================================================
if __name__ == '__main__':
    print("Testing LSTMModel...")
    # Test with wind dataset config (4 features)
    model = LSTMModel(input_size=4, hidden_size=256, num_layers=3, dropout=0.2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Dummy input [B=2, T=20, F=4]
    x = torch.randn(2, 20, 4)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    
    # Test with etth1 dataset config (7 features)
    model2 = LSTMModel(input_size=7, hidden_size=256, num_layers=3, dropout=0.2)
    n_params2 = sum(p.numel() for p in model2.parameters())
    print(f"\nModel parameters (etth1): {n_params2:,}")
    
    x2 = torch.randn(2, 20, 7)
    out2 = model2(x2)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {out2.shape}")
    assert out2.shape == (2, 1), f"Expected (2, 1), got {out2.shape}"
    
    print("\nAll tests passed!")
