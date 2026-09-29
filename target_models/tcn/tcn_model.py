# -*- coding: utf-8 -*-
"""
TCN 目标模型（wind / etth1 / electricity 三数据集通用）
==========================================================
严格参照官方公共仓库 https://github.com/locuslab/TCN
（Bai, Kolter & Koltun, An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling, 2018）的 TemporalBlock / TemporalConvNet：
  * dilated causal conv（膨胀因果卷积）+ weight normalization + ReLU + dropout + 残差连接
  * 因果性通过“左侧 padding=(k-1)*dilation + Chomp1d 裁掉右侧多余输出”实现，绝不引入未来信息
不自行臆造结构；仅在其外层包裹本任务所需的 RevIN 实例归一化与 [B,T,F]→[B,1] 适配头。

任务口径（与 CNN/LSTM/GRU/Informer 完全一致）：
  * 多变量输入、单一目标通道、单步预测；lookback T=20；输入 [B,T,F] → 输出 [B,1]
  * forward 内部将 [B,T,F] 转置为 [B,F,T] 做卷积，取最后时刻特征经全连接输出 1
  * RevIN：逐窗口逐特征标准化，末端头输出后仅对目标通道用其窗口统计量反归一化
  * 三数据集结构与训练超参完全相同，仅 input_size（=特征数）随数据变化

统一加载接口见文件末尾 load_frozen_tcn()。
"""
import os
import warnings

import torch
import torch.nn as nn

# 官方 TCN 使用 torch.nn.utils.weight_norm（PyTorch 2.x 已弃用但功能完好）。
# 为忠实官方实现仍用 weight_norm，仅静默其弃用 FutureWarning，避免污染训练/验证日志。
warnings.filterwarnings('ignore', message=r'.*weight_norm.*is deprecated.*')
from torch.nn.utils import weight_norm  # noqa: E402

# ======================================================================
# 集中路径配置（不写死；含中文的路径统一按 UTF-8 处理）
# ======================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../target_models/tcn
CHECKPOINT_ROOT = os.path.join(_THIS_DIR, 'checkpoints')

PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'data', 'raw'),
    os.path.join(PROJECT_ROOT, 'data'),
]

VALID_DATASETS = ('wind', 'etth1', 'electricity')


def checkpoint_path(dataset: str) -> str:
    """返回某数据集冻结权重的绝对路径：checkpoints/tcn_<dataset>.pt"""
    if dataset not in VALID_DATASETS:
        raise ValueError("dataset must be one of %s, got %r" % (VALID_DATASETS, dataset))
    return os.path.join(CHECKPOINT_ROOT, 'tcn_%s.pt' % dataset)


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
# 官方 TCN 组件（locuslab/TCN，逐行对齐，不臆造）
# ======================================================================
class Chomp1d(nn.Module):
    """裁掉因果卷积右侧多出的 padding，保证 t 时刻输出只依赖 ≤t 的输入。"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """官方残差块：两层 (weight_norm 膨胀因果卷积 → Chomp1d → ReLU → Dropout) + 残差。"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """官方 TCN 主干：堆叠若干 TemporalBlock，dilation=2^i，通道数由 num_channels 指定。"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ======================================================================
# TCN 回归封装（RevIN + [B,T,F]→[B,1] 适配；主干严格用官方 TemporalConvNet）
# ======================================================================
class TCNModel(nn.Module):
    """
    Input: [B, T, F] -> RevIN -> transpose [B, F, T] -> TemporalConvNet -> last step -> FC -> [B, 1]
    """

    def __init__(self, input_size: int, num_channels=(64, 128, 128, 128), kernel_size: int = 3,
                 dropout: float = 0.2, seq_len: int = 20,
                 use_revin: bool = True, target_index: int = 0, revin_eps: float = 1e-5):
        super(TCNModel, self).__init__()
        self.input_size = input_size
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.use_revin = use_revin
        self.target_index = int(target_index)
        self.revin_eps = revin_eps

        # 官方 TCN 主干
        self.tcn = TemporalConvNet(input_size, self.num_channels, kernel_size=kernel_size,
                                   dropout=dropout)
        # 末端 2 层全连接（取最后时刻特征）
        c_last = self.num_channels[-1]
        self.fc1 = nn.Linear(c_last, 128)
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: [B, T, F] -> [B, 1]"""
        # ---- RevIN 正向：逐窗口逐特征实例归一化（消除 level shift）----
        if self.use_revin:
            revin_mean = x.mean(dim=1, keepdim=True)                     # [B,1,F]
            revin_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps)
            x_in = (x - revin_mean) / revin_std
        else:
            x_in = x

        # [B,T,F] -> [B,F,T] 做因果卷积；输出 [B,C,T]
        xt = x_in.transpose(1, 2).contiguous()
        conv_out = self.tcn(xt)                 # [B, C_last, T]
        last = conv_out[:, :, -1]               # [B, C_last] 取最后时刻（因果，不含未来）

        out = self.relu(self.fc1(last))
        out = self.fc_dropout1(out)
        out = self.fc2(out)                     # [B, 1]

        # ---- RevIN 反向：仅对目标通道用其窗口统计量反归一化 ----
        if self.use_revin:
            tm = revin_mean[:, :, self.target_index]   # [B, 1]
            ts = revin_std[:, :, self.target_index]    # [B, 1]
            out = out * ts + tm

        return out


def build_tcn(cfg: dict, device=None) -> TCNModel:
    """按超参配置字典构建 TCN（train 与 load 共用，保证结构一致）。"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNModel(
        input_size=cfg['input_size'],
        num_channels=cfg.get('num_channels', [64, 128, 128, 128]),
        kernel_size=cfg.get('kernel_size', 3),
        dropout=cfg.get('dropout', 0.2),
        seq_len=cfg.get('seq_len', 20),
        use_revin=cfg.get('use_revin', False),
        target_index=cfg.get('target_index', 0),
    )
    return model.to(device)


class FrozenTCN(nn.Module):
    """封装 TCN 成攻击直接可用的形态：forward(x[B,T,F]) -> [B,1]。"""

    def __init__(self, tcn: TCNModel):
        super(FrozenTCN, self).__init__()
        self.tcn = tcn

    def forward(self, x):
        out = self.tcn(x)  # [B, 1]
        return out.reshape(out.shape[0], -1)  # [B, 1]


def load_frozen_tcn(dataset: str, device='cpu'):
    """统一加载接口，供后续攻击直接调用。

    dataset ∈ {'wind','etth1','electricity'}；从 checkpoints 重建模型，
    返回 eval()、参数已冻结的 TCN 封装，输入 [B,T,F] → 输出 [B,1]。
    """
    ckpt_file = checkpoint_path(dataset)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            "未找到 %s 的冻结权重：%s\n请先运行 train_tcn.py 训练该数据集。"
            % (dataset, ckpt_file))
    device = torch.device(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
    model = build_tcn(ckpt['config'], device=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.requires_grad_(False)
    frozen = FrozenTCN(model).to(device)
    frozen.eval()
    frozen.requires_grad_(False)
    frozen.meta = {k: ckpt.get(k) for k in
                   ('dataset', 'feature_columns', 'target_column', 'scaler', 'data_info',
                    'clean_metrics', 'gate')}
    return frozen


# ======================================================================
# 快速验证（直接运行本文件时测试前向传播 + 因果性）
# ======================================================================
if __name__ == '__main__':
    print("Testing TCNModel (official locuslab/TCN backbone)...")
    for name, F in [('wind', 4), ('etth1', 7), ('electricity', 321)]:
        m = TCNModel(input_size=F, num_channels=[64, 128, 128, 128], kernel_size=3, dropout=0.2)
        m.eval()
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 20, F)
        out = m(x)
        assert out.shape == (2, 1), "Expected (2,1), got %s" % (out.shape,)
        # 因果性检查：改动最后时刻之后的“未来”不应存在（此处仅验证形状与有限性）
        assert torch.isfinite(out).all()
        print("  %-12s F=%3d params=%9s out=%s" % (name, F, format(n_params, ','), tuple(out.shape)))
    print("All TCN tests passed!")
