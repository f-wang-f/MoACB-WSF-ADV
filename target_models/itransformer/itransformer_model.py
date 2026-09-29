# -*- coding: utf-8 -*-
"""
iTransformer 目标模型（wind / etth1 / electricity 三数据集通用）
==========================================================
严格参照官方公共仓库 https://github.com/thuml/iTransformer
（Liu et al., iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting, ICLR 2024）的“倒置”结构，不自行臆造：
  * DataEmbedding_inverted：把每个变量的整段历史（T 个时间点）经 nn.Linear(T→d_model)
    嵌入为一个 token（时间维 → d_model），得到 [B, F, d_model]；
  * 自注意力作用在“变量维”（token=变量）上，建模多变量相关性；前馈作用在 d_model（时间/嵌入维）上；
  * projection：nn.Linear(d_model→pred_len) 把每个变量 token 映射回 pred_len；
  * Non-stationary 归一化（逐变量对时间维标准化 = 本项目统一的 RevIN）+ 末端反归一化。
所用 FullAttention / AttentionLayer / EncoderLayer / Encoder 与官方（Time-Series-Library）一致，
亦与本仓库 target_models/informer/informer_model.py 中的忠实官方实现同源。

任务口径（与 CNN/LSTM/GRU/TCN/Informer 完全一致）：
  * 多变量输入、单一目标通道、单步预测；lookback T=20；输入 [B,T,F] → 输出 [B,1]
  * forward：[B,T,F] →（RevIN 逐变量标准化）→ 转置嵌入 [B,F,d_model] → 跨变量注意力编码器
    → 投影 [B,F,pred_len] → 反归一化 → 取 target_index 对应变量通道 → [B,1]
  * 三数据集结构与训练超参完全相同，仅 input_size（=变量数 F）随数据变化；无按数据集名分支。

统一加载接口见文件末尾 load_frozen_itransformer()。
"""
import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# 集中路径配置（不写死；含中文的路径统一按 UTF-8 处理）
# ======================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../target_models/itransformer
CHECKPOINT_ROOT = os.path.join(_THIS_DIR, 'checkpoints')

# 数据根目录候选：任务约定 data/raw/，实际文件当前在 data/，两者都查
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'data', 'raw'),
    os.path.join(PROJECT_ROOT, 'data'),
]

VALID_DATASETS = ('wind', 'etth1', 'electricity')


def checkpoint_path(dataset: str) -> str:
    """返回某数据集冻结权重的绝对路径：checkpoints/itransformer_<dataset>.pt"""
    if dataset not in VALID_DATASETS:
        raise ValueError("dataset must be one of %s, got %r" % (VALID_DATASETS, dataset))
    return os.path.join(CHECKPOINT_ROOT, 'itransformer_%s.pt' % dataset)


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
# 官方组件（Time-Series-Library / iTransformer，逐行对齐，不臆造）
# ======================================================================
class TriangularCausalMask:
    """因果掩码（iTransformer 用 mask_flag=False，不触发；保留以忠实官方 FullAttention）。"""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """官方全注意力。iTransformer 以 mask_flag=False 使用（跨变量注意力，无因果掩码）。"""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """官方注意力封装：Q/K/V/out 线性投影 + 多头。"""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """官方编码器层：自注意力 + 前馈（1x1 Conv 实现）+ 两处 LayerNorm 残差。"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """官方编码器：堆叠 EncoderLayer（iTransformer 不用 conv_layers）+ 末端 LayerNorm。"""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DataEmbedding_inverted(nn.Module):
    """官方倒置嵌入：x[B,T,N] → permute[B,N,T] → Linear(T→d_model) → [B,N,d_model]。

    每个变量（含其整段历史）被嵌入为一个 token；c_in 即 seq_len(T)。
    本任务不使用时间标记（x_mark=None），与统一口径一致。
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)                       # [B,T,N] -> [B,N,T]
        if x_mark is None:
            x = self.value_embedding(x)              # [B,N,d_model]
        else:
            x = self.value_embedding(torch.cat((x, x_mark.permute(0, 2, 1)), 1))
        return self.dropout(x)


# ======================================================================
# iTransformer 回归封装（RevIN + 倒置结构 + 单目标单步适配 [B,T,F]→[B,1]）
# ======================================================================
class ITransformerModel(nn.Module):
    """
    Input: [B, T, F] -> RevIN(逐变量对时间维标准化) -> DataEmbedding_inverted [B,F,d_model]
           -> Encoder(跨变量注意力) -> projection [B,F,pred_len] -> permute [B,pred_len,F]
           -> 反归一化 -> 取 target_index 变量通道 -> [B, 1]
    """

    def __init__(self, input_size: int, seq_len: int = 20, pred_len: int = 1,
                 d_model: int = 128, n_heads: int = 4, e_layers: int = 2, d_ff: int = 256,
                 dropout: float = 0.1, factor: int = 5, activation: str = 'gelu',
                 use_revin: bool = True, target_index: int = 0, revin_eps: float = 1e-5):
        super(ITransformerModel, self).__init__()
        self.input_size = input_size          # 变量数 F
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.target_index = int(target_index)
        self.revin_eps = revin_eps

        # 官方倒置嵌入：Linear(seq_len -> d_model)
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        # 官方编码器：跨变量注意力（mask_flag=False）
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # 官方投影：d_model -> pred_len
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x):
        """x: [B, T, F] -> [B, 1]（严格照官方 forecast 的归一化/嵌入/编码/投影/反归一化）"""
        B, T, N = x.shape

        # ---- Normalization from Non-stationary Transformer（逐变量对时间维，= 本项目 RevIN）----
        if self.use_revin:
            means = x.mean(1, keepdim=True).detach()                       # [B,1,N]
            xc = x - means
            stdev = torch.sqrt(xc.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps)
            xc = xc / stdev
        else:
            xc, means, stdev = x, None, None

        # ---- Embedding：[B,T,N] -> [B,N,d_model] ----
        enc_out = self.enc_embedding(xc, None)
        # ---- Encoder：跨变量注意力 [B,N,d_model] ----
        enc_out, _attns = self.encoder(enc_out, attn_mask=None)
        # ---- Projection：[B,N,d_model] -> [B,N,pred_len] -> permute -> [B,pred_len,N] ----
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # ---- De-Normalization from Non-stationary Transformer ----
        if self.use_revin:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # ---- 取目标变量通道、最后预测步 -> [B,1]（保持单目标接口）----
        out = dec_out[:, -1, self.target_index].unsqueeze(1)
        return out


def build_itransformer(cfg: dict, device=None) -> ITransformerModel:
    """按超参配置字典构建 iTransformer（train 与 load 共用，保证结构一致）。"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ITransformerModel(
        input_size=cfg['input_size'],
        seq_len=cfg.get('seq_len', 20),
        pred_len=cfg.get('pred_len', 1),
        d_model=cfg.get('d_model', 128),
        n_heads=cfg.get('n_heads', 4),
        e_layers=cfg.get('e_layers', 2),
        d_ff=cfg.get('d_ff', 256),
        dropout=cfg.get('dropout', 0.1),
        factor=cfg.get('factor', 5),
        activation=cfg.get('activation', 'gelu'),
        use_revin=cfg.get('use_revin', False),
        target_index=cfg.get('target_index', 0),
    )
    return model.to(device)


class FrozenITransformer(nn.Module):
    """封装 iTransformer 成攻击直接可用的形态：forward(x[B,T,F]) -> [B,1]。"""

    def __init__(self, model: ITransformerModel):
        super(FrozenITransformer, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)  # [B, 1]
        return out.reshape(out.shape[0], -1)  # [B, 1]


def load_frozen_itransformer(dataset: str, device='cpu'):
    """统一加载接口，供后续攻击直接调用。

    dataset ∈ {'wind','etth1','electricity'}；从 checkpoints 重建模型，
    返回 eval()、参数已冻结的 iTransformer 封装，输入 [B,T,F] → 输出 [B,1]。
    """
    ckpt_file = checkpoint_path(dataset)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            "未找到 %s 的冻结权重：%s\n请先运行 train_itransformer.py 训练该数据集。"
            % (dataset, ckpt_file))
    device = torch.device(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
    model = build_itransformer(ckpt['config'], device=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.requires_grad_(False)
    frozen = FrozenITransformer(model).to(device)
    frozen.eval()
    frozen.requires_grad_(False)
    frozen.meta = {k: ckpt.get(k) for k in
                   ('dataset', 'feature_columns', 'target_column', 'scaler', 'data_info',
                    'clean_metrics', 'gate')}
    return frozen


# ======================================================================
# 快速验证（直接运行本文件时测试前向传播）
# ======================================================================
if __name__ == '__main__':
    print("Testing ITransformerModel (official thuml/iTransformer backbone)...")
    for name, Feat in [('wind', 4), ('etth1', 7), ('electricity', 321)]:
        m = ITransformerModel(input_size=Feat, seq_len=20, pred_len=1, d_model=128,
                              n_heads=4, e_layers=2, d_ff=256, dropout=0.1,
                              use_revin=True, target_index=0)
        m.eval()
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 20, Feat)
        out = m(x)
        assert out.shape == (2, 1), "Expected (2,1), got %s" % (out.shape,)
        assert torch.isfinite(out).all()
        print("  %-12s F=%3d params=%9s out=%s" % (name, Feat, format(n_params, ','), tuple(out.shape)))
    print("All iTransformer tests passed!")
