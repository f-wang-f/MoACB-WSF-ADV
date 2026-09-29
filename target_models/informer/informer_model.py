# -*- coding: utf-8 -*-
"""
Informer 目标模型（严格对照官方开源实现 zhouhaoyi/Informer2020）
====================================================================
本文件把官方的 models/attn.py、models/embed.py、models/encoder.py、
models/decoder.py、models/model.py、utils/masking.py 六个文件的核心结构
忠实地合并到单文件内，便于作为“被攻击目标模型”独立加载：

  * ProbSparse 自注意力（ProbAttention）、FullAttention、AttentionLayer —— 与官方 attn.py 一致
  * TokenEmbedding(Conv1d circular)、PositionalEmbedding、DataEmbedding —— 与官方 embed.py 一致
  * ConvLayer(distil)、EncoderLayer、Encoder —— 与官方 encoder.py 一致
  * DecoderLayer、Decoder —— 与官方 decoder.py 一致
  * Informer(enc/dec + projection) —— 与官方 model.py 一致

【相对官方的三处最小改动，均有明确原因】
  1) 单步单目标回归：pred_len=1、c_out=1，projection 直接输出单一目标通道；
     forward 在未显式给出 x_dec 时，自动用“最后 label_len 段 + pred_len 个零占位”
     构造解码器输入（与官方 ETT 数据管线的 dec_inp 构造方式等价）。
  2) 设备兼容修复：官方 ProbAttention/ProbMask 里 torch.arange(...) 未指定 device，
     在 CUDA 上索引 GPU 张量会报设备不匹配；此处统一用张量自身 device 创建索引，
     数值行为与官方完全一致，仅保证 GPU 可运行。
  3) 时间特征嵌入可选（use_temporal，默认 False）：攻击接口要求纯 [B,T,F]->[B,1]，
     时间戳无法从被扰动的输入窗口恢复，故默认关闭 temporal_embedding，
     只保留 token+positional（ProbSparse/distil/enc-dec 主干 100% 保留）。
  4) enc_only（编码器-only + 末端全连接头）：单步预测时官方解码器的“零占位 token”
     与截断后的短 encoder 输出做 cross-attention 信息量很低；enc_only=True 时去掉
     解码器，直接取编码器最后时刻表示过 projection（nn.Linear(d_model, c_out)）输出，
     对所有数据集统一生效（默认 False 保持旧 checkpoint 结构兼容）。
  5) use_revin（RevIN 可逆实例归一化）：对每个输入窗口按时间维逐特征标准化（减去
     窗口均值/除以窗口标准差），编码器处理归一化后的序列，末端头输出后仅对目标
     通道用其窗口均值/标准差反归一化。目的是消除 test 段相对 train 的 level shift
     （季节/趋势漂移），让模型学习“相对最近窗口水平的偏差”而非绝对水平，对三集统一
     生效。RevIN 内置于 forward，对外接口仍为 [B,T,F]->[B,1]（默认 False 兼容旧权重）。

统一加载接口见文件末尾 load_frozen_informer()。
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
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../target_models/informer
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))  # 项目根
CHECKPOINT_ROOT = os.path.join(_THIS_DIR, 'checkpoints')

# 数据根目录候选：任务约定 data/raw/，实际文件当前在 data/，两者都查
DATA_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, 'data', 'raw'),
    os.path.join(PROJECT_ROOT, 'data'),
]

VALID_DATASETS = ('wind', 'etth1', 'electricity')


def checkpoint_path(dataset: str) -> str:
    """返回某数据集冻结权重的绝对路径：checkpoints/informer_<dataset>.pt"""
    if dataset not in VALID_DATASETS:
        raise ValueError("dataset must be one of %s, got %r" % (VALID_DATASETS, dataset))
    return os.path.join(CHECKPOINT_ROOT, 'informer_%s.pt' % dataset)


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
# utils/masking.py （官方，device-safe）
# ======================================================================
class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B, device=device)[:, None, None],
                             torch.arange(H, device=device)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


# ======================================================================
# models/attn.py （官方，ProbAttention 索引改为 device-safe）
# ======================================================================
class FullAttention(nn.Module):
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


class ProbAttention(nn.Module):
    """ProbSparse 自注意力（官方核心创新）。mask_flag=True 时用于解码器因果自注意力。"""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        device = Q.device

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=device)  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q, device=device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B, device=device)[:, None, None],
                     torch.arange(H, device=device)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        device = V.device

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B, device=device)[:, None, None],
                   torch.arange(H, device=device)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B, device=device)[:, None, None],
                  torch.arange(H, device=device)[None, :, None],
                  index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
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


# ======================================================================
# models/embed.py （官方；DataEmbedding 增加 use_temporal 开关）
# ======================================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, use_temporal=False):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.use_temporal = use_temporal
        if use_temporal:
            self.temporal_embedding = (TemporalEmbedding(d_model, embed_type, freq)
                                       if embed_type != 'timeF'
                                       else TimeFeatureEmbedding(d_model, embed_type, freq))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if self.use_temporal and x_mark is not None:
            x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# ======================================================================
# models/encoder.py （官方）
# ======================================================================
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in,
                                  kernel_size=3, padding=padding, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
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


# ======================================================================
# models/decoder.py （官方）
# ======================================================================
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ======================================================================
# models/model.py （官方 Informer，单步单目标适配）
# ======================================================================
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, use_temporal=False,
                 enc_only=False, use_revin=False, target_index=0, revin_eps=1e-5,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention
        self.enc_only = enc_only
        self.use_revin = use_revin
        self.target_index = int(target_index)
        self.revin_eps = revin_eps

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, use_temporal)
        if not enc_only:
            self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, use_temporal)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil and e_layers > 1 else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder（enc_only 时不构建解码器）
        if not enc_only:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=mix),
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout, activation=activation,
                    ) for _ in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B = x_enc.shape[0]
        # 未显式提供解码器输入时，按官方 ETT 管线方式自动构造（见下方 else 分支）
        # ---- RevIN 正向：逐窗口逐特征实例归一化（消除 level shift）----
        if self.use_revin:
            revin_mean = x_enc.mean(dim=1, keepdim=True)                       # [B,1,F]
            revin_std = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps)
            x_in = (x_enc - revin_mean) / revin_std
        else:
            x_in = x_enc
        enc_out = self.enc_embedding(x_in, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if self.enc_only:
            # 编码器-only：取最后时刻表示 -> 末端全连接头（对 pred_len 个步重复输出）
            head = self.projection(enc_out[:, -1:, :])           # [B, 1, c_out]
            if self.use_revin:
                # 仅对目标通道用其窗口统计量反归一化（c_out=1）
                tm = revin_mean[:, :, self.target_index]         # [B, 1]
                ts = revin_std[:, :, self.target_index]          # [B, 1]
                head = (head.squeeze(1) * ts + tm).unsqueeze(1)  # [B, 1, c_out]
            dec_out = head.expand(B, self.pred_len, head.shape[-1])  # [B, pred_len, c_out]
        else:
            # 未显式提供解码器输入时，按官方 ETT 管线方式自动构造：
            #   [编码器最后 label_len 段(已知)] + [pred_len 个零占位(待预测)]
            if x_dec is None:
                label = x_in[:, -self.label_len:, :]
                placeholder = torch.zeros(B, self.pred_len, x_in.shape[-1],
                                          device=x_in.device, dtype=x_in.dtype)
                x_dec = torch.cat([label, placeholder], dim=1)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_out = self.projection(dec_out)               # [B, label_len+pred_len, c_out]
            dec_out = dec_out[:, -self.pred_len:, :]         # [B, pred_len, c_out]
            if self.use_revin:
                tm = revin_mean[:, :, self.target_index]     # [B, 1]
                ts = revin_std[:, :, self.target_index]      # [B, 1]
                dec_out = dec_out[:, :, :1] * ts.unsqueeze(1) + tm.unsqueeze(1)

        if self.output_attention:
            return dec_out, attns
        return dec_out


# ======================================================================
# 攻击友好封装 + 统一加载接口
# ======================================================================
class FrozenInformer(nn.Module):
    """把忠实 Informer 封装成攻击直接可用的形态：forward(x[B,T,F]) -> [B,1]。

    仅做 reshape，不引入任何新参数；本身也应被 eval()+requires_grad_(False)。
    """

    def __init__(self, informer: Informer):
        super(FrozenInformer, self).__init__()
        self.informer = informer

    def forward(self, x):
        out = self.informer(x)                 # [B, pred_len, c_out] = [B, 1, 1]
        return out.reshape(out.shape[0], -1)   # [B, pred_len*c_out] = [B, 1]


def build_informer(cfg: dict, device=None) -> Informer:
    """按超参配置字典构建 Informer（train 与 load 共用，保证结构一致）。"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Informer(
        enc_in=cfg['enc_in'], dec_in=cfg['dec_in'], c_out=cfg['c_out'],
        seq_len=cfg['seq_len'], label_len=cfg['label_len'], out_len=cfg['pred_len'],
        factor=cfg.get('factor', 5), d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        e_layers=cfg['e_layers'], d_layers=cfg['d_layers'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], attn=cfg.get('attn', 'prob'), embed=cfg.get('embed', 'fixed'),
        freq=cfg.get('freq', 'h'), activation=cfg.get('activation', 'gelu'),
        output_attention=False, distil=cfg.get('distil', True), mix=cfg.get('mix', True),
        use_temporal=cfg.get('use_temporal', False), enc_only=cfg.get('enc_only', False),
        use_revin=cfg.get('use_revin', False), target_index=cfg.get('target_index', 0),
        device=device,
    )
    return model.to(device)


def load_frozen_informer(dataset: str, device='cpu'):
    """统一加载接口，供后续攻击直接调用。

    dataset ∈ {'wind','etth1','electricity'}；从 checkpoints 重建模型，
    返回 eval()、参数已冻结的 Informer 封装，输入 [B,T,F] → 输出 [B,1]。
    """
    ckpt_file = checkpoint_path(dataset)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            "未找到 %s 的冻结权重：%s\n请先运行 train_informer.py 训练该数据集。"
            % (dataset, ckpt_file))
    device = torch.device(device)
    # weights_only=True：本 checkpoint 仅含张量与基础类型(dict/list/str/int/float/bool/None)，
    #   安全加载并消除 PyTorch 2.x 的 FutureWarning
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
    model = build_informer(ckpt['config'], device=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.requires_grad_(False)
    frozen = FrozenInformer(model).to(device)
    frozen.eval()
    frozen.requires_grad_(False)
    # 附带归一化/数据元信息，便于攻击侧对齐输入口径
    frozen.meta = {k: ckpt.get(k) for k in
                   ('dataset', 'feature_columns', 'target_column', 'scaler', 'data_info', 'clean_metrics')}
    return frozen
