# -*- coding: utf-8 -*-
"""验证 SA-MOO 集合式交叉/变异算子"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import importlib.util
import numpy as np
import torch
import torch.nn as nn

_spec = importlib.util.spec_from_file_location(
    "lbam", r"C:\Users\30544\Desktop\科研入门\WindSpeedPrediction\MoACB-WSF-ADV\LBA-MOACB-WSF-FixedEncoding.py")
_lbam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbam)
NVITA_NSGA2 = _lbam.NVITA_NSGA2

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.arange(1.0, 16.0).view(1, 15, 1) / 100.0
    def forward(self, x):
        return (x * self.w).sum(dim=(1, 2), keepdim=True)

torch.manual_seed(42)
X = torch.randn(1, 15, 4)
model = DummyModel()
with torch.no_grad():
    y = model(X)

attacker = NVITA_NSGA2(
    n=3, epsilon=0.3, model=model,
    feature_ranges=[3.0, 1.0, 2.0, 2.5],
    maxiter=8, pop_size=12,
)
np.random.seed(7)
seq_len, num_features, ranges = 15, 4, np.array([3.0, 1.0, 2.0, 2.5])
allowed = np.arange(4)

# ---- 1. USX 集合交叉生效率 ----
changed = 0
t_ok = f_ok = p_ok = True
for _ in range(200):
    a = attacker._init_individual(seq_len, num_features, allowed, ranges)
    b = attacker._init_individual(seq_len, num_features, allowed, ranges)
    ca, cb = attacker._sbx_crossover(a, b, seq_len, num_features, ranges)
    if not (np.allclose(ca, a) and np.allclose(cb, b)):
        changed += 1
    for c in (ca, cb):
        for k in range(3):
            t_ok &= 0 <= int(round(c[3*k])) < seq_len
            f_ok &= 0 <= int(round(c[3*k+1])) < num_features
            p_ok &= abs(c[3*k+2]) <= 0.3 * ranges[int(round(c[3*k+1]))] + 1e-9
print(f"[USX 交叉] 200 次生效率: {changed}/200 (期望约 180，crossover_prob=0.9)")
print(f"[USX 交叉] 合法性: t={t_ok} f={f_ok} p预算={p_ok}")

# ---- 2. 分维度变异生效率 ----
mchanged = 0
mt_ok = mf_ok = mp_ok = True
for _ in range(200):
    a = attacker._init_individual(seq_len, num_features, allowed, ranges)
    m = attacker._polynomial_mutation(a, seq_len, num_features, ranges)
    if not np.allclose(m, a):
        mchanged += 1
    for k in range(3):
        mt_ok &= 0 <= int(round(m[3*k])) < seq_len
        mf_ok &= 0 <= int(round(m[3*k+1])) < num_features
        mp_ok &= abs(m[3*k+2]) <= 0.3 * ranges[int(round(m[3*k+1]))] + 1e-9
print(f"[分维度变异] 200 次生效率: {mchanged}/200 (期望约 185+：t位15%×3 + f位15%×3 + p位1/3×3)")
print(f"[分维度变异] 合法性: t={mt_ok} f={mf_ok} p预算={mp_ok}")

# ---- 3. t 位邻域变异统计：变异后的 t 与变异前差值分布 ----
t_diffs = []
for _ in range(2000):
    a = attacker._init_individual(seq_len, num_features, allowed, ranges)
    m = attacker._polynomial_mutation(a, seq_len, num_features, ranges)
    for k in range(3):
        d = int(round(m[3*k])) - int(round(a[3*k]))
        if d != 0:
            t_diffs.append(abs(d))
if t_diffs:
    print(f"[t 邻域变异] 2000 次共 {len(t_diffs)} 个 t 位变化，")
    print(f"   |Δ|=1 占比: {sum(1 for d in t_diffs if d == 1)/len(t_diffs):.2%} (邻域移动+全局重采样可能落邻点)")
    print(f"   |Δ|<=1 占比: {sum(1 for d in t_diffs if d <= 1)/len(t_diffs):.2%}")

# ---- 4. f 位重采样多样性 ----
f_vals = set()
for _ in range(2000):
    a = attacker._init_individual(seq_len, num_features, allowed, ranges)
    m = attacker._polynomial_mutation(a, seq_len, num_features, ranges)
    for k in range(3):
        if int(round(m[3*k+1])) != int(round(a[3*k+1])):
            f_vals.add(int(round(m[3*k+1])))
print(f"[f 重采样] 变异后出现的特征集合: {sorted(f_vals)} (期望覆盖 0~3)")

# ---- 5. 完整 attack() ----
X_adv, best_mse, pareto_sol, pareto_obj, gen_pareto = attacker.attack(X, y, seed=0)
print("\n[attack] 运行完成, X_adv 形状:", tuple(X_adv.shape))
print(f"[attack] 最终前沿解数: {pareto_sol.shape[0]}, 进化记录数: {len(gen_pareto)} (期望 9)")
violations = 0
for i in range(pareto_obj.shape[0]):
    for j in range(pareto_obj.shape[0]):
        if i != j and np.all(pareto_obj[j] <= pareto_obj[i]) and np.any(pareto_obj[j] < pareto_obj[i]):
            violations += 1
print(f"[attack] 前沿内部支配违规: {violations} (期望 0)")
print(f"[attack] 前沿 obj[1] 1/RSE 范围: {pareto_obj[:,1].min():.4f} ~ {pareto_obj[:,1].max():.4f}")
