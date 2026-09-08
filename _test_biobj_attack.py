# -*- coding: utf-8 -*-
"""双目标（L0 约束）改造验收自检脚本（临时，验证后删除）
自检2: 合成样本单测约束支配；自检3: knee tie-break 单测；
自检4: 极小配置端到端（objectives (N,2) / n_actual∈[1,6] / 分层覆盖 / HV 单调 / 各图生成）；
自检5: 每代各层大小 [len(f) for f in fronts] 趋势打印。
"""
import matplotlib
matplotlib.use('Agg')
import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "fixed_enc",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "LBA-MOACB-WSF-FixedEncoding.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

NVITA = mod.NVITA_NSGA2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', '_test_biobj')
os.makedirs(OUT, exist_ok=True)


class DummyModel(nn.Module):
    """合成测试模型：线性映射到标量（与 HybridCNNBiLSTM 输入输出形状一致）"""
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, 1)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))


def make_attacker(**kw):
    params = dict(n_min=1, n_max=6, epsilon=0.3, model=DummyModel(10 * 4),
                  feature_ranges=np.ones(4), maxiter=2, pop_size=6,
                  select_mode='knee', knee_min_rse=2.0)
    params.update(kw)
    return NVITA(**params)


print("=" * 70)
print("自检2: 约束支配合成单测（Deb constrained-domination）")
print("=" * 70)
att = make_attacker()
# 个体0: 可行(cv=0) 且二维支配个体1(可行) => 规则(b) A 胜
# 个体1: 可行但目标更差
# 个体2: 不可行 n=7 (cv=1)，目标数值最"好看" => 仍被可行个体约束支配（规则 a）
# 个体3: 不可行 n=9 (cv=3) => 被个体2 支配（规则 c: cv 1 < 3）
objs = np.array([[0.1, -3.0],
                 [0.2, -2.0],
                 [0.01, -5.0],
                 [0.005, -6.0]])
n_act = np.array([3.0, 5.0, 7.0, 9.0])
cv = att._constraint_violation(n_act)
print(f"cv 向量 = {cv.tolist()}（期望 [0,0,1,3]）")
assert np.allclose(cv, [0, 0, 1, 3]), "cv 计算错误"
fronts, rank = NVITA._fast_non_dominated_sort(objs, cv)
print(f"fronts = {fronts}, rank = {rank.tolist()}")
# 期望：0 支配 1（规则b，[0.1,-3] Pareto 支配 [0.2,-2]）；
#       0、1（可行）均支配 2（不可行 n=7，规则a，即使 2 目标更好看）；
#       2 支配 3（都不可行，cv 1<3，规则c） => 分层 [[0],[1],[2],[3]]
assert fronts == [[0], [1], [2], [3]], f"约束支配分层错误: {fronts}"
assert rank.tolist() == [0, 1, 2, 3]
print("PASS: (b)可行A支配可行B => A胜; (a)可行胜不可行(n=7 目标更好看仍败); (c)cv=1 胜 cv=3")

print()
print("=" * 70)
print("自检3: knee tie-break 单测（复刻 attack() 内 knee 代码路径）")
print("=" * 70)
# 两个二维归一化等距候选：[1,-2] 与 [2,-1]（按每维最大绝对值归一化后 dist 均为 sqrt(0.5²+1²)）
# n_actual 分别为 2 和 5 => 应选 n=2 者（等优更稀疏）
front_idx = np.array([0, 1])
front_obj = np.array([[1.0, -2.0], [2.0, -1.0]])
front_n = np.array([2.0, 5.0])
knee_min_rse = 0.0   # 单测中放开 RSE 有效性过滤，直接检验归一化与 tie-break
front_cv = att._constraint_violation(front_n)
feasible_mask = front_cv <= 0.0
assert feasible_mask.all()
fea_idx = front_idx[feasible_mask]
fea_obj = front_obj[feasible_mask]
cand_mask = fea_obj[:, 1] <= -knee_min_rse
cand_idx = fea_idx[cand_mask]
cand_obj = fea_obj[cand_mask]
cand_n = front_n[feasible_mask][cand_mask]
denom = np.max(np.abs(cand_obj), axis=0)
denom[denom < 1e-12] = 1.0
normed = np.abs(cand_obj) / denom
dist = np.sqrt(np.sum(normed ** 2, axis=1))
print(f"归一化 = {normed.tolist()}, 距离 = {dist.tolist()}")
assert abs(dist[0] - dist[1]) < 1e-8, "两候选距离应相等"
ties = np.where(dist <= dist.min() + 1e-8)[0]
tie_n = cand_n[ties]
min_n_pos = np.where(tie_n <= tie_n.min() + 1e-8)[0]
best_pos = ties[min_n_pos[int(np.argmin(cand_obj[ties][min_n_pos, 0]))]]
best_idx = cand_idx[best_pos]
print(f"ties = {ties.tolist()}, cand_n = {cand_n.tolist()}, 选中个体索引 = {best_idx}（期望 0, n=2）")
assert best_idx == 0, "tie-break 未选中 n_actual=2 的候选"
print("PASS: 等距候选中选中 n_actual=2（而非 5）")

print()
print("=" * 70)
print("自检4: 极小配置端到端（pop_size=6, maxiter=2, 1 个合成样本）")
print("=" * 70)
torch.manual_seed(0)
np.random.seed(0)
model = DummyModel(10 * 4)
model.eval()
att2 = NVITA(n_min=1, n_max=6, epsilon=0.3, model=model, feature_ranges=np.ones(4),
             maxiter=2, pop_size=6, select_mode='knee', knee_min_rse=2.0)
device = torch.device('cpu')
X = torch.rand(1, 10, 4)
Y = torch.rand(1)
X_adv, best_mse, pareto_solutions, pareto_objectives, generation_pareto = att2.attack(X, Y, seed=0)
print(f"pareto_objectives.shape = {pareto_objectives.shape}（期望 (k,2)）")
assert pareto_objectives.ndim == 2 and pareto_objectives.shape[1] == 2
n_sel = [len(s) // 3 for s in pareto_solutions]
print(f"前沿解编码点数 = {n_sel}（期望全部 ∈[1,6]）")
assert all(1 <= n <= 6 for n in n_sel)
# 前沿记录 = (m,3)：前两列 objectives，第三列 n_actual 附带属性
for g, rec_arr in enumerate(generation_pareto):
    assert rec_arr.shape[1] == 3, f"第{g}代记录列数 {rec_arr.shape[1]} != 3"
    assert np.all((rec_arr[:, 2] >= 1) & (rec_arr[:, 2] <= 6)), f"第{g}代 n_actual 越界"
print(f"generation_pareto: {len(generation_pareto)} 条记录, 形状 {[a.shape for a in generation_pareto]}")
# 选中解 n_actual ∈ [1,6]
diff = (X_adv - X).numpy().reshape(-1)
n_sel_final = int(np.count_nonzero(np.abs(diff) > 1e-12))
print(f"选中解 n_actual = {n_sel_final}（期望 ∈[1,6]）")
assert 1 <= n_sel_final <= 6

# 分层覆盖全部个体（用末代前沿记录重建 objectives/cv 验证排序覆盖性）
last = generation_pareto[-1]
fr, rk = NVITA._fast_non_dominated_sort(last[:, :2], att2._constraint_violation(last[:, 2]))
total = sum(len(f) for f in fr)
print(f"末代前沿记录分层: fronts sizes = {[len(f) for f in fr]}, 覆盖 {total}/{last.shape[0]}")
assert total == last.shape[0]

# HV 二维、标量、随代数单调不降（μ+λ 精英保留）；参考点用修复后的 max+0.1|max| 口径
all_pts = np.vstack(generation_pareto)
ref = np.array([all_pts[:, 0].max() + 0.1 * abs(all_pts[:, 0].max()),
                all_pts[:, 1].max() + 0.1 * abs(all_pts[:, 1].max())])
hvs = [mod.hypervolume_2d(a[:, :2], ref) for a in generation_pareto]
print(f"参考点 = {ref.tolist()}, 逐代 HV = {[round(h, 6) for h in hvs]}")
assert all(np.isscalar(h) or isinstance(h, float) for h in hvs)
assert hvs[-1] > 0, "末代 HV 应为正（参考点必须严格劣于所有前沿点）"
assert all(hvs[i + 1] >= hvs[i] - 1e-12 for i in range(len(hvs) - 1)), "HV 未随代数单调不降"
print("PASS: HV 为标量且子代合并后不下降")

# 全局前沿二维筛选（携带 n_actual 附带属性）
gf = mod.extract_first_front(np.vstack([pareto_objectives, last[:, :2]]))
print(f"全局前沿筛选结果 shape = {gf.shape}")
recs = [{'sample_idx': 0, 'generation_pareto': generation_pareto,
         'final_pareto_obj': pareto_objectives, 'final_pareto_solutions': pareto_solutions}]
gm = mod.compute_generation_metrics(recs)
assert gm is not None and gm['ref_point'].shape == (2,)
assert np.allclose(gm['ref_point'], ref), "compute_generation_metrics 参考点口径不一致"
assert np.nanmin(np.asarray(gm['hv_mean'])) > 0, "修复后 hv_mean 应全为正"
print(f"compute_generation_metrics: ref_point shape = {gm['ref_point'].shape}, "
      f"obj_mean_per_sample shape = {gm['obj_mean_per_sample'].shape}, "
      f"n_actual_mean = {np.round(gm['n_actual_mean'], 3).tolist()}")
assert gm['obj_mean_per_sample'].shape[2] == 2
assert 'obj3_mean' not in gm

# 【L0 约束】批量攻击接口的约束满足度指标（三目标版无此项，L0 改为约束后新增）
X_adv_b, metrics_b, _ = mod.run_nsga2_standalone_attack(
    model, X, Y, beta=0.3, n_min=1, n_max=6, maxiter=2, pop_size=6,
    device=device, feature_ranges=np.ones(4))
assert metrics_b['constraint_feasible_cnt'] == metrics_b['total_samples'], "选中解应全部满足 L0 约束"
assert abs(metrics_b['constraint_feasible_rate'] - 1.0) < 1e-12, "可行率应为 100%"
assert 1 <= metrics_b['mean_sel_n_actual'] <= 6, "mean_sel_n_actual 越界"
print(f"run_nsga2_standalone_attack: L0 约束满足率 = "
      f"{metrics_b['constraint_feasible_cnt']}/{metrics_b['total_samples']} "
      f"= {metrics_b['constraint_feasible_rate']:.0%}, "
      f"mean_sel_n_actual = {metrics_b['mean_sel_n_actual']:.2f}")
print("PASS: 约束满足度按双边界 [n_min, n_max] 统计并已写入 metrics")

# 各图生成
mod.plot_pareto_evolution(generation_pareto, 1, 6, 6,
                          save_path=os.path.join(OUT, 'pareto_evolution.png'))
mod.plot_hv_convergence(gm, save_path=os.path.join(OUT, 'hv_convergence.png'))
mod.plot_objective_evolution(gm, save_path=os.path.join(OUT, 'objective_evolution.png'))
mod.plot_pareto_count_evolution(gm, save_path=os.path.join(OUT, 'pareto_count_evolution.png'))
gfront = np.column_stack([pareto_objectives, last[:, 2]])
saved = mod.plot_global_pareto_front(gfront, OUT)
print(f"生成图片: {sorted(os.listdir(OUT))}")
assert os.path.exists(os.path.join(OUT, 'pareto_evolution.png'))
assert os.path.exists(os.path.join(OUT, 'pareto_l2_rse.png'))
assert len(saved) == 1, "主 Pareto 图应只剩一张（原三张两两投影已删除）"
print("PASS: 各图正常生成，无 3D/三目标残留，主图按 n_actual 着色")

print()
print("=" * 70)
print("自检5: 每代各层大小趋势（pop_size=6, maxiter=6，复刻主循环）")
print("=" * 70)
att3 = NVITA(n_min=1, n_max=6, epsilon=0.3, model=model, feature_ranges=np.ones(4),
             maxiter=6, pop_size=6, select_mode='knee', knee_min_rse=2.0)
np.random.seed(1)
seq_len, num_features = 10, 4
allowed = att3._sample_features(num_features)
ranges = np.ones(num_features)
population = [att3._init_individual(seq_len, num_features, allowed, ranges) for _ in range(6)]
objectives = np.zeros((6, 2))
n_actual_pop = np.zeros(6)
for i in range(6):
    objectives[i], n_actual_pop[i] = att3._evaluate_objectives(X.numpy(), Y.item(), population[i], device)
for gen in range(6):
    cv_pop = att3._constraint_violation(n_actual_pop)
    fronts, rank = NVITA._fast_non_dominated_sort(objectives, cv_pop)
    print(f"gen {gen}: [len(f) for f in fronts] = {[len(f) for f in fronts]}  "
          f"(front0/pop_size = {len(fronts[0]) / 6:.2f})")
    crowding = NVITA._crowding_distance(objectives, fronts)
    mating = NVITA._tournament_selection(population, objectives, rank, crowding, 6)
    offspring = []
    for i in range(0, 6, 2):
        c1, c2 = att3._crossover(mating[i], mating[min(i + 1, 5)], seq_len, num_features, ranges, allowed)
        offspring.append(att3._mutation(c1, seq_len, num_features, ranges, allowed))
        offspring.append(att3._mutation(c2, seq_len, num_features, ranges, allowed))
    off_obj = np.zeros((6, 2))
    n_off = np.zeros(6)
    for i in range(6):
        off_obj[i], n_off[i] = att3._evaluate_objectives(X.numpy(), Y.item(), offspring[i], device)
    comb_pop = population + offspring
    comb_obj = np.vstack([objectives, off_obj])
    comb_n = np.concatenate([n_actual_pop, n_off])
    cv_c = att3._constraint_violation(comb_n)
    fr_c, _ = NVITA._fast_non_dominated_sort(comb_obj, cv_c)
    cr_c = NVITA._crowding_distance(comb_obj, fr_c)
    sel = []
    for f in fr_c:
        if len(sel) + len(f) <= 6:
            sel.extend(f)
        else:
            fa = np.array(f)
            sel.extend(fa[np.argsort(-cr_c[f])][:6 - len(sel)].tolist())
            break
    population = [comb_pop[i] for i in sel]
    objectives = comb_obj[sel]
    n_actual_pop = comb_n[sel]

print()
print("全部自检通过 [OK]")
