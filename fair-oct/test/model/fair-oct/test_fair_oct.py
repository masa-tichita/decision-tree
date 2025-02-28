import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.params import params


# (1b) ブランチノードの制約：
#     Σ_f b[n,f] + p[n] + Σ_{m∈A(n), m≠0} p[m] == 1
def test_constraint_1b(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    b = solution.get("b")
    print("bの中身", b)
    p = solution.get("p")
    set_obj = instance.set

    for n in set_obj.B:
        sum_b = sum(b[(n, f)] for f in set_obj.F)
        # 祖先リストから、値が 0 (False) と判定されるノードは除外
        sum_anc = sum(p[m] for m in set_obj.n_A.get(n, []) if m)
        lhs = sum_b + p[n] + sum_anc
        assert lhs == 1


# (1c) 葉ノードの制約：
#     p[n] + Σ_{m∈A(n), m≠0} p[m] == 1
def test_constraint_1c(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    p = solution.get("p")
    set_obj = instance.set

    for n in set_obj.T:
        sum_anc = sum(p[m] for m in set_obj.n_A.get(n, []) if m)
        lhs = p[n] + sum_anc
        assert lhs == 1


# (1d) ブランチノードのフローの制約：
def test_constraint_1d(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    z_a = solution["z_a"]
    z_t = solution["z_t"]
    set_obj = instance.set

    for n in set_obj.B:
        # 祖先リストが存在する場合のみ制約が設定される
        if set_obj.n_A.get(n):
            parent_n = set_obj.n_A[n][-1]
            for i in set_obj.I:
                left_child = set_obj.n_C[n]["left"]
                right_child = set_obj.n_C[n]["right"]
                lhs = z_a[(i, parent_n, n)]
                rhs = (
                    z_a[(i, n, left_child)]
                    + z_a[(i, n, right_child)]
                    + sum(z_t[(i, n, k)] for k in set_obj.K)
                )
                assert lhs == rhs


# (1e) 葉のフローの制約：
def test_constraint_1e(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    z_a = solution["z_a"]
    z_t = solution["z_t"]
    set_obj = instance.set

    for n in set_obj.T:
        if set_obj.n_A.get(n):
            parent_n = set_obj.n_A[n][-1]
            for i in set_obj.I:
                lhs = z_a[(i, parent_n, n)]
                rhs = sum(z_t[(i, n, k)] for k in set_obj.K)
                assert lhs == rhs


# ルートフローの制約：
#     z_root[(i, s, root_child)] <= 1, ここでは s=0, root_child=1 としている
def test_constraint_root_flow(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    z_root = solution["z_s"]
    set_obj = instance.set
    s = 0
    root_child = 1
    for i in set_obj.I:
        val = z_root[(i, s, root_child)]
        assert val <= 1


# (1g) と (1h) 分割ルールの制約：
#     (1g): z_a[i, n, left] <= Σ_{f: x_i_f==0} b[n,f]
#     (1h): z_a[i, n, right] <= Σ_{f: x_i_f==1} b[n,f]
def test_constraint_1g_1h(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    b = solution["b"]
    z_a = solution["z_a"]
    set_obj = instance.set

    for n in set_obj.B:
        left_child = set_obj.n_C[n]["left"]
        right_child = set_obj.n_C[n]["right"]
        for i in set_obj.I:
            # 左側 (x_i_f_value==0)
            expr_left = sum(
                b[(n, f)] for f in set_obj.F if set_obj.x_i_f_value[i][f] == 0
            )
            val_left = z_a[(i, n, left_child)]
            assert val_left <= expr_left
            # 右側 (x_i_f_value==1)
            expr_right = sum(
                b[(n, f)] for f in set_obj.F if set_obj.x_i_f_value[i][f] == 1
            )
            val_right = z_a[(i, n, right_child)]
            assert val_right <= expr_right


# (1i) フローとクラス割り当ての制約：
#     z_t[i, n, k] <= w[n,k]  （n はブランチも葉も含む）
def test_constraint_1i(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    z_t = solution["z_t"]
    w = solution["w"]
    set_obj = instance.set
    nodes = set_obj.B + set_obj.T

    for n in nodes:
        for i in set_obj.I:
            for k in set_obj.K:
                val_z_t = z_t[(i, n, k)]
                val_w = w[(n, k)]
                assert val_z_t <= val_w


# (1j) 葉・ブランチにおける p と w の関係の制約：
#     Σ_k w[n,k] == p[n] （n はブランチも葉も）
def test_constraint_1j(oct_instance_without_fairness_constraint):
    opt_instance, instance = oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    w = solution["w"]
    p = solution["p"]
    set_obj = instance.set
    nodes = set_obj.B + set_obj.T

    for n in nodes:
        sum_w = sum(w[(n, k)] for k in set_obj.K)
        assert sum_w == p[n]


# テスト例1: 統計的パリティの公平性制約が満たされているか
def test_statistical_parity_constraint(oct_instance_without_fairness_constraint):
    _, fair_oct = oct_instance_without_fairness_constraint
    # 公平性制約を追加する（ここでは "statistical_parity" を指定）
    fair_oct.add_fairness_constraints(fairness_types=["statistical_parity"])
    # 公平性制約追加後、再最適化
    opt_result = fair_oct.optimize(time_limit=params.time_limit)
    solution = opt_result["solution"]
    z_t = solution["z_t"]
    set_obj = fair_oct.set
    nodes = set_obj.B + set_obj.T
    groups = set(set_obj.P)
    # 各センシティブグループ間で、クラス1のフローの合計が、制約で指定された範囲内になっているか確認
    for p_val in groups:
        I_p = [i for i in set_obj.I if set_obj.x_i_p.get(i) == p_val]
        for p_prime in groups:
            if p_val == p_prime:
                continue
            I_pprime = [i for i in set_obj.I if set_obj.x_i_p.get(i) == p_prime]
            if not I_p or not I_pprime:
                continue
            expr_p = sum(z_t[(i, n, 1)] for i in I_p for n in nodes)
            expr_pprime = sum(z_t[(i, n, 1)] for i in I_pprime for n in nodes)
            diff = abs(len(I_pprime) * expr_p - len(I_p) * expr_pprime)
            max_allowed = set_obj.delta * len(I_p) * len(I_pprime)
            assert diff <= max_allowed, (
                f"Statistical Parity constraint violated for groups {p_val} and {p_prime}: "
                f"diff={diff}, allowed={max_allowed}"
            )


# テスト例2: 条件付き統計的パリティの公平性制約が満たされているか
def test_conditional_statistical_parity_constraint(
    oct_instance_without_fairness_constraint,
):
    _, fair_oct = oct_instance_without_fairness_constraint
    # "conditional_statistical_parity" を追加
    fair_oct.add_fairness_constraints(fairness_types=["conditional_statistical_parity"])
    opt_result = fair_oct.optimize(time_limit=params.time_limit)
    solution = opt_result["solution"]
    z_t = solution["z_t"]
    set_obj = fair_oct.set
    nodes = set_obj.B + set_obj.T
    # 条件付きの場合、L の各値（正当性属性の値）ごとに制約が適用される
    for legit_val in set_obj.L:
        for p_val in set_obj.P:
            I_p_l = [
                i
                for i in set_obj.I
                if set_obj.x_i_p.get(i) == p_val
                and set_obj.x_i_legit.get(i) == legit_val
            ]
            for p_prime in set_obj.P:
                if p_val == p_prime:
                    continue
                I_pprime_l = [
                    i
                    for i in set_obj.I
                    if set_obj.x_i_p.get(i) == p_prime
                    and set_obj.x_i_legit.get(i) == legit_val
                ]
                if not I_p_l or not I_pprime_l:
                    continue
                expr_p = sum(z_t[(i, n, 1)] for i in I_p_l for n in nodes)
                expr_pprime = sum(z_t[(i, n, 1)] for i in I_pprime_l for n in nodes)
                diff = abs(len(I_pprime_l) * expr_p - len(I_p_l) * expr_pprime)
                max_allowed = set_obj.delta * len(I_p_l) * len(I_pprime_l)
                assert diff <= max_allowed, (
                    f"Conditional Statistical Parity constraint violated for groups {p_val} vs {p_prime} "
                    f"with legit value {legit_val}: diff={diff}, allowed={max_allowed}"
                )


# テスト例3: 均等化オッズの公平性制約が満たされているか
def test_equalized_odds_constraint(oct_instance_without_fairness_constraint):
    _, fair_oct = oct_instance_without_fairness_constraint
    # "equalized_odds" を追加
    fair_oct.add_fairness_constraints(fairness_types=["equalized_odds"])
    opt_result = fair_oct.optimize(time_limit=params.time_limit)
    solution = opt_result["solution"]
    z_t = solution["z_t"]
    set_obj = fair_oct.set
    nodes = set_obj.B + set_obj.T
    for k in set_obj.K:
        for p_val in set_obj.P:
            I_p_k = [
                i
                for i in set_obj.I
                if set_obj.x_i_p.get(i) == p_val and set_obj.x_i_y.get(i) == k
            ]
            for p_prime in set_obj.P:
                if p_val == p_prime:
                    continue
                I_pprime_k = [
                    i
                    for i in set_obj.I
                    if set_obj.x_i_p.get(i) == p_prime and set_obj.x_i_y.get(i) == k
                ]
                if not I_p_k or not I_pprime_k:
                    continue
                expr_p = sum(z_t[(i, n, 1)] for i in I_p_k for n in nodes)
                expr_pprime = sum(z_t[(i, n, 1)] for i in I_pprime_k for n in nodes)
                diff = abs(len(I_pprime_k) * expr_p - len(I_p_k) * expr_pprime)
                max_allowed = set_obj.delta * len(I_p_k) * len(I_pprime_k)
                assert diff <= max_allowed, (
                    f"Equalized Odds constraint violated for class {k} between groups {p_val} and {p_prime}: "
                    f"diff={diff}, allowed={max_allowed}"
                )
