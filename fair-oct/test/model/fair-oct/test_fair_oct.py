import pytest
from math import isclose

TOL = 1e-6  # 浮動小数点の許容誤差


# (1b) ブランチノードの制約：
#     Σ_f b[n,f] + p[n] + Σ_{m∈A(n), m≠0} p[m] == 1
def test_constraint_1b(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
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
        assert isclose(lhs, 1, abs_tol=TOL), (
            f"Constraint 1b failed for branch node {n}: "
            f"sum_b={sum_b}, p[{n}]={p[n]}, sum_anc={sum_anc}, lhs={lhs}"
        )

# (1c) 葉ノードの制約：
#     p[n] + Σ_{m∈A(n), m≠0} p[m] == 1
def test_constraint_1c(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    p = solution.get("p")
    set_obj = instance.set

    for n in set_obj.T:
        sum_anc = sum(p[m] for m in set_obj.n_A.get(n, []) if m)
        lhs = p[n] + sum_anc
        assert isclose(lhs, 1, abs_tol=TOL), (
            f"Constraint 1c failed for leaf node {n}: p[{n}]={p[n]}, sum_anc={sum_anc}, lhs={lhs}"
        )

# (1d) ブランチノードのフローの制約：
def test_constraint_1d(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
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
                rhs = z_a[(i, n, left_child)] + z_a[(i, n, right_child)] \
                      + sum(z_t[(i, n, k)] for k in set_obj.K)
                assert isclose(lhs, rhs, abs_tol=TOL), (
                    f"Constraint 1d failed for i={i}, branch node {n}: "
                    f"lhs (z_a[{i},{parent_n},{n}])={lhs}, rhs={rhs}"
                )

# (1e) 葉のフローの制約：
def test_constraint_1e(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
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
                assert isclose(lhs, rhs, abs_tol=TOL), (
                    f"Constraint 1e failed for i={i}, leaf node {n}: "
                    f"lhs (z_a[{i},{parent_n},{n}])={lhs}, rhs={rhs}"
                )

# ルートフローの制約：
#     z_root[(i, s, root_child)] <= 1, ここでは s=0, root_child=1 としている
def test_constraint_root_flow(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    z_root = solution["z_s"]
    set_obj = instance.set
    s = 0
    root_child = 1
    for i in set_obj.I:
        val = z_root[(i, s, root_child)]
        assert val <= 1 + TOL, (
            f"Constraint root flow failed for i={i}: z_root[(i,0,1)]={val} > 1"
        )

# (1g) と (1h) 分割ルールの制約：
#     (1g): z_a[i, n, left] <= Σ_{f: x_i_f==0} b[n,f]
#     (1h): z_a[i, n, right] <= Σ_{f: x_i_f==1} b[n,f]
def test_constraint_1g_1h(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    b = solution["b"]
    z_a = solution["z_a"]
    set_obj = instance.set

    for n in set_obj.B:
        left_child = set_obj.n_C[n]["left"]
        right_child = set_obj.n_C[n]["right"]
        for i in set_obj.I:
            # 左側 (x_i_f_value==0)
            expr_left = sum(b[(n, f)] for f in set_obj.F if set_obj.x_i_f_value[i][f] == 0)
            val_left = z_a[(i, n, left_child)]
            assert val_left <= expr_left + TOL, (
                f"Constraint 1g failed for i={i}, branch node {n}: "
                f"z_a[{i},{n},{left_child}]={val_left} > expr_left={expr_left}"
            )
            # 右側 (x_i_f_value==1)
            expr_right = sum(b[(n, f)] for f in set_obj.F if set_obj.x_i_f_value[i][f] == 1)
            val_right = z_a[(i, n, right_child)]
            assert val_right <= expr_right + TOL, (
                f"Constraint 1h failed for i={i}, branch node {n}: "
                f"z_a[{i},{n},{right_child}]={val_right} > expr_right={expr_right}"
            )

# (1i) フローとクラス割り当ての制約：
#     z_t[i, n, k] <= w[n,k]  （n はブランチも葉も含む）
def test_constraint_1i(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
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
                assert val_z_t <= val_w + TOL, (
                    f"Constraint 1i failed for i={i}, node {n}, class {k}: "
                    f"z_t={(i,n,k)}={val_z_t} > w[{n},{k}]={val_w}"
                )

# (1j) 葉・ブランチにおける p と w の関係の制約：
#     Σ_k w[n,k] == p[n] （n はブランチも葉も）
def test_constraint_1j(fair_oct_instance_without_fairness_constraint):
    opt_instance, instance = fair_oct_instance_without_fairness_constraint
    solution = opt_instance.get("solution")
    w = solution["w"]
    p = solution["p"]
    set_obj = instance.set
    nodes = set_obj.B + set_obj.T

    for n in nodes:
        sum_w = sum(w[(n, k)] for k in set_obj.K)
        assert isclose(sum_w, p[n], abs_tol=TOL), (
            f"Constraint 1j failed for node {n}: sum_w={sum_w} != p[{n}]={p[n]}"
        )
