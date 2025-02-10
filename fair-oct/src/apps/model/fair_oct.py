import os
import sys

import mip
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from mip import Model, xsum, BINARY, maximize
import polars as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils.const import Cols
from utils.params import params
from modules.prep import (
    select_binary_features,
    create_feature_mapping,
    create_nodes,
    create_ancestors,
    create_children,
)


class Set(BaseModel):
    depth: int = Field(description="木の深さ")
    delta: float = Field(description="公平性制約のバイアス許容値")
    I: List[int] = Field(description="データの集合")
    F: List[str] = Field(description="特徴量の集合")
    K: List[int] = Field(description="予測クラスの集合")
    B: List[int] = Field(description="ブランチノードの集合")
    T: List[int] = Field(description="葉ノードの集合")
    P: List[Any] = Field(default_factory=list, description="センシティブ属性の集合")
    L: List[Any] = Field(
        default_factory=list,
        description="予測に影響を与えるが、公平性には影響しない属性の集合",
    )

    n_A: Dict[int, List[int]] = Field(
        default_factory=dict, description="ノードnと祖先集合A(n)の対応"
    )
    n_C: dict[int, dict[str, int]] = Field(
        default_factory=dict, description="ノードnと子孫集合C(n)の対応"
    )
    x_i_f_value: Dict[int, Dict[str, int]] = Field(
        default_factory=dict, description="データiの特徴量fの値(0 or 1)"
    )
    x_i_y: Dict[int, int] = Field(
        default_factory=dict, description="データiの正解ラベル"
    )
    x_i_p: Dict[int, Any] = Field(
        default_factory=dict, description="データiのセンシティブ属性の値"
    )
    x_i_legit: Dict[int, Any] = Field(
        default_factory=dict, description="データiの正当性属性の値"
    )

    # イミュータブル設定
    model_config = {"frozen": True}

    @classmethod
    def new(
        cls,
        depth: int,
        delta: float,
        I: List[int],
        F: List[str],
        K: List[int],
        B: List[int],
        T: List[int],
        n_A: Dict[int, List[int]],
        n_C: Dict[int, dict[str, int]],
        x_i_f_value: Dict[int, Dict[str, int]],
        # 公平性制約はなくてもインスタンス化できるようにしておく
        P: Optional[List[Any]] = None,
        x_i_p: Optional[Dict[int, Any]] = None,
        x_i_y: Optional[Dict[int, int]] = None,
        L: Optional[List[Any]] = None,
        x_i_legit: Optional[Dict[int, Any]] = None,
    ):
        # Set インスタンスの作成
        data = {
            "depth": depth,
            "delta": delta,
            "I": I,
            "B": B,
            "T": T,
            "F": F,
            "K": K,
            "n_A": n_A,
            "n_C": n_C,
            "x_i_f_value": x_i_f_value,
            "P": P or [],
            "x_i_p": x_i_p or {},
            "x_i_y": x_i_y or {},
            "L": L or [],
            "x_i_legit": x_i_legit or {},
        }
        return cls(**data)


class FairOct(BaseModel):
    set: Set
    _model: Any = None
    _variables: Dict[str, Any] = {}

    def modeling(self) -> Model:
        model = Model(sense=mip.MAXIMIZE, solver_name="cbc")
        nodes = self.set.B + self.set.T

        # (a) b_{n,f}
        b = {
            (n, f): model.add_var(var_type=BINARY, name=f"b_{n}_{f}")
            for n in self.set.B
            for f in self.set.F
        }

        # (b) p_n
        p = {n: model.add_var(var_type=BINARY, name=f"p_{n}") for n in nodes}

        # (d) z_a[i,n,a(n)]
        z_a = {
            (i, n, self.set.n_A[n][-1]): model.add_var(
                var_type=BINARY, name=f"z_a_{i}_{self.set.n_A[n][-1]}_{n}"
            )
            for i in self.set.I
            for n in nodes
            if self.set.n_A[n]  # 祖先が存在
        }

        # (e) z_left, z_right
        # z_left = {
        #     (i, n, self.set.n_C[n]["left"]): model.add_var(
        #         var_type=BINARY, name=f"z_left_{i}_{n}_{self.set.n_C[n]['left']}"
        #     )
        #     for i in self.set.I
        #     for n in self.set.B
        # }
        # z_right = {
        #     (i, n, self.set.n_C[n]["right"]): model.add_var(
        #         var_type=BINARY, name=f"z_right_{i}_{n}_{self.set.n_C[n]['right']}"
        #     )
        #     for i in self.set.I
        #     for n in self.set.B
        # }

        # (f) z_t[i,n,k]
        z_t = {
            (i, n, k): model.add_var(var_type=BINARY, name=f"z_t_{i}_{n}_{k}")
            for i in self.set.I
            for n in nodes
            for k in self.set.K
        }

        # (g) ルートフロー変数 z_root[(i,0,1)]
        #     (root=0 の left子が 1 と想定。もし深さ>1ならルート子は1だけでなく...?)
        z_root = {}
        s = 0
        root_child = 1  # ルート0 の left = 1
        for i in self.set.I:
            z_root[(i, s, root_child)] = model.add_var(
                var_type=BINARY, name=f"z_root_{i}"
            )

        # (h) w_{n,k}
        w = {
            (n, k): model.add_var(var_type=BINARY, name=f"w_{n}_{k}")
            for n in nodes
            for k in self.set.K
        }

        # -----------------------------
        # 2) 制約
        # -----------------------------

        # (1b)
        for n in self.set.B:
            expr = (
                xsum(b[n, f] for f in self.set.F)
                + p[n]
                + xsum(p[m] for m in self.set.n_A.get(n, []) if m)
            )
            model.add_constr(expr == 1, name=f"Constraint_1b_{n}")

        # (1c)
        for n in self.set.T:
            expr = p[n] + xsum(p[m] for m in self.set.n_A.get(n, []) if m)
            model.add_constr(expr == 1, name=f"Constraint_1c_{n}")

        # (1d) ブランチフロー
        for n in self.set.B:
            for i in self.set.I:
                # 左+右+Σ_k z_t = z_a[i,n, parent(n)]
                expr = (
                        z_a[i, self.set.n_C[n]["left"], n]
                    + z_a[i, self.set.n_C[n]["right"], n]
                    + xsum(z_t[i, n, k] for k in self.set.K)
                )
                if self.set.n_A[n]:
                    parent_n = self.set.n_A[n][-1]
                    model.add_constr(
                        z_a[(i, n, parent_n)] == expr, name=f"Constraint_1d_{i}_{n}"
                    )

        # (1e) 葉のフロー
        for n in self.set.T:
            for i in self.set.I:
                expr = xsum(z_t[i, n, k] for k in self.set.K)
                if self.set.n_A[n]:
                    parent_n = self.set.n_A[n][-1]
                    model.add_constr(
                        z_a[(i, n, parent_n)] == expr, name=f"Constraint_1e_{i}_{n}"
                    )

        # (1f) rootフローを必ず 1 にする or データが必ず木を通る想定なら
        #      z_root[(i,0,1)] == 1
        for i in self.set.I:
            model.add_constr(
                z_root[(i, s, root_child)] <= 1, name=f"Constraint_root_flow_{i}"
            )

        # (1g)/(1h)
        for n in self.set.B:
            # left
            for i in self.set.I:
                expr_0 = xsum(
                    b[(n, f)] for f in self.set.F if self.set.x_i_f_value[i][f] == 0
                )
                model.add_constr(
                    z_a[i, self.set.n_C[n]["left"], n] <= expr_0,
                    name=f"Constraint_1g_{i}_{n}",
                )
            # right
            for i in self.set.I:
                expr_1 = xsum(
                    b[(n, f)] for f in self.set.F if self.set.x_i_f_value[i][f] == 1
                )
                model.add_constr(
                    z_a[i, self.set.n_C[n]["right"], n] <= expr_1,
                    name=f"Constraint_1h_{i}_{n}",
                )

        # (1i)
        for n in nodes:
            for i in self.set.I:
                for k in self.set.K:
                    model.add_constr(
                        z_t[(i, n, k)] <= w[(n, k)], name=f"Constraint_1i_{i}_{n}_{k}"
                    )

        # (1j)
        for n in nodes:
            expr = xsum(w[(n, k)] for k in self.set.K)
            model.add_constr(expr == p[n], name=f"Constraint_1j_{n}")

        # -----------------------------
        # 3) 目的関数 (単に正解ラベルz_tを合計)
        # -----------------------------
        model.objective = maximize(
            xsum(z_t[(i, n, self.set.x_i_y[i])] for i in self.set.I for n in nodes)
        )
        # model.objective = -xsum(z_t[(i, n, self.set.x_i_y[i])] for i in self.set.I for n in nodes)

        # 内部保存
        self._model = model
        self._variables = {
            "b": b,
            "p": p,
            "z_a": z_a,
            # "z_left": z_left,
            # "z_right": z_right,
            "z_t": z_t,
            "z_s": z_root,
            "w": w,
        }

        return model

    def _common_fair_constraint(
        self, I_p: List[int], I_pprime: List[int], name_prefix: str
    ) -> None:
        """
        共通の公平性制約（2方向の不均衡を制限する制約）を追加するヘルパーメソッド。
        引数:
          - I_p: グループ p に属するデータポイントの集合
          - I_pprime: グループ p' に属するデータポイントの集合
          - name_prefix: 制約名の接頭辞（例："StatParity_p1_p2" など）
        """
        nodes = self.set.B + self.set.T  # 全ノード
        A = len(I_p)
        B_count = len(I_pprime)
        if A == 0 or B_count == 0:
            return
        expr_p = xsum(self._variables["z_t"][(i, n, 1)] for i in I_p for n in nodes)
        expr_pprime = xsum(
            self._variables["z_t"][(i, n, 1)] for i in I_pprime for n in nodes
        )
        self._model.add_constr(
            B_count * expr_p - A * expr_pprime <= self.set.delta * A * B_count,
            name=f"{name_prefix}_1",
        )
        self._model.add_constr(
            B_count * expr_pprime - A * expr_p <= self.set.delta * A * B_count,
            name=f"{name_prefix}_2",
        )

    def _add_statistical_parity(self) -> None:
        """統計的パリティの制約を追加するメソッド"""
        for p_val in self.set.P:
            I_p = [i for i in self.set.I if self.set.x_i_p.get(i) == p_val]
            if not I_p:
                continue
            for p_prime in self.set.P:
                if p_val == p_prime:
                    continue
                I_pprime = [i for i in self.set.I if self.set.x_i_p.get(i) == p_prime]
                if not I_pprime:
                    continue
                name_prefix = f"StatParity_{p_val}_{p_prime}"
                self._common_fair_constraint(I_p, I_pprime, name_prefix)

    def _add_conditional_statistical_parity(self) -> None:
        """条件付き統計的パリティの制約を追加するメソッド"""
        for legit_val in self.set.L:
            for p_val in self.set.P:
                I_p_l = [
                    i
                    for i in self.set.I
                    if self.set.x_i_p.get(i) == p_val
                    and self.set.x_i_legit.get(i) == legit_val
                ]
                if not I_p_l:
                    continue
                for p_prime in self.set.P:
                    if p_val == p_prime:
                        continue
                    I_pprime_l = [
                        i
                        for i in self.set.I
                        if self.set.x_i_p.get(i) == p_prime
                        and self.set.x_i_legit.get(i) == legit_val
                    ]
                    if not I_pprime_l:
                        continue
                    name_prefix = f"CondStatParity_{p_val}_{p_prime}_l{legit_val}"
                    self._common_fair_constraint(I_p_l, I_pprime_l, name_prefix)

    def _add_equalized_odds(self) -> None:
        """均等化オッズの制約を追加するメソッド"""
        for k in self.set.K:
            for p_val in self.set.P:
                I_p_k = [
                    i
                    for i in self.set.I
                    if self.set.x_i_p.get(i) == p_val and self.set.x_i_y.get(i) == k
                ]
                if not I_p_k:
                    continue
                for p_prime in self.set.P:
                    if p_val == p_prime:
                        continue
                    I_pprime_k = [
                        i
                        for i in self.set.I
                        if self.set.x_i_p.get(i) == p_prime
                        and self.set.x_i_y.get(i) == k
                    ]
                    if not I_pprime_k:
                        continue
                    name_prefix = f"EqualizedOdds_{p_val}_{p_prime}_k{k}"
                    self._common_fair_constraint(I_p_k, I_pprime_k, name_prefix)

    def add_fairness_constraints(self, fairness_types: List[str]) -> None:
        """
        公平性制約の各種を追加するメソッド。
        fairness_types は以下の文字列のいずれか（複数指定可）：
          - "statistical_parity"
          - "conditional_statistical_parity"
          - "equalized_odds"
        """
        # すでにモデルが構築されていない場合は構築
        if self._model is not None:
            if "statistical_parity" in fairness_types:
                self._add_statistical_parity()
            if "conditional_statistical_parity" in fairness_types:
                self._add_conditional_statistical_parity()
            if "equalized_odds" in fairness_types:
                self._add_equalized_odds()
        else:
            raise ValueError("モデルが構築されていません。")

    def optimize(self, time_limit: int = None) -> Dict[str, Any]:
        """
        定式化されたモデルを python-mip で解き、最適解を返す。
        - time_limit: ソルバーの時間制限（秒）
        返り値は、ソルバーの状態、目的関数値、および各変数の値を含む辞書です。
        """
        if self._model is None:
            self.modeling()
        model = self._model
        model.write("fair_oct.lp")
        status = model.optimize(max_seconds=time_limit)

        # 変数の最適解を辞書形式で収集
        solution = {}
        for cat, var_dict in self._variables.items():
            solution[cat] = {key: var_dict[key].x for key in var_dict}

        return {
            "status": str(status),
            "objective": model.objective_value,
            "solution": solution,
        }

    def predict(self, X: pl.DataFrame) -> List[int]:
        """
        学習済みの決定木モデルから、入力データ X (Polars DataFrame) に対して予測を行う。
        ここでは、最適化で得られた決定変数から各ブランチノードでの分割ルールと、
        各葉ノードでのクラス割り当てを抽出し、完全二分木の構造（例: 左子 = 2*n, 右子 = 2*n+1）に基づいて予測を行う。
        """
        # まず、解の決定変数を抽出
        b = self._variables["b"]  # (n, f): value
        w = self._variables["w"]  # (n, k): value
        for key, var in self._variables["b"].items():
            print(key, var.x)
        for key, var in self._variables["w"].items():
            print(key, var.x)
        for key, var in self._variables["p"].items():
            print(key, var.x)
        for key, var in self._variables["z_a"].items():
            print(key, var.x)

        # # ツリー構造を再構築する（ここでは、完全二分木の構造を仮定）
        tree = {}
        # ブランチノード：各ノード n に対して、採用される分割特徴量を決定する
        for n in self.set.B:
            split_feature = None
            for f in self.set.F:
                # b[(n, f)] が1に近い（しきい値0.5以上）ものを選択
                if b[(n, f)].x >= 0.5:
                    split_feature = f
                    break
            tree[n] = {"split_feature": split_feature, "left": None, "right": None}
        # 葉ノード：各ノード n に対して、クラス割り当てを決定する
        for n in self.set.T:
            pred = None
            for k in self.set.K:
                if w[(n, k)].x >= 0.5:
                    pred = k
                    break
            tree[n] = {"prediction": pred}

        # 子ノードの割り当て（完全二分木と仮定：ブランチノード n の左子 = 2*n, 右子 = 2*n+1）
        all_nodes = set(self.set.B + self.set.T)
        for n in self.set.B:
            left = 2 * n
            right = 2 * n + 1
            if left in all_nodes:
                tree[n]["left"] = left
            if right in all_nodes:
                tree[n]["right"] = right

        # 予測処理：各サンプルについて、ルートから葉まで木をたどる
        predictions = []
        X_dicts = X.to_dicts()
        # ルートノードは、仮に最小のブランチノードとする（例：n = min(B)）
        root = min(self.set.B)
        for sample in X_dicts:
            current = root
            while current in self.set.B:
                split_feature = tree[current]["split_feature"]
                # サンプルの該当特徴量の値を取得（存在しない場合は 0 とする）
                value = sample.get(split_feature, 0)
                if value == 0:
                    current = tree[current].get("left")
                else:
                    current = tree[current].get("right")
                if current is None:
                    break
            # 葉ノードに到達した場合、予測値を取得
            if (
                current is not None
                and current in tree
                and "prediction" in tree[current]
            ):
                predictions.append(tree[current]["prediction"])
            else:
                predictions.append(None)
        return predictions


def fair_oct_result(data: pl.LazyFrame, data_fair: pl.DataFrame):
    compas = Cols().compas
    df_features_lazy = select_binary_features(data)
    df_features = df_features_lazy.collect()
    I = [num + 1 for num in range(len(list(df_features.rows())))]
    F = list(df_features.columns)
    B, T = create_nodes(depth=params.depth)
    # 人種
    P = data_fair.select(compas.race).unique().to_series().to_list()
    # 正当性属性の集合
    L = list(data_fair.select(compas.priors_count).unique().to_series())
    n_A = create_ancestors(B=B, T=T)
    n_C = create_children(B=B, Node=B + T)
    x_i_f_value = create_feature_mapping(df_features)
    # 各データポイントの敏感属性
    x_i_p = {i: row[compas.race] for i, row in enumerate(data_fair.to_dicts(), start=1)}
    # 各データポイントの真のラベル
    x_i_y = {
        i: row[compas.is_recid] for i, row in enumerate(data_fair.to_dicts(), start=1)
    }
    # 各データポイントの正当性属性
    x_i_legit = {
        i: row[compas.priors_count]
        for i, row in enumerate(data_fair.to_dicts(), start=1)
    }
    set_obj = Set.new(
        depth=params.depth,
        delta=params.delta,
        I=I,
        F=F,
        K=[0, 1],
        B=B,
        T=T,
        n_A=n_A,
        n_C=n_C,
        x_i_f_value=x_i_f_value,
        P=P,
        x_i_p=x_i_p,
        x_i_y=x_i_y,
        L=L,
        x_i_legit=x_i_legit,
    )
    fair_oct = FairOct(set=set_obj)
    fair_oct.modeling()
    # fair_oct.add_fairness_constraints(
    #     fairness_types=[
    #         "statistical_parity",
    #         "conditional_statistical_parity",
    #         "equalized_odds",
    #     ],
    # )
    result = fair_oct.optimize(time_limit=params.time_limit)
    print("ソルバー状態:", result["status"])
    print("目的関数値:", result["objective"])
    predictions = fair_oct.predict(df_features)
    print("予測結果:", predictions)


if __name__ == "__main__":
    df_compas_one_hot = pl.read_csv(
        "/Users/masaharu/dev/academic-research/decision-tree/fair-oct/data/compas-scores-two-years-ohe.csv"
    )
    df_fair = pl.read_csv(
        "/Users/masaharu/dev/academic-research/decision-tree/fair-oct/data/compas-scores-two-years-filtered.csv"
    )
    # (A) ダミーの One-Hot エンコード済みデータ
    # (1) 上記の小規模データを定義
    # df_compas_one_hot = pl.DataFrame(
    #     {
    #         # "race_AfricanAmerican": [1, 0, 1, 0, 1, 0],
    #         # "race_Caucasian":       [0, 1, 0, 1, 0, 1],
    #         "f1": [0, 1, 0, 1, 0, 1],
    #         "f2": [0, 0, 1, 1, 0, 0],
    #         "is_recid": [0, 1, 0, 1, 0, 1],
    #     }
    # )
    # df_fair = pl.DataFrame(
    #     {
    #         "race": [
    #             "African-American",
    #             "Caucasian",
    #             "African-American",
    #             "Caucasian",
    #             "African-American",
    #             "Caucasian",
    #         ],
    #         "priors_count": [0, 1, 2, 3, 4, 5],
    #         "is_recid": [0, 1, 0, 1, 0, 1],
    #     }
    # )

    # fair_oct_result を呼ぶ
    fair_oct_result(df_compas_one_hot.lazy(), df_fair)
    # fair_oct_result(df_compas_one_hot.lazy(), df_fair)
