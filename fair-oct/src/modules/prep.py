import os
import sys
from typing import Dict, List

import polars as pl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.const import cols

compas = cols.compas


def select_binary_features(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    return lazy_frame.select(
        pl.exclude(
            [
                compas.age_cat,
                compas.c_jail_in,
                compas.c_jail_out,
                compas.is_recid,
                compas.race,
            ]
        )
    )


def get_predicted_value(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    return lazy_frame.select(pl.col(compas.is_recid))


def create_nodes(depth: int) -> tuple[List[int], List[int]]:
    # B: ブランチノード (1 から 2^depth - 1)
    B = list(range(1, int(pow(2, depth))))

    # T: 葉ノード (2^depth から 2^(depth + 1) - 1)
    T = list(range(int(pow(2, depth)), int(pow(2, depth + 1))))
    return B, T


def create_ancestors(B: List[int], T: List[int]) -> Dict[int, List[int]]:
    """
    A(n) を生成する関数。B（ブランチノード）と T（葉ノード）から、
    それぞれのノード n に対する祖先集合 A(n) を返す。

    Args:
        B: ブランチノードのリスト
        T: 葉ノードのリスト
    Returns:
        A: 各ノード n に対する祖先の辞書
    """
    nodes = B + T
    A = {0: []}
    for n in nodes:
        ancestors = []
        parent = n // 2  # 親ノードを計算
        if n == 1:
            A[n] = [0]
        else:
            # ルートノードに達するまで祖先を辿る
            while parent >= 1:
                ancestors.append(parent)
                parent = parent // 2  # さらに上の祖先へ

            A[n] = ancestors[::-1]  # 上位の祖先から順に並べる

    return A


def create_children(B: List[int], Node: List[int]) -> Dict[int, dict[str, int]]:
    """
    C(n) を生成する関数。B（ブランチノード）と T（葉ノード）から、
    それぞれのノード n に対する子ノード集合 C(n) を返す。

    Args:
        B: ブランチノードのリスト
        T: 葉ノードのリスト
    Returns:
        C: 各ノード n に対する子ノードの辞書
    """
    C = {0: {"left": 1}}
    for n in B:
        children = {}
        left_child = 2 * n  # 左の子ノード
        right_child = 2 * n + 1  # 右の子ノード

        # 左の子ノードが範囲内にあれば追加
        if left_child <= Node[-1]:
            children["left"] = left_child

        # 右の子ノードが範囲内にあれば追加
        if right_child <= Node[-1]:
            children["right"] = right_child

        C[n] = children
    return C


def create_feature_mapping(df: pl.DataFrame) -> Dict[int, Dict[str, int]]:
    """
    データポイント i と特徴量 f の値を格納する辞書を作成する関数
    Args:
        df: Polars のデータフレーム（バイナリ特徴量のみ）

    Returns:
        x: {データポイント i: {特徴量 f: 値}} というネストされた辞書
    """
    x = {}
    df_rows = df.to_dicts()  # 各行を辞書形式に変換

    for idx, row in enumerate(df_rows, start=1):
        # 各データポイント i に対して特徴量とその値を格納
        x[idx] = {feature: value for feature, value in row.items()}

    return x


if __name__ == "__main__":
    # compas_data = pl.read_csv(
    #     "/Users/masaharu/dev/academic-research/decision-tree/fair-oct/data/compas-scores-two-years-ohe.csv"
    # )
    # print(compas_data)
    # df_compas_lazy = compas_data.lazy()
    # print(select_binary_features(df_compas_lazy).collect())
    # print(get_predicted_value(df_compas_lazy).collect())
    print(create_nodes(3))
    B, T = create_nodes(3)
    Node = B + T
    print(create_ancestors(B, T))
    print(create_children(B, Node))
