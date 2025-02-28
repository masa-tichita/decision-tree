import os
import sys

from polars.testing import assert_frame_equal

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src/"))
from modules.prep import (
    select_binary_features,
    create_nodes,
    create_date_point_index_and_features,
    create_sensitive_features_and_not_sensitive_features,
    create_ancestors,
    create_children,
    create_feature_mapping,
    create_sensitive_and_no_sensitive_mapping,
    create_true_labels,
)


# 正常系テスト
def test_select_binary_features(create_raw_one_hot_data, create_one_hot_data):
    data = create_raw_one_hot_data.lazy()
    df_correct = create_one_hot_data
    df_result = select_binary_features(data).collect()
    assert_frame_equal(df_result, df_correct)


# 正常系テスト
def test_create_date_point_index_and_features(create_one_hot_data):
    data = create_one_hot_data
    index, features = create_date_point_index_and_features(data)
    assert index == [1, 2, 3, 4, 5, 6, 7, 8]
    assert features == ["f1", "f2", "two_year_recid"]


# 正常系テスト
def test_create_nodes():
    depth = 2
    B, T = create_nodes(depth)
    assert B == [1, 2, 3]
    assert T == [4, 5, 6, 7]


# 正常系テスト
def test_create_sensitive_features_and_not_sensitive_features(
    create_data_include_sensitive_data,
):
    data = create_data_include_sensitive_data
    P, L = create_sensitive_features_and_not_sensitive_features(data)
    print(P)
    assert P == [
        "African-American",
        "Caucasian",
    ]
    assert L == [0, 1]


def test_create_ancestors():
    B = [1, 2, 3]
    T = [4, 5, 6, 7]
    ancestors = create_ancestors(B, T)
    assert ancestors == {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [1, 2],
        5: [1, 2],
        6: [1, 3],
        7: [1, 3],
    }


def test_create_children():
    B = [1, 2, 3]
    T = [4, 5, 6, 7]
    children = create_children(B, T)
    assert children == {
        0: {"left": 1},
        1: {"left": 2, "right": 3},
        2: {"left": 4, "right": 5},
        3: {"left": 6, "right": 7},
    }


def test_create_feature_mapping(create_one_hot_data):
    df = create_one_hot_data
    feature_mapping = create_feature_mapping(df)
    assert feature_mapping == {
        1: {"f1": 0, "f2": 0, "two_year_recid": 0},
        2: {"f1": 1, "f2": 0, "two_year_recid": 0},
        3: {"f1": 0, "f2": 1, "two_year_recid": 1},
        4: {"f1": 1, "f2": 1, "two_year_recid": 1},
        5: {"f1": 0, "f2": 0, "two_year_recid": 0},
        6: {"f1": 1, "f2": 0, "two_year_recid": 0},
        7: {"f1": 0, "f2": 1, "two_year_recid": 1},
        8: {"f1": 1, "f2": 1, "two_year_recid": 1},
    }


def test_create_sensitive_and_no_sensitive_mapping(create_data_include_sensitive_data):
    df = create_data_include_sensitive_data
    sensitive_mapping, no_sensitive_mapping = create_sensitive_and_no_sensitive_mapping(
        df
    )
    assert sensitive_mapping == {
        1: "African-American",
        2: "African-American",
        3: "African-American",
        4: "African-American",
        5: "Caucasian",
        6: "Caucasian",
        7: "Caucasian",
        8: "Caucasian",
    }
    assert no_sensitive_mapping == {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}


def test_create_true_labels(create_data_include_sensitive_data):
    df = create_data_include_sensitive_data
    true_labels = create_true_labels(df)
    assert true_labels == {
        1: 0,
        2: 1,
        3: 0,
        4: 1,
        5: 0,
        6: 1,
        7: 0,
        8: 1,
    }
