import os
import sys

import pytest
import polars as pl
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src/"))
from apps.model.fair_oct import Set, FairOct
from modules.prep import select_binary_features, create_date_point_index_and_features, create_nodes, \
    create_sensitive_features_and_not_sensitive_features, create_ancestors, create_children, create_feature_mapping, \
    create_sensitive_and_no_sensitive_mapping, create_true_labels
from utils.params import params


@pytest.fixture
def create_raw_one_hot_data():
    return pl.DataFrame(
        {
            "age_cat": [
                "25 - 45",
                "Greater than 45",
                "Less than 25",
                "25 - 45",
                "Greater than 45",
                "Less than 25",
            ],
            "c_jail_in": [
                "2013-08-14 06:03:00",
                "2013-01-26 03:45:00",
                "2013-04-13 04:58:00",
                "2013-08-14 06:03:00",
                "2013-01-26 03:45:00",
                "2013-04-13 04:58:00",
            ],
            "c_jail_out": [
                "2015-08-14 06:03:00",
                "2014-01-26 03:45:00",
                "2017-04-13 04:58:00",
                "2016-08-14 06:03:00",
                "2015-01-26 03:45:00",
                "2014-04-13 04:58:00",
            ],
            "is_recid": [0, 1, 0, 1, 0, 1],
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [0, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture
def create_one_hot_data():
    return pl.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [0, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture
def create_data_include_sensitive_data():
    return pl.DataFrame(
        {
            "race": [
                "African-American",
                "Caucasian",
                "African-American",
                "Caucasian",
                "African-American",
                "Caucasian",
            ],
            "priors_count": [0, 1, 2, 3, 4, 5],
            "is_recid": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def fair_oct_instance_without_fairness_constraint(create_one_hot_data, create_data_include_sensitive_data):
    data = create_one_hot_data.lazy()
    data_fair = create_data_include_sensitive_data
    df_one_hot_feature_lazy = select_binary_features(data)
    df_one_hot_feature_binary = df_one_hot_feature_lazy.collect()
    I, F = create_date_point_index_and_features(data=df_one_hot_feature_binary)
    B, T = create_nodes(depth=params.depth)
    # Pはセンシティブ属性の集合、Lは正当性属性の集合
    # 例: P: 人種, L: 前科の回数
    P, L = create_sensitive_features_and_not_sensitive_features(data_fair)
    n_A = create_ancestors(B=B, T=T)
    n_C = create_children(B=B, Node=B + T)
    x_i_f_value = create_feature_mapping(df_one_hot_feature_binary)
    # 各データポイントのセンシティブ属性
    # 各データポイントの正当性属性
    x_i_p, x_i_legit = create_sensitive_and_no_sensitive_mapping(data_fair)
    # 各データポイントの真のラベル
    x_i_y = create_true_labels(data_fair)
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
    return fair_oct.optimize(time_limit=params.time_limit), fair_oct