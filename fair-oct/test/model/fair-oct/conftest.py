import pytest
import polars as pl


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
