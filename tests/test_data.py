import pandas as pd

def test_no_missing_target():
    df = pd.read_csv("data/train.csv")
    assert df["survived"].notna().all()

def test_expected_columns():
    df = pd.read_csv("data/train.csv")
    needed = {"pclass", "sex", "age", "fare", "survived"}
    assert needed <= set(df.columns)

def test_target_is_binary():
    df = pd.read_csv("data/train.csv")
    assert set(df["survived"].unique()) <= {0, 1}
