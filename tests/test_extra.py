import joblib
import pandas as pd

def test_no_negative_fares():
    df = pd.read_csv("data/train.csv")
    assert (df["fare"].dropna() >= 0).all()

def test_prediction_shape():
    model = joblib.load("model.joblib")
    test = pd.read_csv("data/test.csv")
    preds = model.predict(test.drop(columns="survived"))
    assert preds.shape == (len(test),)

def test_beats_baseline():
    # Baseline accuracy = share of the most common class in test["survived"]
    model = joblib.load("model.joblib")
    test = pd.read_csv("data/test.csv")
    baseline = test["survived"].value_counts(normalize=True).max()
    acc = (model.predict(test.drop(columns="survived")) == test["survived"]).mean()
    assert acc > baseline
