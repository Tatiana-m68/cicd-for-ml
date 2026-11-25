import joblib
import pandas as pd

ACCURACY_BAR = 0.75   # example bar — set the one your project needs

def test_accuracy_above_bar():
    model = joblib.load("model.joblib")
    test = pd.read_csv("data/test.csv")
    X = test.drop(columns="survived")
    acc = (model.predict(X) == test["survived"]).mean()
    assert acc >= ACCURACY_BAR, f"accuracy {acc:.3f} is below the {ACCURACY_BAR} bar"

def test_accepts_raw_passenger():
    model = joblib.load("model.joblib")
    row = pd.DataFrame([{"pclass": 3, "sex": "male", "age": None, "fare": 7.25}])
    assert model.predict(row)[0] in (0, 1)
