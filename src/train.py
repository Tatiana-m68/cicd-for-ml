"""Train the model and save it as model.joblib."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["pclass", "sex", "age", "fare"]
TARGET = "survived"

train = pd.read_csv("data/train.csv")
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), ["pclass", "age", "fare"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
])
model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
])
model.fit(train[FEATURES], train[TARGET])
joblib.dump(model, "model.joblib")
print("model.joblib saved")
