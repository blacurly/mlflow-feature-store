from feature_engineering import create_feature_v1, create_feature_v2
from train import train_model
import pandas as pd

df = pd.read_csv("data/athletes.csv")

feature_versions = {
    "v1": create_feature_v1(df.copy()),
    "v2": create_feature_v2(df.copy())
}

param_sets = [
    {"max_depth": 5, "n_estimators": 100},
    {"max_depth": 10, "n_estimators": 200}
]

for version, data in feature_versions.items():
    for params in param_sets:
        train_model(data, version, **params)

