import mlflow
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data, version, max_depth, n_estimators):
    mlflow.set_experiment("Athletes_Prediction")

    X = data.drop(columns=["total_lift"])
    y = data["total_lift"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tracker = EmissionsTracker(output_dir="codecarbon_logs")
    tracker.start()

    with mlflow.start_run(run_name=f"{version}_d{max_depth}_n{n_estimators}"):
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        from math import sqrt
        mse = mean_squared_error(y_test, preds)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, preds)

        mlflow.log_params({
            "version": version,
            "max_depth": max_depth,
            "n_estimators": n_estimators
        })
        mlflow.log_metrics({"rmse": rmse, "r2": r2})

        emissions = tracker.stop()
        mlflow.log_metric("emissions", emissions)

        mlflow.sklearn.log_model(model, "model")

