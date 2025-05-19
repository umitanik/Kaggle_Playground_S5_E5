import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from xgboost import XGBRegressor
from utils.training import evaluate_model

def xgboost_with_optuna(X_train, y_train, X_valid, y_valid, test, n_trials=5, random_state=42):
    test_ids = test["id"]
    test_features = test.drop("id", axis=1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "tree_method": "hist",
            "device": "cuda",
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        y_pred = model.predict(X_valid)
        y_pred = np.maximum(0, y_pred)
        return root_mean_squared_log_error(y_valid, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params | {
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "gpu_hist"
    }

    final_model = XGBRegressor(**best_params)
    final_model, y_pred_val, val_rmsle = evaluate_model(final_model, X_train, y_train, X_valid, y_valid)

    test_pred = np.maximum(0, final_model.predict(test_features))
    pd.DataFrame({"id": test_ids, "Calories": test_pred}).to_csv("submission/submission_XGBoost_Optuna.csv", index=False)

    print(f"Validation RMSLE (XGBoost): {val_rmsle:.5f}")
    return final_model, val_rmsle


