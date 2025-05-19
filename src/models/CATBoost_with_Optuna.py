import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error

from utils.training import evaluate_model

def catboost_with_optuna(X_train, y_train, X_valid, y_valid, test, n_trials=5, random_state=42):
    test_ids = test["id"]
    test_features = test.drop("id", axis=1)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.3),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "task_type": "GPU",
            "devices": "0",
            "verbose": False,
            "random_seed": random_state,
        }
        model = CatBoostRegressor(**params, train_dir=None)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=False, verbose=False)
        y_pred = np.maximum(0, model.predict(X_valid))
        return root_mean_squared_log_error(y_valid, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params | {
        "task_type": "GPU",
        "devices": "0",
        "verbose": False,
        "random_seed": random_state
    }

    final_model = CatBoostRegressor(**best_params)
    final_model, y_pred_val, val_rmsle = evaluate_model(final_model, X_train, y_train, X_valid, y_valid)

    test_pred = np.maximum(0, final_model.predict(test_features))
    pd.DataFrame({"id": test_ids, "Calories": test_pred}).to_csv("submission/submission_CatBoost_Optuna.csv", index=False)

    print(f"Validation RMSLE (CatBoost): {val_rmsle:.5f}")
    return final_model, val_rmsle
