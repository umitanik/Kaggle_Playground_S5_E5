from src.feature_engineering.feature_engineering import *
from src.models.XGBoost_with_Optuna import xgboost_with_optuna
from src.models.CATBoost_with_Optuna import catboost_with_optuna
from src.models.basic_models import evaluate_basic_models
from src.models.tabnet_model import train_tabnet
from utils.config import *
from utils.io import *
from utils.preprocessing import *
from utils.scaler import *
from utils.split import *

if __name__ == '__main__':
    set_options()
    train, test, sample_submission = load_data()
    print(train.head())
    train = get_dummies(train)
    test = get_dummies(test)
    train = engineer_features(train)
    test = engineer_features(test)
    splits = group_k_fold_split(train, target_col="Calories", group_col="id", n_splits=5)

    for fold, (X_train, X_valid, y_train, y_valid) in enumerate(splits):
        print(f"Fold {fold + 1}")

        X_scaled_train, X_scaled_valid = standard_scale(X_train, X_valid)

        results, predictions = evaluate_basic_models(X_scaled_train, y_train, X_scaled_valid, y_valid, test)

        model_xgb, score_xgb = xgboost_with_optuna(X_scaled_train, y_train, X_scaled_valid, y_valid, test, n_trials=3, random_state=42)
        model_cat, score_cat = catboost_with_optuna(X_scaled_train, y_train, X_scaled_valid, y_valid, test, n_trials=3, random_state=42)
        model_tabnet, score_tabnet, _ = train_tabnet(X_scaled_train, y_train, X_scaled_valid, y_valid, test, id_col="id", target_col="Calories", n_splits=5, seed=42)