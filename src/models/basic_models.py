from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import root_mean_squared_log_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


def evaluate_basic_models(X_train, y_train, X_test, y_test, test, random_state=42):
    models = {
         "LinearRegression": LinearRegression(),
         "Ridge": Ridge(),
         "Lasso": Lasso(),
         "ElasticNet": ElasticNet(),
         "DecisionTree": DecisionTreeRegressor(random_state=random_state),
         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
         "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
         "KNeighbors": KNeighborsRegressor()
    }

    results = []
    predictions_dict = {}
    test_ids = test["id"]

    test_features = test.drop("id", axis=1)

    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(0, y_pred)

        rmsle = root_mean_squared_log_error(y_test, y_pred)
        print(f"  RMSLE: {rmsle:.5f}")
        results.append((name, rmsle))

        test_pred = model.predict(test_features.values)
        test_pred = np.maximum(0, test_pred)
        test_pred = test_pred.flatten() if test_pred.ndim > 1 else test_pred
        predictions_dict[name] = test_pred

        submission = pd.DataFrame({
            "id": test_ids,
            "Calories": test_pred
        })
        submission.to_csv(f"submission/submission_{name}.csv", index=False)

    return results, predictions_dict