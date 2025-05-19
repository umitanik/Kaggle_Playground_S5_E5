import numpy as np
from sklearn.metrics import root_mean_squared_log_error


def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_valid)
    y_pred_val = np.maximum(0, y_pred_val)
    rmsle = root_mean_squared_log_error(y_valid, y_pred_val)
    return model, y_pred_val, rmsle
