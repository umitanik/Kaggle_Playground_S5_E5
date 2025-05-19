import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import root_mean_squared_log_error
import torch

def train_tabnet(X_train, y_train, X_valid, y_valid, test_df, id_col="id", target_col="Calories", seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_features = test_df.drop(id_col, axis=1).values.astype(np.float32)
    test_ids = test_df[id_col]

    model = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3,
        gamma=1.3, lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.1),
        mask_type='entmax',
        device_name=device,
        seed=seed
    )

    model.fit(
        X_train.astype(np.float32), y_train,
        eval_set=[(X_valid.astype(np.float32), y_valid)],
        max_epochs=3,
        patience=20,
        batch_size=1024,
        drop_last=False,
        num_workers=0
    )

    y_pred_val = model.predict(X_valid.astype(np.float32))
    y_pred_val = np.maximum(0, y_pred_val)
    val_rmsle = root_mean_squared_log_error(y_valid, y_pred_val)
    print(f"Validation RMSLE: {val_rmsle:.5f}")

    test_preds = model.predict(test_features)
    test_preds = np.maximum(0, test_preds)

    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: test_preds.flatten()
    })
    submission.to_csv("submission/submission_TabNet.csv", index=False)
    print("Tahminler 'submission_TabNet.csv' olarak kaydedildi.")

    return model, val_rmsle, submission
