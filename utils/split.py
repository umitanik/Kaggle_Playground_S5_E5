from sklearn.model_selection import train_test_split, GroupKFold

def simple_train_test_split(df, target_col="Calories", test_size=0.2, random_state=42):

    X = df.drop([target_col, "id"], axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def group_k_fold_split(df, target_col="Calories", group_col="id", n_splits=5):

    X = df.drop([target_col, group_col], axis=1)
    y = df[target_col]
    groups = df[group_col]

    kf = GroupKFold(n_splits=n_splits)
    splits = []

    for train_idx, valid_idx in kf.split(X, y, groups):
        X_train, X_valid = X.iloc[train_idx].to_numpy(), X.iloc[valid_idx].to_numpy()
        y_train, y_valid = y.iloc[train_idx].to_numpy().reshape(-1, 1), y.iloc[valid_idx].to_numpy().reshape(-1, 1)
        splits.append((X_train, X_valid, y_train, y_valid))

    return splits