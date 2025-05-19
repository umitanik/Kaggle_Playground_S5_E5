from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standard_scale(train_data, valid_data):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    valid_scaled = scaler.transform(valid_data)
    return train_scaled, valid_scaled


def minmax_scale(train_data, valid_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    return train_scaled, valid_data
