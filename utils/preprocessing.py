import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_dummies(data, drop_first=True):
    return pd.get_dummies(data, drop_first=drop_first)

def correlation_analysis(data, threshold=0.95, show_plot=True):
    corr_matrix = data.corr()

    if show_plot:
        print("Correlation Matrix:")
        plt.figure(figsize=(15, 15))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="cool warm", square=True, linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        print(f"⚠️ Yüksek korelasyon (> {threshold}) nedeniyle düşürülmesi önerilen sütunlar: {to_drop}")
    else:
        print("✅ Yüksek korelasyonlu sütun bulunamadı.")

    return to_drop