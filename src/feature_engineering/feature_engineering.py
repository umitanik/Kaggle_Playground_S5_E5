import numpy as np
import pandas as pd

def engineer_features(data, eps=1e-6):

    data["BMI"] = data["Weight"] / (data["Height"] / 100) ** 2
    data["Work_per_kg"] = (data["Duration"] * data["Heart_Rate"]) / (data["Weight"] + eps)
    hr_max = 220 - data["Age"]
    data["HR_pctMax"] = data["Heart_Rate"] / (hr_max + eps)
    data["log_heart_rate_duration"] = np.log10(data["Heart_Rate"] + eps) / (data["Duration"] + eps)
    data["age_vs_weight"] = data["Age"] / (data["Weight"] + eps)
    data["age_vs_duration"] = data["Age"] / (data["Duration"] + eps)
    data["weight_vs_heart_rate"] = data["Weight"] / (data["Heart_Rate"] + eps)

    data['age_group'] = pd.cut(data['Age'], bins=[18, 25, 40, 60, np.inf], labels=[1, 2, 3, 4], include_lowest=True).astype(int)

    data['bmi_cat'] = np.select(
        [data['BMI'] < 18.5, data['BMI'] < 25, data['BMI'] < 30, data['BMI'] >= 30],
        [0, 1, 2, 3]
    )

    return data
