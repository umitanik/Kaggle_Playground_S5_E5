import pandas as pd

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    sample_submission_data = pd.read_csv('data/sample_submission.csv')
    return train_data, test_data, sample_submission_data
