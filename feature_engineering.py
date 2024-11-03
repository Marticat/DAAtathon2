import dask.dataframe as dd
import pandas as pd

def generate_features(train_data, test_data):
    train_data['day_of_week'] = pd.to_datetime(train_data['date'].compute()).dt.dayofweek
    test_data['day_of_week'] = pd.to_datetime(test_data['date'].compute()).dt.dayofweek
    train_data['month'] = pd.to_datetime(train_data['date'].compute()).dt.month
    test_data['month'] = pd.to_datetime(test_data['date'].compute()).dt.month
    train_data['is_weekend'] = train_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0, meta=('x', 'int8'))
    test_data['is_weekend'] = test_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0, meta=('x', 'int8'))
    return train_data, test_data
