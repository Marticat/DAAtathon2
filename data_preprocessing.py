import dask.dataframe as dd
import missingno as msno

def load_data(train_path, test_path, bus_path):
    train_data = dd.read_csv(train_path)
    test_data = dd.read_csv(test_path)
    bus_data = dd.read_csv(bus_path)
    return train_data, test_data, bus_data

def optimize_data_types(train_data, test_data, bus_data):
    train_data['bus_route_id'] = train_data['bus_route_id'].astype('int32')
    test_data['bus_route_id'] = test_data['bus_route_id'].astype('int32')
    bus_data['bus_route_id'] = bus_data['bus_route_id'].astype('int32')
    return train_data, test_data, bus_data

def fill_missing_values(train_data, test_data):
    for column in train_data.columns:
        if train_data[column].isnull().sum().compute() > 0:
            if train_data[column].dtype == 'object':
                train_data[column].fillna(train_data[column].mode().compute()[0], inplace=True)
                test_data[column].fillna(test_data[column].mode().compute()[0], inplace=True)
            else:
                train_data[column].fillna(train_data[column].median().compute(), inplace=True)
                test_data[column].fillna(test_data[column].median().compute(), inplace=True)

    return train_data, test_data

def visualize_missing_values(train_data):
    msno.matrix(train_data.compute())
