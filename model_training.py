import dask.dataframe as dd

def optimize_data_types(df):
    """Оптимизация типов данных в DataFrame."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')  # Пример замены int64 на int32
    return df

def prepare_data(train_data, bus_data):
    """Подготовка данных для обучения модели."""
    # Оптимизация типов данных
    train_data = optimize_data_types(train_data)
    bus_data = optimize_data_types(bus_data)

    # Проверка на дубликаты
    train_data = train_data.drop_duplicates()
    bus_data = bus_data.drop_duplicates()

    # Объединение обучающего набора данных с данными о маршрутах
    combined_train_data = train_data.merge(bus_data[['bus_route_id']], how='left', on='bus_route_id', indicator=True)
    return combined_train_data
