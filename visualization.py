import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_matrix(train_data):
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(14, 10))
    sns.heatmap(train_data[numeric_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def plot_predictions(y_val, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_val['6~7_ride'].values, label='Фактические 6~7_ride', marker='o')
    plt.plot(y_pred[:, 0], label='Предсказанные 6~7_ride', marker='x')
    plt.plot(y_val['7~8_ride'].values, label='Фактические 7~8_ride', marker='o')
    plt.plot(y_pred[:, 1], label='Предсказанные 7~8_ride', marker='x')
    plt.title('Фактические и Предсказанные Значения')
    plt.xlabel('Индекс')
    plt.ylabel('Количество пассажиров')
    plt.legend()
    plt.grid()
    plt.show()
