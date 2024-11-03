import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_data, optimize_data_types, fill_missing_values, visualize_missing_values
from feature_engineering import generate_features
from model_training import prepare_data
from visualization import plot_correlation_matrix, plot_predictions

def main():
    # Load datasets using Dask
    train_data, test_data, bus_data = load_data('train.csv', 'test.csv', 'bus_bts.csv')

    # Reset index for Dask DataFrame
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Optimize data types
    train_data, test_data, bus_data = optimize_data_types(train_data, test_data, bus_data)

    # Fill missing values
    train_data, test_data = fill_missing_values(train_data, test_data)

    # Generate new features
    train_data, test_data = generate_features(train_data, test_data)

    # Prepare data
    combined_train_data = prepare_data(train_data, bus_data)

    # Split back into training and test sets
    train_data = combined_train_data[combined_train_data['_merge'] == 'both'].drop(columns=['_merge'])
    test_data = combined_train_data[combined_train_data['_merge'] == 'right_only'].drop(columns=['_merge'])

    # Prepare for modeling
    X = train_data[['day_of_week', 'month', 'is_weekend', 'bus_route_id']]  # Features
    y = train_data[['6~7_ride', '7~8_ride']]  # Target variable

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X.compute(), y.compute(), test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate results
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

    # Predict on test data
    X_test = test_data[['day_of_week', 'month', 'is_weekend', 'bus_route_id']]  # Test features
    predictions = model.predict(X_test.compute())

    # Visualize results
    plot_correlation_matrix(train_data)
    plot_predictions(y_val, y_pred)

if __name__ == "__main__":
    main()
