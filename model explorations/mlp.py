import numpy as np
import clean_data_exploration as clean_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


def get_input_targets(data):
    # Extract the target vector
    t = np.array(data['Label'])
    city_map = {
        'Dubai': 0,
        'Rio de Janeiro': 1,
        'New York City': 2,
        'Paris': 3
    }
    t = np.array([city_map[label] for label in t]).reshape(-1, 1)

    # Extract the feature matrix
    X_fets = np.array(data[['Q1', 'Q2', 'Q3', 'Q4', 'Friends', 'Co-worker', 'Siblings', 'Partner', 'Q7', 'Q8', 'Q9',
                            'Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']])
    X_fets[:, 4:8] = X_fets[:, 4:8].astype(int)

    # Add a column of ones for bias
    n = len(data)
    one_col = np.ones((n, 1), dtype=int)
    X = np.concatenate((X_fets, one_col), axis=1)
    return X, t


def predict_all(filename: str):
    train_data, validation_data, test_data = clean_data.split_data(validation_size=0)

    X_train, t_train = get_input_targets(train_data)
    X_test, t_test = get_input_targets(test_data)

    # Solve for weights using MLP
    mlp = MLPClassifier(hidden_layer_sizes=(300, 200, 100), alpha=5, max_iter=1000)
    mlp.fit(X_train, t_train.flatten())

    # Make predictions
    prediction = mlp.predict(X_test)

    # Calculate MSE and accuracy
    mse_value = mean_squared_error(t_test.flatten(), prediction)
    accuracy = accuracy_score(t_test.flatten(), prediction)

    print('Prediction:', prediction)
    print('MSE:', mse_value)
    print('Accuracy:', accuracy)

    return


if __name__ == '__main__':
    predict_all('data/clean_dataset.csv')
