from sklearn.linear_model import LogisticRegression
import clean_data_exploration as clean_data
import numpy as np
import pickle


def sigmoid(x):
    """
    Apply the sigmoid activation to a numpy matrix `x` of any shape.
    """
    return 1 / (1 + np.exp(-x))


def pred(w, X):
    """
    Compute the prediction made by a logistic regression model with weights `w`
    on the data set with input data matrix `X`. Recall that N is the number of
    samples and D is the number of features. The +1 accounts for the bias term.

    Parameters:
        `w` - a numpy array of shape (D+1)
        `X` - data matrix of shape (N, D+1)

    Returns: Prediction vector `y` of shape (N). Each value in `y` should be betwen 0 and 1.
    """
    return sigmoid(np.dot(X, w))


def label_encode(labels):
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = [label_map[label] for label in labels]
    return encoded_labels


def label_decode(labels):
    label_map = {0: 'Dubai', 1: 'New York City', 2: 'Paris', 3: 'Rio de Janeiro'}
    decoded_labels = [label_map[label] for label in labels]
    return decoded_labels


if __name__ == "__main__":
    # --------------------------------- USING SKLEARN ----------------------------------------
    #
    test_set, validation_set, training_set = clean_data.split_data(0.2, 0)

    # Assuming the last column is the target variable
    y_train = label_encode(training_set['Label'])  # Target variable
    X_train = training_set.drop(columns=['Label'])
    y_test = label_encode(test_set['Label'])  # Target variable
    X_test = test_set.drop(columns=['Label'])

    # Initialize the logistic regression model
    logistic_reg = LogisticRegression()

    # Fit the model on the training data
    logistic_reg.fit(X_train, y_train)

    # Get the weights after fitting
    weights = logistic_reg.coef_
    bias_term = logistic_reg.intercept_

    # Predict on the testing data
    y_pred = logistic_reg.predict(X_test)

    # Evaluate the model
    accuracy = logistic_reg.score(X_test, y_test)
    print("Accuracy:", accuracy)  # 93% is the best possible

    linear_combination = np.dot(X_test, weights.T) + bias_term
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y_pred1 = np.argmax(probabilities, axis=1)

    count = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            count += 1
    print(count / len(y_test))

    count = 0
    for i in range(len(y_test)):
        if y_pred1[i] == y_test[i]:
            count += 1
    print(count / len(y_test))

    # with open('weights.pkl', 'wb') as f:
    #     pickle.dump(weights.T, f)
    #
    # with open('bias.pkl', 'wb') as f:
    #     pickle.dump(bias_term, f)

    # --------------------------------- WITHOUT SKLEARN ----------------------------------------

    # Encoding the label column into numerical values
    y_train = training_set['Label']
    X_train = training_set.drop(columns=['Label'])
    y_train_encoded = label_encode(y_train)

    y_valid = validation_set['Label']
    X_valid = validation_set.drop(columns=['Label'])
    y_valid_encoded = label_encode(y_valid)

    y_test = test_set['Label']
    X_test = test_set.drop(columns=['Label'])
    y_test_encoded = label_encode(y_test)

    # W = train(X_train, y_train_encoded,X_valid,y_valid_encoded, learning_rate=0.01, num_epochs=2000)
    predictions = np.argmax(X_test.dot(weights.T), axis=1)

    count = 0
    for i in range(len(y_test_encoded)):
        if predictions[i] == y_test_encoded[i]:
            count += 1
    print(count / len(y_test_encoded))
