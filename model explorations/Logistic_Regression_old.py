import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import clean_data
import numpy as np
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def pred(X, w):
    """
    #     Compute the prediction made by a logistic regression model with weights `w`
    #     on the data set with input data matrix `X`. Recall that N is the number of
    #     samples and D is the number of features. The +1 accounts for the bias term.
    #
    #     Parameters:
    #         `w` - a numpy array of shape (D+1)
    #         `X` - data matrix of shape (N, D+1)
    #
    #     Returns: Prediction vector `y` of shape (N). Each value in `y` should be betwen 0 and 1.
    #     """
    return softmax(np.dot(X, w))

def loss(X, y, w):
    """
    #     Compute the average cross-entropy loss of a logistic regression model
    #     with weights `w` on the data set with input data matrix `X` and
    #     targets `t`. Please use the function `np.logaddexp` for numerical
    #     stability.
    #
    #     Parameters:
    #         `w` - a numpy array of shape (D+1)
    #         `X` - data matrix of shape (N, D+1)
    #         `t` - target vector of shape (N)
    #
    #     Returns: a scalar cross entropy loss value, computed using the numerically
    #              stable np.logaddexp function.
    #     """
    N = X.shape[0]
    probs = pred(X, w)
    correct_probs = -np.log(probs[np.arange(N), y])
    return np.mean(correct_probs)

def gradient(X, y, w):
    '''
    #     Return the gradient of the cost function at `w`. The cost function
    #     is the average cross-entropy loss across the data set `X` and the
    #     target vector `y`.
    #
    #     Parameters:
    #         `w` - a current "guess" of what our weights should be,
    #                    a numpy array of shape (D+1)
    #         `X` - matrix of shape (N,D+1) of input features
    #         `y` - target y values of shape (N)
    #
    #     Returns: gradient vector of shape (D+1)
    #     '''
    N = X.shape[0]
    probs = pred(X, w)
    probs[np.arange(N), y] -= 1
    return np.dot(X.T, probs) / N

def train(X_train, y_train, X_valid, y_valid, learning_rate=0.01, num_epochs=1000, regularization_strength=0.1):
    '''
    #            Given `alpha` - the learning rate
    #           `num_epochs` - the number of iterations of gradient descent to run
    #           `X_train` - the data matrix to use for training
    #           `t_train` - the target vector to use for training
    #           `X_valid` - the data matrix to use for validation
    #           `t_valid` - the target vector to use for validation
    #
    #     Solves for logistic regression weights via full batch gradient descent.
    #     Return weights after `niter` iterations.
    '''
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    w = np.zeros((num_features, num_classes))

    best_valid_loss = float('inf')
    best_W = None

    for epoch in range(num_epochs):
        grad = gradient(X_train, y_train, w) + regularization_strength * w
        w -= learning_rate * grad

        if epoch % 100 == 0:
            train_loss = loss(X_train, y_train, w)
            valid_loss = loss(X_valid, y_valid, w)
            print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {valid_loss}")

            # Check if validation loss improved, if so, update best W
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_W = w.copy()

    return best_W
def accuracy1(w, X, t, thres=0.5):
    """
    Compute the accuracy of a logistic regression model with weights `w`
    on the data set with input data matrix `X` and targets `t`

    If the logistic regression model prediction is y >= thres, then
    predict that t = 1. Otherwise, predict t = 0.
    (Note that this is an arbitrary decision that we are making, and
    it makes virtually no difference if we decide to predict t = 0 if
    y == thres exactly, since the chance of y == thres is highly
    improbable.)

    Parameters:
        `w` - a numpy array of shape (D+1)
        `X` - data matrix of shape (N, D+1)
        `t` - target vector of shape (N)
        `thres` - a value between 0 and 1

    Returns: accuracy value, between 0 and 1
    """
    y = pred(w, X)
    predictions = (y >= thres).astype(int)
    return np.mean(predictions == t)
def label_encode(labels):
    unique_labels = sorted(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = [label_map[label] for label in labels]
    return encoded_labels

if __name__ == "__main__":
# --------------------------------- USING SKLEARN ----------------------------------------
#
#     # Load the data from CSV file into a DataFrame
#     test_set, validation_set, training_set = clean_data.split_data(0.2,0)
#
#     # Assuming the last column is the target variable
#     y_train = training_set['Label']                  # Target variable
#     X_train = training_set.drop(columns=['Label'])
#     y_test = test_set['Label']  # Target variable
#     X_test = test_set.drop(columns=['Label'])
#
#
#     # Initialize the logistic regression model
#     logistic_reg = LogisticRegression(max_iter=1000)
#
#     # Fit the model on the training data
#     logistic_reg.fit(X_train, y_train)
#
#     # Predict on the testing data
#     y_pred = logistic_reg.predict(X_test)
#
#     # Evaluate the model
#     accuracy = logistic_reg.score(X_test, y_test)
#     print("Accuracy:", accuracy) #93% is the best possible
#     from sklearn.linear_model import LogisticRegression
#
#     # Assuming the last column is the target variable
#     y_train = training_set['Label']  # Target variable
#     y_train_encoded = label_encode(y_train)
#     X_train = training_set.drop(columns=['Label'])
#     y_test = test_set['Label']  # Target variable
#     X_test = test_set.drop(columns=['Label'])
#     y_test_encoded = label_encode(y_test)
#
#     # Initialize the logistic regression model
#     logistic_reg = LogisticRegression(max_iter=1000)
#
#     # Fit the model on the training data
#     logistic_reg.fit(X_train, y_train_encoded)
#
#     # Get the weights after fitting
#     weights = logistic_reg.coef_
#     print("shape",weights.shape)
#     print("Weights:", weights)
#
#     # Predict on the testing data
#     y_pred = logistic_reg.predict(X_test)
#
#     # Evaluate the model
#     accuracy = logistic_reg.score(X_test, y_test_encoded)
#     print("Accuracy:", accuracy)  # 93% is the best possible
# --------------------------------- WITHOUT SKLEARN ----------------------------------------

    test_set, validation_set, training_set = clean_data.split_data(0.2,0.1,42)

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

    W = train(X_train, y_train_encoded,X_valid,y_valid_encoded, learning_rate=0.01, num_epochs=2000)
    # print("my shape",W.shape)

    predictions = np.argmax(X_test.dot(W), axis=1)
    print(X_test.shape)


    # print(predictions[10],y_test_encoded[10])


    count = 0
    for i in range(len(y_test_encoded)):
        if predictions[i]==y_test_encoded[i]:
            count+=1
    print(count/len(y_test_encoded))
