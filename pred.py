# basic python imports are permitted
import sys
import csv
import random
import pickle
import clean_data

# numpy and pandas are also permitted
import numpy as np
import pandas as pd


def label_decode(labels):
    label_map = {0: 'Dubai', 1: 'New York City', 2: 'Paris', 3: 'Rio de Janeiro'}
    decoded_labels = [label_map[label] for label in labels]
    return decoded_labels


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    with open('data/weights.pkl', 'rb') as f:
        weights = pickle.load(f)

    with open('data/bias.pkl', 'rb') as f:
        bias = pickle.load(f)

    data = pd.read_csv(filename)
    data = clean_data.clean_data(data)

    linear_combination = np.dot(data, weights) + bias
    probabilities = 1 / (1 + np.exp(-linear_combination))
    predictions = np.argmax(probabilities, axis=1)

    return label_decode(predictions)
