import pandas as pd
import re
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, confusion_matrix, mean_squared_error
def get_data_from_column(df, column_name):
    data = []
    for index, row in df.iterrows():
        quote = row[column_name]
        data.append(quote)
    return data


def get_unique_words_from_column(df, column_name):
    vocab = set()
    for index, row in df.iterrows():
        quote = row[column_name]
        if quote is not None:
            words = quote.split()
            vocab.update(words)
    return list(vocab)


def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


def make_bow(data, vocab):
    X = np.zeros([len(data), len(vocab)])

    # Map words to indices
    word_to_index = {word: index for index, word in enumerate(vocab)}

    for i, quote in enumerate(data):
        if quote is not None:
            words = quote.split()
            # Update the matrix X based on the presence of words in the vocab.
            for word in words:
                if word in word_to_index:  # Check if the word is in our vocabulary
                    X[i, word_to_index[word]] = 1  # Mark as present
    return X


# Function to clean and normalize text
def clean_text(text):
    if pd.isna(text):  # Check for NaN values
        return None
    # Remove quotations, excessive white space, non-alphanumeric characters, and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.replace('"', '').replace('\n', '').strip().lower())
    return cleaned_text


def clean_data(data):
    data.drop(columns=['id'], inplace=True)

    q5_split = ["Partner", "Friends", "Siblings", "Co-worker"]
    for category in q5_split:
        data[category] = data['Q5'].apply(lambda entry: 1 if category in str(entry) else 0)
    data.drop(columns=['Q5'], inplace=True)

    # Splitting the Q6 column into individual columns
    q6_split = data['Q6'].str.split(',', expand=True)

    new_column_names = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']

    # Extracting values after '=>' and creating new columns
    for col in q6_split.columns:
        data[new_column_names[col]] = pd.to_numeric(q6_split[col].str.split('=>').str[1])

    # Drop the original Q6 column
    data.drop(columns=['Q6'], inplace=True)

    data["Q7"] = data["Q7"].apply(to_numeric)
    data.fillna({'Q7': data['Q7'].mean()}, inplace=True)

    data.fillna({'Q8': data['Q8'].mean()}, inplace=True)
    data['Q8'] = np.round(data['Q8']).astype(int)

    data["Q9"] = data["Q9"].apply(to_numeric)
    data.fillna({'Q9': data['Q9'].mean()}, inplace=True)
    data['Q9'] = np.round(data['Q9']).astype(int)

    for column in ['Q7', 'Q8', 'Q9']:
        col_mean = data[column].mean()
        col_std = data[column].std()
        data[column] = (data[column] - col_mean) / col_std
        outliers = (data[column] < -2) | (data[column] > 2)
        data[column] = data[column] * col_std + col_mean
        data[column] = np.round(data[column]).astype(float)
        data[column] = data[column].mask(outliers, round(col_mean))

    for column in ['Q1', 'Q2', 'Q3', 'Q4', 'Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']:
        mean_value = round(data[column].mean())
        data.fillna({column: mean_value}, inplace=True)
        data[column] = data[column].astype(int)

    # Apply the clean_text function to 'Q10' column
    data['Q10'] = data['Q10'].apply(clean_text)

    # Add bag of words
    data_column = get_data_from_column(data, "Q10")

    # Get unique words from the specified column
    unique_words = get_unique_words_from_column(data, "Q10")
    unique_words.sort()

    # Create bag of words representation
    X_bow = make_bow(data_column, unique_words)
    # Convert bag of words representation to DataFrame
    bow_df = pd.DataFrame(X_bow, columns=unique_words)

    # Concatenate the bag of words DataFrame with the original DataFrame
    data = pd.concat([data, bow_df], axis=1)

    with open('vocab.pkl', 'wb') as file:
        pickle.dump(unique_words, file)

    data.drop(columns=['Q10'], inplace=True)
    return data


def split_data(test_size: float = 0.2, validation_size: float = 0.2, random_seed: int = 42):
    data = pd.read_csv('data/clean_dataset.csv')

    data = clean_data(data)

    # Shuffle the data
    np.random.seed(random_seed)  # Set random seed for reproducibility
    shuffled_indices = np.random.permutation(len(data))
    data = data.iloc[shuffled_indices]

    # Calculate sizes for test and validation sets
    num_samples = len(data)
    num_test_samples = int(test_size * num_samples)
    num_validation_samples = int(validation_size * num_samples)

    # Split the data
    test_data = data.iloc[:num_test_samples]
    validation_data = data.iloc[num_test_samples:num_test_samples + num_validation_samples]
    train_data = data.iloc[num_test_samples + num_validation_samples:]

    return test_data, validation_data, train_data


if __name__ == "__main__":
    # Split the data into training and testing sets
    test_data, validation_data, train_data = split_data()

    # Define X_train, y_train, X_test, y_test
    X_train = train_data.drop(columns=['Label'])
    y_train = train_data['Label']
    X_val = validation_data.drop(columns=['Label'])
    y_val = validation_data['Label']
    X_test = test_data.drop(columns=['Label'])
    y_test = test_data['Label']

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_knn_model = grid_search.best_estimator_

    print("best parameter", best_params)
    print("best knn model", best_knn_model)

    # Predict the labels for training data using the best model
    y_pred_train = best_knn_model.predict(X_train)

    # Evaluate the model on training data
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Training Accuracy:", accuracy_train)

    # Perform on validation set
    y_pred_val = best_knn_model.predict(X_val)

    # Evaluate the model on validation data
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Validation Accuracy:", accuracy_val)

    # Predict the labels for test data using the best model
    y_pred_test = best_knn_model.predict(X_test)

    # Evaluate the model on test data
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Test Accuracy:", accuracy_test)

    # Precision -> I think Micro most relevant
    precision_micro = precision_score(y_test, y_pred_test, average='micro')
    precision_macro = precision_score(y_test, y_pred_test, average='macro')
    precision_weighted = precision_score(y_test, y_pred_test, average='weighted')
    print("Precision micro:", precision_micro)
    print("Precision macro:", precision_macro)
    print("Precision weighted:", precision_weighted)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Encode city labels into integers
    label_encoding = {
        'Dubai': 0,
        'Rio de Janeiro': 1,
        'New York City': 2,
        'Paris': 3
    }
    y_test_encoded = y_test.map(label_encoding)
    y_pred_test_encoded = pd.Series(y_pred_test).map(label_encoding)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test_encoded, y_pred_test_encoded)
    print("Mean Squared Error (MSE):", mse)
