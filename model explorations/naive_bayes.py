from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import re
import numpy as np
import pickle


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


def separate_features_and_targets(data):
    features = data.iloc[:,:-1]
    targets = data.iloc[:, -1]
    return features, targets


if __name__ == "__main__":
    data = clean_data(pd.read_csv('data/clean_dataset.csv'))

    label_to_index = {
        'Dubai': 0,
        'Rio de Janeiro': 1,
        'New York City': 2,
        'Paris': 3
    }

    data['Label'] = data['Label'].map(label_to_index)
    X = data.drop(columns=['Label'])
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB(var_smoothing=0.0008111308307896872)

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sklearn model - Test Accuracy: {accuracy:.4f}")

    mse = mean_squared_error(y_test, y_pred)
    print(f"Sklearn model - Test MSE: {mse:.4f}")

    class GaussianNaiveBayes:
        def __init__(self):
            self.means = {}
            self.variances = {}
            self.priors = {}

        def fit(self, X, y):
            self.classes = np.unique(y)
            for cls in self.classes:
                X_cls = X[y == cls]
                self.means[cls] = X_cls.mean(axis=0)
                self.variances[cls] = X_cls.var(axis=0)
                self.priors[cls] = X_cls.shape[0] / X.shape[0]

        def predict(self, X):
            predictions = []
            for x in X:
                posteriors = []
                for cls in self.classes:
                    prior = np.log(self.priors[cls])
                    conditional = self.log_pdf(cls, x)
                    posterior = prior + conditional
                    posteriors.append(posterior)
                predictions.append(self.classes[np.argmax(posteriors)])
            return np.array(predictions)

        def log_pdf(self, cls, x):
            mean = self.means[cls]
            var = self.variances[cls]
            log_numerator = -0.5 * np.sum(((x - mean) ** 2) / (var + 1e-9))
            log_denominator = x.size / 2. * np.log(2 * np.pi) + 0.5 * np.sum(
                np.log(var + 1e-9))
            return log_numerator - log_denominator


    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.astype(
            float).values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.astype(
            float).values

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # y_pred_train = gnb.predict(X_train)
    # train_accuracy = accuracy(y_train, y_pred_train)
    # print(f"Training Accuracy: {train_accuracy:.4f}")

    y_pred_test = gnb.predict(X_test)
    test_accuracy = accuracy(y_test, y_pred_test)
    print(f"Manually Implemented Model - Test Accuracy: {test_accuracy:.4f}")

    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Manually Implemented Model - Test MSE: {mse_test:.4f}")

