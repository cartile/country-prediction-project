import pandas as pd
import re
import numpy as np
import pickle


def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


# Function to clean and normalize text
def clean_text(text):
    if pd.isna(text):  # Check for NaN values
        return None
    # Remove quotations, excessive white space, non-alphanumeric characters, and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.replace('"', '').replace('\n', '').strip().lower())
    return cleaned_text


def bow(data, vocab):
    X_bow = np.zeros([len(data), len(vocab)])
    word_to_index = {word: index for index, word in enumerate(vocab)}
    for i, quote in enumerate(data['Q10']):
        if quote is not None:
            words = quote.split()
            for word in words:
                if word in word_to_index:
                    X_bow[i, word_to_index[word]] = 1
    X_bow_df = pd.DataFrame(X_bow, columns=vocab)
    return pd.concat([data.reset_index(), X_bow_df], axis=1)


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

    with open('data/vocab.pkl', 'rb') as file:
        vocab = pickle.load(file)
        data = bow(data, vocab)
        data.drop(columns=['index'], inplace=True)

    data.drop(columns=['Q10'], inplace=True)
    return data
