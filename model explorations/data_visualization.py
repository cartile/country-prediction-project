import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

if __name__ == "__main__":
    data = csv.DictReader(open("data/clean_dataset.csv",encoding="utf-8"))
    data_list = list(data)
    df = pd.DataFrame(data_list)

    summary_stats = df.describe()

    # Write the summary statistics to the file
    with open('data_outs.txt', 'w', encoding="utf-8") as f:  # Open the file in append mode
        f.write("\nSummary Statistics:\n")
        f.write(summary_stats.to_string())

    # Bar graph plots of questions
    order_number = ['1', '2', '3', '4', '5']
    df_filtered_Q1 = df[df['Q1'].str.strip() != '']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered_Q1, x='Q1', hue='Label', palette='viridis', order=order_number)
    plt.title('q1 Popularity Ratings of Cities')
    plt.xlabel('City Rating')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.xticks(rotation=0)
    plt.show()

    city_rating_counts = df_filtered_Q1.groupby('Q1').size()
    # Calculating the mean of the counts
    mean_count = np.mean(city_rating_counts)
    print("Q1 Mean count:", mean_count)

    df_filtered_Q2 = df[df['Q2'].str.strip() != '']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered_Q2, x='Q2', hue='Label', palette='viridis', order=order_number)
    plt.title('q2 Every day occurance to viral')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.xticks(rotation=0)
    plt.show()
    city_rating_counts = df_filtered_Q2.groupby('Q2').size()
    # Calculating the mean of the counts
    mean_count = np.mean(city_rating_counts)
    print("Q2 Mean count:", mean_count)

    df_filtered_Q3 = df[df['Q3'].str.strip() != '']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered_Q3, x='Q3', hue='Label', palette='viridis', order=order_number)
    plt.title('q3 Architectural uniqueness')
    plt.xlabel('rating')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.xticks(rotation=0)
    plt.show()

    city_rating_counts = df_filtered_Q3.groupby('Q3').size()
    # Calculating the mean of the counts
    mean_count = np.mean(city_rating_counts)
    print("Q3 Mean count:", mean_count)

    df_filtered_Q4 = df[df['Q4'].str.strip() != '']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered_Q4, x='Q4', hue='Label', palette='viridis', order=order_number)
    plt.title('q4 Enthusiasm for spontaneous street parties')
    plt.xlabel('rating')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.xticks(rotation=0)
    plt.show()

    city_rating_counts = df_filtered_Q4.groupby('Q4').size()
    # Calculating the mean of the counts
    mean_count = np.mean(city_rating_counts)
    print("Q4 Mean count:", mean_count)

    # Grouping data by 'Q5' and 'Label' columns and counting occurrences
    # Split the values in the 'Q5' column by commas and count the occurrences of each element
    df_filtered_Q5 = df[df['Q5'].str.strip() != '']
    q5_counts = Counter(df_filtered_Q5['Q5'].str.split(',').explode())

    # Convert the Counter object to a dictionary for plotting
    q5_dict = dict(q5_counts)

    # Bar graph plots of questions counting each element separately
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df.explode('Q5'), x='Q5', hue='Label', palette='viridis', dodge=True, order=q5_counts.keys())
    plt.title('q5 Who would likely travel with you here')
    plt.xlabel('Who')
    plt.ylabel('Count')
    plt.legend(title='Label')
    plt.xticks(rotation=90)  # Rotate x-labels vertically
    plt.show()

    # Assuming "df" is your DataFrame and "City" is the column containing city names, and "Q6" contains lists of words
    word_counts = df.explode('Q6').groupby('Label')['Q6'].value_counts()
    with open('data_outs.txt', 'a') as f:
        for index, value in word_counts.items():
            f.write(f"{index}: {value}\n")

    df['Q7'] = pd.to_numeric(df['Q7'], errors='coerce')

    # Filter the DataFrame to remove values outside the range of -100 to 100 in the 'Q7' column
    df_filtered = df[(df['Q7'] >= -100) & (df['Q7'] <= 100)]

    # Assuming "df_filtered" is the filtered DataFrame and you want to create a boxplot for each label
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x='Label', y='Q7', hue='Label', palette='viridis')
    plt.title('Q7 Boxplot per City')
    plt.xlabel('City')
    plt.ylabel('Temperature')
    plt.xticks(rotation=45)  # Rotate x-labels for better readability if needed
    plt.ylim(-25, 50)  # Set y-axis limits from -100 to 100
    plt.show()

    # Filter
    df['Q8'] = pd.to_numeric(df['Q8'], errors='coerce')
    df_filtered = df[(df['Q8'] >= -100) & (df['Q8'] <= 100)]
    # Assuming "df_filtered" is the filtered DataFrame and you want to create a boxplot for each label
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Label', y='Q8', hue='Label', palette='viridis')
    plt.title('Q8 Boxplot per City')
    plt.xlabel('City')
    plt.ylabel('Languages')
    plt.xticks(rotation=45)  # Rotate x-labels for better readability if needed
    plt.ylim(0, 20)  # Set y-axis limits from -100 to 100
    plt.show()

    # Filter
    df['Q9'] = pd.to_numeric(df['Q9'], errors='coerce')
    df_filtered = df[(df['Q9'] >= -40) & (df['Q9'] <= 40)]
    # Assuming "df_filtered" is the filtered DataFrame and you want to create a boxplot for each label
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Label', y='Q9', hue='Label', palette='viridis')
    plt.title('Q9 Boxplot per City')
    plt.xlabel('City')
    plt.ylabel('Fashion styles')
    plt.xticks(rotation=45)  # Rotate x-labels for better readability if needed
    plt.ylim(0, 30)  # Set y-axis limits from -100 to 100
    plt.show()

    word_counts = df.explode('Q10').groupby('Label')['Q10'].value_counts()
    with open('data_outs.txt', 'a') as f:
        for index, value in word_counts.items():
            f.write(f"{index}: {value}\n")
