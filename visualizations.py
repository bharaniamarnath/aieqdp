import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_histogram(data_values):
    plt.figure(figsize=(10, 6))
    sns.histplot(data_values['age'], bins=30, kde=True)
    plt.title('Distribution of Building Ages')
    plt.xlabel('Age of Buildings (Years)')
    plt.ylabel('Frequency')
    st.pyplot(plt)  # Use Streamlit to display the plot

def plot_countplot(data_values, column):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data_values, x=column, order=data_values[column].value_counts().index)
    plt.title(f'{column} Count')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(plt)  # Use Streamlit to display the plot

def plot_correlation_heatmap(dataframe):
    plt.figure(figsize=(14, 10))
    correlation = dataframe.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap of Dataset Features')
    st.pyplot(plt)  # Use Streamlit to display the plot
