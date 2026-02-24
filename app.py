import streamlit as st
import pandas as pd
from data_loader import load_data
from data_analysis import analyze_data
from visualizations import plot_histogram, plot_countplot, plot_correlation_heatmap
from model import train_model, preprocess_data

# Load dataset paths from the user or predefine them
labels_path = 'datasets/train_labels.csv'
values_path = 'datasets/train_values.csv'

# Load the data
data_labels, data_values = load_data(labels_path, values_path)

# Title of the app
st.title('Earthquake Damage Level Prediction')

# Create tabs for different sections
tabs = st.tabs(["Data Analysis", "Visualizations", "Model Training"])

# Data Analysis Section
with tabs[0]:
    if st.button("Analyze Data"):
        head, info, desc, missing_details = analyze_data(data_values)  # Get details from analyze_data

        st.subheader("First Few Rows of the Dataset:")
        st.dataframe(data_values.head())  # Display the head as a DataFrame

        # Display dataset information
        st.subheader("Dataset Information:")
        info_df = pd.DataFrame({'Info': [info]})  # Convert info to DataFrame
        st.dataframe(info_df)

        # Display dataset description
        st.subheader("Dataset Description:")
        desc_df = data_values.describe()  # Get the description as a DataFrame
        st.dataframe(desc_df)

        # Check for missing values and display them in tabular format
        st.subheader("Missing Values:")
        missing_values = data_values.isnull().sum()
        missing_details_df = pd.DataFrame({'Missing Values': missing_values[missing_values > 0]})
        if not missing_details_df.empty:
            st.dataframe(missing_details_df.reset_index())  # Display in a DataFrame
        else:
            st.write("No missing values in the dataset.")

# Visualization Section
with tabs[1]:
    st.header("Visualizations")
    visualization_choice = st.selectbox("Choose a visualization", ["Histogram", "Countplot", "Correlation Heatmap"])

    if visualization_choice == "Histogram":
        plot_histogram(data_values)
    elif visualization_choice == "Countplot":
        column = st.selectbox("Choose column for countplot", data_values.columns)
        plot_countplot(data_values, column)
    elif visualization_choice == "Correlation Heatmap":
        processed_data = preprocess_data(data_values, data_labels)  # Preprocess data
        plot_correlation_heatmap(processed_data)  # Call the updated function

# Model Training Section
with tabs[2]:
    if st.button("Train Model"):
        model, conf_matrix, class_report_df = train_model(data_values, data_labels)

        # Display results in the Streamlit app
        st.success("Model training completed.")
        
        st.subheader("Confusion Matrix:")
        st.write(conf_matrix)
        
        st.subheader("Classification Report:")
        st.dataframe(class_report_df)  # Display the classification report as a DataFrame
