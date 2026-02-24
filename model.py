import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def get_categorical_columns(dataframe):
    return dataframe.select_dtypes(include=['object']).columns.tolist()

def preprocess_data(data_values, data_labels):
    # Merge the datasets
    pdata = pd.merge(data_values, data_labels)
    
    # Replace the deprecated fillna method with .ffill()
    pdata.ffill(inplace=True)  # Forward fill missing values

    # Identify categorical columns
    categorical_cols = get_categorical_columns(pdata)

    # One-hot encoding categorical variables
    pdata = pd.get_dummies(pdata, columns=categorical_cols, drop_first=True)

    return pdata


def train_model(data_values, data_labels):
    pdata = preprocess_data(data_values, data_labels)

    # Define features and target variable
    X = pdata.drop(columns=['damage_grade', 'building_id'], errors='ignore')
    y = pdata['damage_grade']

    # No need to check for NaN now as we've handled it in preprocess_data
    X.fillna(0, inplace=True)  # Additional safety

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Get report as a dictionary

    # Convert classification report to DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()

    # Return the model, confusion matrix, and classification report DataFrame
    return model, conf_matrix, class_report_df