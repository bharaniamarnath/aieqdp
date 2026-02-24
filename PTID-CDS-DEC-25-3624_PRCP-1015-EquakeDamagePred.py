#!/usr/bin/env python
# coding: utf-8

# # PTID-CDS-DEC-25-3624_PRCP-1015-EquakeDamagePred

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# configurations
import warnings
warnings.filterwarnings("ignore")

# seaborn plot style
sns.set(style="whitegrid")


# In[3]:


# load dataset
url_labels = 'datasets/train_labels.csv'
data_labels = pd.read_csv(url_labels)


# In[4]:


url_values = 'datasets/train_values.csv'
data_values = pd.read_csv(url_values)


# In[5]:


print(data_labels.columns.tolist())


# In[6]:


print(data_values.columns.tolist())


# ## Data Aanalysis

# In[7]:


# display the first few rows of the dataset
print("First few rows of the dataset:")
display(data_values.head())

# display dataset information
print("Dataset Information:")
print(data_values.info())

# summary statistics
print("Summary Statistics:")
display(data_values.describe())

# check for missing values
missing_values = data_values.isnull().sum()
print("Missing Values:")
display(missing_values[missing_values > 0])

# visualizations

# histogram of the age of buildings
plt.figure(figsize=(10, 6))
sns.histplot(data_values['age'], bins=30, kde=True)
plt.title('Distribution of Building Ages')
plt.xlabel('Age of Buildings (Years)')
plt.ylabel('Frequency')
plt.show()

# countplot for foundation types
plt.figure(figsize=(12, 6))
sns.countplot(data=data_values, x='foundation_type', order=data_values['foundation_type'].value_counts().index)
plt.title('Foundation Types of Buildings')
plt.xlabel('Foundation Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# correlation heatmap
plt.figure(figsize=(12, 10))
correlation = data_values.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# boxplot for age vs count of floors
plt.figure(figsize=(12, 6))
sns.boxplot(x='count_floors_pre_eq', y='age', data=data_values)
plt.title('Boxplot of Age by Count of Floors')
plt.xlabel('Count of Floors (Pre-Earthquake)')
plt.ylabel('Age (Years)')
plt.show()

# ddditional analysis based on secondary use
secondary_use_columns = [
    'has_secondary_use', 
    'has_secondary_use_agriculture',
    'has_secondary_use_hotel',
    'has_secondary_use_rental',
    'has_secondary_use_institution'
]

for column in secondary_use_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data_values, x=column)
    plt.title(f'Count of Buildings with {column.replace("_", " ").title()}')
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel('Count')
    plt.show()


# # Prediction

# In[8]:


# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# load the dataset
pvdata = data_values.copy()
pldata = data_labels.copy()

pdata = pd.merge(pvdata, pldata)

# handle missing values
pdata.fillna(method='ffill', inplace=True)  # forward fill or use an appropriate method

# check if damage_grade is present
if 'damage_grade' not in pdata.columns:
    raise ValueError("Column 'damage_grade' does not exist in the dataset.")

# encode categorical variables
categorical_cols = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']

# drop columns that are not in the pvdataset if any
valid_categorical_cols = [col for col in categorical_cols if col in pdata.columns]

pdata = pd.get_dummies(pdata, columns=valid_categorical_cols, drop_first=True)

# define features and target variable
# using errors='ignore' to avoid errors
X = pdata.drop(columns=['damage_grade', 'building_id'], errors='ignore')
y = pdata['damage_grade']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Model training completed.")


# # Suggestions

# In[9]:


# Load the dataset
svdata = data_values.copy()
sldata = data_labels.copy()

sdata = pd.merge(svdata, sldata)

# Display the first few rows of the dataset
print(sdata.head())

# Display dataset info to understand sssdata types and missing values
print(sdata.info())


# In[10]:


# check current columns in dataframe
print("Columns in DataFrame:", sdata.columns.tolist())

# convert categorical variables to numerical
# check for non-numeric columns
non_numeric_cols = sdata.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# check unique values in non-numeric columns for unexpected strings
for col in non_numeric_cols:
    print(f"Unique values in {col}:", sdata[col].unique())

# define the columns to one-hot encode
categorical_cols = [
    'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']

# ensure only existing categorical columns are converted
existing_categorical_cols = [col for col in categorical_cols if col in sdata.columns]

# using pandas' get_dummies to convert categorical variables to dummy variables
sdata = pd.get_dummies(sdata, columns=existing_categorical_cols, drop_first=True)

# convert damage_grade column to numeric
sdata['damage_grade'] = pd.to_numeric(sdata['damage_grade'], errors='coerce')

# remove rows with nan values
sdata.dropna(inplace=True)

# ensure damage_grade is now an integer or float
sdata['damage_grade'] = sdata['damage_grade'].astype(int)

# proceed to create visualizations
# correlation heatmap for dataset features
plt.figure(figsize=(14, 8))
corr_matrix = sdata.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Dataset Features')
plt.show()

# damage grade vs building height
plt.figure(figsize=(10, 5))
sns.boxplot(x='damage_grade', y='height_percentage', data=sdata)
plt.title('Damage Grade Distribution by Height Percentage')
plt.xlabel('Damage Grade')
plt.ylabel('Height Percentage')
plt.grid()
plt.show()

# damage grade vs foundation type 
existing_foundation_cols = [col for col in sdata.columns if 'foundation_type' in col]
plt.figure(figsize=(15, 7))
if existing_foundation_cols:  # Check if foundation_type columns exist
    sns.countplot(x=existing_foundation_cols[0], hue='damage_grade', data=sdata)
    plt.title('Foundation Type vs Damage Grade')
    plt.xlabel('Foundation Type')
    plt.ylabel('Count')
    plt.legend(title='Damage Grade')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
else:
    print("No foundation type columns to plot.")

# suggestions based on analysis results
suggestions = {
    "Improved Building Codes": (
        "Implement stricter building codes that take into account various structural factors such as height and foundation type to enhance resilience against earthquakes."
    ),
    "Targeted Retrofitting": (
        "Prioritize retrofitting older or vulnerable buildings that are likely to suffer more damage based on foundation types and building height."
    ),
    "Risk Assessment": (
        "Conduct geological and structural assessments to identify and reinforce buildings located in high-risk zones."
    ),
    "Public Policy": (
        "Create mandatory inspection policies for older buildings to ensure they meet current safety standards."
    ),
    "Community Education": (
        "Educate the public on the importance of building practices and materials that contribute to earthquake resilience."
    ),
    "Emergency Preparedness Plans": (
        "Develop comprehensive emergency response plans that account for various building types and their associated risks."
    )
}

# Display the suggestions in a more structured manner
for title, suggestion in suggestions.items():
    print(f"{title}:\n{suggestion}\n")


# In[ ]:




