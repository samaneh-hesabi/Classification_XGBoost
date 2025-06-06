import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Load the Red Wine Quality dataset
print("Loading the Red Wine Quality dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# Step 2: Explore the dataset
print("\nDataset exploration:")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head(1))

print("\nData information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

# Step 3: Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

# Step 4: Clean the data
print("\nCleaning the data...")
# No missing values to handle in this dataset, but we'll add code for completeness
if missing_values.sum() > 0:
    # Fill numeric columns with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

# Step 5: Feature Engineering
print("\nPerforming feature engineering...")
# Create a new feature: total acidity (fixed + volatile)
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']

# Step 6: Handle outliers using IQR method
print("\nHandling outliers...")
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace outliers with bounds
    df[column] = np.where(df[column] < lower_bound, df[column].median(), df[column])
    df[column] = np.where(df[column] > upper_bound, df[column].median(), df[column])

# Step 7: Normalize the features
print("\nNormalizing features...")
scaler = StandardScaler()
features = df.drop('quality', axis=1)
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with scaled features
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['quality'] = df['quality']  # Add back the target variable

# Step 8: Save the cleaned dataset
print("\nSaving cleaned dataset...")
if not os.path.exists('data'):
    os.makedirs('data')

# Save both unscaled (but cleaned) and scaled versions
df.to_csv('data/cleaned_red_wine.csv', index=False)
scaled_df.to_csv('data/cleaned_scaled_red_wine.csv', index=False)

print("Data cleaning complete! Files saved as:")
print("- data/cleaned_red_wine.csv (cleaned but unscaled)")
print("- data/cleaned_scaled_red_wine.csv (cleaned and scaled)")

# Return the DataFrames for direct use
print("\nDatasets are ready for use!") 