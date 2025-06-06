<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Wine Classification Data</div>

# 1. Data Files

This directory contains the data files used in the Wine Classification project:

## 1.1 Raw Data
- The raw data is sourced from the UCI Machine Learning Repository's Wine Quality dataset
- Original URL: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

## 1.2 Processed Data
- `cleaned_red_wine.csv`: The cleaned but unscaled dataset
- `cleaned_scaled_red_wine.csv`: The cleaned and standardized dataset (used for model training)

# 2. Data Description

The dataset contains various chemical properties of red wine samples and their quality ratings:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- total_acidity (engineered feature)
- quality (target variable)

# 3. Data Processing

The data processing steps include:
1. Loading the raw data
2. Handling missing values
3. Feature engineering (adding total_acidity)
4. Outlier detection and handling
5. Feature scaling using StandardScaler

These steps are implemented in the `data_preparation.py` script in the `scr` directory. 