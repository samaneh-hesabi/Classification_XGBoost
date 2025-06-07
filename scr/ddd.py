import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the cleaned Red Wine Quality dataset
print("Step 1: Loading the cleaned Red Wine Quality dataset...")
# Load the cleaned and scaled dataset that was prepared in data_preparation.py
df = pd.read_csv('data/cleaned_scaled_red_wine.csv')
print("First few rows of the cleaned dataset:")
print(df.head(1))

# Features are all columns except quality
X = df.drop('quality', axis=1).values
# Target is the quality column
y = df['quality'].values
feature_names = df.drop('quality', axis=1).columns.tolist()

# Print dataset information
print(f"Dataset shape: {X.shape} (samples, features)")
#print(f"Number of classes: {len(np.unique(y))}")
print(f"Original class labels: {np.unique(y)}")

# Transform the labels to start from 0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#print(f"Transformed class labels: {np.unique(y_encoded)}")
#print(f"Number of encoded classes: {len(np.unique(y_encoded))}")

# Step 2: Split data into training and testing sets
print("\nStep 2: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,  # Use encoded labels
    test_size=0.2,  # Use 20% of data for testing
    random_state=42  # For reproducible results
)
#The code specifically wants to print just the number of training samples (rows), not the number of features (columns)
print(f"Training set size: {X_train.shape[0]} samples") # 
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 3: Create and train the XGBoost model
print("\nStep 3: Creating and training the XGBoost model...")
# Get the number of unique classes in the target
y_encoded_classes = len(np.unique(y_encoded))
print(f"Number of wine quality classes: {y_encoded_classes}")

model = xgb.XGBClassifier(
    objective='multi:softmax',  # For multi-class classification
    num_class=y_encoded_classes,      # Number of wine quality classes
    learning_rate=0.1,          # Controls how quickly the model learns
    max_depth=5,                # Maximum depth of trees
    n_estimators=200,           # Number of trees to build
    random_state=42             # For reproducible results
)

# Train the model
model.fit(X_train, y_train)


# Step 4: Make predictions
print("\nStep 4: Making predictions on test data...")
y_pred_encoded = model.predict(X_test)

# Step 5: Evaluate the model
print("\nStep 5: Evaluating model performance...")
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_encoded, zero_division=0))

# Convert encoded predictions back to original labels for better interpretability
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

# Step 6: Create and display a confusion matrix
print("\nStep 6: Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_encoded)
print("Confusion Matrix (encoded labels):")
print(cm)

# Display the confusion matrix with original labels
plt.figure(figsize=(8, 6))
cm_original = confusion_matrix(y_test_original, y_pred_original)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(np.unique(y)), 
            yticklabels=sorted(np.unique(y)))
plt.title('Confusion Matrix (Original Labels)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 7: Display feature importance
print("\nStep 7: Feature importance...")
importance = model.feature_importances_  ###ex:[0.2, 0.5, 0.1, 0.7]
# Sort features by importance
indices = np.argsort(importance) ## ex:[2, 0, 1, 3]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Print feature importance scores
print("\nFeature Importance Scores:")
for name, score in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}") 