import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',  # for multi-class classification
    num_class=3,               # number of classes in Wine dataset
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            annot_kws={'size': 14}, 
            cbar_kws={'label': 'Number of Samples'})
plt.title('Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve for multi-class
n_classes = 3
y_test_bin = label_binarize(y_test, classes=range(n_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], 
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})',
             linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Each Class', pad=20, fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance with actual feature names
importance = model.feature_importances_
sorted_idx = importance.argsort()

plt.figure()
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance', pad=20, fontsize=16, fontweight='bold')
plt.xlabel('F Score', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the model
model.save_model('wine_xgboost_model.json')

# Print additional metrics
print("\nAdditional Metrics:")
print(f"Average ROC AUC: {np.mean(list(roc_auc.values())):.4f}")
print("\nConfusion Matrix:")
print(cm)

# Print feature names and their importance scores
print("\nFeature Importance Scores:")
for name, score in zip(feature_names, importance):
    print(f"{name}: {score:.4f}") 