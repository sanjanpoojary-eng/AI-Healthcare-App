import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the datasets
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing_modified.csv')

# Drop unnecessary columns if present
train_df = train_df.drop(columns=['Unnamed: 133'], errors='ignore')
test_df = test_df.drop(columns=['Unnamed: 133'], errors='ignore')

# Preprocess the data
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Encode target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Initialize models
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
}

# Train models and collect predictions
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions[model_name] = model.predict(X_test)

# Generate confusion matrices for each model
confusion_matrices = {}
for model_name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm

# Compute ensemble predictions (majority voting)
ensemble_predictions = []
for i in range(len(y_test)):
    votes = [predictions[model_name][i] for model_name in models.keys()]
    ensemble_predictions.append(np.bincount(votes).argmax())

# Generate confusion matrix for the ensemble model
ensemble_cm = confusion_matrix(y_test, ensemble_predictions)
confusion_matrices["Ensemble"] = ensemble_cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, class_labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Plot confusion matrices for all models
for model_name, cm in confusion_matrices.items():
    class_labels = label_encoder.classes_
    plot_confusion_matrix(cm, model_name, class_labels)

print("Confusion matrices have been drawn.")


