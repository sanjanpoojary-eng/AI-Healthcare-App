# Model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle

# Load the dataset
data = pd.read_csv('Training_modified.csv')
if 'Unnamed: 133' in data.columns:
    data = data.drop('Unnamed: 133', axis=1)

# Encode target labels
encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])

X = data.drop(columns=['prognosis'])
y = data['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train each model
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
    'LogisticRegression': LogisticRegression(max_iter=200),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save the label encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
