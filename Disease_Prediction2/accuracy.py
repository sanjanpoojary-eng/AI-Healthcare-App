import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the datasets and drop unnamed columns
training_data = pd.read_csv('Training.csv').dropna(axis=1, how='all')
testing_data = pd.read_csv('Testing.csv').dropna(axis=1, how='all')

# Drop any unnamed columns that might have been added
if 'Unnamed: 133' in training_data.columns:
    training_data = training_data.drop(columns=['Unnamed: 133'])

if 'Unnamed: 133' in testing_data.columns:
    testing_data = testing_data.drop(columns=['Unnamed: 133'])

# Separate features and target variable from training data
X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']

# Separate features and target variable from testing data
X_test = testing_data.drop(columns=['prognosis'])
y_test = testing_data['prognosis']

# Encode the target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Model Accuracy: {nb_accuracy * 100:.2f}%")
