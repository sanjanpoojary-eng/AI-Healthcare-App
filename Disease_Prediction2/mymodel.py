# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Load datasets
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing_modified.csv')

# Drop the unnecessary 'Unnamed' column
train_df = train_df.drop(columns=['Unnamed: 133'], errors='ignore')
test_df = test_df.drop(columns=['Unnamed: 133'], errors='ignore')

# Preprocess the data (Assuming the last column is the target 'prognosis' and the rest are symptoms)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Encode the target (prognosis/disease)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Initialize a dictionary to store the results
results = {}

# Function to evaluate and store results of a model
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# 1. Decision Tree
dt = DecisionTreeClassifier()
evaluate_model('Decision Tree', dt, X_train, y_train, X_test, y_test)

# 2. Random Forest
rf = RandomForestClassifier()
evaluate_model('Random Forest', rf, X_train, y_train, X_test, y_test)

# 3. Naive Bayes
nb = GaussianNB()
evaluate_model('Naive Bayes', nb, X_train, y_train, X_test, y_test)

# 4. Logistic Regression
lr = LogisticRegression(max_iter=200)
evaluate_model('Logistic Regression', lr, X_train, y_train, X_test, y_test)

# 5. Support Vector Machine (SVM)
svm = SVC()
evaluate_model('SVM', svm, X_train, y_train, X_test, y_test)

# 6. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
evaluate_model('KNN', knn, X_train, y_train, X_test, y_test)

# 7. XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
evaluate_model('XGBoost', xgb_model, X_train, y_train, X_test, y_test)


# Display all results in a DataFrame for comparison
results_df = pd.DataFrame(results).T
print(results_df)
