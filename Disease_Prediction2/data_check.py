import pandas as pd

# Load the datasets
training_data = pd.read_csv('Training.csv')
testing_data = pd.read_csv('Testing.csv')

# Display the first few rows and check the number of columns
print("Training Data:")
print(training_data.head())
print("\nNumber of Features in Training Data:", training_data.shape[1])

print("\nTesting Data:")
print(testing_data.head())
print("\nNumber of Features in Testing Data:", testing_data.shape[1])
