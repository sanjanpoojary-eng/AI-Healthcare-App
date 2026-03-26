from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from collections import Counter
import os

app = Flask(__name__)

# Load models and encoder
model_names = ['DecisionTree', 'RandomForest', 'NaiveBayes', 'LogisticRegression', 'SVM', 'KNN', 'XGBoost']
models = {}
for name in model_names:
    with open(f'{name}_model.pkl', 'rb') as f:
        models[name] = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load symptom data
data = pd.read_csv('Training.csv')
if 'Unnamed: 133' in data.columns:
    data = data.drop('Unnamed: 133', axis=1)
symptoms = data.columns[:-1]
symptom_index = {symptom: i for i, symptom in enumerate(symptoms)}

# Configure Gemini API with embedded key
API_KEY = "AIzaSyCLrmKkKhlzs3qU6Dnt455guNimFIF_SfA"  # Replace with your actual Gemini 2.0 Flash API key
genai.configure(api_key=API_KEY)

def predict_disease(input_symptoms):
    input_data = [0] * len(symptoms)
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
    # Get probability predictions from all models
    model_probabilities = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([input_data])[0]
        else:
            prediction = model.predict([input_data])[0]
            probabilities = [0] * len(encoder.classes_)
            probabilities[prediction] = 1
        model_probabilities.append(probabilities)
    # Average the probabilities across models
    avg_probabilities = np.mean(model_probabilities, axis=0)
    final_prediction_index = np.argmax(avg_probabilities)
    final_disease = encoder.inverse_transform([final_prediction_index])[0]
    # Map each model to its predicted disease
    model_predictions = {name: encoder.inverse_transform([np.argmax(prob)])[0] for name, prob in zip(models.keys(), model_probabilities)}
    return model_predictions, final_disease

def fetch_disease_info(disease, gender, age, selected_symptoms, model_predictions, additional_info):
    model = genai.GenerativeModel("gemini-2.0-flash")
    symptoms_str = ", ".join(selected_symptoms)
    models_str = "\n".join([f"- {model}: {prediction}" for model, prediction in model_predictions.items()])
    prompt = f"""
Based on the following information:

- Input symptoms: {symptoms_str}
- Predictions from machine learning models:
{models_str}
- Final predicted disease: {disease}
- User's age: {age}
- User's gender: {gender}
- Additional information: {additional_info}

Please provide the following in HTML format:

1. <h4>Disease Description</h4>
   Provide a detailed summary about the final predicted disease ({disease}) tailored to the user's age, gender, and additional information. Include sections on:
   - <h5>Description</h5>
   - <h5>Causes</h5>
   - <h5>Symptoms</h5>
   - <h5>Remedies</h5>
   - <h5>Medication</h5>

2. <h4>Evaluation of Machine Learning Predictions</h4>
   Evaluate the machine learning predictions based on the input symptoms and the predictions from each model. Discuss whether the final predicted disease seems accurate.

3. <h4>LLM's Prediction</h4>
   Provide your own prediction of the disease based on the input symptoms and additional information. Include a brief explanation for your prediction.

Ensure the entire response is formatted using proper HTML tags.
"""
    response = model.generate_content(prompt)
    text = response.text.strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_symptoms = [request.form['symptom1'], request.form['symptom2'], request.form['symptom3']]
        gender = request.form['gender']
        age = request.form['age']
        additional_info = request.form.get('additional_info', '')
        model_predictions, final_disease = predict_disease(selected_symptoms)
        disease_info = fetch_disease_info(final_disease, gender, age, selected_symptoms, model_predictions, additional_info)
        return render_template('index.html', symptoms=symptoms, model_predictions=model_predictions, final_disease=final_disease, disease_info=disease_info, selected_symptoms=selected_symptoms, gender=gender, age=age)
    return render_template('index.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)