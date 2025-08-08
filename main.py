import os
import sys
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import random

app = Flask(__name__)

# -------------------------------------------
# Load necessary datasets with robust, absolute paths
# -------------------------------------------
# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths
symptoms_path = os.path.join(BASE_DIR, "dataset", "symtoms_df.csv")
precautions_path = os.path.join(BASE_DIR, "dataset", "precautions_df.csv")
workout_path = os.path.join(BASE_DIR, "dataset", "workout_dff.csv")
description_path = os.path.join(BASE_DIR, "dataset", "description.csv")
medications_path = os.path.join(BASE_DIR, "dataset", "medicationss.csv")
diets_path = os.path.join(BASE_DIR, "dataset", "dietss.csv")
severity_path = os.path.join(BASE_DIR, "dataset", "symptom-severity.csv")

try:
    sym_des = pd.read_csv(symptoms_path)
    precautions = pd.read_csv(precautions_path)
    workout = pd.read_csv(workout_path, on_bad_lines='skip')
    description = pd.read_csv(description_path)
    medications = pd.read_csv(medications_path)
    diets = pd.read_csv(diets_path)
    symptom_severity = pd.read_csv(severity_path)
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    # If a core data file is missing, the app cannot run.
    print(f"FATAL ERROR: A data file was not found. {e}")
    sys.exit(1) # Exit the application immediately
except Exception as e:
    print(f"Error loading datasets: {e}")
    sys.exit(1) # Exit the application immediately

# -------------------------------------------
# Build a set of disease names from symtoms_df.csv
# -------------------------------------------
disease_names = set(sym_des['Disease'].unique())

# -------------------------------------------
# Generate symptoms dictionary (only actual symptoms, excluding disease names)
# -------------------------------------------
def generate_symptoms_dict(df):
    unique_symptoms = set()
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        for s in df[col].dropna().unique():
            if s not in disease_names:
                unique_symptoms.add(s.lower())  # Normalize to lower-case
    return {symptom: idx for idx, symptom in enumerate(sorted(unique_symptoms))}

symptoms_dict = generate_symptoms_dict(sym_des)
print("Total unique symptoms loaded:", len(symptoms_dict))

# -------------------------------------------
# Create a dictionary for symptom severity with lower-case keys
# -------------------------------------------
severity_dict = dict(zip(symptom_severity['Symptom'].str.lower(), symptom_severity['weight']))
print("Symptom severity dictionary:", severity_dict)

# -------------------------------------------
# Reduce the symptoms displayed to the user
# Only include symptoms with severity >= threshold and then take the top 50
# -------------------------------------------
def get_symptom_weight(symptom):
    return severity_dict.get(symptom, 1)

threshold = 2  # Only consider symptoms with weight 2 or more
filtered_symptoms = {k: v for k, v in symptoms_dict.items() if get_symptom_weight(k) >= threshold}
if not filtered_symptoms:
    filtered_symptoms = symptoms_dict  # Fallback if no symptom qualifies
display_symptoms = dict(sorted(filtered_symptoms.items(), key=lambda item: (-get_symptom_weight(item[0]), item[0]))[:50])
print("Display symptoms (top 50):", display_symptoms)

# -------------------------------------------
# Build a dictionary mapping each disease to its set of symptoms
# -------------------------------------------
disease_symptoms = {}
for _, row in sym_des.iterrows():
    disease = row['Disease']
    row_symptoms = []
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        if pd.notna(row[col]) and row[col] not in disease_names:
            row_symptoms.append(row[col].strip().lower())
    if disease in disease_symptoms:
        disease_symptoms[disease].update(row_symptoms)
    else:
        disease_symptoms[disease] = set(row_symptoms)
print("Disease to symptoms mapping:", {d: list(s) for d, s in disease_symptoms.items()})

# -------------------------------------------
# Function to predict disease based on input symptoms using weighted scoring
# -------------------------------------------
def get_predicted_value(patient_symptoms):
    # Initialize scores for each disease
    disease_scores = {disease: 0 for disease in disease_symptoms.keys()}
    
    # For each disease, sum the severity weights of matching symptoms
    for disease, symptoms_set in disease_symptoms.items():
        score = 0
        for symptom in patient_symptoms:
            if symptom in symptoms_set:
                score += severity_dict.get(symptom, 1)
        # REMOVED: Random noise to break ties. max() will handle ties deterministically.
        # score += random.uniform(0, 0.5) 
        disease_scores[disease] = score
    print("Disease scores:", disease_scores)
    if disease_scores:
        predicted_disease = max(disease_scores, key=disease_scores.get)
    else:
        predicted_disease = "General Illness"
    print("Predicted Disease:", predicted_disease)
    return predicted_disease

# -------------------------------------------
# Helper function to get disease details from datasets
# -------------------------------------------
def helper(dis):
    print("Fetching details for disease:", dis)
    
    # Description: get the first available description.
    desc_series = description[description['Disease'] == dis]['Description']
    desc = desc_series.iloc[0] if not desc_series.empty else "No description available."
    
    # Precautions: retrieve the first row, convert to list, and wrap in a list if necessary.
    prec_df = precautions[precautions['Disease'] == dis]
    if not prec_df.empty:
        prec_list = [str(x) for x in prec_df.iloc[0] if pd.notna(x)]
        prec = prec_list
    else:
        prec = ["No precautions available."]
    
    # Medications: retrieve the first entry, wrap in list if it's a string.
    med_series = medications[medications['Disease'] == dis]['Medication']
    if not med_series.empty:
        med_val = med_series.iloc[0]
        if isinstance(med_val, str):
            med = [med_val]
        else:
            med = list(med_val)
    else:
        med = ["No medications available."]
    
    # Diet: retrieve the first entry, wrap in list if it's a string.
    diet_series = diets[diets['Disease'] == dis]['Diet']
    if not diet_series.empty:
        diet_val = diet_series.iloc[0]
        if isinstance(diet_val, str):
            diet = [diet_val]
        else:
            diet = list(diet_val)
    else:
        diet = ["No diet available."]
    
    # Workout: retrieve the first entry, wrap in list if it's a string.
    wrkout_series = workout[workout['disease'] == dis]['workout'] if 'disease' in workout.columns else pd.Series()
    if not wrkout_series.empty:
        w_val = wrkout_series.iloc[0]
        if isinstance(w_val, str):
            wrkout = [w_val]
        else:
            wrkout = list(w_val)
    else:
        wrkout = ["No workout available."]
    
    return desc, prec, med, diet, wrkout

# -------------------------------------------
# Flask Routes
# -------------------------------------------
@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/homepage")
def homepage_route():
    return render_template("homepage.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/index")
def index():
    return render_template("index2.html", symptoms_dict=display_symptoms)

@app.route("/emergency")
def emergency():
    return render_template("emergency.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms_text = request.form.get('symptoms')
        print("Received Symptoms Text:", symptoms_text)
        
        # Check if the input is empty
        if not symptoms_text:
            message = "Please enter at least one symptom."
            return render_template('index2.html', message=message, symptoms_dict=display_symptoms)
        
        # Normalize input: convert to lower-case and strip whitespace
        selected_symptoms = [s.strip().lower() for s in symptoms_text.split(',') if s.strip()]
        print("Selected Symptoms:", selected_symptoms)
        
        # Check if any valid symptoms were extracted after cleaning
        if not selected_symptoms:
            message = "Please enter valid symptoms."
            return render_template('index2.html', message=message, symptoms_dict=display_symptoms)
        
        predicted_disease = get_predicted_value(selected_symptoms)
        dis_des, prec, meds, diet, work = helper(predicted_disease)
        
        return render_template('index2.html', predicted_disease=predicted_disease,
                               dis_des=dis_des, my_precautions=prec,
                               medications=meds, my_diet=diet,
                               workout=work, symptoms_dict=display_symptoms)
    
    return render_template('index2.html', symptoms_dict=display_symptoms)

# -------------------------------------------
# Run the Flask app
# -------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)