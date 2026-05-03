import pandas as pd
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split

print("Initializing V4 Realistic Symptom Generator...\n")

# 1. The Overlapping Knowledge Base with Explicit Probabilities
# Format: 'symptom_name': probability (0.0 to 1.0)
clinical_knowledge = {
    # --- RESPIRATORY CLUSTER ---
    'Common Cold': {
        'specialist': 'General Physician', 
        'symptoms': {'cough': 0.95, 'fatigue': 0.85, 'loss_of_smell': 0.15, 'runny_nose': 0.90, 'continuous_sneezing': 0.80, 'sore_throat': 0.70, 'mild_fever': 0.60}
    },
    'Influenza (Flu)': {
        'specialist': 'General Physician', 
        'symptoms': {'cough': 0.90, 'high_fever': 0.95, 'fatigue': 0.95, 'chills': 0.80, 'headache': 0.75, 'muscle_pain': 0.85}
    },
    'COVID-19': {
        'specialist': 'General Physician', 
        'symptoms': {'cough': 0.85, 'fatigue': 0.85, 'loss_of_smell': 0.60, 'loss_of_taste': 0.60, 'high_fever': 0.70, 'breathlessness': 0.40, 'muscle_pain': 0.50}
    },
    'Bronchitis': {
        'specialist': 'Pulmonologist', 
        'symptoms': {'cough': 0.98, 'mucoid_sputum': 0.80, 'chest_pain': 0.60, 'mild_fever': 0.50, 'fatigue': 0.60}
    },
    
    # --- SKIN CLUSTER ---
    'Fungal Infection': {
        'specialist': 'Dermatologist', 
        'symptoms': {'skin_rash': 0.95, 'itching': 0.95, 'nodal_skin_eruptions': 0.60, 'dischromic_patches': 0.50}
    },
    'Eczema': {
        'specialist': 'Dermatologist', 
        'symptoms': {'skin_rash': 0.90, 'itching': 0.98, 'dry_skin': 0.90, 'red_spots_over_body': 0.70}
    },
    'Psoriasis': {
        'specialist': 'Dermatologist', 
        'symptoms': {'skin_rash': 0.90, 'itching': 0.70, 'skin_peeling': 0.85, 'silver_like_dusting': 0.80, 'joint_pain': 0.30}
    },
    'Acne': {
        'specialist': 'Dermatologist', 
        'symptoms': {'skin_rash': 0.90, 'pus_filled_pimples': 0.85, 'blackheads': 0.80, 'scurring': 0.40}
    },

    # --- GI CLUSTER ---
    'GERD': {
        'specialist': 'Gastroenterologist', 
        'symptoms': {'stomach_pain': 0.40, 'acidity': 0.95, 'chest_pain': 0.60, 'vomiting': 0.30, 'cough': 0.20}
    },
    'Peptic Ulcer': {
        'specialist': 'Gastroenterologist', 
        'symptoms': {'stomach_pain': 0.95, 'vomiting': 0.60, 'indigestion': 0.80, 'loss_of_appetite': 0.70, 'internal_itching': 0.10}
    },
    'Gastroenteritis': {
        'specialist': 'Gastroenterologist', 
        'symptoms': {'stomach_pain': 0.90, 'diarrhoea': 0.95, 'vomiting': 0.85, 'dehydration': 0.60, 'mild_fever': 0.40}
    },

    # --- OTHERS ---
    'Allergy': {
        'specialist': 'Allergist', 
        'symptoms': {'continuous_sneezing': 0.90, 'watering_from_eyes': 0.85, 'itchy_nose': 0.80, 'skin_rash': 0.30}
    },
    'Asthma': {
        'specialist': 'Pulmonologist', 
        'symptoms': {'breathlessness': 0.95, 'wheezing': 0.90, 'cough': 0.80, 'chest_tightness': 0.75}
    },
    'Migraine': {
        'specialist': 'Neurologist', 
        'symptoms': {'headache': 0.98, 'sensitivity_to_light': 0.85, 'blurred_and_distorted_vision': 0.50, 'nausea': 0.70, 'dizziness': 0.40}
    },
    'Hypertension': {
        'specialist': 'Cardiologist', 
        'symptoms': {'headache': 0.60, 'dizziness': 0.70, 'loss_of_balance': 0.40, 'chest_pain': 0.30}
    },
    'Type 2 Diabetes': {
        'specialist': 'Endocrinologist', 
        'symptoms': {'frequent_urination': 0.90, 'fatigue': 0.85, 'increased_thirst': 0.85, 'excessive_hunger': 0.70, 'weight_loss': 0.50, 'blurred_and_distorted_vision': 0.40}
    },
    'Urinary Tract Infection': {
        'specialist': 'Urologist', 
        'symptoms': {'frequent_urination': 0.95, 'burning_micturition': 0.95, 'bladder_discomfort': 0.80, 'foul_smell_of_urine': 0.70, 'mild_fever': 0.30}
    },
    'Osteoarthritis': {
        'specialist': 'Rheumatologist', 
        'symptoms': {'joint_pain': 0.95, 'knee_pain': 0.80, 'neck_pain': 0.60, 'swelling_joints': 0.70}
    },
    'Conjunctivitis (Pink Eye)': {
        'specialist': 'Ophthalmologist', 
        'symptoms': {'redness_of_eyes': 0.98, 'itching_eyes': 0.95, 'watering_from_eyes': 0.85, 'pain_in_eye': 0.70}
    },
    'Tonsillitis': {
        'specialist': 'ENT Specialist', 
        'symptoms': {'sore_throat': 0.98, 'difficulty_swallowing': 0.90, 'high_fever': 0.70, 'patches_in_throat': 0.60}
    }
}

all_symptoms = set()
specialist_map = {"UNKNOWN_OR_LOW_CONFIDENCE": "General Physician"}

for disease, info in clinical_knowledge.items():
    for sym in info['symptoms'].keys():
        all_symptoms.add(sym)
    specialist_map[disease] = info['specialist']

all_symptoms = sorted(list(all_symptoms))

# 3. Generate Patients with Realistic Probabilities
NUM_PATIENTS = 10000
NOISE_LEVEL = 0.05 # Small background noise for misdiagnoses

data = []
disease_names = list(clinical_knowledge.items())

for _ in range(NUM_PATIENTS):
    disease, info = random.choice(disease_names)
    patient_row = {'prognosis': disease}
    
    for sym in all_symptoms:
        patient_row[sym] = 0
        
    # Apply symptoms based on their actual probabilities!
    for sym, prob in info['symptoms'].items():
        if random.random() < prob:
            patient_row[sym] = 1
            
    # CORRELATED SYMPTOM LOGIC
    # E.g., if diarrhoea is present, dehydration becomes highly likely
    if patient_row.get('diarrhoea', 0) == 1 and random.random() < 0.85:
        patient_row['dehydration'] = 1
        
    # If vomiting is present, nausea is almost guaranteed
    if patient_row.get('vomiting', 0) == 1 and random.random() < 0.95:
        patient_row['nausea'] = 1
        
    # If high_fever is present, chills and fatigue are highly likely
    if patient_row.get('high_fever', 0) == 1 and random.random() < 0.80:
        patient_row['chills'] = 1
        patient_row['fatigue'] = 1
            
    # Add pure background noise 
    for sym in all_symptoms:
        if random.random() < NOISE_LEVEL:
            patient_row[sym] = 1
            
    data.append(patient_row)

df = pd.DataFrame(data)
cols = [c for c in df.columns if c != 'prognosis'] + ['prognosis']
df = df[cols]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('Training.csv', index=False)
test_df.to_csv('Testing.csv', index=False)
with open('specialist_mapping.json', 'w') as f:
    json.dump(specialist_map, f, indent=4)
with open('symptoms_prompt_list.txt', 'w') as f:
    f.write(str(all_symptoms))

print(f"✅ V4 Generation Complete. Realistic probabilistic modeling applied to {len(all_symptoms)} symptoms.")