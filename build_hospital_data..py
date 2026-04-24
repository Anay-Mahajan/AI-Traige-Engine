import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split

print("Initializing V3 Controlled Ambiguity Generator...\n")

# 1. The Overlapping Knowledge Base
# Notice how diseases in the same "cluster" now share mandatory symptoms.
clinical_knowledge = {
    # --- RESPIRATORY CLUSTER (The model will confuse these) ---
    'Common Cold': {'specialist': 'General Physician', 'mandatory': ['cough', 'fatigue', 'loss_of_smell'], 'optional': ['runny_nose', 'continuous_sneezing', 'sore_throat', 'mild_fever']},
    'Influenza (Flu)': {'specialist': 'General Physician', 'mandatory': ['cough', 'high_fever', 'fatigue'], 'optional': ['chills', 'headache', 'muscle_pain']},
    'COVID-19': {'specialist': 'General Physician', 'mandatory': ['cough', 'fatigue'], 'optional': ['loss_of_smell', 'loss_of_taste', 'high_fever', 'breathlessness']},
    'Bronchitis': {'specialist': 'Pulmonologist', 'mandatory': ['cough'], 'optional': ['mucoid_sputum', 'chest_pain', 'mild_fever']},
    
    # --- SKIN CLUSTER (The model will confuse these) ---
    'Fungal Infection': {'specialist': 'Dermatologist', 'mandatory': ['skin_rash', 'itching'], 'optional': ['nodal_skin_eruptions', 'dischromic_patches']},
    'Eczema': {'specialist': 'Dermatologist', 'mandatory': ['skin_rash', 'itching'], 'optional': ['dry_skin', 'red_spots_over_body']},
    'Psoriasis': {'specialist': 'Dermatologist', 'mandatory': ['skin_rash', 'itching'], 'optional': ['skin_peeling', 'silver_like_dusting', 'joint_pain']},
    'Acne': {'specialist': 'Dermatologist', 'mandatory': ['skin_rash'], 'optional': ['pus_filled_pimples', 'blackheads', 'scurring']},

    # --- GI CLUSTER (The model will confuse these) ---
    'GERD': {'specialist': 'Gastroenterologist', 'mandatory': ['stomach_pain', 'acidity'], 'optional': ['chest_pain', 'vomiting']},
    'Peptic Ulcer': {'specialist': 'Gastroenterologist', 'mandatory': ['stomach_pain', 'vomiting'], 'optional': ['indigestion', 'loss_of_appetite']},
    'Gastroenteritis': {'specialist': 'Gastroenterologist', 'mandatory': ['stomach_pain', 'diarrhoea', 'vomiting'], 'optional': ['dehydration']},

    # --- OTHERS (More distinct, higher confidence) ---
    'Allergy': {'specialist': 'Allergist', 'mandatory': ['continuous_sneezing', 'watering_from_eyes'], 'optional': ['itchy_nose']},
    'Asthma': {'specialist': 'Pulmonologist', 'mandatory': ['breathlessness', 'wheezing'], 'optional': ['cough', 'chest_tightness']},
    'Migraine': {'specialist': 'Neurologist', 'mandatory': ['headache'], 'optional': ['sensitivity_to_light', 'blurred_and_distorted_vision', 'nausea']},
    'Hypertension': {'specialist': 'Cardiologist', 'mandatory': ['headache', 'dizziness'], 'optional': ['loss_of_balance', 'chest_pain']},
    'Type 2 Diabetes': {'specialist': 'Endocrinologist', 'mandatory': ['frequent_urination', 'fatigue'], 'optional': ['increased_thirst', 'excessive_hunger', 'weight_loss']},
    'Urinary Tract Infection': {'specialist': 'Urologist', 'mandatory': ['frequent_urination', 'burning_micturition'], 'optional': ['bladder_discomfort', 'foul_smell_of_urine']},
    'Osteoarthritis': {'specialist': 'Rheumatologist', 'mandatory': ['joint_pain'], 'optional': ['knee_pain', 'neck_pain', 'swelling_joints']},
    'Conjunctivitis (Pink Eye)': {'specialist': 'Ophthalmologist', 'mandatory': ['redness_of_eyes', 'itching_eyes'], 'optional': ['watering_from_eyes', 'pain_in_eye']},
    'Tonsillitis': {'specialist': 'ENT Specialist', 'mandatory': ['sore_throat'], 'optional': ['difficulty_swallowing', 'high_fever', 'patches_in_throat']}
}

all_symptoms = set()
specialist_map = {"UNKNOWN_OR_LOW_CONFIDENCE": "General Physician"}

for disease, info in clinical_knowledge.items():
    all_symptoms.update(info['mandatory'])
    all_symptoms.update(info['optional'])
    specialist_map[disease] = info['specialist']

all_symptoms = sorted(list(all_symptoms))

# 3. Generate Patients with "Messy" Probabilities
NUM_PATIENTS = 8000
NOISE_LEVEL = 0.04   # Bumping global noise to 4% (adds random realistic errors)

data = []
disease_names = list(clinical_knowledge.items())

for _ in range(NUM_PATIENTS):
    disease, info = random.choice(disease_names)
    patient_row = {'prognosis': disease}
    
    for sym in all_symptoms:
        patient_row[sym] = 0
        
    # Mandatory Symptoms (Dropped to 90% chance to simulate patients forgetting to mention it)
    for sym in info['mandatory']:
        if random.random() < 0.90:
            patient_row[sym] = 1
            
    # Optional Symptoms (Patient only mentions 1 or 2)
    num_optional = random.randint(1, 3)
    available_optional = info['optional']
    chosen_optional = random.sample(available_optional, min(num_optional, len(available_optional)))
    
    for sym in chosen_optional:
        patient_row[sym] = 1
        
    # Add pure noise 
    for sym in all_symptoms:
        if random.random() < NOISE_LEVEL:
            patient_row[sym] = 1
            
    data.append(patient_row)

df = pd.DataFrame(data)
cols = [c for c in df.columns if c != 'prognosis'] + ['prognosis']
df = df[cols]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('Training_Realistic.csv', index=False)
test_df.to_csv('Testing_Realistic.csv', index=False)
with open('specialist_mapping.json', 'w') as f:
    json.dump(specialist_map, f, indent=4)
with open('symptoms_prompt_list.txt', 'w') as f:
    f.write(str(all_symptoms))

print(f"✅ V3 Generation Complete. Controlled Ambiguity injected into {len(all_symptoms)} symptoms.")