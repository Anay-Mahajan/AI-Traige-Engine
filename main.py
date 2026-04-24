from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json

app = FastAPI(title="AI Triage Engine")


model = joblib.load('triage_model.pkl')
symptoms_list = joblib.load('symptoms_list.pkl')

with open('specialist_mapping.json', 'r') as f:
    specialist_map = json.load(f)

# 2. Define the Input Schema
class PatientInput(BaseModel):
    symptoms: list[str]


@app.post("/triage")
def run_triage(patient: PatientInput):
    input_data = np.zeros(len(symptoms_list))
    recognized = []
    
    for sym in patient.symptoms:
        if sym in symptoms_list:
            index = symptoms_list.index(sym)
            input_data[index] = 1
            recognized.append(sym)
            
    input_df = pd.DataFrame([input_data], columns=symptoms_list)
    

    probabilities = model.predict_proba(input_df)[0]
    disease_probs = list(zip(model.classes_, probabilities))
    disease_probs.sort(key=lambda x: x[1], reverse=True)    
    top_disease = disease_probs[0][0]
    top_confidence = float(disease_probs[0][1])
    response = {
        "recognized_symptoms": recognized,
        "top_disease": top_disease,
        "confidence": round(top_confidence * 100, 1),
        "differential": [
            {"disease": disease_probs[0][0], "prob": round(float(disease_probs[0][1]) * 100, 1)},
            {"disease": disease_probs[1][0], "prob": round(float(disease_probs[1][1]) * 100, 1)},
            {"disease": disease_probs[2][0], "prob": round(float(disease_probs[2][1]) * 100, 1)}
        ]
    }
    if top_confidence < 0.50:
        response["action"] = "TRIGGER_LLM"
        response["specialist"] = "Pending"
        response["llm_instruction"] = f"The ML model is unsure. It suspects {disease_probs[0][0]} or {disease_probs[1][0]}. Ask the patient a highly specific follow-up question to tell these two apart."
    else:
        response["action"] = "ROUTE_PATIENT"
        response["specialist"] = specialist_map.get(top_disease, "General Physician")
        response["llm_instruction"] = "Diagnosis confident. Summarize the chat for the doctor."

    return response