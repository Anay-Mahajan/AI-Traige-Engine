from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import json

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
from google import genai

app = FastAPI(title="AI Healthcare Microservice")

import os
from dotenv import load_dotenv

load_dotenv()

# 2. NEW CLIENT SETUP 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

# Load Machine Learning Artifacts
try:
    model = joblib.load('triage_model.pkl')
    symptoms_list = joblib.load('symptoms_list.pkl')
    with open('specialist_mapping.json', 'r') as f:
        specialist_map = json.load(f)
except Exception as e:
    print(f"Error loading ML files: {e}. Please ensure .pkl and .json files are in the directory.")


# ==========================================
# 2. DATA MODELS (Communication with Java)
# ==========================================
class ChatRequest(BaseModel):
    user_message: str
    current_symptoms: List[str] = []
    denied_symptoms: Optional[List[str]] = []  # Added this so Java doesn't crash!
    chat_history: Optional[List[str]] = []
    weight_kg: Optional[float] = None
    height_m: Optional[float] = None


# ==========================================
# 3. LLM HELPER FUNCTIONS
# ==========================================
def extract_symptoms(message: str, chat_history: list) -> dict:
    recent_context = "\\n".join(chat_history[-4:]) if chat_history else "No previous context."
    prompt = f"""
    You are an expert clinical data extractor. 
    
    Recent Chat History for Context:
    {recent_context}
    
    Read the latest patient message: "{message}"
    
    Look at the AI's immediately preceding question in the chat history. If the patient answers with a short "no", "nope", or "none", you MUST figure out which symptom the AI was asking about.
    Map that symptom to the EXACT closest match in this strict list of allowed snake_case symptoms: {symptoms_list}
    
    Identify which symptoms the patient HAS (present) and which they EXPLICITLY DENY (denied).
    Output ONLY a valid JSON object. Do not use markdown backticks.
    
    Example Output:
    {{"present": ["fatigue"], "denied": ["headache", "loss_of_smell"]}}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt
        )
        raw_text = response.text.strip()
        
        # Strip markdown if Gemini disobeys
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()
            
        print(f"--- DEBUG: LLM Extraction --- \n{raw_text}")
        
        # Parse the JSON string into a Python dictionary
        extracted_data = json.loads(raw_text)
        return extracted_data
    except Exception as e:
        print(f"--- DEBUG: Extraction Error --- \n{e}")
        return {"present": [], "denied": []}


def generate_followup(instruction: str, current_symptoms: list, denied_symptoms: list,chat_history: list) -> str:
    recent_context = "\\n".join(chat_history[-4:]) if chat_history else "No previous context."
    prompt = f"""
    You are an empathetic triage nurse. The diagnostic system says: "{instruction}"
    
    Recent Chat History:
    {recent_context}
    
    PATIENT CHART:
    - Confirmed Symptoms: {current_symptoms}
    - Denied Symptoms: {denied_symptoms}
    
    CRITICAL INSTRUCTION: Do NOT ask if the patient has any of the symptoms on EITHER of those lists. 
    You must ask about a DIFFERENT symptom to break the tie. Translate this into a single, polite question.
    """
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=prompt
    )
    return response.text.strip()


def generate_diet_plan(disease: str, bmi: float) -> str:
    category = "Normal"
    if bmi > 25: category = "Overweight"
    if bmi < 18.5: category = "Underweight"
    
    prompt = f"""
    You are a Clinical Nutritionist. Write a brief 2-day diet plan for a patient with suspected {disease}. 
    Their BMI is {bmi:.1f} ({category}). Tailor the diet to both their disease and BMI. 
    Use clean bullet points. Keep it under 100 words.
    """
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=prompt
    )
    return response.text.strip()


def finalize_diagnosis(ml_predictions: list, current_symptoms: list, denied_symptoms: list) -> str:
    prompt = f"""
    You are an expert diagnostician. 
    A patient has confirmed these symptoms: {current_symptoms}
    The patient has EXPLICITLY DENIED these symptoms: {denied_symptoms}
    
    A statistical model suggests these top possibilities: {ml_predictions}
    
    Considering the denied symptoms, the statistical model might be wrong! For example, if it predicted Migraine but the patient denied nausea and light sensitivity, it is likely just a Tension Headache or Normal Headache.
    
    Select the most accurate diagnosis from the list of possibilities, OR suggest a more accurate one based on the denied symptoms.
    Output ONLY the name of the final disease. Do not explain.
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip().replace("\"", "")
    except:
        return ml_predictions[0]


# ==========================================
# 4. THE MASTER ENDPOINT
# ==========================================
@app.post("/api/v1/chat")
def handle_chat(request: ChatRequest):
    print(f"\nIncoming Message: {request.user_message}")
    print(f"Previous Present Symptoms: {request.current_symptoms}")
    print(f"Previous Denied Symptoms: {request.denied_symptoms}")

    # Step 1: Use LLM to extract symptoms (returns a dictionary)
    extraction = extract_symptoms(request.user_message, request.chat_history)
    print(f"Newly Extracted Symptoms: {extraction}")
    
    # Safely pull the lists out of the dictionary
    new_present = extraction.get("present", [])
    new_denied = extraction.get("denied", [])
    
    # Merge new PRESENT symptoms with existing ones from Java
    all_present = list(set(request.current_symptoms + new_present))
    
    # Merge new DENIED symptoms
    existing_denied = getattr(request, "denied_symptoms", [])
    if existing_denied is None:
        existing_denied = []
    all_denied = list(set(existing_denied + new_denied))
    
    print(f"Combined Master Present List: {all_present}")
    print(f"Combined Master Denied List: {all_denied}")
    
    # If no symptoms are present at all yet, prompt the user
    if not all_present:
        return {
            "status": "CONTINUE_CHAT",
            "bot_reply": "I'm sorry, I didn't quite catch your symptoms. Could you describe how you are feeling?",
            "tracked_symptoms": [],
            "denied_symptoms": all_denied
        }

    # Step 2: Prepare data for ML Model
    input_data = np.zeros(len(symptoms_list))
    for sym in all_present:
        if sym in symptoms_list:
            index = symptoms_list.index(sym)
            input_data[index] = 1
            
    # Convert to DataFrame (Many scikit-learn models expect DataFrames with valid feature names)
    input_df = pd.DataFrame([input_data], columns=symptoms_list)
    
    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    disease_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    
    top_disease = disease_probs[0][0]
    top_confidence = disease_probs[0][1]
    
    print(f"ML Prediction: {top_disease} at {top_confidence:.2%} confidence")

    # Step 3: Routing Logic
    # We will ask follow-ups ONLY if confidence is low AND the user hasn't denied too many symptoms.
    # This prevents the AI from endlessly interrogating the patient if they keep saying "No".
    if top_confidence < 0.65 and len(all_denied) < 2:
        # Model is unsure. Ask a follow-up question.
        instruction = f"The model suspects {disease_probs[0][0]} or {disease_probs[1][0]}. Ask a specific question to tell them apart."
        
        bot_reply = generate_followup(instruction, all_present, all_denied,request.chat_history)
        
        return {
            "status": "ASKING_QUESTION",
            "bot_reply": bot_reply,
            "tracked_symptoms": all_present,
            "denied_symptoms": all_denied,
            "predicted_disease": "Pending",
            "confidence": round(top_confidence * 100, 1)
        }
        
    else:
        # Model is confident or we reached denial cap. 
        # Have Gemini finalize the triage to account for denied symptoms (since ML model ignores them).
        top_3_diseases = [d[0] for d in disease_probs[:3]]
        final_disease = finalize_diagnosis(top_3_diseases, all_present, all_denied)
        
        diet_plan = "No BMI data provided for diet generation."
        if request.weight_kg is not None and request.height_m is not None and request.height_m > 0:
            bmi = request.weight_kg / (request.height_m ** 2)
            diet_plan = generate_diet_plan(final_disease, bmi)
            
        recommended_doctor = specialist_map.get(final_disease, "General Physician")
        
        return {
            "status": "TRIAGE_COMPLETE",
            "bot_reply": f"Based on your symptoms, I suspect you may have {final_disease}. Please book an appointment with a {recommended_doctor}. I have also generated a customized diet plan for your recovery.",
            "tracked_symptoms": all_present,
            "denied_symptoms": all_denied,
            "predicted_disease": final_disease,
            "confidence": round(top_confidence * 100, 1),
            "diet_plan": diet_plan,
            "recommended_specialist": recommended_doctor
        }