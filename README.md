# 🏥 AI Triage Engine

An intelligent, hybrid clinical decision-support microservice that combines the mathematical precision of **Deterministic Machine Learning** with the contextual flexibility of **Large Language Models (LLMs)**. 

This engine is designed to interact with patients, extract complex symptom sets, compute differential diagnoses with calibrated statistical models, resolve borderline classifications through conversational follow-ups, and generate customized recovery diet plans.

---

## 🧠 The Hybrid AI Approach: Best of Both Worlds

In clinical triage, purely LLM-based systems or purely ML-based systems suffer from distinct, critical limitations. This project introduces a **Hybrid Architecture** that bridges these paradigms:

```
                  +----------------------------------------------+
                  |            Patient Natural Text              |
                  +----------------------------------------------+
                                         |
                                         v
   +----------------------------------------------------------------------------+
   |                        Conversational LLM (Gemini)                         |
   |  - Parses unstructured raw text into structured symptoms                   |
   |  - Deduces context (e.g., matching "no, none" to the asked symptom)       |
   |  - Extracts both Confirmed (Present) & Denied (Absent) symptoms            |
   +----------------------------------------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |         Structured Binary Feature Vector     |
                  |             [1, 0, 0, 1, 0, 1, 0...]         |
                  +----------------------------------------------+
                                         |
                                         v
   +----------------------------------------------------------------------------+
   |                    Deterministic ML Classifier (SciKit)                    |
   |  - Computes probability distribution across 20 clinical conditions         |
   |  - Guarantees calibrated, bounded, non-hallucinated probabilities          |
   |  - Evaluates feature importances & correlations mathematically             |
   +----------------------------------------------------------------------------+
                                         |
                       +-----------------+-----------------+
                       | < 65% Confidence                  | >= 65% Confidence
                       v                                   v
   +---------------------------------------+   +---------------------------------------+
   |        LLM Tie-Breaker Query          |   |       LLM Clinical Review Board       |
   | - Analyzes borderline classes         |   | - Cross-references predictions        |
   | - Asks empathetic follow-up question  |   |   against denied symptoms (negatives) |
   | - Loop repeats up to 2 times          |   | - Finalizes diagnosis & routes doctor |
   +---------------------------------------+   | - Generates BMI recovery diet plan    |
                                               +---------------------------------------+
```

### 🔬 Why This Hybrid Design is Superior:

1. **Why Pure LLMs Fail at Diagnosis**: 
   * **Hallucination & Probability Violations**: LLMs cannot guarantee mathematical probability bounds (i.e. the sum of all differential diagnoses equals $1.0$). They can hallucinate random statistics.
   * **Unbounded Decision Boundaries**: LLMs lack consistent decision boundaries, making their diagnoses fluctuate based on phrasing.

2. **Why Pure ML Models Fail at Patient Interaction**:
   * **Dialogue Blindness**: ML classifiers require a static, binary input vector. They cannot parse free text like *"My throat is scratchy and I feel like I've been hit by a truck."*
   * **The "Negative Space" Ignorance**: Standard ML models operate on positive indicator arrays. If a patient is diagnosed with *Migraine*, the model does not mathematically discount it if the patient explicitly says *"I do NOT have nausea."* It simply looks at the positive indicators.

3. **How the Hybrid Engine Combines Them**:
   * **The LLM (Gemini)** acts as the **Sensory Cortex** (Natural Language Parsing, Context, Empathy, Reasoning about "negative" symptoms).
   * **The ML Model (Random Forest / Logistic Regression)** acts as the **Subcortical Brain** (computing deterministic statistical fits, feature importance, and diagnostic probabilities).
   * **The LLM Clinical Review Board** checks the ML output against the **denied symptoms** to make a final expert decision, bypassing the ML model's blind spots.

---

## 📂 Project Repository Structure

* **`build_hospital_data..py`**: Generates a high-quality probabilistic synthetic dataset of 10,000 patient records using real-world clinical relationships (e.g., linking fever to chills, and diarrhoea to dehydration).
* **`Training.csv` & `Testing.csv`**: Synthesized datasets with an 80/20 train/test split.
* **`ml_model.py` / `ml_model.ipynb`**: Trains and serializes a robust **Random Forest Classifier** (`triage_model.pkl`), generating academic evaluation metrics (`report_metrics.txt`, `confusion_matrix_report.png`, `feature_importances_report.csv`).
* **`main.py`**: The core FastAPI backend integrating the machine learning models with the **Google GenAI Client SDK** (`google-genai`).
* **`specialist_mapping.json`**: Clinical routing database linking diagnoses to medical departments (e.g., Gastroenterologist, Pulmonologist).
* **`symptoms_list.pkl`**: Serialized list of symptoms matching the ML model's training columns.
* **`.env`**: Stores local environment secrets (e.g., `GEMINI_API_KEY`).

---

## 🛠️ Setup & Installation

### 1. Clone & Initialize Environment
Set up a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```
*(Dependencies include: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `google-genai`, `python-dotenv`, `matplotlib`, `seaborn`)*

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Synthesize Data & Train the Model
Generate realistic patient data and train the classifier:
```bash
python build_hospital_data..py
python ml_model.py
```
This will train the calibrated Random Forest model, write metrics to `report_metrics.txt`, save the confusion matrix to `confusion_matrix_report.png`, and serialize the pipeline to `triage_model.pkl` and `symptoms_list.pkl`.

---

## 🚀 Running the FastAPI Service

Start the Uvicorn development server:
```bash
uvicorn main:app --reload --port 8000
```
The microservice API will be live at `http://127.0.0.1:8000`. You can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

---

## 🔗 API Documentation

### **Endpoint**: `POST /api/v1/chat`

Handles the stateful-like multi-turn conversational triage flow.

#### **Request Body Schema**:
```json
{
  "user_message": "I have a sudden high fever and chills, and my body really aches.",
  "current_symptoms": [],
  "denied_symptoms": [],
  "chat_history": [],
  "weight_kg": 72.5,
  "height_m": 1.78
}
```

#### **Responses**:

##### **Case 1: Under-Confident Diagnosis (Confidence < 65%)**
If the ML model is torn between conditions, it triggers an interactive tie-breaking question.
* **Response**:
  ```json
  {
    "status": "ASKING_QUESTION",
    "bot_reply": "I see. Along with your fever and body aches, are you experiencing any dry skin or skin rashes?",
    "tracked_symptoms": ["high_fever", "chills", "muscle_pain"],
    "denied_symptoms": [],
    "predicted_disease": "Pending",
    "confidence": 42.8
  }
  ```

##### **Case 2: Triage Complete (Confidence >= 65%)**
If the ML model is confident, Gemini filters the prediction against explicitly denied symptoms, routes to a specialist, and returns a personalized recovery diet plan using the patient's BMI.
* **Response**:
  ```json
  {
    "status": "TRIAGE_COMPLETE",
    "bot_reply": "Based on your symptoms, I suspect you may have Influenza (Flu). Please book an appointment with a General Physician. I have also generated a customized diet plan for your recovery.",
    "tracked_symptoms": ["high_fever", "chills", "muscle_pain", "headache"],
    "denied_symptoms": ["skin_rash"],
    "predicted_disease": "Influenza (Flu)",
    "confidence": 88.5,
    "recommended_specialist": "General Physician",
    "diet_plan": "### 2-Day recovery Diet Plan (BMI: 22.9 - Normal)\n* **Day 1**: Warm vegetable broth with ginger, chamomile tea, oatmeal with banana.\n* **Day 2**: Baked chicken breast with steamed carrots, coconut water for hydration, scrambled eggs."
  }
  ```

---

## ⚖️ License & Attribution
 Integrates custom probabilistic symptom modeling with Google GenAI capabilities.
