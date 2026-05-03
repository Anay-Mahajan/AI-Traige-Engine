import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load and Clean the Data
train_df = pd.read_csv('Training.csv') 

train_df = train_df.fillna(0)
train_df = train_df.drop_duplicates()
X_train = train_df.drop('prognosis', axis=1) 
y_train = train_df['prognosis']              

# 2. Extract Top Symptoms
top_symptoms = list(X_train.columns)
print(f"Total symptoms used: {len(top_symptoms)}\n")

# 5. Train the FINAL model
# Logistic Regression produces strictly calibrated probabilities
final_model = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
final_model.fit(X_train, y_train)

# 6. The Prediction Function (For Testing)
def predict_smart(user_symptoms):
    input_data = np.zeros(len(top_symptoms))

    recognized_symptoms = []
    for symptom in user_symptoms:
        if symptom in top_symptoms:
            index = top_symptoms.index(symptom)
            input_data[index] = 1
            recognized_symptoms.append(symptom)
            
    input_df = pd.DataFrame([input_data], columns=top_symptoms)

    probabilities = final_model.predict_proba(input_df)[0]
    disease_probs = list(zip(final_model.classes_, probabilities))
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAnalyzing recognized symptoms: {recognized_symptoms}")
    print("Differential Diagnosis:")

    # Print the top 3 possibilities
    for disease, prob in disease_probs[:3]:
        if prob > 0:
            print(f"- {disease}: {prob * 100:.1f}%")
    print("-" * 40)

# --- Test it out! ---
my_symptoms = ['skin_rash', 'itching']
predict_smart(my_symptoms)

my_other_symptoms = ['cough', 'high_fever', 'loss_of_smell']
predict_smart(my_other_symptoms)


print("\n" + "="*50)
print("PHASE 3: MODEL EVALUATION ON TESTING DATA")
print("="*50)

# 1. Load the Testing dataset
test_df = pd.read_csv('Testing.csv') 

# 2. Clean the Data
test_df = test_df.fillna(0)

# 3. Separate Features and Target
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

# 4. Make Predictions on the unseen data
y_pred = final_model.predict(X_test)

# 5. Calculate Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%\n")

# 6. Generate a Classification Report
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and the symptom list
joblib.dump(final_model, 'triage_model.pkl')
joblib.dump(top_symptoms, 'symptoms_list.pkl')
print("Model pipeline saved for the API!")
