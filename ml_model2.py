import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# PHASE 1: LOAD AND CLEAN THE DATA
# ==========================================
train_df = pd.read_csv('Training.csv') 

train_df = train_df.fillna(0)
train_df = train_df.drop_duplicates()
X_train = train_df.drop('prognosis', axis=1) 
y_train = train_df['prognosis']              

# Extract Top Symptoms
top_symptoms = list(X_train.columns)
print(f"Total symptoms used: {len(top_symptoms)}\n")

# ==========================================
# PHASE 2: TRAIN THE MODEL
# ==========================================
# Logistic Regression produces strictly calibrated probabilities
final_model = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
final_model.fit(X_train, y_train)

# The Prediction Function (For Testing)
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

# --- Quick Sanity Checks ---
my_symptoms = ['skin_rash', 'itching']
predict_smart(my_symptoms)

my_other_symptoms = ['cough', 'high_fever', 'loss_of_smell']
predict_smart(my_other_symptoms)


# ==========================================
# PHASE 3: MODEL EVALUATION FOR REPORT
# ==========================================
print("\n" + "="*50)
print("PHASE 3: MODEL EVALUATION ON TESTING DATA")
print("="*50)

# 1. Load and Clean the Testing dataset
test_df = pd.read_csv('Testing.csv') 
test_df = test_df.fillna(0)

# 2. Separate Features and Target
X_test = test_df.drop('prognosis', axis=1)
# Ensure columns align perfectly with training data
X_test = X_test.reindex(columns=top_symptoms, fill_value=0)
y_test = test_df['prognosis']

# 3. Make Predictions
y_pred = final_model.predict(X_test)

# 4. Calculate & Save Overall Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Output text file for report copy-pasting
with open("report_metrics.txt", "w") as f:
    f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report (Precision, Recall, F1-Score):\n")
    f.write(report)
print("-> Saved 'report_metrics.txt' (Use this for Chapter 4 tables).")

# 5. Generate & Save Confusion Matrix Image
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(18, 16)) # Large size to accommodate all disease names
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', 
            xticklabels=final_model.classes_, 
            yticklabels=final_model.classes_)
plt.title('Confusion Matrix - Predictive Diagnostic Engine', fontsize=16)
plt.ylabel('Actual Diagnosis', fontsize=12)
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('confusion_matrix_report.png', dpi=300) # 300 DPI is required for academic reports
plt.close()
print("-> Saved 'confusion_matrix_report.png' (Insert this image into Chapter 4).")

# 6. Extract & Save Feature Importance (Model Coefficients)
# For Logistic Regression, we find the maximum absolute coefficient for each symptom across all diseases
importance = np.max(np.abs(final_model.coef_), axis=0)
feature_importance_df = pd.DataFrame({
    'Symptom': top_symptoms,
    'Importance_Weight': importance
}).sort_values(by='Importance_Weight', ascending=False)

print("\nTop 10 Most Important Symptoms (Highest Predictive Weight):")
print(feature_importance_df.head(10).to_string(index=False))

# Export to CSV for report appendix or tables
feature_importance_df.to_csv('feature_importances_report.csv', index=False)
print("-> Saved 'feature_importances_report.csv' (Discuss these top symptoms in Chapter 4).")

# ==========================================
# PHASE 4: SAVE ARTIFACTS FOR API
# ==========================================
joblib.dump(final_model, 'triage_model.pkl')
joblib.dump(top_symptoms, 'symptoms_list.pkl')
print("\n" + "="*50)
print("SUCCESS: Model pipeline artifacts saved for the FastAPI backend!")
print("="*50)