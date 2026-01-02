import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the Dataset
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
data = pd.read_csv(url)

# Data Cleaning Process
empty_col= ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[empty_col] = data[empty_col].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Features scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
    
# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Get Input from the User
print("\nEnter Patient Details:")
pregnancies = float(input("Number of Pregnancies: "))
glucose = float(input("Glucose Level: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))
patient = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]],
                        columns=X.columns)
patient_scaled = scaler.transform(patient)
prediction = model.predict(patient_scaled)

print("\n🩺 Diabetes Prediction Result:")
if prediction[0] == 1:
    print("The patient is DIABETIC")
else:
    print("The patient is NOT DIABETIC")
