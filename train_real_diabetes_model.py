import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

# Load the real dataset
df = pd.read_csv("diabetes_data_upload.csv")  # make sure it's in your project folder

# Encode all categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Features and label
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "model/lightgbm_real_diabetes_model.pkl")
print("✅ Model saved to model/lightgbm_real_diabetes_model.pkl")
