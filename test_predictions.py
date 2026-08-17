import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print('Testing predictions with compatible XGBoost version...')
print()

# Load training data and prepare scaler
DATA_PATH = os.path.join('data', 'Customer-Dataset.csv')
df_train = pd.read_csv(DATA_PATH)
df_train.dropna(subset=['CreditScore'], inplace=True)

gender_encoder = joblib.load(os.path.join('models', 'gender_encoder.pkl'))
df_train.drop(columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname'], inplace=True)
df_train = df_train.replace(gender_encoder)

geo_encoder = OneHotEncoder(sparse_output=False)
geo_features = df_train[['Geography']]
geo_encoded = geo_encoder.fit_transform(geo_features)
geo_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
df_train = df_train.reset_index(drop=True)
geo_df = geo_df.reset_index(drop=True)
df_train = pd.concat([df_train, geo_df], axis=1)
df_train.drop(columns=['Geography'], inplace=True)

scaler = StandardScaler()
scaler.fit(df_train[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])

# Get the training feature order
training_features = df_train.drop('churn', axis=1)
feature_order = list(training_features.columns)

model = joblib.load(os.path.join('models', 'xgb_classifier_model.pkl'))

# Test A: High-risk customer
print('TEST A: High-Risk Profile')
print('-' * 50)
sample_data_a = {
    'CreditScore': 450,
    'Gender': 'Male', 
    'Age': 60, 
    'Tenure': 2, 
    'Balance': 0,
    'NumOfProducts': 3, 
    'HasCrCard': 'No',
    'IsActiveMember': 'No',
    'EstimatedSalary': 50000,
    'Geography': 'Germany',
    'Surname': 'TestA'
}

df_a = pd.DataFrame([sample_data_a])
df_a = df_a.replace(gender_encoder)

geo_encoded_a = geo_encoder.transform(df_a[['Geography']])
geo_df_a = pd.DataFrame(geo_encoded_a, columns=geo_encoder.get_feature_names_out(['Geography']))
df_a = df_a.reset_index(drop=True)
geo_df_a = geo_df_a.reset_index(drop=True)
df_a = pd.concat([df_a, geo_df_a], axis=1)
df_a.drop(columns=['Geography'], inplace=True)

hasCrCard_encoder = joblib.load(os.path.join('models', 'hasCrCard_encoder.pkl'))
isActiveMember_encoder = joblib.load(os.path.join('models', 'isActiveMember_encoder.pkl'))
df_a = df_a.replace(hasCrCard_encoder)
df_a = df_a.replace(isActiveMember_encoder)

df_a[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.transform(
    df_a[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
)

features_a = df_a[feature_order]
input_a = np.array(features_a).reshape(1, -1)
pred_a = model.predict(input_a)[0]

print('Customer Details:')
print(f'  Age: 60, Credit Score: 450, Tenure: 2, Balance: 0')
print(f'  NumOfProducts: 3, HasCrCard: No, IsActiveMember: No')
print(f'  EstimatedSalary: 50000, Geography: Germany')
pred_text_a = "CHURN" if pred_a == 1 else "NO CHURN"
print(f'Prediction: {pred_text_a} (Code: {pred_a})')
print()

# Test B: Low-risk customer
print('TEST B: Low-Risk Profile')
print('-' * 50)
sample_data_b = {
    'CreditScore': 750,
    'Gender': 'Female', 
    'Age': 30, 
    'Tenure': 8, 
    'Balance': 150000,
    'NumOfProducts': 1, 
    'HasCrCard': 'Yes',
    'IsActiveMember': 'Yes',
    'EstimatedSalary': 100000,
    'Geography': 'France',
    'Surname': 'TestB'
}

df_b = pd.DataFrame([sample_data_b])
df_b = df_b.replace(gender_encoder)

geo_encoded_b = geo_encoder.transform(df_b[['Geography']])
geo_df_b = pd.DataFrame(geo_encoded_b, columns=geo_encoder.get_feature_names_out(['Geography']))
df_b = df_b.reset_index(drop=True)
geo_df_b = geo_df_b.reset_index(drop=True)
df_b = pd.concat([df_b, geo_df_b], axis=1)
df_b.drop(columns=['Geography'], inplace=True)

df_b = df_b.replace(hasCrCard_encoder)
df_b = df_b.replace(isActiveMember_encoder)

df_b[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.transform(
    df_b[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
)

features_b = df_b[feature_order]
input_b = np.array(features_b).reshape(1, -1)
pred_b = model.predict(input_b)[0]

print('Customer Details:')
print(f'  Age: 30, Credit Score: 750, Tenure: 8, Balance: 150000')
print(f'  NumOfProducts: 1, HasCrCard: Yes, IsActiveMember: Yes')
print(f'  EstimatedSalary: 100000, Geography: France')
pred_text_b = "CHURN" if pred_b == 1 else "NO CHURN"
print(f'Prediction: {pred_text_b} (Code: {pred_b})')
print()
print('[OK] Both predictions executed successfully!')
