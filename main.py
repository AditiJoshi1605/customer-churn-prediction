import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

# Define paths to the model and encoders
MODEL_PATH = os.path.join('models', 'xgb_classifier_model.pkl')
GENDER_ENCODER_PATH = os.path.join('models', 'gender_encoder.pkl')
HAS_CR_CARD_ENCODER_PATH = os.path.join('models', 'hasCrCard_encoder.pkl')
IS_ACTIVE_MEMBER_ENCODER_PATH = os.path.join('models', 'isActiveMember_encoder.pkl')
DATA_PATH = os.path.join('data', 'Customer-Dataset.csv')

# Load the model and encoders
model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)
hasCrCard_encoder = joblib.load(HAS_CR_CARD_ENCODER_PATH)
isActiveMember_encoder = joblib.load(IS_ACTIVE_MEMBER_ENCODER_PATH)


@st.cache_resource
def load_and_fit_scaler():
    """
    Load training data and fit the StandardScaler on the numerical columns.
    This ensures that inference uses the exact same scaling parameters as training.
    """
    # Load training data
    df_train = pd.read_csv(DATA_PATH)
    
    # Handle missing values (same as training)
    df_train.dropna(subset=['CreditScore'], inplace=True)
    
    # Drop irrelevant columns (same as training)
    df_train.drop(columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname'], inplace=True)
    
    # Apply binary encoding for Gender (same as training)
    df_train = df_train.replace(gender_encoder)
    
    # Apply one-hot encoding for Geography (same as training)
    geo_encoder = OneHotEncoder(sparse_output=False)
    geo_features = df_train[['Geography']]
    geo_encoded = geo_encoder.fit_transform(geo_features)
    geo_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
    df_train = df_train.reset_index(drop=True)
    geo_df = geo_df.reset_index(drop=True)
    df_train = pd.concat([df_train, geo_df], axis=1)
    df_train.drop(columns=['Geography'], inplace=True)
    
    # Fit scaler on numerical columns (same as training)
    scaler = StandardScaler()
    scaler.fit(df_train[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])
    
    return scaler, geo_encoder


# Load the scaler and geography encoder
scaler, geo_encoder = load_and_fit_scaler()

def main():
    st.set_page_config(page_title="Bank Customer Churn Prediction", layout="centered")

    st.title("Bank Customer Churn Prediction")
    st.caption("Predict customer churn based on account and customer information.")

    st.subheader("Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        Surname = st.text_input("Surname: ")
        Age = st.number_input("Age: ", 18, 100)
        Gender = st.radio("Gender: ", ["Male", "Female"])
        Geography = st.radio("Geography: ", ['France', 'Spain', 'Germany'])
        Tenure = st.selectbox("Tenure: ", list(range(1, 11)))

    with col2:
        Balance = st.number_input("Balance: ", 0, 10000000)
        NumOfProducts = st.selectbox("Number of Products:", [1, 2, 3, 4])
        HasCrCard = st.radio("Credit Card:", ["Yes", "No"])
        IsActiveMember = st.radio("Active Member:", ["Yes", "No"])
        EstimatedSalary = st.number_input("Estimated Salary: ", 0, 10000000)
        CreditScore = st.number_input("Credit Score: ", 300, 850)

    # Create customer data with EXACT feature order from training
    # Training order: CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography
    # Note: Surname is collected in the UI but is NOT a model feature, so it is excluded here.
    data = {
        'CreditScore': int(CreditScore),
        'Gender': Gender, 
        'Age': int(Age), 
        'Tenure': int(Tenure), 
        'Balance': int(Balance),
        'NumOfProducts': NumOfProducts, 
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': int(EstimatedSalary),
        'Geography': Geography,
    }

    # Create dataframe with proper column ordering
    df = pd.DataFrame([data])

    # Apply binary encoding for Gender (same as training)
    df = df.replace(gender_encoder)
    
    # Apply one-hot encoding for Geography (using the fitted encoder from training)
    geo_features = df[['Geography']]
    geo_encoded = geo_encoder.transform(geo_features)
    geo_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
    df = df.reset_index(drop=True)
    geo_df = geo_df.reset_index(drop=True)
    df = pd.concat([df, geo_df], axis=1)
    df.drop(columns=['Geography'], inplace=True)
    
    # Apply binary encoding for HasCrCard
    df = df.replace(hasCrCard_encoder)
    
    # Apply binary encoding for IsActiveMember
    df = df.replace(isActiveMember_encoder)
    
    # IMPORTANT: Scale only - do NOT fit on customer input
    # Use the scaler fitted on training data
    df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.transform(
        df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']]
    )

    # Prediction button
    st.markdown("---")
    if st.button('Make Prediction'):
        # Reorder features to match training order exactly
        # Training order (after dropping Surname and churn): 
        # CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        # Geography_France, Geography_Germany, Geography_Spain
        feature_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                        'Geography_France', 'Geography_Germany', 'Geography_Spain']
        features = df[feature_order]
        result = makePrediction(features)
        st.markdown("### Prediction Result")
        if result == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is unlikely to churn.")
        st.caption("Prediction is based on the customer information provided above.")




# Prediction function
def makePrediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()