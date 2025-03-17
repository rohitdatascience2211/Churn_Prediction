import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


st.title("Customer Churn Prediction")

def load_data():
    data = pd.read_csv('churn_prediction.csv')  
    st.write("Loaded data:")
    st.write(data.head())
    
    if 'churn' not in data.columns:
        st.error("The dataset does not contain a 'churn' column!")
        return None
    
    le_gender = LabelEncoder()
    data['gender'] = le_gender.fit_transform(data['gender'])
    
    le_country = LabelEncoder()
    data['country'] = le_country.fit_transform(data['country'])
    
    data.fillna(data.mean(), inplace=True)
    
    X = data.drop(columns=['customer_id', 'churn'])  
    y = data['churn']
    
    columns_order = X.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler, le_gender, le_country, columns_order

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    with open('churn_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model

def load_model():
    try:
        with open('churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("Model not found! Please train the model first.")
        model = None
    return model

def user_input_form():
    st.header("Enter Customer Data")
    
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Tenure", min_value=0, step=1)
    balance = st.number_input("Balance", min_value=0, step=100)
    products_number = st.number_input("Number of Products", min_value=1, step=1)
    credit_card = st.number_input("Has Credit Card (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1)
    active_member = st.number_input("Is Active Member (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1)
    estimated_salary = st.number_input("Estimated Salary", min_value=10000, step=100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["France", "Spain"])

    user_data = pd.DataFrame({
        'credit_score': [credit_score],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary],
        'gender': [gender],
        'country': [country]
    })
    
    return user_data

def preprocess_user_input(user_data, scaler, le_gender, le_country, columns_order):
    user_data['gender'] = le_gender.transform(user_data['gender'])
    user_data['country'] = le_country.transform(user_data['country'])
    
    user_data = user_data[columns_order]
    
    user_data_scaled = scaler.transform(user_data)
    
    return user_data_scaled

def main():
    X, y, scaler, le_gender, le_country, columns_order = load_data()
    
    if X is None or y is None:
        st.error("Please ensure that the dataset is correct and contains a 'churn' column.")
        return
    
    model = load_model()
    if model is None:
        st.write("Training model...")
        model = train_model(X, y)
    
    user_data = user_input_form()
    
    user_data_scaled = preprocess_user_input(user_data, scaler, le_gender, le_country, columns_order)
    
    if st.button("Predict Churn"):
        if model:
            prediction = model.predict(user_data_scaled)
            churn_status = "Yes" if prediction[0] == 1 else "No"
            st.write(f"The predicted churn status is: **{churn_status}**")

if __name__ == '__main__':
    main()
