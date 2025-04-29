import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data

st.title("🔍 Customer Churn Prediction")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Raw Data", df.head())

    df_clean = preprocess_data(df)
    
    if 'Churn' not in df_clean.columns:
        st.error("Dataset must include a 'Churn' column.")
    else:
        X = df_clean.drop('Churn', axis=1)
        y = df_clean['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ Model trained with {acc * 100:.2f}% accuracy!")

        result_df = X_test.copy()
        result_df['Actual Churn'] = y_test
        result_df['Predicted Churn'] = preds
        st.write("📈 Predictions", result_df.head(20))