import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Auto Churn Prediction", layout="wide")
st.title("📊 Auto Churn Prediction")
st.write("Upload a customer churn dataset and predict whether a customer will churn or not.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df.dropna(inplace=True)

    # Encode categorical variables
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object'):
        if col != 'customerID':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    st.subheader("📊 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Churn Distribution")
        fig1 = px.pie(df, names="Churn", title="Churn vs Not Churn")
        st.plotly_chart(fig1)

    with col2:
        selected_col = st.selectbox("Select a Column to Visualize", df.columns)
        if df[selected_col].dtype == 'object':
            fig2 = px.bar(df[selected_col].value_counts().reset_index(),
                          x='index', y=selected_col,
                          labels={'index': selected_col, selected_col: 'Count'})
            st.plotly_chart(fig2)
        else:
            fig3 = px.histogram(df, x=selected_col)
            st.plotly_chart(fig3)

    st.subheader("🧠 Churn Prediction Model")

    X = df_encoded.drop(columns=['customerID', 'Churn'])
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("📌 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    st.subheader("👤 Predict for a New Customer")

    new_data = {}
    for col in df.columns:
        if col in ['customerID', 'Churn']:
            continue
        if df[col].dtype == 'object':
            new_data[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            new_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    # Convert input into DataFrame
    new_df = pd.DataFrame([new_data])

    # Encode input the same way
    for col in new_df.select_dtypes(include='object'):
        le = LabelEncoder()
        le.fit(df[col])
        new_df[col] = le.transform(new_df[col])

    prediction = model.predict(new_df)[0]

    if st.button("Predict Churn"):
        st.success("✅ Prediction: Customer will CHURN" if prediction == 1 else "✅ Prediction: Customer will NOT churn")
