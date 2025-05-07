import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from preprocess import preprocess_data

st.set_page_config(page_title="Auto Churn Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("sample_dataset.csv")

raw_data = load_data()
st.title("📊 Auto Churn Prediction App")

# 1. Churn Distribution Pie Chart
st.subheader("1️⃣ Churn Distribution")
fig_pie = px.pie(raw_data, names='Churn', title='Churned vs Not Churned', color='Churn')
st.plotly_chart(fig_pie)

# 2. Feature Distribution
st.subheader("2️⃣ Feature Distribution")
selected_feature = st.selectbox("Choose a feature to visualize", raw_data.columns[:-1])

if raw_data[selected_feature].dtype == 'object':
    value_counts_df = raw_data[selected_feature].value_counts().reset_index()
    value_counts_df.columns = [selected_feature, 'Count']
    fig_bar = px.bar(value_counts_df, x=selected_feature, y='Count')
    st.plotly_chart(fig_bar)
else:
    fig_hist = px.histogram(raw_data, x=selected_feature)
    st.plotly_chart(fig_hist)

# 3. Train Model
st.subheader("3️⃣ Train Churn Prediction Model")
processed_data = preprocess_data(raw_data)

X = processed_data.drop("Churn_Yes", axis=1)
y = processed_data["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Confusion Matrix
st.subheader("4️⃣ Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

# 5. Classification Report
st.subheader("5️⃣ Classification Report")
st.text(classification_report(y_test, y_pred))

# 6. Predict for New User
st.subheader("6️⃣ Predict Churn for New Customer")

user_input = {}
for col in raw_data.columns:
    if col in ['customerID', 'Churn']:
        continue
    if raw_data[col].dtype == 'object':
        user_input[col] = st.selectbox(f"{col}", raw_data[col].unique())
    else:
        user_input[col] = st.number_input(f"{col}", float(raw_data[col].min()), float(raw_data[col].max()))

user_df = pd.DataFrame([user_input])
user_df_full = pd.concat([raw_data.drop(['customerID', 'Churn'], axis=1), user_df], axis=0)
processed_user_data = preprocess_data(
    pd.concat([raw_data, user_df.assign(customerID='0000', Churn='No')], ignore_index=True)
)
user_processed = processed_user_data.iloc[[-1]].drop(columns=['Churn_Yes'])

if st.button("Predict Churn"):
    result = model.predict(user_processed)[0]
    st.success("✅ Prediction: Churn" if result == 1 else "✅ Prediction: Not Churn")