
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
    return pd.read_csv("customer_churn.csv")

raw_data = load_data()
data = preprocess_data(raw_data.copy())

st.title("📊 Auto Churn Prediction App")

st.subheader("1️⃣ Churn Distribution")
fig_pie = px.pie(raw_data, names='Churn', title='Churned vs Not Churned', color='Churn')
st.plotly_chart(fig_pie)

st.subheader("2️⃣ Feature Distribution (Bar/Histogram)")
selected_feature = st.selectbox("Choose a feature to visualize", raw_data.columns[:-1])

if raw_data[selected_feature].dtype == 'object':
    fig_bar = px.bar(raw_data[selected_feature].value_counts().reset_index(),
                     x='index', y=selected_feature,
                     labels={'index': selected_feature, selected_feature: 'Count'})
    st.plotly_chart(fig_bar)
else:
    fig_hist = px.histogram(raw_data, x=selected_feature)
    st.plotly_chart(fig_hist)

st.subheader("3️⃣ Train Churn Prediction Model")
X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("4️⃣ Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("5️⃣ Classification Report")
st.text(classification_report(y_test, y_pred))
