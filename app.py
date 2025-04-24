import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page configuration for better layout
st.set_page_config(page_title="Medicosa", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stSlider>div>div>div {
        color: #34495e;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    .sidebar .stRadio>div>label {
        font-size: 16px;
        color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar .stRadio>div>label:hover {
        background-color: #dfe6e9;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 40px;
        padding: 20px;
        background-color: #f7f9fb;
        border-top: 1px solid #dfe6e9;
    }
    .stAlert {
        border-radius: 8px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation with Icons
st.sidebar.markdown("### Navigation Menu")
rad = st.sidebar.radio(
    "Navigate to a section:",
    ["Home", "Covid-19", "Diabetes", "Heart Disease", "Plots"],
    format_func=lambda x: f"🏠 {x}" if x == "Home" else f"🩺 {x}" if x in ["Covid-19", "Diabetes", "Heart Disease"] else f"📊 {x}"
)

# Load datasets and train models (same as original)
# COVID-19
df1 = pd.read_csv("Covid-19 Predictions.csv")
x1 = df1.drop("Infected with Covid19", axis=1)
x1 = np.array(x1)
y1 = pd.DataFrame(df1["Infected with Covid19"])
y1 = np.array(y1)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)
model1 = RandomForestClassifier()
model1.fit(x1_train, y1_train)

# Diabetes
df2 = pd.read_csv("Diabetes Predictions.csv")
x2 = df2.iloc[:, [1, 4, 5, 7]].values
x2 = np.array(x2)
y2 = df2.iloc[:, [-1]].values
y2 = np.array(y2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=0)
model2 = RandomForestClassifier()
model2.fit(x2_train, y2_train)

# Heart Disease
df3 = pd.read_csv("Heart Disease Predictions.csv")
x3 = df3.iloc[:, [2, 3, 4, 7]].values
x3 = np.array(x3)
y3 = df3.iloc[:, [-1]].values
y3 = np.array(y3)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=0)
model3 = RandomForestClassifier()
model3.fit(x3_train, y3_train)

# Home Page
if rad == "Home":
    st.markdown('<div class="main-header">Medicosa</div>', unsafe_allow_html=True)
    
    # Display image with centered alignment
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("background.jpg", use_column_width=True)
    
    st.markdown('<div class="section-header">Available Predictions</div>', unsafe_allow_html=True)
    st.markdown("""
    - 🦠 **COVID-19 Infection Predictions**: Assess your risk based on symptoms like dry cough and fever.
    - 🩺 **Diabetes Predictions**: Check your likelihood of diabetes using metrics like glucose and BMI.
    - ❤️ **Heart Disease Predictions**: Evaluate heart disease risk with factors like blood pressure and cholesterol.
    """)

# COVID-19 Prediction Page
if rad == "Covid-19":
    st.markdown('<div class="section-header">Know If You Are Affected By COVID-19</div>', unsafe_allow_html=True)
    st.info("Enter values within the specified ranges. Higher values indicate more severe symptoms.")

    with st.form(key="covid_form"):
        st.markdown("### Symptom Input")
        col1, col2 = st.columns(2)
        with col1:
            drycough = st.slider(
                "Rate of Dry Cough (0-20)",
                min_value=0, max_value=20, value=0, step=1,
                help="0 = No cough, 20 = Severe cough"
            )
            fever = st.slider(
                "Rate of Fever (0-20)",
                min_value=0, max_value=20, value=0, step=1,
                help="0 = No fever, 20 = High fever"
            )
        with col2:
            sorethroat = st.slider(
                "Rate of Sore Throat (0-20)",
                min_value=0, max_value=20, value=0, step=1,
                help="0 = No sore throat, 20 = Severe sore throat"
            )
            breathingprob = st.slider(
                "Rate of Breathing Problem (0-20)",
                min_value=0, max_value=20, value=0, step=1,
                help="0 = No breathing issues, 20 = Severe breathing issues"
            )
        
        submit_button = st.form_submit_button(label="Predict COVID-19 Risk")

    if submit_button:
        with st.spinner("Predicting..."):
            prediction1 = model1.predict([[drycough, fever, sorethroat, breathingprob]])[0]
            if prediction1 == "Yes":
                st.warning("⚠️ You Might Be Affected By COVID-19")
            elif prediction1 == "No":
                st.success("✅ You Are Safe")

# Diabetes Prediction Page
if rad == "Diabetes":
    st.markdown('<div class="section-header">Know If You Are Affected By Diabetes</div>', unsafe_allow_html=True)
    st.info("Enter values within the specified ranges to assess your risk.")

    with st.form(key="diabetes_form"):
        st.markdown("### Health Metrics")
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider(
                "Glucose Level (0-200)",
                min_value=0, max_value=200, value=0, step=1,
                help="Typical range: 70-140 mg/dL for healthy individuals"
            )
            insulin = st.slider(
                "Insulin Level (0-850)",
                min_value=0, max_value=850, value=0, step=1,
                help="Normal range: 2-25 µIU/mL (fasting)"
            )
        with col2:
            bmi = st.slider(
                "Body Mass Index (BMI) (0-70)",
                min_value=0, max_value=70, value=0, step=1,
                help="Normal BMI: 18.5-24.9"
            )
            age = st.slider(
                "Age (20-80)",
                min_value=20, max_value=80, value=20, step=1,
                help="Age in years"
            )
        
        submit_button = st.form_submit_button(label="Predict Diabetes Risk")

    if submit_button:
        with st.spinner("Predicting..."):
            prediction2 = model2.predict([[glucose, insulin, bmi, age]])[0]
            if prediction2 == 1:
                st.warning("⚠️ You Might Be Affected By Diabetes")
            elif prediction2 == 0:
                st.success("✅ You Are Safe")

# Heart Disease Prediction Page
if rad == "Heart Disease":
    st.markdown('<div class="section-header">Know If You Are Affected By Heart Disease</div>', unsafe_allow_html=True)
    st.info("Enter values within the specified ranges to assess your risk.")

    with st.form(key="heart_disease_form"):
        st.markdown("### Health Metrics")
        col1, col2 = st.columns(2)
        with col1:
            chestpain = st.slider(
                "Chest Pain Level (1-4)",
                min_value=1, max_value=4, value=1, step=1,
                help="1 = Minimal, 4 = Severe"
            )
            bp = st.slider(
                "Blood Pressure (95-200 mmHg)",
                min_value=95, max_value=200, value=95, step=1,
                help="Normal BP: ~120/80 mmHg"
            )
        with col2:
            cholestrol = st.slider(
                "Cholesterol Level (125-565 mg/dL)",
                min_value=125, max_value=565, value=125, step=1,
                help="Normal: <200 mg/dL"
            )
            maxhr = st.slider(
                "Maximum Heart Rate (70-200 bpm)",
                min_value=70, max_value=200, value=70, step=1,
                help="Normal range varies by age"
            )
        
        submit_button = st.form_submit_button(label="Predict Heart Disease Risk")

    if submit_button:
        with st.spinner("Predicting..."):
            prediction3 = model3.predict([[chestpain, bp, cholestrol, maxhr]])[0]
            if str(prediction3) == "Presence":
                st.warning("⚠️ You Might Be Affected By Heart Disease")
            elif str(prediction3) == "Absence":
                st.success("✅ You Are Safe")

# Plots Page
if rad == "Plots":
    st.markdown('<div class="section-header">Data Visualizations</div>', unsafe_allow_html=True)
    st.info("Select a dataset to visualize relationships between key features and outcomes.")

    type = st.selectbox(
        "Choose a Dataset to Visualize:",
        ["Covid-19", "Diabetes", "Heart Disease"],
        help="Select a dataset to see a scatter plot of key features."
    )

    with st.expander("View Plot", expanded=True):
        if type == "Covid-19":
            st.markdown("#### COVID-19: Breathing Difficulty vs Infection Status")
            fig = px.scatter(df1, x="Difficulty in breathing", y="Infected with Covid19", 
                            color="Infected with Covid19", size="Difficulty in breathing",
                            title="COVID-19 Infection vs Breathing Difficulty")
            st.plotly_chart(fig, use_container_width=True)

        elif type == "Diabetes":
            st.markdown("#### Diabetes: Glucose Level vs Outcome")
            fig = px.scatter(df2, x="Glucose", y="Outcome", color="Outcome", size="Glucose",
                            title="Diabetes Outcome vs Glucose Level")
            st.plotly_chart(fig, use_container_width=True)

        elif type == "Heart Disease":
            st.markdown("#### Heart Disease: Blood Pressure vs Disease Presence")
            fig = px.scatter(df3, x="BP", y="Heart Disease", color="Heart Disease", size="BP",
                            title="Heart Disease vs Blood Pressure")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div class="footer">
        Medical Predictions App | Built with ❤️ using Streamlit | © 2025
    </div>
""", unsafe_allow_html=True)