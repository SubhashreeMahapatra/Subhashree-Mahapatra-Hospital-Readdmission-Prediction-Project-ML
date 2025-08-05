import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load model
model = joblib.load("xgboost_readmission_model.pkl")

# Set page config
st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide")

# HTML & CSS Styling
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: 700;
            color: #4e73df;
        }
        .sub-text {
            font-size: 1.1em;
            color: #444;
            margin-bottom: 2rem;
        }
        .result-box {
            background-color: #f8f9fc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 5px solid #4e73df;
            margin-top: 1rem;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Diabetes Readmission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict whether a diabetic patient will be readmitted within 30 days of discharge using hospital visit data.</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📋 Enter Patient Details")

def user_input():
    race = st.sidebar.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    
    # Changed from selectbox to numeric input
    age = st.sidebar.number_input("Age (in years)", min_value=1, max_value=100, value=45, step=1)

    time_in_hospital = st.sidebar.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=5)
    num_lab_procedures = st.sidebar.number_input("Lab Procedures", min_value=0, max_value=132, value=40)
    num_procedures = st.sidebar.number_input("Other Procedures", min_value=0, max_value=6, value=1)
    num_medications = st.sidebar.number_input("Medications", min_value=1, max_value=81, value=20)
    number_outpatient = st.sidebar.number_input("Outpatient Visits", min_value=0, max_value=42, value=0)
    number_emergency = st.sidebar.number_input("Emergency Visits", min_value=0, max_value=76, value=0)
    number_inpatient = st.sidebar.number_input("Inpatient Visits", min_value=0, max_value=21, value=0)
    number_diagnoses = st.sidebar.number_input("Diagnoses Count", min_value=1, max_value=16, value=5)

    insulin = st.sidebar.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
    change = st.sidebar.selectbox("Change in Medication", ['No', 'Ch'])
    diabetesMed = st.sidebar.selectbox("Diabetes Medication Prescribed", ['Yes', 'No'])

    # Submit Button inside Sidebar
    submit = st.sidebar.button("🚀 Submit for Prediction")

    # For consistent encoding later, age converted back to range category if needed
    age_group = '[90-100)'
    if age < 10:
        age_group = '[0-10)'
    elif age < 20:
        age_group = '[10-20)'
    elif age < 30:
        age_group = '[20-30)'
    elif age < 40:
        age_group = '[30-40)'
    elif age < 50:
        age_group = '[40-50)'
    elif age < 60:
        age_group = '[50-60)'
    elif age < 70:
        age_group = '[60-70)'
    elif age < 80:
        age_group = '[70-80)'
    elif age < 90:
        age_group = '[80-90)'

    data = {
        'race': race,
        'gender': gender,
        'age': age_group,  # Keep consistent with model expectations
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'insulin': insulin,
        'change': change,
        'diabetesMed': diabetesMed,
        'glipizide': 'No',  # fixed for demo
        'metformin': 'No'   # fixed for demo
    }

    return pd.DataFrame([data]), submit
    
# Get user input + submit status
input_df, submit = user_input()

# Predict and show result only if user clicks Submit
if submit:

    def preprocess_input(df):
        df_encoded = pd.get_dummies(df)
        model_cols = model.get_booster().feature_names
        for col in model_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_cols]
        return df_encoded.astype(float)

    X_final = preprocess_input(input_df)
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1]

    # Prediction result
    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.markdown('<div class="result-box"><b>⚠ Likely Readmission:</b> The patient is likely to be <b>readmitted</b> within 30 days.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box"><b>✅ No Readmission:</b> The patient is <b>not likely</b> to be readmitted within 30 days.</div>', unsafe_allow_html=True)

    # Confidence Display
    st.markdown("### 🧮 Prediction Confidence")
    st.progress(int(prob * 100))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 2),
        title={'text': "Confidence (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4e73df"},
            'steps': [
                {'range': [0, 50], 'color': "#d4edda"},
                {'range': [50, 100], 'color': "#f8d7da"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)



st.title("🌍 Global Diabetes Prevalence Trends and Forecasts")

# Updated historical + forecast data for global prevalence (1990–2045)
trend_data = {
    "Year": list(range(1990, 2050, 5)),  # 1990 to 2045 in 5-year steps
    "Prevalence (%)": [4.7, 5.1, 5.6, 6.2, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8, 10.2, 10.5, 10.9]
}

df_trend = pd.DataFrame(trend_data)

# Global line chart
st.subheader("📈 Global Diabetes Prevalence (1990–2045)")
st.line_chart(df_trend.set_index("Year"), use_container_width=True)

# Country-wise prevalence in 2045 (static map)
map_data = {
    "Country": ["Pakistan", "Mexico", "Bangladesh", "India", "China", "Brazil"],
    "Prevalence_2045": [33.6, 20.0, 16.0, 11.0, 12.0, 10.0]
}
df_map = pd.DataFrame(map_data)

st.subheader("🗺 Projected Country-wise Diabetes Prevalence (2045)")

# Plot static choropleth for 2045
fig = px.choropleth(
    df_map,
    locations="Country",
    locationmode="country names",
    color="Prevalence_2045",
    range_color=(0, 35),
    color_continuous_scale="Reds",
    title="Diabetes Prevalence by Country in 2045"
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

st.plotly_chart(fig, use_container_width=True)

# Explanation
st.markdown("""
### 📊 Summary

*Global Prevalence (Adults 20–79 yrs)*  
- 1990: *4.7%*
- 2010: *8.3%*
- 2019: *9.3%*
- 2045: *10.9%*

*Country Forecasts for 2045*  
- *Pakistan*: ~33.6%  
- *India*: ~11.0%  
- *Bangladesh*: ~16.0%  
- *Mexico*: ~20.0%  
- *China*: ~12.0%  
- *Brazil*: ~10.0%

Source: International Diabetes Federation (IDF), WHO Projections
""")




# Patient summary section
st.markdown("### 🧾 Patient Summary Report")

with st.expander("View Detailed Patient Summary", expanded=True):
    st.markdown(f"""
    <div style="padding:1rem; background-color:#f1f2f6; border-left:5px solid #4e73df; border-radius:0.5rem">
        <h5 style="color:#4e73df;">🧍 Demographics:</h5>
        <ul>
            <li><b>Race:</b> {input_df['race'][0]}</li>
            <li><b>Gender:</b> {input_df['gender'][0]}</li>
            <li><b>Age Range:</b> {input_df['age'][0]}</li>
        </ul>
        <h5 style="color:#4e73df;">🏥 Visit & Diagnosis Info:</h5>
        <ul>
            <li><b>Time in Hospital:</b> {input_df['time_in_hospital'][0]} days</li>
            <li><b>Diagnoses:</b> {input_df['number_diagnoses'][0]}</li>
            <li><b>Outpatient Visits:</b> {input_df['number_outpatient'][0]}</li>
            <li><b>Emergency Visits:</b> {input_df['number_emergency'][0]}</li>
            <li><b>Inpatient Visits:</b> {input_df['number_inpatient'][0]}</li>
        </ul>
        <h5 style="color:#4e73df;">💊 Treatment Details:</h5>
        <ul>
            <li><b>Medications:</b> {input_df['num_medications'][0]}</li>
            <li><b>Lab Procedures:</b> {input_df['num_lab_procedures'][0]}</li>
            <li><b>Other Procedures:</b> {input_df['num_procedures'][0]}</li>
            <li><b>Insulin:</b> {input_df['insulin'][0]}</li>
            <li><b>Medication Change:</b> {input_df['change'][0]}</li>
            <li><b>Diabetes Medication:</b> {input_df['diabetesMed'][0]}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.table(input_df.T.rename(columns={0: "Value"}))

    
