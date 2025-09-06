import streamlit as st
import numpy as np
import joblib
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hr_rf2.pkl")
model = joblib.load(MODEL_PATH)

# ----------------------------
# Dark Blue-Black Theme Styling
# ----------------------------
st.markdown("""
<style>
/* Full-page dark blue to black gradient */
body {
    background: linear-gradient(135deg, #0b0c1a, #0d1b2a, #000000);
    color: #E0FFFF;
}

/* Main container with glass effect */
.main .block-container {
    backdrop-filter: blur(10px);
    background: rgba(10, 15, 25, 0.75);
    border-radius: 20px;
    padding: 30px 50px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* Title Bar Gradient (dark blue → black) */
.title-bar {
    background: linear-gradient(90deg, #1B263B, #0D1B2A, #000000);
    padding: 20px 30px;
    border-radius: 15px;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    font-size: 34px;
    font-weight: bold;
    color: #00FFF7;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
    margin-bottom: 30px;
}

/* Subheadings & sections */
h2, h3, h4 {
    font-family: 'Segoe UI', sans-serif;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.6);
}

/* Input fields - dark neumorphism */
.stSlider, .stNumberInput, .stRadio, .stSelectbox {
    background: rgba(20, 25, 40, 0.7) !important;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 15px;
    color: #E0FFFF;
}

/* Button Gradient & hover glow */
div.stButton > button {
    background: linear-gradient(135deg, #1B263B, #0D1B2A);
    color: #00FFF7;
    border-radius: 12px;
    padding: 10px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,255,247,0.3);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0D1B2A, #000000);
    box-shadow: 0 0 20px #00FFF7;
    transform: scale(1.05);
}

/* Result Card */
.result-card {
    background: linear-gradient(145deg, #0B0C1A, #1B263B);
    padding: 25px;
    border-radius: 20px;
    color: #00FFF7;
    font-family: 'Segoe UI', sans-serif;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    text-align: center;
    width: 90%;
    max-width: 1000px;
    margin: 15px auto;
}

/* Suggestions Panel */
.suggestion-card {
    background: linear-gradient(145deg, #0D1B2A, #1B263B);
    padding: 25px;
    border-radius: 25px;
    color: #E0FFFF;
    font-family: 'Segoe UI', sans-serif;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    line-height: 1.7;
    width: 90%;
    max-width: 1000px;
    margin: 25px auto;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Hero Section inside Gradient Bar
# ----------------------------
st.markdown("""
<div class="title-bar">
    💼 HR Employee Attrition Prediction
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center; font-size:18px; color:#00FFF7;">
    Enter employee details or use quick scenarios to predict if an employee will 
    <b style="color:#FF6B6B;">Leave</b> or <b style="color:#00FFA3;">Stay</b>.
</p>
""", unsafe_allow_html=True)

# ----------------------------
# Session State Defaults
# ----------------------------
def init_session_state():
    defaults = {
        "last_evaluation": 0.7,
        "number_project": 4,
        "tenure": 3,
        "work_accident": "No",
        "promotion_last_5years": "No",
        "average_montly_hours": 160,
        "salary": "Medium",
        "department": "Sales",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
init_session_state()

# ----------------------------
# Layout
# ----------------------------
st.subheader("📌 Employee Information")
col1, col2 = st.columns(2)

with col1:
    st.session_state['last_evaluation'] = st.slider(
        "📊 Last Evaluation Score", 0.0, 1.0,
        st.session_state['last_evaluation'], 0.01
    )
    st.session_state['number_project'] = st.number_input(
        "📂 Number of Projects", 1, 10, st.session_state['number_project']
    )
    st.session_state['tenure'] = st.number_input(
        "⏳ Tenure (Years at Company)", 1, 20, st.session_state['tenure']
    )
    st.session_state['work_accident'] = st.radio(
        "⚠️ Work Accident?", ["No", "Yes"],
        index=0 if st.session_state['work_accident']=="No" else 1
    )

with col2:
    st.session_state['promotion_last_5years'] = st.radio(
        "🎖️ Promotion in Last 5 Years?", ["No", "Yes"],
        index=0 if st.session_state['promotion_last_5years']=="No" else 1
    )
    st.session_state['average_montly_hours'] = st.number_input(
        "🕒 Average Monthly Hours", 50, 310,
        st.session_state['average_montly_hours']
    )
    st.session_state['salary'] = st.radio(
        "💰 Salary Level", ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(st.session_state['salary'])
    )
    st.session_state['department'] = st.selectbox(
        "🏢 Department", [
            "IT", "RandD", "Accounting", "HR", "Management",
            "Marketing", "Product_mng", "Sales", "Support", "Technical"
        ],
        index=[
            "IT", "RandD", "Accounting", "HR", "Management",
            "Marketing", "Product_mng", "Sales", "Support", "Technical"
        ].index(st.session_state['department'])
    )

st.markdown(
    "**Note:** `overworked` is computed automatically from "
    "`Average Monthly Hours` (overworked = avg_hours > 175)."
)

# ----------------------------
# Quick Scenarios
# ----------------------------
st.subheader("⚡ Quick Scenarios")
colA, colB, colC = st.columns(3)
with colA:
    if st.button("🔴 Extreme Risk"):
        st.session_state.update({
            "last_evaluation": 0.15,
            "number_project": 7,
            "tenure": 7,
            "work_accident": "No",
            "promotion_last_5years": "No",
            "average_montly_hours": 280,
            "salary": "Low",
            "department": "Sales",
        })
with colB:
    if st.button("🟡 Likely to Leave"):
        st.session_state.update({
            "last_evaluation": 0.35,
            "number_project": 8,
            "tenure": 1,
            "work_accident": "No",
            "promotion_last_5years": "No",
            "average_montly_hours": 250,
            "salary": "Low",
            "department": "Support",
        })
with colC:
    if st.button("🟢 Likely to Stay"):
        st.session_state.update({
            "last_evaluation": 0.9,
            "number_project": 3,
            "tenure": 2,
            "work_accident": "No",
            "promotion_last_5years": "Yes",
            "average_montly_hours": 160,
            "salary": "High",
            "department": "RandD",
        })

# ----------------------------
# Feature Mapping
# ----------------------------
def map_features(state):
    overworked = 1 if state['average_montly_hours'] > 175 else 0
    salary_map = {"Low": 0, "Medium": 1, "High": 2}
    dept_list = [
        "IT", "RandD", "Accounting", "HR", "Management",
        "Marketing", "Product_mng", "Sales", "Support", "Technical"
    ]
    dept_encoded = [1 if state['department'] == d else 0 for d in dept_list]

    features = [
        state['last_evaluation'],
        state['number_project'],
        state['tenure'],
        1 if state['work_accident']=="Yes" else 0,
        1 if state['promotion_last_5years']=="Yes" else 0,
        salary_map[state['salary']],
        *dept_encoded,
        overworked
    ]
    return np.array(features).reshape(1, -1)

# ----------------------------
# Suggestions Generator
# ----------------------------
def generate_suggestions(state, prediction, prob_leave):
    explanation = ""
    if prediction == 1:
        explanation += f"⚠️ <b>High risk of attrition ({prob_leave*100:.2f}%)</b><br><br>"
        explanation += "<b>Areas to improve:</b><br>"
        if state['last_evaluation'] < 0.5:
            explanation += "- Performance is low. Provide coaching or mentorship.<br>"
        if state['number_project'] > 6:
            explanation += "- Overloaded with projects. Consider redistributing workload.<br>"
        if state['tenure'] < 2:
            explanation += "- Employee is new. Enhance onboarding and engagement.<br>"
        if state['promotion_last_5years'] == "No":
            explanation += "- No recent promotion. Discuss career growth opportunities.<br>"
        if state['salary'] == "Low":
            explanation += "- Compensation is low. Review salary.<br>"
        if state['average_montly_hours'] > 175:
            explanation += "- Overworked. Reduce hours to prevent burnout.<br>"
    else:
        explanation += f"✅ <b>Low risk of attrition ({(1-prob_leave)*100:.2f}%)</b><br><br>"
        explanation += "<b>Strengths to maintain:</b><br>"
        if state['last_evaluation'] > 0.8:
            explanation += "- High performance. Encourage challenging projects.<br>"
        if 5 <= state['number_project'] <= 6:
            explanation += "- Balanced workload. Maintain current assignments.<br>"
        if state['promotion_last_5years'] == "Yes":
            explanation += "- Recently promoted. Continue recognizing achievements.<br>"
        if state['salary'] in ["Medium","High"]:
            explanation += "- Competitive salary. Keep rewarding performance.<br>"
        if state['average_montly_hours'] <= 175:
            explanation += "- Work hours balanced. Maintain work-life balance.<br>"
    return explanation

# ----------------------------
# Predict Button & Result
# ----------------------------
st.subheader("🔮 Predict Attrition")
if st.button("🔮 Predict Now"):
    features_array = map_features(st.session_state)
    prediction = model.predict(features_array)[0]
    prob_leave = model.predict_proba(features_array)[0][1]

    if prediction == 1:
        result_html = f"""
        <div class="result-card">
            <h2>⚠️ Employee Likely to Leave</h2>
            <p style='font-size:18px;'>Probability: <b>{prob_leave*100:.2f}%</b></p>
        </div>
        """
    else:
        result_html = f"""
        <div class="result-card" style="background: linear-gradient(145deg, #0B0C1A, #1B263B);">
            <h2>✅ Employee Likely to Stay</h2>
            <p style='font-size:18px;'>Probability: <b>{(1-prob_leave)*100:.2f}%</b></p>
        </div>
        """
    st.markdown(result_html, unsafe_allow_html=True)

    suggestions = generate_suggestions(st.session_state, prediction, prob_leave)
    suggestions_html = f"""
    <div class="suggestion-card">
        <h3>💡 Detailed Suggestions & Improvement Areas</h3>
        <p style="white-space: pre-wrap; word-wrap: break-word;">{suggestions}</p>
    </div>
    """
    st.markdown(suggestions_html, unsafe_allow_html=True)

# ----------------------------
# Sidebar Info
# ----------------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    "This HR Attrition Predictor helps organizations estimate employee attrition risk "
    "based on various features. Use employee details or predefined scenarios. "
    "Detailed suggestions are generated based on the model's prediction."
)
