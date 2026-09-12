import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load('student_performance_model/lasso_pipeline.pkl')


@st.cache_data
def load_metadata():
    with open('student_performance_model/metadata.json', 'r') as f:
        return json.load(f)


try:
    model = load_model()
    meta = load_metadata()
except Exception as e:
    st.error(f"Could not load model or metadata: {e}")
    st.stop()


# ---------- Styling ----------
st.markdown("""
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f9fafb;
        text-align: center;
        margin-top: 1rem;
    }
    .result-grade {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.2rem 0;
    }
    .result-label {
        color: #6b7280;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-tag {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .tag-excellent { background-color: #d1fae5; color: #065f46; }
    .tag-good      { background-color: #dbeafe; color: #1e40af; }
    .tag-satisfactory { background-color: #fef3c7; color: #92400e; }
    .tag-needs     { background-color: #fed7aa; color: #9a3412; }
    .tag-risk      { background-color: #fee2e2; color: #991b1b; }
    .section-head {
        font-size: 1.05rem;
        font-weight: 600;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.4rem;
        margin-bottom: 0.9rem;
        margin-top: 1rem;
    }
    .note {
        background-color: #f3f4f6;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #374151;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------- Header ----------
st.markdown('<div class="main-title">Student Grade Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the final grade (G3, 0-20) from student background and academic records.</div>',
    unsafe_allow_html=True
)

# ---------- Model Info Strip ----------
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Model", "Lasso Regression")
info_col2.metric("R² Score", f"{meta['test_r2']:.4f}")
info_col3.metric("Mean Abs. Error", f"{meta['test_mae']:.4f}")
info_col4.metric("Accuracy (±1 mark)", f"{meta['test_accuracy_tolerance_1']:.1%}")

st.divider()


# ---------- Helper for numeric default ----------
def default_of(col):
    return int(meta[f'{col}_median'])


# ---------- Personal Info ----------
st.markdown('<div class="section-head">Personal Information</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    school = st.selectbox("School", meta['school_values'])
with c2:
    sex = st.selectbox("Sex", meta['sex_values'])
with c3:
    age = st.number_input("Age",
                          min_value=int(meta['age_range'][0]),
                          max_value=int(meta['age_range'][1]),
                          value=default_of('age'))
with c4:
    address = st.selectbox("Address", meta['address_values'],
                           help="U = Urban, R = Rural")

c1, c2, c3, c4 = st.columns(4)
with c1:
    famsize = st.selectbox("Family Size", meta['famsize_values'],
                           help="LE3 = 3 or fewer, GT3 = more than 3")
with c2:
    Pstatus = st.selectbox("Parents Status", meta['Pstatus_values'],
                           help="T = Together, A = Apart")
with c3:
    guardian = st.selectbox("Guardian", meta['guardian_values'])
with c4:
    romantic = st.selectbox("In Romantic Relationship", meta['romantic_values'])


# ---------- Family Background ----------
st.markdown('<div class="section-head">Family Background</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4],
                        index=default_of('Medu'),
                        help="0=None, 1=Primary, 2=5th-9th, 3=Secondary, 4=Higher")
with c2:
    Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4],
                        index=default_of('Fedu'),
                        help="0=None, 1=Primary, 2=5th-9th, 3=Secondary, 4=Higher")
with c3:
    Mjob = st.selectbox("Mother's Job", meta['Mjob_values'])
with c4:
    Fjob = st.selectbox("Father's Job", meta['Fjob_values'])

c1, c2, c3, c4 = st.columns(4)
with c1:
    reason = st.selectbox("Reason for School Choice", meta['reason_values'])
with c2:
    famrel = st.selectbox("Family Relationship Quality (1-5)", [1, 2, 3, 4, 5],
                          index=default_of('famrel') - 1)
with c3:
    famsup = st.selectbox("Family Support", meta['famsup_values'])
with c4:
    schoolsup = st.selectbox("School Support", meta['schoolsup_values'])


# ---------- Academic Info ----------
st.markdown('<div class="section-head">Academic Information</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    traveltime = st.selectbox("Travel Time to School (1-4)", [1, 2, 3, 4],
                              index=default_of('traveltime') - 1,
                              help="1=<15 min, 2=15-30 min, 3=30min-1h, 4=>1h")
with c2:
    studytime = st.selectbox("Weekly Study Time (1-4)", [1, 2, 3, 4],
                             index=default_of('studytime') - 1,
                             help="1=<2h, 2=2-5h, 3=5-10h, 4=>10h")
with c3:
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4],
                            index=default_of('failures'))
with c4:
    absences = st.number_input("Number of Absences",
                               min_value=int(meta['absences_range'][0]),
                               max_value=min(int(meta['absences_range'][1]), 75),
                               value=default_of('absences'))

c1, c2, c3, c4 = st.columns(4)
with c1:
    G1 = st.number_input("First Period Grade (G1, 0-20)",
                         min_value=0, max_value=20, value=default_of('G1'))
with c2:
    G2 = st.number_input("Second Period Grade (G2, 0-20)",
                         min_value=0, max_value=20, value=default_of('G2'))
with c3:
    paid = st.selectbox("Paid Classes", meta['paid_values'])
with c4:
    activities = st.selectbox("Extracurricular Activities", meta['activities_values'])

c1, c2, c3, c4 = st.columns(4)
with c1:
    nursery = st.selectbox("Attended Nursery", meta['nursery_values'])
with c2:
    higher = st.selectbox("Wants Higher Education", meta['higher_values'])
with c3:
    internet = st.selectbox("Internet at Home", meta['internet_values'])
with c4:
    st.write("")  # spacer


# ---------- Lifestyle ----------
st.markdown('<div class="section-head">Lifestyle Factors</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    freetime = st.selectbox("Free Time (1-5)", [1, 2, 3, 4, 5],
                            index=default_of('freetime') - 1)
with c2:
    goout = st.selectbox("Going Out (1-5)", [1, 2, 3, 4, 5],
                         index=default_of('goout') - 1)
with c3:
    Dalc = st.selectbox("Workday Alcohol (1-5)", [1, 2, 3, 4, 5],
                        index=default_of('Dalc') - 1)
with c4:
    Walc = st.selectbox("Weekend Alcohol (1-5)", [1, 2, 3, 4, 5],
                        index=default_of('Walc') - 1)
with c5:
    health = st.selectbox("Health Status (1-5)", [1, 2, 3, 4, 5],
                          index=default_of('health') - 1)


# ---------- Predict ----------
st.divider()

if st.button("Predict Final Grade", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'school': school, 'sex': sex, 'age': age, 'address': address,
        'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
        'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher,
        'internet': internet, 'romantic': romantic, 'famrel': famrel,
        'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
        'health': health, 'absences': absences, 'G1': G1, 'G2': G2
    }])

    input_data = input_data[meta['all_features']]
    prediction = float(np.clip(model.predict(input_data)[0], 0, 20))

    # Categorize
    if prediction >= 16:
        tag, tag_class = "Excellent", "tag-excellent"
    elif prediction >= 14:
        tag, tag_class = "Good", "tag-good"
    elif prediction >= 10:
        tag, tag_class = "Satisfactory", "tag-satisfactory"
    elif prediction >= 8:
        tag, tag_class = "Needs Improvement", "tag-needs"
    else:
        tag, tag_class = "At Risk", "tag-risk"

    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Final Grade (G3)</div>
            <div class="result-grade">{prediction:.1f} / 20</div>
            <div class="result-tag {tag_class}">{tag}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Range", f"{int(prediction-1)} – {int(prediction+1)}")
    m2.metric("Change from G2", f"{prediction - G2:+.1f}")
    if absences > 10:
        m3.metric("Flag", "High Absences", f"{absences} days")
    elif failures > 1:
        m3.metric("Flag", "Past Failures", f"{failures}")
    elif studytime <= 2:
        m3.metric("Flag", "Low Study Time")
    else:
        m3.metric("Flag", "None")

    if prediction < 10:
        st.markdown(
            '<div class="note"><b>Note:</b> The predicted grade is below the passing threshold. '
            'Consider additional academic support, reducing absences, or tutoring.</div>',
            unsafe_allow_html=True
        )
    elif prediction < 13:
        st.markdown(
            '<div class="note"><b>Note:</b> Borderline performance. Increased study time and '
            'engagement could improve the outcome.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="note"><b>Note:</b> The student is on track for a positive result.</div>',
            unsafe_allow_html=True
        )