import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Check", page_icon="🩸", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
if not os.path.exists(os.path.join(OUT, "model.joblib")):
    OUT = BASE  # files were uploaded flat (no outputs folder)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 1150px; }
.hero { background: linear-gradient(135deg,#0ea5e9 0%,#2563eb 55%,#1e1b4b 100%); border-radius: 24px;
  padding: 2.5rem 2.8rem; color: #fff; margin-bottom: 1.4rem; box-shadow: 0 18px 40px rgba(37,99,235,.28); }
.hero h1 { margin: 0 0 .4rem 0; font-size: 2.4rem; font-weight: 800; color: #fff; }
.hero p { margin: 0; font-size: 1.05rem; opacity: .92; max-width: 560px; }
.pill { display: inline-block; background: rgba(255,255,255,.2); padding: .3rem .85rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; margin-bottom: .9rem; }
.sec-title { font-size: 1.02rem; font-weight: 700; margin-bottom: .6rem; }
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: .55rem 1.3rem; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #e0f2fe; color: #1d4ed8 !important; }
.stButton > button { background: linear-gradient(135deg,#0ea5e9,#2563eb); color: #fff; border: none;
  border-radius: 14px; padding: .8rem 2rem; font-size: 1.05rem; font-weight: 700; width: 100%;
  box-shadow: 0 10px 24px rgba(37,99,235,.3); }
.stButton > button:hover { color: #fff; transform: translateY(-2px); }
.result { border-radius: 20px; padding: 1.5rem 1.7rem; color: #1f2937; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
.r-label { font-size: .78rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; opacity: .6; }
.r-level { font-size: 2.1rem; font-weight: 800; margin: .15rem 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap: .8rem; margin: .6rem 0 1rem; }
.tile { background: #fff; border: 1px solid #cfe3f7; border-radius: 16px; padding: 1rem 1.1rem; }
.tile .v { font-size: 1.6rem; font-weight: 800; color: #1d4ed8; }
.tile .l { font-size: .76rem; font-weight: 600; color: #6b7280; text-transform: uppercase; }
.note { text-align: center; color: #9ca3af; font-size: .8rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

if not os.path.exists(f"{OUT}/model.joblib"):
    st.error("Model files not found. Run  python train_diabetes.py  first, then restart the app.")
    st.write("Files next to the app:", sorted(os.listdir(BASE)))
    st.stop()


@st.cache_resource
def load():
    return joblib.load(f"{OUT}/model.joblib"), joblib.load(f"{OUT}/cluster_model.joblib")


model, cluster_model = load()


def tiles(items):
    html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in items)
    st.markdown(f'<div class="tiles">{html}</div>', unsafe_allow_html=True)


st.markdown("""<div class="hero"><div class="pill">AI-POWERED RISK ESTIMATE</div>
<h1>Diabetes Risk Check</h1>
<p>Enter a few health measurements and get an instant estimate of diabetes risk,
along with the patient group that best matches the profile.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🩺  Risk Check", "📊  Model Insights", "👥  Patient Groups"])

with tab1:
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1.container(border=True):
        st.markdown('<div class="sec-title">👤 Personal</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 90, 35)
        preg = st.slider("Number of pregnancies", 0, 17, 1, help="Use 0 for males")
        bmi = st.number_input("BMI", 10.0, 70.0, 27.0, 0.1)
    with c2.container(border=True):
        st.markdown('<div class="sec-title">🩸 Blood Tests</div>', unsafe_allow_html=True)
        glucose = st.slider("Glucose (mg/dl)", 40, 300, 110)
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80, help="Enter 0 if unknown")
        dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.45, 0.01,
                              help="Family history score: higher means stronger family history")
    with c3.container(border=True):
        st.markdown('<div class="sec-title">💓 Body Measures</div>', unsafe_allow_html=True)
        bp = st.slider("Blood pressure (mm Hg)", 30, 140, 72)
        skin = st.number_input("Skin thickness (mm)", 0, 100, 25, help="Enter 0 if unknown")

    st.write("")
    _, mid, _ = st.columns([1.2, 1, 1.2])
    if mid.button("Check my risk"):
        row = pd.DataFrame([{"Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
                             "SkinThickness": skin if skin > 0 else np.nan,
                             "Insulin": insulin if insulin > 0 else np.nan, "BMI": bmi,
                             "DiabetesPedigreeFunction": dpf, "Age": age}])
        prob = float(model.predict_proba(row)[0][1])
        if prob < 0.35:
            level, color, bg, msg = "Low Risk", "#16a34a", "#ecfdf3", "Values suggest a low likelihood of diabetes. Keep a healthy lifestyle and regular checkups."
        elif prob < 0.65:
            level, color, bg, msg = "Moderate Risk", "#d97706", "#fffbeb", "Some indicators point to a possible risk. A routine blood sugar test is a good idea."
        else:
            level, color, bg, msg = "High Risk", "#dc2626", "#fef2f2", "Several indicators are linked to diabetes. Please consult a healthcare professional."
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob * 100, 1), number={"suffix": "%", "font": {"size": 44}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.28}, "borderwidth": 0,
                       "steps": [{"range": [0, 35], "color": "#dcfce7"}, {"range": [35, 65], "color": "#fef3c7"},
                                 {"range": [65, 100], "color": "#fee2e2"}]}))
            fig.update_layout(height=290, margin=dict(l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(f"""<div class="result" style="background:{bg};border-left:8px solid {color};">
            <div class="r-label">Assessment</div><div class="r-level" style="color:{color};">{level}</div>
            <div style="font-weight:600;margin-bottom:.5rem;">{prob * 100:.1f}% estimated probability</div>
            <p style="margin:0;">{msg}</p></div>""", unsafe_allow_html=True)
        g = int(cluster_model.predict(row)[0])
        if os.path.exists(f"{OUT}/cluster_profile.csv"):
            p = pd.read_csv(f"{OUT}/cluster_profile.csv", index_col=0).loc[g]
            st.markdown(f"#### Matching patient group: Group {g}")
            tiles([("Patients in group", int(p["Patients"])), ("Average age", p["Avg_Age"]),
                   ("Average glucose", p["Avg_Glucose"]), ("Diabetes rate", f"{p['Diabetes_Rate_Percent']}%")])

with tab2:
    if os.path.exists(f"{OUT}/model_comparison.csv"):
        comp = pd.read_csv(f"{OUT}/model_comparison.csv")
        top = comp.iloc[0]
        st.markdown(f"#### Best performing model: {top['Model']}")
        tiles([("Accuracy", f"{top['accuracy']:.0%}"), ("Precision", f"{top['precision']:.0%}"),
               ("Recall", f"{top['recall']:.0%}"), ("F1 score", f"{top['f1']:.0%}"), ("ROC-AUC", f"{top['roc_auc']:.2f}")])
        with st.expander("See all models compared"):
            st.dataframe(comp, use_container_width=True, hide_index=True)
    cols = st.columns(3, gap="medium")
    for col, img, cap in zip(cols, ["confusion_matrix.png", "roc_curve.png", "feature_importance.png"],
                             ["Confusion matrix", "ROC curve", "Most important factors"]):
        if os.path.exists(f"{OUT}/{img}"):
            with col.container(border=True):
                st.image(f"{OUT}/{img}", caption=cap, use_container_width=True)
    if os.path.exists(f"{OUT}/association_rules.csv"):
        st.markdown("#### Patterns linked to diabetes")
        st.dataframe(pd.read_csv(f"{OUT}/association_rules.csv"), use_container_width=True, hide_index=True)

with tab3:
    if os.path.exists(f"{OUT}/cluster_profile.csv"):
        prof = pd.read_csv(f"{OUT}/cluster_profile.csv", index_col=0)
        st.markdown("#### Group overview")
        st.dataframe(prof, use_container_width=True)
        fig = go.Figure(go.Bar(x=[f"Group {i}" for i in prof.index], y=prof["Diabetes_Rate_Percent"],
                               marker_color="#2563eb", text=prof["Diabetes_Rate_Percent"].astype(str) + "%",
                               textposition="outside"))
        fig.update_layout(title="Diabetes rate by group", height=340, margin=dict(t=50, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    if os.path.exists(f"{OUT}/clusters_pca.png"):
        with st.container(border=True):
            st.image(f"{OUT}/clusters_pca.png", caption="Patient groups (2D view)", width=560)

st.markdown('<div class="note">For informational purposes only. Not a substitute for professional medical advice.</div>',
            unsafe_allow_html=True)
