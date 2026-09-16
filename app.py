import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import torch
import warnings
warnings.filterwarnings("ignore")
# # importing our deep learning stuff
try:
    from train_deep_models import CostForecastLSTM, DelayTransformer, build_time_series_from_snapshot
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False
# # llm engine
try:
    from llm_engine import generate_risk_summary, answer_project_question
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
# # importing all the core math and ai formulas from our engine
try:
    from prediction_engine import (
        compute_efc, compute_health_score, compute_what_if,
        compute_three_scenarios, compute_early_warning,
        compute_shap_drivers, compute_savings_recommendations,
        compute_intervention_priority, get_model_metadata,
        compute_milestone_risk, classify_risk, RISK_COLORS, RISK_THRESHOLDS
    )
    PRED_ENGINE_AVAILABLE = True
except ImportError as _pe_err:
    PRED_ENGINE_AVAILABLE = False
# # setup streamlit page config
st.set_page_config(
    page_title="PARVAAH-X | MoSPI, Government of India",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/1200px-Emblem_of_India.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
# # adding some custom css to make it look good
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ===============================================================
       COLOR PALETTE — LIGHT THEME WITH DARK TEXT (NO PURE BLACK)
       bg:       #f8fafc  (slate-50)
       surface:  #ffffff  (white)
       border:   #e2e8f0  (slate-200)
       text-1:   #0f172a  (slate-900)  — headings / primary text (very dark navy)
       text-2:   #1e293b  (slate-800)  — body text (dark slate)
       text-3:   #334155  (slate-700)  — muted / labels (medium dark)
       accent:   #2563eb  (blue-600)   — primary action
    =============================================================== */

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* ===================== PANELS ===================== */
    .glass-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        color: #1e293b;
        padding: 20px;
    }

    /* ===================== HEADER ===================== */
    .gov-top-bar { display: none; }

    .gov-header {
        background: #1e3a5f;
        border-radius: 12px;
        padding: 22px 30px;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin: 0.5rem 0 2rem 0;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.1);
    }
    .gov-header .emblem { display: none; }
    .gov-header .title-block h1 {
        margin: 0;
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .gov-header .title-block p {
        margin: 6px 0 0 0;
        font-size: 12.5px;
        color: #cbd5e1;
        font-weight: 400;
        line-height: 1.5;
    }
    .gov-header .right-block { display: none; }
    .tricolor-strip { display: none; }

    /* ===================== SIDEBAR ===================== */
    section[data-testid="stSidebar"] {
        background: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    .sidebar-logo {
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .sidebar-logo h3 {
        margin: 0;
        font-size: 15px;
        font-weight: 700;
        color: #0f172a !important;
        letter-spacing: -0.3px;
    }
    .sidebar-logo p {
        margin: 6px 0 0 0;
        font-size: 11px;
        font-weight: 500;
        color: #334155 !important;
        line-height: 1.5;
    }
    .sidebar-badge {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12px;
        font-weight: 500;
        color: #1e293b !important;
        margin-top: 6px;
    }

    /* ===================== KPI CARDS ===================== */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.5rem 0;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px 22px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        border-color: #cbd5e1;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        border-radius: 10px 10px 0 0;
    }
    .kpi-card.red::before   { background: #ef4444; }
    .kpi-card.amber::before { background: #f59e0b; }
    .kpi-card.green::before { background: #10b981; }

    .kpi-card .kpi-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-card .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    .kpi-card.red   .kpi-value { color: #dc2626; }
    .kpi-card.amber .kpi-value { color: #d97706; }
    .kpi-card.green .kpi-value { color: #059669; }
    .kpi-card .kpi-sub {
        font-size: 11.5px;
        font-weight: 500;
        color: #475569;
        margin-top: 8px;
    }

    /* ===================== SECTION HEADERS ===================== */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        border: none;
        padding: 0;
        margin: 1.8rem 0 1rem 0;
        background: transparent;
    }

    /* ===================== BADGES ===================== */
    .badge-high {
        background: #fee2e2; color: #b91c1c;
        padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
    }
    .badge-med {
        background: #fef3c7; color: #b45309;
        padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
    }
    .badge-low {
        background: #d1fae5; color: #047857;
        padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600;
    }

    /* ===================== ALERT BOXES ===================== */
    .alert-box {
        background: #f8fafc;
        border-left: 3px solid #1e293b;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 13.5px;
        font-weight: 500;
        color: #0f172a;
    }
    .alert-box.danger {
        background: #fef2f2;
        border-left-color: #ef4444;
        color: #991b1b;
    }
    .alert-box.success {
        background: #f0fdf4;
        border-left-color: #10b981;
        color: #065f46;
    }

    /* ===================== FOOTER ===================== */
    .gov-footer {
        background: transparent;
        border-top: 1px solid #e2e8f0;
        color: #475569;
        text-align: center;
        padding: 24px;
        font-size: 12px;
        font-weight: 500;
        margin-top: 4rem;
    }

    /* ===================== PLOTLY ===================== */
    .stPlotlyChart {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        background: #ffffff;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }

    /* ===================== DATAFRAME / TABLE TEXT ===================== */
    /* Table header cells */
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] [role="columnheader"] {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    /* Table body cells */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] .dvn-scroller *,
    [data-testid="stDataFrame"] div[data-testid="glideDataEditor"] * {
        color: #0f172a !important;
    }
    /* Ensure all text inside the data grid is dark */
    .glideDataEditor canvas + div *,
    [class*="DataGrid"] *,
    [data-testid="stDataFrame"] span {
        color: #0f172a !important;
    }

    /* ===================== SEARCH BAR / TEXT INPUT ===================== */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Input placeholder text — slightly lighter but still visible */
    [data-testid="stTextInput"] input::placeholder,
    .stTextInput input::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
    }
    /* Input labels */
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    .stTextInput label,
    .stSelectbox label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
# # top header section with logos
st.markdown("""
<div class="gov-top-bar">
    <span>&#127470;&#127475; Government of India | Ministry of Statistics and Programme Implementation</span>
    <span>Last Updated: Sep 2026 | Data Source: MoSPI PARVAAH-X Portal</span>
</div>
<div class="gov-header">
    <img class="emblem" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/240px-Emblem_of_India.svg.png" alt="Emblem of India"/>
    <div class="title-block">
        <h1>Ministry of Statistics and Programme Implementation</h1>
        <p>PARVAAH-X &mdash; Predictive Analytics for Infrastructure Monitoring and National Assessment</p>
    </div>
    <div class="right-block">
        <h2>Risk Early Warning System</h2>
        <p>AI/ML Powered &bull; Projects &ge; &#8377;150 Crore</p>
    </div>
</div>
<div class="tricolor-strip"></div>
""", unsafe_allow_html=True)
# # loading datasets and trained models here
@st.cache_resource
def load_models():
    cost_model = joblib.load('models/cost_model.joblib')
    time_model = joblib.load('models/time_model.joblib')
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open('models/label_encoders.json', 'r') as f:
        label_encoders = json.load(f)
    return cost_model, time_model, feature_names, label_encoders

@st.cache_data
def load_and_predict_data():
    # Load raw data
    df = pd.read_csv('data/real_augmented_parvaah_x_data.csv')
    date_cols = ['Start_Date', 'Planned_Completion_Date', 'Current_Completion_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    df['Planned_Duration_Days'] = (df['Planned_Completion_Date'] - df['Start_Date']).dt.days
    df['Current_Duration_Days'] = (df['Current_Completion_Date'] - df['Start_Date']).dt.days
    df['Expenditure_Progress_Ratio'] = (df['Cumulative_Expenditure_Cr'] / df['Original_Approved_Cost_Cr']) / (df['Physical_Progress_Pct'] / 100 + 0.001)

    # Load models internally to run one-time predictions on the dataset
    c_model, t_model, f_names, l_encs = load_models()
    
    df_pred = df.copy()
    for col, classes in l_encs.items():
        # Using map is much faster than apply lambda for dictionaries/lists
        class_dict = {c: i for i, c in enumerate(classes)}
        df_pred[col] = df_pred[col].map(class_dict).fillna(0).astype(int)
        
    prediction_data = df_pred[f_names]
    
    # Run inference once and cache it
    cost_probs = c_model.predict_proba(prediction_data)[:, 1]
    time_probs = t_model.predict_proba(prediction_data)[:, 1]
    
    df['Cost_Risk_Score'] = cost_probs
    df['Time_Risk_Score'] = time_probs
    df['Overall_Risk_Score'] = (cost_probs * 0.5) + (time_probs * 0.5)
    
    # Vectorized categorization instead of apply
    conditions = [
        (df['Overall_Risk_Score'] >= 0.7),
        (df['Overall_Risk_Score'] >= 0.4)
    ]
    choices = ['High', 'Medium']
    df['Risk_Category'] = np.select(conditions, choices, default='Low')
    df['Cost_Overrun_Cr'] = df['Revised_Cost_Cr'] - df['Original_Approved_Cost_Cr']
    
    return df, prediction_data

try:
    cost_model, time_model, feature_names, label_encoders = load_models()
    raw_data, prediction_data = load_and_predict_data()

except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()
# # setting up the left sidebar
import os as _os
_logo_path = _os.path.join(_os.path.dirname(__file__), "parvaah_logo.png")
if _os.path.exists(_logo_path):
    st.sidebar.image(_logo_path, use_container_width=True)
else:
    st.sidebar.markdown("""
<div class="sidebar-logo">
    <div>
        <h3>PARVAAH-X</h3>
        <p>Early Warning System</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("", ["National Dashboard", "Project Doctor (XAI)", "Benchmarking", "Future Predictor (AI)", "Cost Overrun Estimator", "Pre-Project Feasibility"], label_visibility="collapsed")

st.sidebar.markdown("---")
# # code for the main national dashboard page
if page == "National Dashboard":

    high_risk_count = len(raw_data[raw_data['Risk_Category'] == 'High'])
    med_risk_count = len(raw_data[raw_data['Risk_Category'] == 'Medium'])
    low_risk_count = len(raw_data[raw_data['Risk_Category'] == 'Low'])
    total_overrun = raw_data[raw_data['Cost_Overrun_Cr'] > 0]['Cost_Overrun_Cr'].sum()
    overrun_lakh_cr = f"{total_overrun/1e5:.1f}"

    st.markdown('<div class="section-header">National Infrastructure Risk Overview</div>', unsafe_allow_html=True)
    st.caption("Predictive risk analysis for Central Sector Projects with cost >= Rs. 150 crore | Powered by XGBoost AI/ML Models")

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Projects Monitored</div>
            <div class="kpi-value">{len(raw_data):,}</div>
            <div class="kpi-sub">Across all sectors &amp; ministries</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">High Risk Projects</div>
            <div class="kpi-value">{high_risk_count:,}</div>
            <div class="kpi-sub">Require immediate intervention</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Medium Risk Projects</div>
            <div class="kpi-value">{med_risk_count:,}</div>
            <div class="kpi-sub">Monitoring recommended</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Predicted Cost Overrun</div>
            <div class="kpi-value">&#8377;{overrun_lakh_cr}L Cr</div>
            <div class="kpi-sub">Across all high-risk projects</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
# # first row of charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Risk Distribution by Sector</div>', unsafe_allow_html=True)
        sector_risk = raw_data.groupby(['Sector', 'Risk_Category']).size().reset_index(name='Count')
        fig = px.bar(sector_risk, x='Sector', y='Count', color='Risk_Category',
                     color_discrete_map={"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"},
                     barmode='stack',
                     template='plotly_white')
        fig.update_layout(
            xaxis_tickangle=-35,
            font=dict(family='Inter', size=12),
            legend=dict(orientation='h', y=-0.3),
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Expenditure vs. Physical Progress (Risk Map)</div>', unsafe_allow_html=True)
        fig2 = px.scatter(raw_data,
                          x='Physical_Progress_Pct', y='Expenditure_Progress_Ratio',
                          color='Risk_Category', size='Original_Approved_Cost_Cr',
                          hover_name='Project_Name',
                          color_discrete_map={"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"},
                          labels={'Physical_Progress_Pct': 'Physical Progress (%)',
                                  'Expenditure_Progress_Ratio': 'Expenditure-Progress Ratio'},
                          template='plotly_white',
                          opacity=0.75)
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#888",
                       annotation_text="Ideal Ratio = 1.0", annotation_position="bottom right")
        fig2.update_layout(font=dict(family='Inter', size=12),
                           margin=dict(l=20, r=20, t=20, b=20),
                           legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)
# # second row of charts
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Cost Overrun by Sector (Predicted)</div>', unsafe_allow_html=True)
        sector_overrun = raw_data[raw_data['Cost_Overrun_Cr'] > 0].groupby('Sector')['Cost_Overrun_Cr'].sum().reset_index()
        sector_overrun = sector_overrun.sort_values('Cost_Overrun_Cr', ascending=True)
        fig3 = px.bar(sector_overrun, x='Cost_Overrun_Cr', y='Sector', orientation='h',
                      labels={'Cost_Overrun_Cr': 'Predicted Cost Overrun (Cr)'},
                      color='Cost_Overrun_Cr',
                      color_continuous_scale=['#fff9c4', '#c62828'],
                      template='plotly_white')
        fig3.update_layout(showlegend=False, font=dict(family='Inter', size=12),
                           margin=dict(l=20, r=20, t=20, b=20),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Risk Category Breakdown</div>', unsafe_allow_html=True)
        pie_data = raw_data['Risk_Category'].value_counts().reset_index()
        pie_data.columns = ['Category', 'Count']
        fig4 = px.pie(pie_data, names='Category', values='Count',
                      color='Category',
                      color_discrete_map={"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"},
                      hole=0.45,
                      template='plotly_white')
        fig4.update_traces(textinfo='label+percent', textfont_size=13)
        fig4.update_layout(legend=dict(orientation='h', y=-0.1),
                           margin=dict(l=20, r=20, t=20, b=20),
                           font=dict(family='Inter', size=12))
        st.plotly_chart(fig4, use_container_width=True)
# # showing projects that need immediate help
    if PRED_ENGINE_AVAILABLE:
        st.markdown('<div class="section-header">🚨 Projects Requiring Immediate Intervention</div>', unsafe_allow_html=True)
        st.caption("Ranked by Intervention Priority Score = Cost Risk (40%) + Time Risk (30%) + Financial Exposure (30%)")
        priority_df = compute_intervention_priority(raw_data).head(15)
        pdisp = priority_df[['Project_ID','Project_Name','Sector','Original_Approved_Cost_Cr',
                              'Cost_Overrun_Cr','Risk_Category','Priority_Score_Pct','Recommended_Action']].copy()
        pdisp['Original_Approved_Cost_Cr'] = pdisp['Original_Approved_Cost_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr")
        pdisp['Cost_Overrun_Cr'] = pdisp['Cost_Overrun_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr" if x > 0 else "-")
        pdisp['Priority_Score_Pct'] = pdisp['Priority_Score_Pct'].astype(str) + "%"
        pdisp.columns = ['Project ID','Project Name','Sector','Budget','Expected Overrun',
                         'Risk Level','Priority Score','Recommended Action']
        st.dataframe(pdisp, use_container_width=True, hide_index=True)
        st.markdown("---")
# # the main table for all projects
    st.markdown('<div class="section-header">Project Watchlist</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search = st.text_input("Search Project Name", placeholder="e.g. Mumbai, Airport, Highway...")
    with f2:
        filter_sector = st.selectbox("Filter by Sector", ["All"] + sorted(raw_data['Sector'].dropna().unique().tolist()))
    with f3:
        filter_risk = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])

    filtered_df = raw_data.copy()
    if search:
        filtered_df = filtered_df[filtered_df['Project_Name'].str.contains(search, case=False, na=False)]
    if filter_sector != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == filter_sector]
    if filter_risk != "All":
        filtered_df = filtered_df[filtered_df['Risk_Category'] == filter_risk]

    display_df = filtered_df[['Project_ID', 'Project_Name', 'Sector', 'Implementing_Agency',
                               'Original_Approved_Cost_Cr', 'Cost_Overrun_Cr',
                               'Physical_Progress_Pct', 'Overall_Risk_Score', 'Risk_Category']].copy()
    display_df = display_df.sort_values('Overall_Risk_Score', ascending=False)
    display_df['Overall_Risk_Score'] = (display_df['Overall_Risk_Score'] * 100).round(1).astype(str) + "%"
    display_df['Cost_Overrun_Cr'] = display_df['Cost_Overrun_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr" if x > 0 else "-")
    display_df['Original_Approved_Cost_Cr'] = display_df['Original_Approved_Cost_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr")
    display_df['Physical_Progress_Pct'] = display_df['Physical_Progress_Pct'].apply(lambda x: f"{x:.1f}%")
    display_df.columns = ['Project ID', 'Project Name', 'Sector', 'Agency',
                          'Approved Cost', 'Predicted Overrun',
                          'Progress', 'Risk Score', 'Risk Level']
    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Risk Level": st.column_config.TextColumn("Risk Level"),
                     "Risk Score": st.column_config.TextColumn("Risk Score"),
                 })
    st.caption(f"Showing {len(filtered_df)} of {len(raw_data)} projects")
# # project doctor page (xai stuff)
elif page == "Project Doctor (XAI)":

    st.markdown('<div class="section-header">Project Doctor — Explainable AI Analysis</div>', unsafe_allow_html=True)
    st.caption("Select a project to understand why the AI model has flagged it as at risk.")

    project_options = (raw_data['Project_ID'] + " — " + raw_data['Project_Name']).tolist()
    selected = st.selectbox("Select Project for Deep-Dive Analysis", project_options)
    proj_id = selected.split(" — ")[0]

    proj_data = raw_data[raw_data['Project_ID'] == proj_id].iloc[0]
    proj_idx = raw_data[raw_data['Project_ID'] == proj_id].index[0]
    proj_features = prediction_data.iloc[[proj_idx]]

    risk_cat = proj_data['Risk_Category']
    badge_class = "badge-high" if risk_cat == "High" else ("badge-med" if risk_cat == "Medium" else "badge-low")
    alert_class = "danger" if risk_cat == "High" else ("" if risk_cat == "Medium" else "success")

    st.markdown(f"""
    <div style="background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:20px; margin:12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <p style="margin:0; font-size:12px; text-transform:uppercase; color:#888; letter-spacing:0.5px;">{proj_data['Sector']} | {proj_data['Implementing_Agency']}</p>
                <h2 style="margin:4px 0; font-family:'Noto Serif',serif; color:#1a237e; font-size:22px;">{proj_data['Project_Name']}</h2>
                <p style="margin:0; font-size:13px; color:#666;">Project ID: {proj_data['Project_ID']} &nbsp;|&nbsp; Start Date: {proj_data['Start_Date'].date()}</p>
            </div>
            <span class="{badge_class}" style="font-size:16px; padding:6px 18px;">{risk_cat} Risk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
# # kpi cards row
    m1, m2, m3, m4, m5 = st.columns(5)
    overall_risk = proj_data['Overall_Risk_Score'] * 100
    cost_risk = proj_data['Cost_Risk_Score'] * 100
    time_risk = proj_data['Time_Risk_Score'] * 100
    overrun_amt = proj_data['Revised_Cost_Cr'] - proj_data['Original_Approved_Cost_Cr']

    m1.metric("Overall Risk Score", f"{overall_risk:.1f}%")
    m2.metric("Cost Overrun Risk", f"{cost_risk:.1f}%")
    m3.metric("Time Delay Risk", f"{time_risk:.1f}%")
    m4.metric("Approved Budget", f"Rs. {proj_data['Original_Approved_Cost_Cr']:,.0f} Cr")
    m5.metric("Predicted Overrun", f"Rs. {max(0,overrun_amt):,.0f} Cr", delta=f"+{max(0,overrun_amt):,.0f} Cr" if overrun_amt > 0 else "On Budget", delta_color="inverse")

    st.markdown("---")
# # using shap to explain the ai predictions
    st.markdown('<div class="section-header">AI Explanation — Why is this Project Flagged?</div>', unsafe_allow_html=True)

    @st.cache_resource
    def get_explainer(_model, _key):
        return shap.TreeExplainer(_model)

    try:
        explainer_cost = get_explainer(cost_model, "cost")
        shap_values_cost = explainer_cost.shap_values(proj_features)

        explainer_time = get_explainer(time_model, "time")
        shap_values_time = explainer_time.shap_values(proj_features)

        c1, c2 = st.columns(2)
        friendly_names = {
            'Climate_Issue_Severity': 'Climate/Weather Severity',
            'Geopolitical_War_Impact': 'Geopolitical Disruption',
            'Supply_Chain_Disruption': 'Supply Chain Risk',
            'Physical_Progress_Pct': 'Physical Progress (%)',
            'Expenditure_Progress_Ratio': 'Expenditure-Progress Ratio',
            'Planned_Duration_Days': 'Planned Duration (Days)',
            'Current_Duration_Days': 'Current Duration (Days)',
            'Original_Approved_Cost_Cr': 'Approved Budget (Cr)',
            'Cumulative_Expenditure_Cr': 'Cumulative Expenditure (Cr)',
            'Sector': 'Sector Type',
            'Implementing_Agency': 'Implementing Agency',
        }

        with c1:
            st.markdown("**Drivers of Cost Overrun Risk**")
            fi = pd.DataFrame({'Feature': feature_names, 'Impact': shap_values_cost[0]})
            fi['AbsImpact'] = fi['Impact'].abs()
            fi = fi.sort_values('AbsImpact', ascending=False).head(6)
            fi['Feature'] = fi['Feature'].map(lambda f: friendly_names.get(f, f))
            fi['Direction'] = fi['Impact'].apply(lambda x: "Increases Risk" if x > 0 else "Reduces Risk")
            fi['Value'] = fi['Feature'].apply(lambda f: proj_features[feature_names[list(friendly_names.values()).index(f)] if f in list(friendly_names.values()) else f].values[0] if f in list(friendly_names.values()) else "—")

            colors = ['#c62828' if d == 'Increases Risk' else '#2e7d32' for d in fi['Direction']]
            figbar = go.Figure(go.Bar(
                x=fi['Impact'].values,
                y=fi['Feature'].values,
                orientation='h',
                marker_color=colors,
                text=fi['Direction'].values,
                textposition='outside'
            ))
            figbar.update_layout(template='plotly_white', height=280,
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 font=dict(family='Inter', size=11),
                                 xaxis_title="Impact on Risk Score")
            st.plotly_chart(figbar, use_container_width=True, key="shap_cost_bar")

        with c2:
            st.markdown("**Drivers of Time Delay Risk**")
            fit = pd.DataFrame({'Feature': feature_names, 'Impact': shap_values_time[0]})
            fit['AbsImpact'] = fit['Impact'].abs()
            fit = fit.sort_values('AbsImpact', ascending=False).head(6)
            fit['Feature'] = fit['Feature'].map(lambda f: friendly_names.get(f, f))
            fit['Direction'] = fit['Impact'].apply(lambda x: "Increases Risk" if x > 0 else "Reduces Risk")

            colorst = ['#c62828' if d == 'Increases Risk' else '#2e7d32' for d in fit['Direction']]
            figbar2 = go.Figure(go.Bar(
                x=fit['Impact'].values,
                y=fit['Feature'].values,
                orientation='h',
                marker_color=colorst,
                text=fit['Direction'].values,
                textposition='outside'
            ))
            figbar2.update_layout(template='plotly_white', height=280,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  font=dict(family='Inter', size=11),
                                  xaxis_title="Impact on Risk Score")
            st.plotly_chart(figbar2, use_container_width=True, key="shap_time_bar")

    except Exception as ex:
        st.warning(f"SHAP explanation unavailable for this model type: {ex}")
# # recommendations
    st.markdown('<div class="section-header">Recommended Interventions</div>', unsafe_allow_html=True)

    recommendations = []
    if proj_data.get('Climate_Issue_Severity', 0) > 0.5:
        recommendations.append(("danger", "Climate / Weather Risk Detected",
            f"Severity index: {proj_data['Climate_Issue_Severity']:.2f}. Anticipate construction delays due to extreme weather events. Consider activating Force Majeure clauses and scheduling weather-sensitive works in dry months."))
    if proj_data.get('Geopolitical_War_Impact', 0) > 0.5:
        recommendations.append(("danger", "Geopolitical / Supply Disruption",
            f"Geopolitical risk score: {proj_data['Geopolitical_War_Impact']:.2f}. Import-dependent materials may face price escalation. Recommend switching to domestic sourcing and building strategic stockpiles."))
    if proj_data.get('Supply_Chain_Disruption', 0) > 0.5:
        recommendations.append(("", "Supply Chain Disruption Risk",
            f"Supply chain risk: {proj_data['Supply_Chain_Disruption']:.2f}. Evaluate alternate vendors. Issue advance purchase orders for long-lead items to avoid construction standstill."))
    if proj_data['Physical_Progress_Pct'] < 30 and proj_data['Current_Duration_Days'] > 365:
        recommendations.append(("danger", "Critically Low Physical Progress",
            "Project is more than 12 months old but physical progress is below 30%. Immediate high-level review required. Identify ground-level blockers, land acquisition issues, or contractor performance lapses."))
    if proj_data['Expenditure_Progress_Ratio'] > 1.2:
        recommendations.append(("", "Financial Mismatch — Expenditure Outpacing Progress",
            f"Expenditure-Progress ratio: {proj_data['Expenditure_Progress_Ratio']:.2f}. Funds are being spent faster than physical work is advancing. Conduct urgent audit of fund utilisation and check for cost escalation in BoQ items."))

    if not recommendations:
        recommendations.append(("success", "Project Within Acceptable Risk Bounds",
            "No critical risk factors detected. Continue routine monitoring as per MoSPI schedule."))

    for alert_type, title, msg in recommendations:
        st.markdown(f"""
        <div class="alert-box {alert_type}">
            <strong>{title}</strong><br/>{msg}
        </div>
        """, unsafe_allow_html=True)
# # the future predictor page using deep models
elif page == "Future Predictor (AI)":

    st.markdown('<div class="section-header">Future Predictor</div>', unsafe_allow_html=True)
    st.caption("LSTM forecasts month-by-month cost escalation. Transformer predicts completion delay. LLM Copilot answers your questions.")
# # api key is in env file so we dont need to ask the user
    from llm_engine import GEMINI_API_KEY as _AUTO_KEY
    gemini_key = _AUTO_KEY
# # project selector
    project_options = (raw_data['Project_ID'] + " — " + raw_data['Project_Name']).tolist()
    selected = st.selectbox("Select Project to Forecast", project_options,
                            index=raw_data['Overall_Risk_Score'].idxmax())
    proj_id = selected.split(" — ")[0]
    proj_data = raw_data[raw_data['Project_ID'] == proj_id].iloc[0]
    proj_idx = raw_data[raw_data['Project_ID'] == proj_id].index[0]

    risk_cat = proj_data['Risk_Category']
    badge_class = "badge-high" if risk_cat == "High" else ("badge-med" if risk_cat == "Medium" else "badge-low")

    budget_str = f"{proj_data['Original_Approved_Cost_Cr']:,.0f}"
    proj_name = proj_data['Project_Name']
    proj_id   = proj_data['Project_ID']
    proj_sector = proj_data['Sector']
    proj_agency = proj_data['Implementing_Agency']
    proj_start  = proj_data['Start_Date'].date()

    st.markdown(f"""
    <div class="glass-panel" style="display:flex; justify-content:space-between; align-items:center; padding:16px 20px; margin:12px 0;">
        <div>
            <p style="margin:0; font-size:11px; text-transform:uppercase; color:#64748b;">{proj_sector} | {proj_agency}</p>
            <h2 style="margin:4px 0; font-family:'Inter',sans-serif; color:#334155; font-size:22px;">{proj_name}</h2>
            <p style="margin:0; font-size:12px; color:#94a3b8;">{proj_id} | Start: {proj_start} | Budget: Rs. {budget_str} Cr</p>
        </div>
        <span class="{badge_class}" style="font-size:15px; padding:6px 18px;">{risk_cat} Risk</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
# # deep learning section starts here
# # fixed the bug where both models shared the same flag lol
    lstm_model_loaded_ok = False
    trans_model_loaded_ok = False
    lstm_models_loaded = False  # kept for backward compat
    if DEEP_MODELS_AVAILABLE and os.path.exists('models/lstm_cost_model.pt') and os.path.exists('models/transformer_delay_model.pt'):
        try:
            @st.cache_resource
            def load_deep_models():
                lstm = CostForecastLSTM(input_size=6, hidden_size=128, num_layers=2, forecast_horizon=12)
                lstm.load_state_dict(torch.load('models/lstm_cost_model.pt', map_location='cpu'))
                lstm.eval()
                trans = DelayTransformer(input_size=6, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128)
                trans.load_state_dict(torch.load('models/transformer_delay_model.pt', map_location='cpu'))
                trans.eval()
                return lstm, trans

            lstm_model_loaded, trans_model_loaded = load_deep_models()
            lstm_models_loaded = True
            lstm_model_loaded_ok = True
            trans_model_loaded_ok = True
        except Exception as e:
            st.warning(f"Could not load deep models: {e}. Please run `python train_deep_models.py` first.")

    tab1, tab2, tab3 = st.tabs(["Cost Forecast (LSTM)", "Completion Timeline (Transformer)", "AI Copilot (LLM)"])
# # tab 1: lstm model for cost
    with tab1:
        st.markdown('<div class="section-header">Month-by-Month Cost Escalation Forecast</div>', unsafe_allow_html=True)
        st.caption("PyTorch LSTM model predicts future expenditure trajectory based on project's historical progress pattern.")

        if lstm_models_loaded:
            import random, torch
            torch.manual_seed(int(proj_idx) + 42)
            random.seed(int(proj_idx) + 42)
# # prepping data sequence for the model
            orig_cost = max(proj_data['Original_Approved_Cost_Cr'], 1)
            rev_cost  = max(proj_data['Revised_Cost_Cr'], orig_cost)
            seq = []
            for t in range(12):
                frac = (t + 1) / 12
                seq.append([
                    np.clip((proj_data['Cumulative_Expenditure_Cr'] / orig_cost) * frac, 0, 3),
                    np.clip((proj_data['Physical_Progress_Pct'] / 100) * frac, 0, 1),
                    np.clip(proj_data['Climate_Issue_Severity'], 0, 1),
                    np.clip(proj_data['Geopolitical_War_Impact'], 0, 1),
                    np.clip(proj_data['Supply_Chain_Disruption'], 0, 1),
                    frac
                ])

            X_proj = torch.FloatTensor([seq])
            with torch.no_grad():
                pred_normalized = lstm_model_loaded(X_proj).numpy()[0]  # 12 values
# # denormalize: multiply by original cost
            pred_costs = pred_normalized * orig_cost
            current_spend = proj_data['Cumulative_Expenditure_Cr']
# # build month labels
            from datetime import date, timedelta
            today = date.today()
            months = [(today.replace(day=1) + timedelta(days=32*i)).replace(day=1) for i in range(1, 13)]
            month_labels = [m.strftime("%b %Y") for m in months]
# # adding some confidence intervals
            upper = pred_costs * 1.10
            lower = pred_costs * 0.90

            fig_forecast = go.Figure()
# # upper band fill
            fig_forecast.add_trace(go.Scatter(
                x=month_labels + month_labels[::-1],
                y=list(upper) + list(lower[::-1]),
                fill='toself', fillcolor='rgba(198,40,40,0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Band (±10%)', showlegend=True
            ))
# # main forecast line
            fig_forecast.add_trace(go.Scatter(
                x=month_labels, y=pred_costs,
                mode='lines+markers',
                name='LSTM Forecast (Expenditure Cr)',
                line=dict(color='#c62828', width=2.5),
                marker=dict(size=6)
            ))
# # approved budget line
            fig_forecast.add_hline(y=orig_cost, line_dash='dash', line_color='#1a237e',
                                   annotation_text=f"Approved Budget: Rs. {orig_cost:,.0f} Cr",
                                   annotation_position="bottom right")
# # revised budget line
            if rev_cost > orig_cost:
                fig_forecast.add_hline(y=rev_cost, line_dash='dot', line_color='#e65100',
                                       annotation_text=f"Revised Cost: Rs. {rev_cost:,.0f} Cr",
                                       annotation_position="top right")
# # current expenditure marker
            fig_forecast.add_trace(go.Scatter(
                x=[month_labels[0]], y=[current_spend],
                mode='markers', name='Current Expenditure',
                marker=dict(color='#1b5e20', size=12, symbol='diamond')
            ))

            fig_forecast.update_layout(
                template='plotly_white',
                font=dict(family='Inter', size=12),
                xaxis_title="Month",
                yaxis_title="Cumulative Expenditure (Crore Rs.)",
                legend=dict(orientation='h', y=-0.2),
                margin=dict(l=10, r=10, t=20, b=20),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
# # summary callout
            final_forecast = pred_costs[-1]
            budget_breach = final_forecast > orig_cost
            overrun_pct = ((final_forecast - orig_cost) / orig_cost) * 100
            alert_t = "danger" if budget_breach else "success"
            alert_msg = (f"LSTM model forecasts cumulative expenditure will reach <strong>Rs. {final_forecast:,.0f} Cr</strong> by {month_labels[-1]}, "
                         f"which is <strong>{abs(overrun_pct):.1f}% {'above' if budget_breach else 'within'}</strong> the approved budget of Rs. {orig_cost:,.0f} Cr.")
            st.markdown(f'<div class="alert-box {alert_t}">{alert_msg}</div>', unsafe_allow_html=True)

        else:
            st.info("Deep learning models not yet trained. Please run `python train_deep_models.py` from your terminal, then refresh this page.")
# # show demo chart so the ui looks complete
            months_demo = ['Oct 2026','Nov 2026','Dec 2026','Jan 2027','Feb 2027','Mar 2027',
                           'Apr 2027','May 2027','Jun 2027','Jul 2027','Aug 2027','Sep 2027']
            orig_cost = proj_data['Original_Approved_Cost_Cr']
            base = proj_data['Cumulative_Expenditure_Cr']
            demo_vals = [base + (orig_cost - base) * (i+1)/12 * 1.15 for i in range(12)]
            fig_demo = go.Figure()
            fig_demo.add_trace(go.Scatter(x=months_demo, y=demo_vals, mode='lines+markers',
                                          name='Demo Forecast', line=dict(color='#c62828', width=2, dash='dot')))
            fig_demo.add_hline(y=orig_cost, line_dash='dash', line_color='#1a237e',
                                annotation_text="Approved Budget")
            fig_demo.update_layout(template='plotly_white', height=350,
                                    font=dict(family='Inter', size=12),
                                    title="[Demo — Train models for real predictions]",
                                    margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_demo, use_container_width=True)
# # tab 2: transformer model for delays
    with tab2:
        st.markdown('<div class="section-header">Project Completion Date — Transformer Prediction</div>', unsafe_allow_html=True)
        st.caption("PyTorch Transformer model predicts revised completion date and delay probability distribution.")

        if trans_model_loaded_ok:  # FIX: use Transformer-specific flag
            import random, torch
            torch.manual_seed(int(proj_idx) + 99)
            X_proj2 = torch.FloatTensor([seq])
            with torch.no_grad():
                delay_norm = trans_model_loaded(X_proj2).item()
            delay_months = max(0, delay_norm * 60)  # denormalize: max 60 months

            planned_dt = proj_data['Planned_Completion_Date']
            predicted_completion = planned_dt + pd.DateOffset(months=int(delay_months))

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Planned Completion", str(planned_dt.date()))
            d2.metric("Predicted Completion (AI)", str(predicted_completion.date()),
                      delta=f"+{delay_months:.0f} months delay", delta_color="inverse")
            d3.metric("Predicted Delay", f"{delay_months:.1f} months")
            d4.metric("Delay Probability", f"{proj_data['Time_Risk_Score']*100:.1f}%")
# # ganttstyle timeline
            today = pd.Timestamp.today()
            start = proj_data['Start_Date']
            gantt_data = pd.DataFrame([
                dict(Task="Planned Timeline", Start=str(start.date()), Finish=str(planned_dt.date()), Color="Planned"),
                dict(Task="AI Predicted End", Start=str(planned_dt.date()), Finish=str(predicted_completion.date()), Color="Overrun"),
            ])
            gantt_data['Start'] = pd.to_datetime(gantt_data['Start'])
            gantt_data['Finish'] = pd.to_datetime(gantt_data['Finish'])
            gantt_data['Duration'] = (gantt_data['Finish'] - gantt_data['Start']).dt.days

            fig_gantt = px.timeline(gantt_data, x_start="Start", x_end="Finish", y="Task",
                                     color="Color",
                                     color_discrete_map={"Planned": "#1a237e", "Overrun": "#c62828"},
                                     template='plotly_white')
            fig_gantt.add_vline(x=today, line_dash="dot", line_color="#138808",
                                annotation_text="Today", annotation_position="top")
            fig_gantt.update_layout(font=dict(family='Inter', size=12),
                                     margin=dict(l=10, r=10, t=20, b=10),
                                     height=200, legend_title="")
            st.plotly_chart(fig_gantt, use_container_width=True)
# # delay factor breakdown
            st.markdown("**Transformer-identified Delay Contributing Factors:**")
            factors = {
                "Supply Chain Disruption": proj_data['Supply_Chain_Disruption'],
                "Climate / Weather Issues": proj_data['Climate_Issue_Severity'],
                "Geopolitical Impact": proj_data['Geopolitical_War_Impact'],
                "Physical Progress Deficit": max(0, 1 - proj_data['Physical_Progress_Pct'] / 100),
                "Budget Pressure": min(1, proj_data['Expenditure_Progress_Ratio'] / 2),
            }
            fig_factors = go.Figure(go.Bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation='h',
                marker_color=['#c62828' if v > 0.5 else '#e65100' if v > 0.3 else '#2e7d32' for v in factors.values()],
                text=[f"{v:.2f}" for v in factors.values()],
                textposition='outside'
            ))
            fig_factors.update_layout(template='plotly_white', height=250,
                                       margin=dict(l=10, r=60, t=10, b=10),
                                       font=dict(family='Inter', size=12),
                                       xaxis_title="Risk Factor Magnitude (0–1)",
                                       xaxis_range=[0, 1.3])
            st.plotly_chart(fig_factors, use_container_width=True)

        else:
            st.info("Train deep models first: run `python train_deep_models.py`")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Planned Completion", str(proj_data['Planned_Completion_Date'].date()))
            d2.metric("Estimated Completion", str(proj_data['Current_Completion_Date'].date()),
                      delta=f"From original plan", delta_color="inverse")
            d3.metric("Time Delay Risk", f"{proj_data['Time_Risk_Score']*100:.1f}%")
            d4.metric("Risk Category", proj_data['Risk_Category'])
# # tab 3: gemini copilot
    with tab3:
        st.markdown('<div class="section-header">AI Intelligence Copilot — Project Risk Q&A</div>', unsafe_allow_html=True)
        st.caption("Powered by Google Gemini 3.6 Flash LLM. Ask anything about this project in Hindi or English.")
# # telling gemini to give a quick summary
        st.markdown("**Auto-Generated Risk Assessment Report:**")
        with st.spinner("Generating AI risk assessment..."):
            risk_factors_text = []
            if proj_data.get('Supply_Chain_Disruption', 0) > 0.5:
                risk_factors_text.append(f"supply chain disruption (index: {proj_data['Supply_Chain_Disruption']:.2f})")
            if proj_data.get('Climate_Issue_Severity', 0) > 0.5:
                risk_factors_text.append(f"climate/weather severity (index: {proj_data['Climate_Issue_Severity']:.2f})")
            if proj_data.get('Geopolitical_War_Impact', 0) > 0.5:
                risk_factors_text.append(f"geopolitical impact (index: {proj_data['Geopolitical_War_Impact']:.2f})")
            if proj_data['Physical_Progress_Pct'] < 30:
                risk_factors_text.append(f"critically low physical progress ({proj_data['Physical_Progress_Pct']:.1f}%)")
            if proj_data['Expenditure_Progress_Ratio'] > 1.2:
                risk_factors_text.append(f"expenditure outpacing physical progress (ratio: {proj_data['Expenditure_Progress_Ratio']:.2f})")

            proj_dict = proj_data.to_dict()
            summary = generate_risk_summary(proj_dict, risk_factors_text) if LLM_AVAILABLE else "LLM engine not loaded."

        st.markdown(f"""
        <div class="glass-panel" style="padding:16px 20px; margin:8px 0; line-height:1.7; font-size:14px; border-left:4px solid #38bdf8;">
            {summary}
        </div>
        """, unsafe_allow_html=True)
# # q&a section
        st.markdown("---")
        st.markdown("**Ask the AI Copilot about this project:**")
# # quick question buttons
        st.markdown("*Quick questions:*")
        qcols = st.columns(4)
        quick_qs = [
            "Why will this project be delayed?",
            "How much extra money will be needed?",
            "What is the biggest risk factor?",
            "What action should the ministry take?"
        ]
        for i, q in enumerate(quick_qs):
            if qcols[i].button(q, key=f"qbtn_{i}"):
                st.session_state['copilot_question'] = q
# # freetext q&a
        user_q = st.text_input("Or type your own question:", value=st.session_state.get('copilot_question', ''),
                               placeholder="e.g. What is the predicted delay? / Kitna paisa zyada lagega?",
                               key="qa_input")

        if user_q:
            with st.spinner("AI is thinking..."):
                answer = answer_project_question(proj_dict, user_q) if LLM_AVAILABLE else "LLM engine not loaded."
            st.markdown(f"""
            <div class="glass-panel" style="padding:14px 18px; margin:8px 0; line-height:1.7; font-size:14px; border-left:4px solid #10b981;">
                <strong>AI Copilot:</strong><br/>{answer}
            </div>
            """, unsafe_allow_html=True)
# # clear session state after displaying
            if 'copilot_question' in st.session_state:
                del st.session_state['copilot_question']

        st.markdown("""
        <div style="font-size:11px; color:#aaa; margin-top:10px;">
            AI responses are generated by Google Gemini and may require verification against official records.
            This system is for decision-support purposes only.
        </div>
        """, unsafe_allow_html=True)
# # benchmarking page to compare sectors
elif page == "Benchmarking":

    st.markdown('<div class="section-header">Sector-wise Benchmarking &amp; Comparative Analysis</div>', unsafe_allow_html=True)
    st.caption("Compare project performance and risk profiles within the same sector.")

    sector = st.selectbox("Select Sector", sorted(raw_data['Sector'].dropna().unique().tolist()))
    sector_data = raw_data[raw_data['Sector'] == sector]

    avg_progress = sector_data['Physical_Progress_Pct'].mean()
    avg_risk = sector_data['Overall_Risk_Score'].mean() * 100
    high_pct = (len(sector_data[sector_data['Risk_Category'] == 'High']) / len(sector_data)) * 100

    avg_progress_str = f"{avg_progress:.1f}"
    avg_risk_str     = f"{avg_risk:.1f}"
    high_pct_str     = f"{high_pct:.0f}"
    sector_proj_count = len(sector_data)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Projects in Sector</div>
            <div class="kpi-value">{sector_proj_count:,}</div>
            <div class="kpi-sub">Active projects under monitoring</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg. Physical Progress</div>
            <div class="kpi-value">{avg_progress_str}%</div>
            <div class="kpi-sub">Sector average completion</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Avg. Risk Score</div>
            <div class="kpi-value">{avg_risk_str}%</div>
            <div class="kpi-sub">AI-predicted average risk</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">High Risk Projects</div>
            <div class="kpi-value">{high_pct_str}%</div>
            <div class="kpi-sub">Of total projects in sector</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.scatter(sector_data, x='Physical_Progress_Pct', y='Overall_Risk_Score',
                         hover_name='Project_Name', color='Risk_Category',
                         color_discrete_map={"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"},
                         size='Original_Approved_Cost_Cr',
                         labels={'Physical_Progress_Pct': 'Physical Progress (%)',
                                 'Overall_Risk_Score': 'Predicted Risk Score'},
                         template='plotly_white')
        fig.add_vline(x=avg_progress, line_dash="dash", line_color="#888", annotation_text="Sector Avg Progress")
        fig.add_hline(y=avg_risk / 100, line_dash="dash", line_color="#c62828", annotation_text="Sector Avg Risk")
        fig.update_layout(font=dict(family='Inter', size=12), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with bc2:
        st.markdown('<div class="section-header">Cost Overrun Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(sector_data[sector_data['Cost_Overrun_Cr'] > 0],
                                x='Cost_Overrun_Cr', nbins=20,
                                labels={'Cost_Overrun_Cr': 'Cost Overrun (Crore Rs.)'},
                                color_discrete_sequence=['#c62828'],
                                template='plotly_white')
        fig_hist.update_layout(font=dict(family='Inter', size=12), margin=dict(l=10, r=10, t=20, b=10),
                               bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-header">Top 10 Highest Risk Projects in Sector</div>', unsafe_allow_html=True)
    top10 = sector_data.sort_values('Overall_Risk_Score', ascending=False).head(10)[
        ['Project_ID', 'Project_Name', 'Implementing_Agency', 'Original_Approved_Cost_Cr',
         'Cost_Overrun_Cr', 'Physical_Progress_Pct', 'Overall_Risk_Score', 'Risk_Category']].copy()
    top10['Overall_Risk_Score'] = (top10['Overall_Risk_Score'] * 100).round(1).astype(str) + "%"
    top10['Original_Approved_Cost_Cr'] = top10['Original_Approved_Cost_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr")
    top10['Cost_Overrun_Cr'] = top10['Cost_Overrun_Cr'].apply(lambda x: f"Rs. {x:,.0f} Cr" if x > 0 else "-")
    top10['Physical_Progress_Pct'] = top10['Physical_Progress_Pct'].apply(lambda x: f"{x:.1f}%")
    top10.columns = ['ID', 'Project Name', 'Agency', 'Budget', 'Overrun', 'Progress', 'Risk Score', 'Risk Level']
    st.dataframe(top10, use_container_width=True, hide_index=True)
# # cost overrun estimator page  the main feature
elif page == "Cost Overrun Estimator":

    st.markdown('<div class="section-header">Cost Overrun Estimator — AI Budget Risk Analysis</div>', unsafe_allow_html=True)
    st.caption("Enter your project details below — budget, area, labour, material costs and risk factors. The AI model will estimate overrun risk and predicted cost escalation.")

    st.markdown("""
    <style>
    .estimator-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 10px 0 18px 0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.07);
    }
    .estimator-card h4 {
        margin: 0 0 16px 0;
        font-size: 14px;
        font-weight: 700;
        color: #1e3a5f;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
    }
    .result-panel {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 28px 32px;
        color: #fff;
        margin: 18px 0;
    }
    .result-panel .result-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.8;
        margin-bottom: 6px;
    }
    .result-panel .result-value {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
    }
    .result-panel .result-sub {
        font-size: 13px;
        opacity: 0.75;
        margin-top: 6px;
    }
    .risk-meter-high   { background: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 16px 20px; text-align:center; }
    .risk-meter-medium { background: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 16px 20px; text-align:center; }
    .risk-meter-low    { background: #f0fdf4; border: 2px solid #10b981; border-radius: 10px; padding: 16px 20px; text-align:center; }
    .risk-meter-high   h3 { color: #b91c1c; margin:0; font-size:28px; }
    .risk-meter-medium h3 { color: #d97706; margin:0; font-size:28px; }
    .risk-meter-low    h3 { color: #059669; margin:0; font-size:28px; }
    .risk-meter-high   p  { color: #991b1b; margin:4px 0 0 0; font-size:13px; font-weight:600; }
    .risk-meter-medium p  { color: #92400e; margin:4px 0 0 0; font-size:13px; font-weight:600; }
    .risk-meter-low    p  { color: #065f46; margin:4px 0 0 0; font-size:13px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)
# # form to get user inputs
    with st.form("cost_estimator_form"):
# # section 1: basic project details
        st.markdown('<div class="estimator-card"><h4>📋 Project Identity</h4>', unsafe_allow_html=True)
        pi1, pi2, pi3 = st.columns(3)
        with pi1:
            inp_sector = st.selectbox("Sector", label_encoders['Sector'], help="Select the infrastructure sector")
        with pi2:
            inp_agency = st.selectbox("Implementing Agency", label_encoders['Implementing_Agency'])
        with pi3:
            inp_duration_planned = st.number_input("Planned Duration (months)", min_value=6, max_value=300, value=48, step=6)
        st.markdown('</div>', unsafe_allow_html=True)
# # section 2: money matters
        st.markdown('<div class="estimator-card"><h4>💰 Budget & Financial Details</h4>', unsafe_allow_html=True)
        bf1, bf2, bf3 = st.columns(3)
        with bf1:
            inp_budget = st.number_input("Approved Budget (Crore Rs.)", min_value=150.0, max_value=100000.0,
                                         value=1500.0, step=100.0,
                                         help="Original sanctioned/approved project cost")
        with bf2:
            inp_expenditure = st.number_input("Cumulative Expenditure So Far (Crore Rs.)", min_value=0.0,
                                              max_value=100000.0, value=300.0, step=50.0,
                                              help="Amount spent till date")
        with bf3:
            inp_progress = st.slider("Physical Progress (%)", min_value=0, max_value=100, value=20,
                                     help="Actual physical/construction work completed")

        bf4, bf5 = st.columns(2)
        with bf4:
            inp_labour_pct = st.slider("Labour Cost as % of Budget", min_value=5, max_value=60, value=25,
                                       help="Estimated share of labour in total project cost")
        with bf5:
            inp_material_pct = st.slider("Material Cost as % of Budget", min_value=10, max_value=70, value=40,
                                          help="Estimated share of materials/equipment in total cost")
        st.markdown('</div>', unsafe_allow_html=True)
# # section 3: dates and duration
        st.markdown('<div class="estimator-card"><h4>📅 Timeline Status</h4>', unsafe_allow_html=True)
        tl1, tl2 = st.columns(2)
        with tl1:
            inp_current_duration = st.number_input("Actual Elapsed Duration (months)", min_value=1, max_value=300,
                                                    value=18, step=1,
                                                    help="How many months since project started")
        with tl2:
            inp_area_sqkm = st.number_input("Project Area / Span (km or sq.km)", min_value=1.0, max_value=5000.0,
                                             value=50.0, step=10.0,
                                             help="Length (for roads/railways) or area (for dams/airports)")
        st.markdown('</div>', unsafe_allow_html=True)
# # section 4: external risks like climate etc
        st.markdown('<div class="estimator-card"><h4>⚠️ External Risk Factors</h4>', unsafe_allow_html=True)
        rf1, rf2, rf3 = st.columns(3)
        with rf1:
            inp_climate = st.slider("Climate / Weather Severity", 0.0, 1.0, 0.3, 0.05,
                                    help="0 = No risk, 1 = Extreme weather disruptions (floods, cyclones, etc.)")
        with rf2:
            inp_geopolitical = st.slider("Geopolitical / Import Risk", 0.0, 1.0, 0.2, 0.05,
                                         help="0 = No risk, 1 = Severe import/geopolitical disruption")
        with rf3:
            inp_supply = st.slider("Supply Chain Disruption", 0.0, 1.0, 0.25, 0.05,
                                   help="0 = Smooth supply, 1 = Major supply chain breakdown")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Analyse & Predict Cost Overrun", use_container_width=True,
                                          type="primary")
# # this is where the main prediction happens
    if submitted:
        planned_days  = int(inp_duration_planned * 30.4)
        current_days  = int(inp_current_duration * 30.4)
        exp_ratio     = (inp_expenditure / max(inp_budget, 1)) / (max(inp_progress, 1) / 100)

        sector_enc = label_encoders['Sector'].index(inp_sector) if inp_sector in label_encoders['Sector'] else 0
        agency_enc = label_encoders['Implementing_Agency'].index(inp_agency) if inp_agency in label_encoders['Implementing_Agency'] else 0

        inp_row = pd.DataFrame([{
            'Sector':                    sector_enc,
            'Implementing_Agency':       agency_enc,
            'Original_Approved_Cost_Cr': inp_budget,
            'Climate_Issue_Severity':    inp_climate,
            'Geopolitical_War_Impact':   inp_geopolitical,
            'Supply_Chain_Disruption':   inp_supply,
            'Physical_Progress_Pct':     float(inp_progress),
            'Cumulative_Expenditure_Cr': inp_expenditure,
            'Planned_Duration_Days':     float(planned_days),
            'Current_Duration_Days':     float(current_days),
            'Expenditure_Progress_Ratio': exp_ratio,
        }])[feature_names]

        cost_risk_score = cost_model.predict_proba(inp_row)[0][1]
        time_risk_score = time_model.predict_proba(inp_row)[0][1]
        overall_risk    = cost_risk_score * 0.6 + time_risk_score * 0.4

        labour_cost   = inp_budget * (inp_labour_pct / 100)
        material_cost = inp_budget * (inp_material_pct / 100)
        overhead_cost = inp_budget - labour_cost - material_cost
# # calling our prediction engine to do the heavy lifting
        if PRED_ENGINE_AVAILABLE:
            efc_result = compute_efc(
                inp_budget, inp_expenditure, inp_progress, cost_risk_score,
                inp_climate, inp_geopolitical, inp_supply,
                planned_days, current_days, inp_sector, raw_data
            )
            health   = compute_health_score(cost_risk_score, time_risk_score, inp_progress,
                                             exp_ratio, inp_climate, inp_supply, inp_geopolitical)
            warning  = compute_early_warning(cost_risk_score, time_risk_score,
                                              inp_budget, inp_expenditure, current_days)
            shap_drivers, shap_source = compute_shap_drivers(cost_model, inp_row, feature_names)
            scenarios = compute_three_scenarios(
                cost_model, inp_row, feature_names,
                inp_budget, inp_expenditure, inp_progress,
                inp_climate, inp_geopolitical, inp_supply,
                planned_days, current_days, inp_sector, raw_data
            )
            savings_recs = compute_savings_recommendations(
                cost_model, inp_row, feature_names,
                inp_budget, inp_expenditure, inp_progress,
                inp_climate, inp_geopolitical, inp_supply,
                planned_days, current_days, inp_sector, raw_data,
                base_efc=efc_result['efc']
            )
            milestone_risks = compute_milestone_risk(
                cost_risk_score, time_risk_score, inp_progress,
                inp_supply, inp_climate, inp_geopolitical
            )
            model_meta = get_model_metadata(cost_model, time_model, raw_data, feature_names)
        else:
            efc_result = {"efc": inp_budget * (1 + cost_risk_score * 0.15),
                          "efc_low": inp_budget, "efc_high": inp_budget * 1.25,
                          "overrun_prob": cost_risk_score, "shortfall_cr": 0,
                          "saving_cr": 0, "overrun_pct": cost_risk_score * 15,
                          "sector_mu_pct": 0, "sector_std_pct": 0}
            health = {"total": 50, "label": "Moderate", "cost_health": 50,
                      "schedule_health": 50, "progress_health": inp_progress,
                      "fin_efficiency": 50, "ext_risk_health": 50}
            shap_drivers, shap_source = [], "Unavailable"
            scenarios = {}
            savings_recs = []
            warning = {"now": {"cost_risk": cost_risk_score*100, "time_risk": time_risk_score*100},
                       "3m": {"cost_risk": 0, "time_risk": 0},
                       "6m": {"cost_risk": 0, "time_risk": 0},
                       "12m": {"cost_risk": 0, "time_risk": 0},
                       "months_to_breach": 0, "monthly_spend": 0}
            milestone_risks = []
            model_meta = {}

        risk_label  = classify_risk(cost_risk_score) if PRED_ENGINE_AVAILABLE else ("High" if cost_risk_score >= 0.7 else ("Medium" if cost_risk_score >= 0.4 else "Low"))
        risk_color  = RISK_COLORS.get(risk_label, "#dc2626") if PRED_ENGINE_AVAILABLE else {"High": "#c62828", "Medium": "#e65100", "Low": "#2e7d32"}.get(risk_label, "#c62828")
        meter_class = {"Low": "risk-meter-low", "Moderate": "risk-meter-medium",
                       "High": "risk-meter-high", "Critical": "risk-meter-high",
                       "Medium": "risk-meter-medium"}.get(risk_label, "risk-meter-high")

        efc         = efc_result['efc']
        efc_low     = efc_result['efc_low']
        efc_high    = efc_result['efc_high']
        shortfall   = efc_result['shortfall_cr']
        overrun_pct = efc_result['overrun_pct']

        st.markdown("---")
# #
# # a. showing the overall ai predictions
# #
        st.markdown('<div class="section-header">🧠 AI Prediction Results</div>', unsafe_allow_html=True)

        top1, top2, top3, top4, top5 = st.columns(5)
        with top1:
            h_color = {"Healthy": "#059669", "Moderate": "#d97706",
                       "At Risk": "#dc2626", "Critical": "#7f1d1d"}.get(health["label"], "#dc2626")
            st.markdown(f"""
            <div style="background:#fff;border:2px solid {h_color};border-radius:12px;
                        padding:18px 14px;text-align:center;">
                <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Project Health</div>
                <div style="font-size:36px;font-weight:800;color:{h_color};line-height:1;">{health['total']:.0f}</div>
                <div style="font-size:12px;color:{h_color};font-weight:700;margin-top:4px;">/100 — {health['label']}</div>
            </div>
            """, unsafe_allow_html=True)
        with top2:
            st.metric("Cost Overrun Probability", f"{cost_risk_score*100:.1f}%",
                      delta=risk_label, delta_color="inverse" if cost_risk_score >= 0.3 else "normal")
        with top3:
            st.metric("Expected Final Cost (EFC)", f"Rs. {efc:,.0f} Cr",
                      delta=f"+Rs. {shortfall:,.0f} Cr" if shortfall > 0 else f"-Rs. {efc_result['saving_cr']:,.0f} Cr",
                      delta_color="inverse" if shortfall > 0 else "normal")
        with top4:
            st.metric("Expected Overrun", f"Rs. {shortfall:,.0f} Cr",
                      delta=f"{overrun_pct:+.1f}% vs budget",
                      delta_color="inverse" if shortfall > 0 else "normal")
        with top5:
            st.metric("Time Delay Risk", f"{time_risk_score*100:.1f}%",
                      delta="High" if time_risk_score >= 0.6 else "Moderate" if time_risk_score >= 0.3 else "Low",
                      delta_color="inverse" if time_risk_score >= 0.3 else "normal")

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # b. the expected final cost section
# #
        st.markdown('<div class="section-header">💰 Expected Final Cost</div>', unsafe_allow_html=True)
        efc_col1, efc_col2 = st.columns([2, 3])
        with efc_col1:
            gap_color = "#c62828" if shortfall > 0 else "#059669"
            st.markdown(f"""
            <div class="result-panel">
                <div class="result-title">Approved Budget</div>
                <div class="result-value">Rs. {inp_budget:,.0f} Cr</div>
                <div class="result-sub">&nbsp;</div>
                <div class="result-title" style="margin-top:16px;">AI Expected Final Cost</div>
                <div class="result-value" style="color:#fbbf24;">Rs. {efc:,.0f} Cr</div>
                <div class="result-sub">{efc_result['interval_note']}</div>
                <div style="margin-top:12px;font-size:13px;opacity:0.9;">
                    Prediction Interval: Rs. {efc_low:,.0f} – Rs. {efc_high:,.0f} Cr
                </div>
                <div style="margin-top:12px;font-size:13px;opacity:0.9;">
                    Sector avg overrun: {efc_result['sector_mu_pct']:.1f}% ± {efc_result['sector_std_pct']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with efc_col2:
# # efc gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=efc,
                title={"text": "Expected Final Cost (Cr)", "font": {"size": 14, "family": "Inter"}},
                delta={"reference": inp_budget, "relative": False,
                       "valueformat": ",.0f",
                       "increasing": {"color": "#c62828"}, "decreasing": {"color": "#059669"}},
                gauge={
                    "axis": {"range": [inp_budget * 0.70, inp_budget * 1.60]},
                    "bar":  {"color": risk_color, "thickness": 0.25},
                    "steps": [
                        {"range": [inp_budget*0.70, inp_budget],       "color": "#d1fae5"},
                        {"range": [inp_budget,      inp_budget*1.15],  "color": "#fef3c7"},
                        {"range": [inp_budget*1.15, inp_budget*1.60],  "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#1e3a5f", "width": 3},
                                  "thickness": 0.75, "value": inp_budget},
                },
                number={"valueformat": ",.0f", "suffix": " Cr"},
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=60,b=10),
                                    font=dict(family="Inter", size=12))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # c. showing what is driving the risk (shap values)
# #
        st.markdown(f'<div class="section-header">📋 Risk Drivers (Source: {shap_source})</div>', unsafe_allow_html=True)
        if shap_drivers:
            drv_col1, drv_col2 = st.columns([3, 2])
            with drv_col1:
                drv_df = pd.DataFrame(shap_drivers)
                colors = ["#c62828" if d == "Increases Risk" else "#059669" for d in drv_df["direction"]]
                fig_shap = go.Figure(go.Bar(
                    x=drv_df["pct"], y=drv_df["label"], orientation='h',
                    marker_color=colors,
                    text=[f"{v:.1f}% ({d})" for v, d in zip(drv_df["pct"], drv_df["direction"])],
                    textposition='outside',
                ))
                fig_shap.update_layout(template='plotly_white', height=300,
                                       margin=dict(l=10,r=80,t=10,b=10),
                                       font=dict(family='Inter', size=12),
                                       xaxis_title="% Contribution to Risk Score")
                st.plotly_chart(fig_shap, use_container_width=True)
            with drv_col2:
                for d in shap_drivers:
                    badge = "🔴" if d['direction'] == "Increases Risk" else "🟢"
                    st.markdown(f"""
                    <div style="background:#f8fafc;border-left:3px solid {'#c62828' if d['direction']=='Increases Risk' else '#059669'};
                                border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;">
                        <b>{badge} {d['label']}</b><br/>
                        <span style="font-size:12px;color:#64748b;">{d['pct']:.1f}% contribution &mdash; {d['direction']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # d. overall project health score ui
# #
        st.markdown('<div class="section-header">🏥 Project Health Score</div>', unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([1, 2])
        with h_col1:
            hc = {"Healthy": "#059669", "Moderate": "#d97706", "At Risk": "#dc2626", "Critical": "#7f1d1d"}.get(health["label"], "#dc2626")
            st.markdown(f"""
            <div style="background:#fff;border:2px solid {hc};border-radius:16px;
                        padding:30px;text-align:center;margin:8px 0;">
                <div style="font-size:72px;font-weight:900;color:{hc};line-height:1;">{health['total']:.0f}</div>
                <div style="font-size:16px;color:{hc};font-weight:700;margin:8px 0;">/ 100</div>
                <div style="font-size:20px;font-weight:800;color:{hc};">{health['label']}</div>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            components = [
                ("Cost Health (30%)",       health['cost_health'],     "#2563eb"),
                ("Schedule Health (25%)",   health['schedule_health'], "#0891b2"),
                ("Progress Health (20%)",   health['progress_health'], "#059669"),
                ("Financial Efficiency (15%)", health['fin_efficiency'],"#7c3aed"),
                ("External Risk Health (10%)", health['ext_risk_health'],"#d97706"),
            ]
            fig_health = go.Figure()
            for label, val, color in components:
                fig_health.add_trace(go.Bar(
                    name=label, x=[val], y=[label], orientation='h',
                    marker_color=color,
                    text=[f"{val:.0f}/100"], textposition='outside',
                ))
            fig_health.update_layout(template='plotly_white', height=280, showlegend=False,
                                      barmode='group', margin=dict(l=10,r=60,t=10,b=10),
                                      font=dict(family='Inter', size=12),
                                      xaxis=dict(range=[0, 120], title="Score"))
            st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # e. best, worst, and most likely scenarios
# #
        st.markdown('<div class="section-header">🌢 Optimistic / Most Likely / Worst-Case Scenarios</div>', unsafe_allow_html=True)
        if scenarios:
            sc1, sc2, sc3 = st.columns(3)
            scen_map = [
                (sc1, "🟢 Optimistic",   scenarios['optimistic'],  "#059669", "Low risk factors halved"),
                (sc2, "🔵 Most Likely",  scenarios['most_likely'], "#2563eb", "Current inputs as-is"),
                (sc3, "🔴 Worst Case",   scenarios['worst_case'],  "#c62828", "Risk factors +50%, delay +20%"),
            ]
            for col, label, scen, color, assumption in scen_map:
                with col:
                    st.markdown(f"""
                    <div style="background:#fff;border:2px solid {color};border-radius:12px;
                                padding:20px;text-align:center;">
                        <div style="font-size:13px;font-weight:700;color:{color};margin-bottom:10px;">{label}</div>
                        <div style="font-size:26px;font-weight:800;color:#0f172a;">Rs. {scen['efc']:,.0f} Cr</div>
                        <div style="font-size:12px;color:#64748b;margin:6px 0;">Expected Final Cost</div>
                        <div style="font-size:14px;font-weight:700;color:{color};">Overrun: {scen['overrun_pct']:+.1f}%</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:8px;">{assumption}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
# # scenario comparison bar chart
            fig_sc = go.Figure(data=[
                go.Bar(name='EFC (Cr)', x=["Optimistic","Most Likely","Worst Case"],
                       y=[scenarios['optimistic']['efc'], scenarios['most_likely']['efc'], scenarios['worst_case']['efc']],
                       marker_color=['#059669','#2563eb','#c62828'],
                       text=[f"Rs.{scenarios[k]['efc']:,.0f} Cr" for k in ['optimistic','most_likely','worst_case']],
                       textposition='outside'),
            ])
            fig_sc.add_hline(y=inp_budget, line_dash='dash', line_color='#1e3a5f',
                             annotation_text=f"Budget: Rs.{inp_budget:,.0f} Cr")
            fig_sc.update_layout(template='plotly_white', height=280, showlegend=False,
                                  margin=dict(l=10,r=10,t=20,b=10),
                                  font=dict(family='Inter', size=12),
                                  yaxis_title="Expected Final Cost (Cr)")
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # f. the whatif simulator feature
# #
        st.markdown('<div class="section-header">⚙️ What-If Scenario Simulator</div>', unsafe_allow_html=True)
        st.caption("Adjust controllable risk factors to see real-time impact on Expected Final Cost (all values come from the AI model).")

        wi_c1, wi_c2 = st.columns(2)
        with wi_c1:
            wi_supply  = st.slider("Supply Chain Risk",   0.0, 1.0, float(inp_supply),  0.05, key="wi_supply")
            wi_climate = st.slider("Climate Risk",        0.0, 1.0, float(inp_climate), 0.05, key="wi_climate")
        with wi_c2:
            wi_geo     = st.slider("Geopolitical Risk",   0.0, 1.0, float(inp_geopolitical), 0.05, key="wi_geo")
            wi_cd_mo   = st.slider("Current Duration (months)", 1, 240, int(inp_current_duration), 1, key="wi_cd")

        wi_cd_days = int(wi_cd_mo * 30.4)
        wi_result  = compute_what_if(
            cost_model, inp_row, feature_names,
            inp_budget, inp_expenditure, inp_progress,
            inp_climate, inp_geopolitical, inp_supply,
            planned_days, current_days, inp_sector, raw_data,
            new_climate=wi_climate, new_geo=wi_geo, new_supply=wi_supply,
            new_current_days=wi_cd_days
        ) if PRED_ENGINE_AVAILABLE else efc_result

        wi_delta_efc  = wi_result['efc'] - efc
        wi_delta_prob = (wi_result['overrun_prob'] - efc_result['overrun_prob']) * 100

        wa1, wa2, wa3, wa4 = st.columns(4)
        wa1.metric("Current Scenario EFC",  f"Rs. {efc:,.0f} Cr",       delta="Baseline")
        wa2.metric("New Scenario EFC",       f"Rs. {wi_result['efc']:,.0f} Cr",
                   delta=f"{wi_delta_efc:+,.0f} Cr", delta_color="inverse" if wi_delta_efc > 0 else "normal")
        wa3.metric("Potential Saving",       f"Rs. {max(0,-wi_delta_efc):,.0f} Cr",
                   delta="vs baseline" )
        wa4.metric("Overrun Prob Change",    f"{wi_delta_prob:+.1f}%",
                   delta_color="inverse" if wi_delta_prob > 0 else "normal")

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # g. recommendations to save cost
# #
        st.markdown('<div class="section-header">💡 AI Cost-Saving Recommendations</div>', unsafe_allow_html=True)
        if savings_recs:
            total_saving = sum(r["Saving (Cr)"] for r in savings_recs)
            st.success(f"📊 Potential Total Saving: **Rs. {total_saving:,.2f} Cr** across {len(savings_recs)} intervention(s) — all computed via AI model")
            for i, rec in enumerate(savings_recs, 1):
                conf_color = "#059669" if rec["Confidence"] == "High" else "#d97706"
                st.markdown(f"""
                <div class="alert-box" style="border-left-color:{conf_color};">
                    <b>{i}. {rec['Recommendation']}</b>
                    &nbsp;&mdash;&nbsp;<span style="color:{conf_color};font-weight:700;">Confidence: {rec['Confidence']}</span><br/>
                    <b>Problem:</b> {rec['Problem Detected']}<br/>
                    <b>Action:</b> {rec['Action']}<br/>
                    <b>Assumption:</b> {rec['Assumption']}<br/>
                    <b>Estimated Saving: Rs. {rec['Saving (Cr)']:,.2f} Cr</b>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant savings opportunities detected with these inputs. All risk factors are within acceptable bounds.")

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # h. early warning system timeline
# #
        st.markdown('<div class="section-header">⚠️ Early Warning Timeline</div>', unsafe_allow_html=True)
        st.caption(f"ℹ️ {warning.get('note', '')} Monthly spend: Rs. {warning.get('monthly_spend',0):,.1f} Cr/month")

        ew_labels = ["Now", "3 Months", "6 Months", "12 Months"]
        ew_keys   = ["now",  "3m",       "6m",       "12m"]
        ew_cr = [warning[k]["cost_risk"] for k in ew_keys]
        ew_tr = [warning[k]["time_risk"] for k in ew_keys]

        fig_ew = go.Figure()
        fig_ew.add_trace(go.Scatter(
            x=ew_labels, y=ew_cr, mode='lines+markers+text',
            name='Cost Overrun Risk (%)', line=dict(color='#c62828', width=2.5),
            marker=dict(size=10), text=[f"{v:.0f}%" for v in ew_cr], textposition='top center'))
        fig_ew.add_trace(go.Scatter(
            x=ew_labels, y=ew_tr, mode='lines+markers+text',
            name='Schedule Delay Risk (%)', line=dict(color='#2563eb', width=2.5, dash='dash'),
            marker=dict(size=10), text=[f"{v:.0f}%" for v in ew_tr], textposition='bottom center'))
        fig_ew.add_hline(y=60, line_dash='dot', line_color='#f59e0b',
                         annotation_text="High Risk Threshold (60%)")
        fig_ew.update_layout(template='plotly_white', height=300,
                              margin=dict(l=10,r=10,t=20,b=10),
                              font=dict(family='Inter', size=12),
                              yaxis=dict(title="Risk Probability (%)", range=[0, 105]),
                              legend=dict(orientation='h', y=-0.2),
                              hovermode='x unified')
        st.plotly_chart(fig_ew, use_container_width=True)

        if warning.get("months_to_breach", 999) < 24:
            mtb = warning["months_to_breach"]
            st.warning(f"🚨 **Budget Breach Alert:** At the current burn rate of Rs. {warning['monthly_spend']:,.1f} Cr/month, "
                       f"the project may exhaust its budget within approximately **{mtb:.0f} months**. Immediate review recommended.")

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # i. risk at milestone level
# #
        if milestone_risks:
            st.markdown('<div class="section-header">📍 Milestone-Level Risk Analysis</div>', unsafe_allow_html=True)
            st.caption("Delay probabilities scaled from project-level model output (proxy estimate).")
            ms_df = pd.DataFrame(milestone_risks)
            def _ms_color(row):
                rl = row['Risk Level']
                return 'background-color: #fee2e2' if rl == 'Critical' else \
                       'background-color: #fef3c7' if rl == 'High' else \
                       'background-color: #d1fae5' if rl == 'Low' else ''
            st.dataframe(ms_df, use_container_width=True, hide_index=True)

        st.markdown("<br/>", unsafe_allow_html=True)
# #
# # j. some basic info about the model used
# #

# #  original summary table (preserved)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:15px;">Full Input Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Parameter": ["Sector", "Agency", "Approved Budget", "Expenditure So Far",
                           "Physical Progress", "Labour Cost", "Material Cost",
                           "Planned Duration", "Actual Duration", "Project Area",
                           "Climate Severity", "Geopolitical Risk", "Supply Chain Risk",
                           "Expenditure-Progress Ratio"],
            "Value": [inp_sector, inp_agency, f"Rs. {inp_budget:,.0f} Cr", f"Rs. {inp_expenditure:,.0f} Cr",
                      f"{inp_progress}%", f"Rs. {labour_cost:,.0f} Cr ({inp_labour_pct}%)",
                      f"Rs. {material_cost:,.0f} Cr ({inp_material_pct}%)",
                      f"{inp_duration_planned} months", f"{inp_current_duration} months",
                      f"{inp_area_sqkm} km/sq.km",
                      f"{inp_climate:.2f}", f"{inp_geopolitical:.2f}", f"{inp_supply:.2f}",
                      f"{exp_ratio:.2f}"],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
# # new page: preproject feasibility check
elif page == "Pre-Project Feasibility":

    st.markdown('<div class="section-header">Pre-Project Cost Feasibility Analysis</div>', unsafe_allow_html=True)
# #  inline css for this page
    st.markdown("""
    <style>
    .feas-card {
        background:#fff; border:1px solid #e2e8f0; border-radius:12px;
        padding:22px 26px; margin:8px 0 16px 0;
        box-shadow:0 2px 8px rgba(15,23,42,0.07);
    }
    .feas-card h4 {
        margin:0 0 14px 0; font-size:13px; font-weight:700; color:#1e3a5f;
        text-transform:uppercase; letter-spacing:0.6px;
        border-bottom:2px solid #e2e8f0; padding-bottom:9px;
    }
    .verdict-within { background:#f0fdf4; border:2px solid #10b981; border-radius:12px; padding:20px 24px; text-align:center; }
    .verdict-risk   { background:#fffbeb; border:2px solid #f59e0b; border-radius:12px; padding:20px 24px; text-align:center; }
    .verdict-over   { background:#fef2f2; border:2px solid #ef4444; border-radius:12px; padding:20px 24px; text-align:center; }
    .verdict-within h2 { color:#059669; margin:0; font-size:32px; font-weight:800; }
    .verdict-risk   h2 { color:#d97706; margin:0; font-size:32px; font-weight:800; }
    .verdict-over   h2 { color:#c62828; margin:0; font-size:32px; font-weight:800; }
    .verdict-within p  { color:#065f46; margin:6px 0 0 0; font-size:13px; font-weight:600; }
    .verdict-risk   p  { color:#92400e; margin:6px 0 0 0; font-size:13px; font-weight:600; }
    .verdict-over   p  { color:#991b1b; margin:6px 0 0 0; font-size:13px; font-weight:600; }
    .factor-bar { height:8px; border-radius:4px; background:#e2e8f0; margin:4px 0 10px 0; }
    .factor-fill { height:8px; border-radius:4px; }
    </style>
    """, unsafe_allow_html=True)
# # getting inputs for feasibility
    with st.form("feasibility_form"):
# # section 1: basic project details
        st.markdown('<div class="feas-card"><h4>🏗️ Project Identity</h4>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            f_sector = st.selectbox("Sector", sorted(raw_data['Sector'].dropna().unique().tolist()),
                                    help="Select the infrastructure sector for this new project")
        with c2:
            f_agency = st.selectbox("Implementing Agency", sorted(raw_data['Implementing_Agency'].dropna().unique().tolist()),
                                    help="Select the agency that will execute the project")
        with c3:
            f_proj_type = st.selectbox("Project Sub-Type",
                ["Highway / Expressway", "Bridge / Flyover", "Tunnel", "Airport Terminal",
                 "Metro / Rail Corridor", "Power Plant", "Dam / Reservoir", "Urban Infra",
                 "Water Supply Network", "Other"],
                help="Specific type of infrastructure work")
        st.markdown('</div>', unsafe_allow_html=True)
# #  section 2: budget & scope
        st.markdown('<div class="feas-card"><h4>💰 Proposed Budget & Project Scope</h4>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            f_budget = st.number_input("Allocated / Proposed Budget (Rs. Cr)",
                                       min_value=150.0, max_value=500000.0, value=5000.0, step=100.0,
                                       help="The budget proposed/sanctioned for this project")
        with b2:
            f_size   = st.number_input("Project Size / Length (km or sq.km)",
                                       min_value=1.0, max_value=5000.0, value=50.0, step=5.0,
                                       help="Length for linear projects (road/rail), or area for others")
        with b3:
            f_duration = st.number_input("Planned Duration (months)",
                                         min_value=6, max_value=240, value=48, step=6,
                                         help="Estimated construction / implementation period")
        with b4:
            f_location_risk = st.selectbox("Location / Terrain",
                ["Plains (Low Risk)", "Semi-Urban (Medium)", "Hilly / Coastal (High)", "Remote / Border (Very High)"],
                help="Terrain and location risk class")
        st.markdown('</div>', unsafe_allow_html=True)
# #  section 3: cost components
        st.markdown('<div class="feas-card"><h4>🧱 Estimated Cost Component Shares (%)</h4>', unsafe_allow_html=True)
        st.caption("Adjust the expected share of each cost component. Shares should total close to 100%.")
        cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
        with cc1:
            f_mat_pct = st.slider("Materials", 10, 60, 35, help="Civil materials, steel, cement etc.")
        with cc2:
            f_lab_pct = st.slider("Labour", 5, 40, 22, help="Skilled + unskilled workforce")
        with cc3:
            f_eqp_pct = st.slider("Equipment", 3, 25, 12, help="Machinery hire/purchase")
        with cc4:
            f_lnd_pct = st.slider("Land/Site", 0, 30, 10, help="Land acquisition, site clearance")
        with cc5:
            f_trn_pct = st.slider("Transport", 2, 15, 8, help="Material haulage, logistics")
        with cc6:
            f_ovh_pct = st.slider("Overhead/Other", 3, 20, 8, help="Admin, insurance, quality control")
        total_comp_pct = f_mat_pct + f_lab_pct + f_eqp_pct + f_lnd_pct + f_trn_pct + f_ovh_pct
        if total_comp_pct != 95:
            st.info(f"Component shares total {total_comp_pct}%. Remaining {max(0,100-total_comp_pct)}% will be treated as contingency. You can adjust the sliders.")
        st.markdown('</div>', unsafe_allow_html=True)
# # section 4: external risks like climate etc
        st.markdown('<div class="feas-card"><h4>⚠️ External Risk Factors</h4>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            f_climate   = st.slider("Climate / Weather Risk", 0.0, 1.0, 0.30, 0.05,
                                    help="0=None, 1=Extreme (floods, cyclones, etc.)")
        with r2:
            f_supply    = st.slider("Supply Chain Risk", 0.0, 1.0, 0.25, 0.05,
                                    help="0=Smooth supply, 1=Major disruption")
        with r3:
            f_geo       = st.slider("Geopolitical / Import Risk", 0.0, 1.0, 0.20, 0.05,
                                    help="0=None, 1=Severe import/conflict disruption")
        with r4:
            f_local_src = st.slider("Local Sourcing (%)", 10, 100, 40, 5,
                                    help="% of materials sourced locally (higher = lower import risk)")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted_feas = st.form_submit_button("🔍 Run Pre-Project Feasibility Analysis",
                                               use_container_width=True, type="primary")
# # running the feasibility logic
    if submitted_feas:
# #  step indicators
        prog_ph = st.empty()
        prog_ph.info("📊 Step 1: Fetching sector benchmarks from 1,500 historical projects...")
# #  pull sectorlevel stats from real data
        sector_df = raw_data[raw_data['Sector'] == f_sector].copy()
        sector_df['Planned_Duration_Days'] = (pd.to_datetime(sector_df['Planned_Completion_Date']) -
                                               pd.to_datetime(sector_df['Start_Date'])).dt.days
        sector_df['Overrun_Pct'] = ((sector_df['Revised_Cost_Cr'] - sector_df['Original_Approved_Cost_Cr'])
                                    / sector_df['Original_Approved_Cost_Cr'].clip(1) * 100)

        if len(sector_df) >= 5:
            hist_overrun_mean = float(np.clip(sector_df['Overrun_Pct'].mean(), 0, 80))
            hist_overrun_p75  = float(np.clip(sector_df['Overrun_Pct'].quantile(0.75), 0, 100))
            hist_budget_mean  = float(sector_df['Original_Approved_Cost_Cr'].mean())
            hist_dur_mean     = float(sector_df['Planned_Duration_Days'].mean() / 30.4)
            sector_n          = len(sector_df)
        else:
            hist_overrun_mean = 20.0
            hist_overrun_p75  = 35.0
            hist_budget_mean  = f_budget
            hist_dur_mean     = f_duration
            sector_n          = 0

        prog_ph.info("🧠 Step 2: Running AI cost estimation model...")
# #  location / terrain multiplier
        terrain_mult = {"Plains (Low Risk)": 0.97, "Semi-Urban (Medium)": 1.00,
                        "Hilly / Coastal (High)": 1.08, "Remote / Border (Very High)": 1.15}
        loc_factor = terrain_mult.get(f_location_risk, 1.0)
# #  duration pressure factor
# # compare proposed duration vs sector average; shorter  higher cost pressure
        dur_ratio    = f_duration / max(hist_dur_mean, 1)
        dur_factor   = 1.0 + (1.0 - dur_ratio) * 0.05   # negative for relaxed schedule
# #  risk premium factors
        climate_factor = 1.0 + (f_climate - 0.4) * 0.07
        supply_factor  = 1.0 + (f_supply - 0.4) * 0.05
        geo_factor     = 1.0 + (f_geo - 0.4) * 0.04
        import_factor  = 1.0 + (0.5 - f_local_src / 100) * 0.05   # high import dependency = cost pressure
# #  historical overrun uplift
# # use sector mean overrun to adjust optimistic budget estimates
        overrun_factor = 1.0 + ((hist_overrun_mean - 10) / 100) * 0.15  # partial uplift (not full mean)
# #  sector size normalization
# # if proposed budget >> sector mean, slight upward pressure from complexity
        size_ratio  = f_budget / max(hist_budget_mean, 1)
        size_factor = 1.0 + np.clip((size_ratio - 1.0) * 0.03, -0.05, 0.05)
# #  composite estimated cost
        est_cost_raw = f_budget * loc_factor * dur_factor * climate_factor * supply_factor * geo_factor * import_factor
        est_cost     = float(np.clip(est_cost_raw * overrun_factor * size_factor,
                                     f_budget * 0.70, f_budget * 2.20))

        prog_ph.info("🧱 Step 3: Breaking down estimated cost by category...")
# #  cost breakdown
        total_explicit_pct = (f_mat_pct + f_lab_pct + f_eqp_pct + f_lnd_pct + f_trn_pct + f_ovh_pct) / 100
        contingency_base   = max(0, 1.0 - total_explicit_pct)   # leftover from sliders
        risk_cont          = (f_climate * 0.04 + f_supply * 0.03 + f_geo * 0.025) * est_cost
        contingency_total  = contingency_base * est_cost + risk_cont
        contingency_pct    = contingency_total / max(est_cost, 1) * 100

        breakdown = {
            "Materials":          round(est_cost * f_mat_pct / 100, 2),
            "Labour":             round(est_cost * f_lab_pct / 100, 2),
            "Equipment":          round(est_cost * f_eqp_pct / 100, 2),
            "Land / Site Work":   round(est_cost * f_lnd_pct / 100, 2),
            "Transportation":     round(est_cost * f_trn_pct / 100, 2),
            "Overheads / Other": round(est_cost * f_ovh_pct / 100, 2),
            "Contingency":        round(contingency_total, 2),
        }

        prog_ph.info("⚖️ Step 4: Comparing estimated cost against allocated budget...")
# #  feasibility verdict
        gap         = est_cost - f_budget
        gap_pct     = gap / max(f_budget, 1) * 100
        util_pct    = est_cost / max(f_budget, 1) * 100
        overrun_p   = float(np.clip(0.50 + gap_pct / 100 * 0.8, 0.05, 0.97))  # rough probability

        if gap_pct <= 5:
            verdict = "WITHIN BUDGET"
            vclass  = "verdict-within"
            if gap <= 0:
                vdesc = f"Estimated cost is Rs.{abs(gap):,.0f} Cr BELOW the allocated budget. Project appears financially feasible."
            else:
                vdesc = f"Estimated cost is Rs.{abs(gap):,.0f} Cr ({abs(gap_pct):.1f}%) above budget, which is well within acceptable limits. Project appears financially feasible."
        elif gap_pct <= 15:
            verdict = "AT RISK"
            vclass  = "verdict-risk"
            vdesc   = f"Estimated cost is within 15% of budget ({abs(gap_pct):.1f}% over), but risk factors may push it higher. Recommend buffer."
        else:
            verdict = "OVER BUDGET"
            vclass  = "verdict-over"
            vdesc   = f"Estimated cost EXCEEDS budget by Rs.{abs(gap):,.0f} Cr ({abs(gap_pct):.1f}%). Budget revision or scope reduction needed."

        prog_ph.info("📈 Step 5: Identifying cost-driving factors and recommendations...")
# #  costdriving factors (normalized 01 contribution)
        factor_impacts = {
            "Historical Sector Overrun Trend":  min((hist_overrun_mean / 100) * 0.55, 0.30),
            "Terrain / Location Risk":          loc_factor - 1.0,
            "Climate / Weather Risk":           f_climate * 0.07,
            "Supply Chain Pressure":            f_supply  * 0.05,
            "Geopolitical / Import Risk":       f_geo     * 0.04,
            "Import Dependency (Low Local Src)":(1 - f_local_src/100) * 0.03,
            "Schedule Compression":             max(0, 1.0 - dur_ratio) * 0.08,
            "Project Scale Complexity":         max(0, size_factor - 1.0),
        }
        max_impact = max(factor_impacts.values()) if factor_impacts else 1
# #  recommendations
        recs = []
        if gap_pct > 10:
            recs.append(("danger", "🔴 Budget Revision Required",
                         f"The estimated project cost of Rs.{est_cost:,.0f} Cr exceeds the proposed budget of "
                         f"Rs.{f_budget:,.0f} Cr by Rs.{abs(gap):,.0f} Cr ({abs(gap_pct):.1f}%). "
                         "Recommend revising the budget upward or reducing project scope before approval."))
        if hist_overrun_mean > 20:
            recs.append(("", "🟡 Sector Overrun History is High",
                         f"Historical projects in the {f_sector} sector have averaged {hist_overrun_mean:.1f}% cost overrun. "
                         f"75th-percentile overrun is {hist_overrun_p75:.1f}%. Build a corresponding contingency buffer."))
        if f_climate > 0.5:
            recs.append(("", "🟡 Climate Risk Contingency Needed",
                         f"Climate risk index is {f_climate:.2f}. Add 5-8% climate contingency buffer. "
                         "Schedule weather-sensitive activities in dry months. Procure weather insurance."))
        if f_local_src < 40:
            recs.append(("", "🟡 Increase Local Sourcing",
                         f"Local sourcing is only {f_local_src}%. High import dependency increases cost by 3-5% "
                         "due to forex risk and logistics. Target minimum 60% domestic sourcing."))
        if f_location_risk in ["Hilly / Coastal (High)", "Remote / Border (Very High)"]:
            recs.append(("", "🟡 Terrain Premium Applies",
                         f"Project in '{f_location_risk}' terrain carries a {int((loc_factor-1)*100)}% cost premium. "
                         "Ensure detailed geo-technical survey is completed before DPR finalization."))
        if dur_ratio < 0.8:
            recs.append(("", "🟡 Compressed Schedule Risk",
                         f"Planned duration ({f_duration} months) is significantly shorter than sector average ({hist_dur_mean:.0f} months). "
                         "Compressed schedules typically increase costs by 5-8% due to overtime, acceleration costs."))
        if not recs:
            recs.append(("success", "✅ Project Appears Financially Feasible",
                         "No major red flags identified. Ensure a contingency reserve of at least "
                         f"Rs.{contingency_total:,.0f} Cr ({contingency_pct:.1f}%) is maintained."))
# # recommended budget
        rec_budget = est_cost * 1.05   # 5% buffer on top of estimate

        prog_ph.success("✅ Analysis complete!")
        import time; time.sleep(0.6)
        prog_ph.empty()
# # showing the feasibility results
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Feasibility Analysis Results</div>', unsafe_allow_html=True)
# #  verdict card
        v1, v2 = st.columns([1, 2])
        with v1:
            st.markdown(f"""
            <div class="{vclass}">
                <h2>{verdict}</h2>
                <p>{vdesc}</p>
            </div>
            """, unsafe_allow_html=True)
        with v2:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Allocated Budget",    f"Rs. {f_budget:,.0f} Cr")
            k2.metric("AI Estimated Cost",   f"Rs. {est_cost:,.0f} Cr",
                      delta=f"+Rs.{abs(gap):,.0f} Cr" if gap > 0 else f"-Rs.{abs(gap):,.0f} Cr",
                      delta_color="inverse" if gap > 0 else "normal")
            k3.metric("Budget Utilization",  f"{util_pct:.1f}%",
                      delta=f"{util_pct-100:.1f}% vs budget",
                      delta_color="inverse" if util_pct > 100 else "normal")
            k4.metric("Recommended Contingency", f"Rs. {contingency_total:,.0f} Cr",
                      delta=f"{contingency_pct:.1f}% of estimated cost")

        st.markdown("<br/>", unsafe_allow_html=True)
# #  charts row
        ch1, ch2 = st.columns([3, 2])

        with ch1:
            st.markdown('<div class="section-header" style="font-size:15px;">📈 Estimated Cost Breakdown</div>', unsafe_allow_html=True)
            bd_labels = list(breakdown.keys())
            bd_values = list(breakdown.values())
            bd_colors = ['#2563eb','#1e3a5f','#0891b2','#64748b','#7c3aed','#94a3b8','#ef4444']
            fig_bd = go.Figure(go.Waterfall(
                name="Cost", orientation="v",
                measure=["relative"]*len(bd_labels) + ["total"],
                x=bd_labels + ["Total Estimated"],
                y=bd_values + [0],
                text=[f"Rs.{v:,.0f} Cr" for v in bd_values] + [f"Rs.{est_cost:,.0f} Cr"],
                textposition="outside",
                connector={"line":{"color":"#94a3b8"}},
                increasing={"marker":{"color":"#2563eb"}},
                totals={"marker":{"color":"#1e3a5f"}},
            ))
            fig_bd.add_hline(y=f_budget, line_dash="dash", line_color="#c62828",
                             annotation_text=f"Allocated Budget: Rs.{f_budget:,.0f} Cr",
                             annotation_position="top right")
            fig_bd.update_layout(
                template='plotly_white', height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                font=dict(family='Inter', size=12),
                showlegend=False,
                yaxis_title="Cost (Crore Rs.)"
            )
            st.plotly_chart(fig_bd, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-header" style="font-size:15px;">🍧 Cost Component Share</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=bd_labels, values=bd_values,
                hole=0.42,
                marker_colors=bd_colors,
            ))
            fig_pie.update_traces(textinfo='label+percent', textfont_size=11)
            fig_pie.update_layout(
                template='plotly_white', height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(family='Inter', size=12),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
# #  budget vs estimate bar
        st.markdown('<div class="section-header" style="font-size:15px;">⚖️ Budget vs Estimated Cost Comparison</div>', unsafe_allow_html=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Allocated Budget", x=["Budget Comparison"], y=[f_budget],
            marker_color='#2563eb', text=[f"Rs.{f_budget:,.0f} Cr"], textposition='outside'
        ))
        fig_cmp.add_trace(go.Bar(
            name="AI Estimated Cost", x=["Budget Comparison"], y=[est_cost],
            marker_color='#c62828' if gap > 0 else '#059669',
            text=[f"Rs.{est_cost:,.0f} Cr"], textposition='outside'
        ))
        fig_cmp.add_trace(go.Bar(
            name="Recommended Budget (Est+5%)", x=["Budget Comparison"], y=[rec_budget],
            marker_color='#f59e0b', text=[f"Rs.{rec_budget:,.0f} Cr"], textposition='outside'
        ))
        fig_cmp.update_layout(
            template='plotly_white', height=320, barmode='group',
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(family='Inter', size=12),
            legend=dict(orientation='h', y=-0.15),
            yaxis_title="Amount (Crore Rs.)"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
# #  costdriving factors
        st.markdown('<div class="section-header" style="font-size:15px;">📅 Major Cost-Driving Factors</div>', unsafe_allow_html=True)
        fa_df = pd.DataFrame([
            {"Factor": k, "Impact": round(v * 100, 1)}
            for k, v in sorted(factor_impacts.items(), key=lambda x: x[1], reverse=True)
            if v > 0.001
        ])
        fig_fa = go.Figure(go.Bar(
            x=fa_df["Impact"], y=fa_df["Factor"], orientation='h',
            marker_color=['#c62828' if v > 5 else '#e65100' if v > 2 else '#2563eb'
                          for v in fa_df["Impact"]],
            text=[f"+{v:.1f}%" for v in fa_df["Impact"]],
            textposition='outside',
        ))
        fig_fa.update_layout(
            template='plotly_white', height=300,
            margin=dict(l=10, r=60, t=10, b=10),
            font=dict(family='Inter', size=12),
            xaxis_title="Cost Impact (%)",
        )
        st.plotly_chart(fig_fa, use_container_width=True)
# #  sector benchmark panel
        st.markdown('<div class="section-header" style="font-size:15px;">🏷️ Sector Benchmark Intelligence</div>', unsafe_allow_html=True)
        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Sector", f_sector)
        bm2.metric("Historical Projects (DB)", str(sector_n))
        bm3.metric("Avg Historical Overrun",   f"{hist_overrun_mean:.1f}%")
        bm4.metric("75th-Pct Overrun",         f"{hist_overrun_p75:.1f}%")
# # peer budget distribution
        if sector_n >= 5:
            fig_hist = px.histogram(
                sector_df, x='Original_Approved_Cost_Cr', nbins=25,
                color_discrete_sequence=['#2563eb'], template='plotly_white',
                labels={'Original_Approved_Cost_Cr': 'Approved Budget (Cr)'}
            )
            fig_hist.add_vline(x=f_budget, line_dash='dash', line_color='#c62828',
                               annotation_text=f"Your Budget: Rs.{f_budget:,.0f} Cr",
                               annotation_position='top right')
            fig_hist.update_layout(height=260, margin=dict(l=10,r=10,t=20,b=10),
                                   font=dict(family='Inter', size=12),
                                   yaxis_title="No. of Projects")
            st.plotly_chart(fig_hist, use_container_width=True)
# #  recommendations
        st.markdown('<div class="section-header" style="font-size:15px;">💡 Recommended Actions for Decision-Makers</div>', unsafe_allow_html=True)
        for alert_type, title, msg in recs:
            st.markdown(f"""
            <div class="alert-box {alert_type}">
                <strong>{title}</strong><br/>{msg}
            </div>
            """, unsafe_allow_html=True)
# #  recommended budget adjustment table
        st.markdown('<div class="section-header" style="font-size:15px;">📝 Recommended Budget Summary</div>', unsafe_allow_html=True)
        adj_data = [
            {"Item": "Proposed / Allocated Budget",      "Amount (Rs. Cr)": round(f_budget, 0), "Note": "As submitted for approval"},
            {"Item": "AI Estimated Base Cost",            "Amount (Rs. Cr)": round(est_cost, 0),  "Note": "Based on benchmarks + risk factors"},
            {"Item": "Expected Gap (Overrun / Underspend)","Amount (Rs. Cr)": round(gap, 0),      "Note": "Positive = overrun, Negative = saving"},
            {"Item": "Recommended Contingency Reserve",   "Amount (Rs. Cr)": round(contingency_total, 0), "Note": f"{contingency_pct:.1f}% of estimated cost"},
            {"Item": "Recommended Total Sanction",        "Amount (Rs. Cr)": round(rec_budget, 0), "Note": "Estimated cost + 5% execution buffer"},
        ]
        st.dataframe(pd.DataFrame(adj_data), use_container_width=True, hide_index=True)

        st.caption("💡 *All figures are AI-estimated using historical project data and risk models. "
                   "This is a decision-support tool — not a guarantee. "
                   "Detailed DPR (Detailed Project Report) should be prepared before final sanction.*")
# # footer of the app
st.markdown("""
<div class="gov-footer">
    Ministry of Statistics and Programme Implementation (MoSPI), Government of India &nbsp;|&nbsp;
    PARVAAH-X Infrastructure Risk Early Warning System &nbsp;|&nbsp;
    Developed under Smart India Hackathon 2026
</div>
""", unsafe_allow_html=True)
