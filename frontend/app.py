import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Application configuration
st.set_page_config(
    page_title="Epidemiological Modeling Dashboard",
    page_icon="[O]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    h1, h2, h3 {
        color: #f3f4f6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## Model Configuration")

st.sidebar.markdown("Hardware Status: **CUDA ACTIVE**")

iterations = st.sidebar.slider("Training Cycles", 1000, 20000, 10000)
disease_select = st.sidebar.selectbox("Target Pathogen", ["COVID-19", "Influenza", "Ebola", "Zika"])

if st.sidebar.button("Run Mechanistic Inference"):
    with st.spinner("Processing trajectory via CUDA..."):
        try:
            params = {"disease": disease_select, "epochs": iterations}
            response = requests.get("http://localhost:8000/api/surveillance", params=params, timeout=300)
            res_data = response.json()
            st.session_state['data'] = res_data
            st.session_state['last_run'] = datetime.now()
            st.success("Calibration complete.")
        except Exception as e:
            st.error(f"Execution error: {str(e)}")

# Header
st.title("Epidemiological Modeling and Forecasting")
st.markdown("Automated parameter identification using Physics-Informed Neural Networks")

if 'data' in st.session_state:
    data = st.session_state['data']
    modeling = data.get('modeling', {})
    kinetics = modeling.get('metadata', {}).get('inferred_kinetics', {})

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("R0 (Reproduction Number)", f"{kinetics.get('r_naught', 0):.2f}")
    with m2:
        st.metric("Beta (Transmission)", f"{kinetics.get('beta', 0):.4f}")
    with m3:
        st.metric("Gamma (Recovery)", f"{kinetics.get('gamma', 0):.4f}")
    with m4:
        st.metric("Cycles", f"{iterations}")

    # Visualization
    st.markdown("## Trajectory Analysis")
    
    historical = modeling.get('historical', [])
    full_traj = modeling.get('full_trajectory', [])
    
    # Date alignment
    total_days = len(full_traj)
    start_date = datetime.now() - timedelta(days=len(historical))
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    
    fig = go.Figure()

    # Empirical data
    fig.add_trace(go.Scatter(
        x=dates[:len(historical)],
        y=historical,
        mode='markers',
        name='Observed Infections',
        marker=dict(color='#ef4444', size=7),
        opacity=0.5
    ))

    # Model fit
    fig.add_trace(go.Scatter(
        x=dates,
        y=full_traj,
        mode='lines',
        name='Mechanistic SIR Fit',
        line=dict(color='#3b82f6', width=2),
    ))

    # Forecast area
    fig.add_vrect(
        x0=dates[len(historical)-1], 
        x1=dates[-1],
        fillcolor="rgba(59, 130, 246, 0.1)", 
        annotation_text="Forecast Horizon",
        annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Telemetry and Statistics
    t1, t2 = st.columns([2, 1])
    with t1:
        st.markdown("### Clinical Alerts (ProMED)")
        st.dataframe(pd.DataFrame(data.get('alerts', [])), height=250)
    
    with t2:
        st.markdown("### Global Metrics")
        stats_raw = data.get('stats', {})
        st.json({
            "Cases": stats_raw.get('cases'),
            "Recovered": stats_raw.get('recovered'),
            "Deaths": stats_raw.get('deaths'),
            "Active": stats_raw.get('active')
        })

else:
    st.info("System ready. Configure parameters in the sidebar to begin mechanistic calibration.")

# Footer
st.markdown("---")
st.markdown("Project Repository: [GitHub Link](https://github.com/SourishSenapati/Omics-Optimizers)")
