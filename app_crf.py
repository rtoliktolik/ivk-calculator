import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Статические утверждённые данные для демонстрационного стенда
# ---------------------------------------------------------------------------
IVK_VALUE = 75.99
PREDICTED_CRF = 0.94
DELTA_L = 10.34
DELTA_AB = 75.28

# Точки для интерактивного графика регрессии
XP_POINTS = [12.5, 33.5, 47.0, 58.5, 80.0]
FP_POINTS = [1.19, 1.03, 1.00, 0.975, 0.93]

# Настройка веб-интерфейса
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

# Стилизация шрифтов против троеточий
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; }
    </style>
""", unsafe_allow_html=True)

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=300)
else:
    st.title("FARRATE-X | ANALYTICAL IVK CALCULATOR")

st.markdown("---")

# --- СЕКЦИЯ НАСТРОЕК В БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("⚙️ Database Settings")
db_tolerance = st.sidebar.slider("Cloud tolerance radius (± IVK):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("💰 Insurance Profile")

currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])

base_premium_annual = st.sidebar.number_input(label=f"Base Annual Premium ({currency_symbol}):", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)
base_premium_monthly = base_premium_annual / 12.0

# --- ДИНАМИЧЕСКИЙ ФИНАНСОВЫЙ РАСЧЕТ ---
val_annual = base_premium_annual * PREDICTED_CRF
val_monthly = val_annual / 12.0
d_annual = val_annual - base_premium_annual
d_monthly = val_monthly - base_premium_monthly

# --- ОСНОВНОЙ КОНТЕНТ ---
uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    with col_right_data:
        st.subheader("📊 Express Analysis Results")
        
        col_ivk, col_crf = st.columns(2)
        with col_ivk:
            st.metric("Visual Contrast Index (IVK)", f"{IVK_VALUE:.2f}")
        with col_crf:
            st.metric("Color Risk Factor (CRF)", f"{PREDICTED_CRF:.2f}")
        
        st.write("**Current Visibility Status:** LOW RISK 👍")
        
        # --- БЛОК СТРАХОВОЙ ПРЕМИИ ---
        st.markdown("---")
        st.subheader("➡️ Smart Insurance Premium Adjustment")
        st.write(f"Base profile: **{base_premium_annual:.2f} {currency_symbol}/year** ({base_premium_monthly:.2f} {currency_symbol}/month).")
        
        st.metric(label="Adjusted Annual Premium", value=f"{val_annual:.2f} {currency_symbol}/yr", delta=f"{d_annual:.2f} {currency_symbol}/yr", delta_color="inverse")
        st.metric(label="Adjusted Monthly Premium", value=f"{val_monthly:.2f} {currency_symbol}/mo", delta=f"{d_monthly:.2f} {currency_symbol}/mo", delta_color="inverse")
        
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Light Contrast ΔL", f"{DELTA_L:.2f}")
        m2.metric("Chromatic Contrast Δab", f"{DELTA_AB:.2f}")
        
        st.write("**Detected Car Body Color (RGB):** 225, 85, 36")
        
        # Стабильная отрисовка прямоугольника цвета кузова машины
        pure_color_block = np.zeros((60, 400, 3), dtype=np.uint8)
        pure_color_block[:] = [225, 85, 36]
        st.image(pure_color_block, caption="Isolated Paint Shade")
        
        st.markdown("---")
        st.subheader("🔮 Predictive Evaluation by Databases")
        st.write(f"Found **8,294,260** registered vehicles in the tolerance cloud ({IVK_VALUE:.2f} ± {db_tolerance}).")
        st.caption("Related Statistical Groups: Red, White, Yellow")
        
        st.markdown("---")
        st.write("### Continuous Accident Risk Regression Curve")
        
        # Стабильная отрисовка интерактивного графика Plotly
        ivk_axis = np.linspace(10.0, 90.0, 300)
        crf_axis = np.interp(ivk_axis, XP_POINTS, FP_POINTS)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ivk_axis, y=crf_axis, mode='lines', name='Regression Curve', line=dict(color='royalblue', width=3)))
        fig.add_trace(go.Scatter(
            x=[IVK_VALUE], y=[PREDICTED_CRF], mode='markers+text', name='Current Car',
            text=[f"Your Car ({PREDICTED_CRF:.2f})"], textposition="top center",
            marker=dict(color='crimson', size=14, symbol='diamond', line=dict(color='white', width=2))
        ))
        fig.update_layout(xaxis_title="Visual Contrast Index (IVK)", yaxis_title="Color Risk Factor (CRF)", margin=dict(l=40, r=40, t=20, b=40), hovermode="x unified", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
