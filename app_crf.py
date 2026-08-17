import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go

# Fixed reference road background constant (Asphalt)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

# Точки для интерактивного графика регрессии рисков
XP_POINTS = [12.5, 33.5, 47.0, 58.5, 80.0]
FP_POINTS = [1.19, 1.03, 1.00, 0.975, 0.93]

# Статические утверждённые демонстрационные данные
IVK_VALUE = 75.99
PREDICTED_CRF = 0.94
DELTA_L = 10.34
DELTA_AB = 75.28

# Изолированный оранжево-рыжий цвет кузова автомобиля со скриншота
R_VAL, G_VAL, B_VAL = 225, 85, 36

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def create_checkerboard_pattern(width, height, square_size=15):
    base = np.zeros((square_size * 2, square_size * 2, 3), dtype=np.uint8)
    base[0:square_size, 0:square_size] = (240, 240, 240)
    base[square_size:, square_size:] = (240, 240, 240)
    base[0:square_size, square_size:] = (200, 200, 200)
    base[square_size:, 0:square_size] = (200, 200, 200)
    return np.tile(base, (int(np.ceil(height / (square_size * 2))), int(np.ceil(width / (square_size * 2))), 1))[0:height, 0:width]

# ---------------------------------------------------------------------------
# Web Interface Configuration
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

# Оптимизировали размер шрифта метрик, чтобы текст валюты больше НЕ обрезался точками!
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

# Динамический выбор символа валюты
currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])

# Интерактивное поле ввода базовой годовой премии
base_premium_annual = st.sidebar.number_input(
    label=f"Base Annual Premium ({currency_symbol}):", 
    min_value=1.0, 
    max_value=1000000.0, 
    value=850.0, 
    step=10.0
)
base_premium_monthly = base_premium_annual / 12.0

# --- МАТЕМАТИЧЕСКИЙ ДИНАМИЧЕСКИЙ РАСЧЕТ ПРЕМИИ ---
val_annual = base_premium_annual * PREDICTED_CRF
val_monthly = val_annual / 12.0
d_annual = val_annual - base_premium_annual
d_monthly = val_monthly - base_premium_monthly

# --- ОСНОВНОЙ КОНТЕНТ ПРИЛОЖЕНИЯ ---
uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        st.markdown("**Crosshair coordinates:**")
        cx = st.slider("Horizontal (X)", 0, w, int(w * 0.34), step=2)
        cy = st.slider("Vertical (Y)", 0, h, int(h * 0.48), step=2)
        
        # Эмуляция выделения точки прицела
        final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = max(0, cx - 10), max(0, cy - 10)
        x2, y2 = min(w, cx + 10), min(h, cy + 10)
        final_calculated_mask[y1:y2, x1:x2] = 1
        
        visual_img = img.copy()
        ch_p = create_checkerboard_pattern(w, h)
        visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
        cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
        
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

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
        
        col_fin_y, col_fin_m = st.columns(2)
        with col_fin_y:
            st.metric(label="Adjusted Annual Premium", value=f"{val_annual:.2f} {currency_symbol}/yr", delta=f"{d_annual:.2f} {currency_symbol}/yr", delta_color="inverse")
        with col_fin_m:
            st.metric(label="Adjusted Monthly Premium", value=f"{val_monthly:.2f} {currency_symbol}/mo", delta=f"{d_monthly:.2f} {currency_symbol}/mo", delta_color="inverse")
        
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Light Contrast ΔL", f"{DELTA_L:.2f}")
        m2.metric("Chromatic Contrast Δab", f"{DELTA_AB:.2f}")
        
        # Блок отображения цвета кузова
        st.write(f"**Detected Car Body Color (RGB):** {R_VAL}, {G_VAL}, {B_VAL}")
        
        # ДОБАВЛЕНО: Стабильная и красивая отрисовка прямоугольника изолированного цвета краски
        pure_color_block = np.zeros((60, 400, 3), dtype=np.uint8)
        pure_color_block[:] = [R_VAL, G_VAL, B_VAL]
        st.image(pure_color_block, caption="Isolated Paint Shade")
        
        st.markdown("---")
        st.subheader("🔮 Predictive Evaluation by Databases")
        st.write(f"Found **8,294,260** registered vehicles in the tolerance cloud ({IVK_VALUE:.2f} ± {db_tolerance}).")
        st.caption("Related Statistical Groups: Red, White, Yellow")
        
        st.markdown("---")
        st.write("### Continuous Accident Risk Regression Curve")
        
        # ДОБАВЛЕНО: Стабильное интерактивное построение графика регрессии Plotly
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
