import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go

# Справочный фон дороги — асфальт в пространстве CIELAB
BG_L = 44.40
BG_A = 0.00
BG_B = 0.00

# Точки для интерполяции кривой риска аварийности (CRF)
XP_POINTS = [12.5, 33.5, 47.0, 58.5, 80.0]
FP_POINTS = [1.19, 1.03, 1.00, 0.975, 0.93]

def predict_crf_by_function(target_ivk: float) -> float:
    predicted_crf = float(np.interp(target_ivk, XP_POINTS, FP_POINTS))
    return float(np.round(predicted_crf, 2))

def rgb_to_lab(r, g, b):
    # Математически точный конвертер RGB -> CIELAB для живого анализа цвета
    var_R = (r / 255.0)
    var_G = (g / 255.0)
    var_B = (b / 255.0)

    if var_R > 0.04045: var_R = ((var_R + 0.055) / 1.055) ** 2.4
    else: var_R = var_R / 12.92
    if var_G > 0.04045: var_G = ((var_G + 0.055) / 1.055) ** 2.4
    else: var_G = var_G / 12.92
    if var_B > 0.04045: var_B = ((var_B + 0.055) / 1.055) ** 2.4
    else: var_B = var_B / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    X = X / 95.047
    Y = Y / 100.000
    Z = Z / 108.883

    if X > 0.008856: X = X ** (1/3)
    else: X = (7.787 * X) + (16 / 116)
    if Y > 0.008856: Y = Y ** (1/3)
    else: Y = (7.787 * Y) + (16 / 116)
    if Z > 0.008856: Z = Z ** (1/3)
    else: Z = (7.787 * Z) + (16 / 116)

    L = (116 * Y) - 16
    a = 500 * (X - Y)
    sub_b = 200 * (Y - Z)
    return L, a, sub_b

def simulate_database_lookup(target_ivk: float, tolerance: float) -> dict:
    COLOR_STATS_DATABASE = [
        {"name": "Grey",   "count": 3597270, "ivk_min": 0.0,  "ivk_max": 25.0},
        {"name": "Black",  "count": 2634864, "ivk_min": 25.0, "ivk_max": 42.0},
        {"name": "Blue",   "count": 1382228, "ivk_min": 42.0, "ivk_max": 48.0},
        {"name": "Others", "count": 772997,  "ivk_min": 48.0, "ivk_max": 52.0},
        {"name": "Red",    "count": 654054,  "ivk_min": 52.0, "ivk_max": 57.0},
        {"name": "White",  "count": 1639041, "ivk_min": 57.0, "ivk_max": 65.0},
        {"name": "Yellow", "count": 96277,   "ivk_min": 65.0, "ivk_max": 150.0},
    ]
    ivk_min = max(0.0, target_ivk - tolerance)
    ivk_max = target_ivk + tolerance
    total_cars_in_cloud = 0
    matched_groups = []
    
    for group in COLOR_STATS_DATABASE:
        overlap_min = max(ivk_min, group["ivk_min"])
        overlap_max = min(ivk_max, group["ivk_max"])
        if overlap_min < overlap_max:
            group_span = group["ivk_max"] - group["ivk_min"]
            overlap_span = overlap_max - overlap_min
            ratio = overlap_span / group_span if group_span > 0 else 1.0
            cars_in_sample = int(group["count"] * ratio)
            if cars_in_sample > 0:
                total_cars_in_cloud += cars_in_sample
                matched_groups.append(group['name'])
                
    if total_cars_in_cloud == 0:
        return {"total_cars": 0, "groups": ["Unique Shade"]}
    return {"total_cars": total_cars_in_cloud, "groups": matched_groups}

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

# --- ОСНОВНОЙ КОНТЕНТ ПРИЛОЖЕНИЯ ---
uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        # Ручной высокоточный прицел замера цвета включен по умолчанию
        st.markdown("**🎯 Position Crosshair over target paintwork:**")
        cx = st.slider("Horizontal (X)", 0, w, int(w * 0.34), step=2)
        cy = st.slider("Vertical (Y)", 0, h, int(h * 0.48), step=2)
        
        # Чтение реального живого пикселя из картинки в формате BGR
        b_raw, g_raw, r_raw = img[cy, cx]
        r_val, g_val, b_val = int(r_raw), int(g_raw), int(b_raw)
        
        # Эмуляция маски зоны сканирования для визуала
        final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = max(0, cx - 12), max(0, cy - 12)
        x2, y2 = min(w, cx + 12), min(h, cy + 12)
        final_calculated_mask[y1:y2, x1:x2] = 1
        
        visual_img = img.copy()
        ch_p = create_checkerboard_pattern(w, h)
        visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
        cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
        
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    with col_right_data:
        # НАСТОЯЩИЙ перевод живого замерянного пикселя в LAB пространство
        p_L, p_a, p_b = rgb_to_lab(r_val, g_val, b_val)
        
        # Честные формулы вычисления индексов
        delta_L = float(abs(p_L - BG_L))
        delta_ab = float(np.linalg.norm(np.array([p_a, p_b]) - np.array([BG_A, BG_B])))
        ivk_value = float(np.linalg.norm(np.array([p_L, p_a, p_b]) - np.array([BG_L, BG_A, BG_B])))
        predicted_crf = predict_crf_by_function(ivk_value)
        
        # Живой пересчет финансовой премии под тариф пользователя
        val_annual = base_premium_annual * predicted_crf
        val_monthly = val_annual / 12.0
        d_annual = val_annual - base_premium_annual
        d_monthly = val_monthly - base_premium_monthly
        
        st.subheader("📊 Express Analysis Results")
        
        col_ivk, col_crf = st.columns(2)
        with col_ivk:
            st.metric("Visual Contrast Index (IVK)", f"{ivk_value:.2f}")
        with col_crf:
            st.metric("Color Risk Factor (CRF)", f"{predicted_crf:.2f}")
        
        status_text = "LOW RISK 👍" if predicted_crf < 1.0 else ("HIGH RISK ⚠️" if predicted_crf > 1.0 else "NORMAL")
        st.write(f"**Current Visibility Status:** {status_text}")
        
        # --- БЛОК СТРАХОВОЙ ПРЕМИИ ---
        st.markdown("---")
        st.subheader("➡️ Smart Insurance Premium Adjustment")
        st.write(f"Base profile: **{base_premium_annual:.2f} {currency_symbol}/year** ({base_premium_monthly:.2f} {currency_symbol}/month).")
        
        col_metrics_y, col_metrics_m = st.columns(2)
        with col_metrics_y:
            st.metric(label="Adjusted Annual Premium", value=f"{val_annual:.2f} {currency_symbol}/yr", delta=f"{d_annual:.2f} {currency_symbol}/yr", delta_color="inverse")
        with col_metrics_m:
            st.metric(label="Adjusted Monthly Premium", value=f"{val_monthly:.2f} {currency_symbol}/mo", delta=f"{d_monthly:.2f} {currency_symbol}/mo", delta_color="inverse")
        
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Light Contrast ΔL", f"{delta_L:.2f}")
        m2.metric("Chromatic Contrast Δab", f"{delta_ab:.2f}")

    # --- СЕКЦИЯ ПОДВАЛА НА ВСЮ ШИРИНУ ---
    st.markdown("---")
    st.write("### 🎨 Isolated Paintwork Color Block")
    st.write(f"**Detected Car Body Color (RGB):** {r_val}, {g_val}, {b_val}")
    
    # ТЕПЕРЬ ПРЯМОУГОЛЬНИК СТАНЕТ РЕАЛЬНО СИНИМ (ИЛИ ЛЮБЫМ ДРУГИМ ЖИВЫМ ЦВЕТОМ)
    st.markdown(f'<div style="background-color: rgb({r_val},{g_val},{b_val}); width: 100%; height: 75px; border-radius: 6px; border: 1px solid #bbb; margin-bottom: 12px;"></div>', unsafe_allow_html=True)
    st.caption("Isolated Paint Shade Workspace")
    
    st.markdown("---")
    st.write("### 🔮 Predictive Evaluation by Databases")
    db_res = simulate_database_lookup(ivk_value, db_tolerance)
    st.write(f"Found **{db_res['total_cars']:,}** registered vehicles in the tolerance cloud ({ivk_value:.2f} ± {db_tolerance}).")
    st.caption(f"Related Statistical Groups: {', '.join(db_res['groups'])}")
    
    st.markdown("---")
    st.write("### 📈 Continuous Accident Risk Regression Curve")
    
    # Живой график регрессии Plotly
    ivk_axis = np.linspace(10.0, 90.0, 300)
    crf_axis = np.interp(ivk_axis, XP_POINTS, FP_POINTS)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ivk_axis, y=crf_axis, mode='lines', name='Regression Curve', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(
