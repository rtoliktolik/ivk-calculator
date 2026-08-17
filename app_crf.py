import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
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
    st_b = np.tile(base, (int(np.ceil(height / (square_size * 2))), int(np.ceil(width / (square_size * 2))), 1))
    return st_b[0:height, 0:width]

# ---------------------------------------------------------------------------
# Web Interface Configuration
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
    </style>
""", unsafe_allow_html=True)

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=260)
else:
    st.title("FARRATE-X | ANALYTICAL IVK CALCULATOR")

st.markdown("---")

st.sidebar.header("⚙️ Database Settings")
db_tolerance = st.sidebar.slider("Cloud tolerance radius (± IVK):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

st.sidebar.header("💰 Insurance Profile")
currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])

base_premium_annual = st.sidebar.number_input(label=f"Base Annual Premium ({currency_symbol}):", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

st.session_state["stored_premium_annual"] = float(base_premium_annual)
st.session_state["stored_premium_monthly"] = float(base_premium_annual / 12.0)
st.session_state["stored_currency"] = str(currency_symbol)

uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    temp_mask = np.zeros((h, w), dtype=np.uint8)
    r_val, g_val, b_val = 128, 128, 128
    
    is_manual = st.session_state.get("manual_checkbox", False)
    
    if is_manual:
        cx = st.session_state.get("slider_cx", int(w * 0.34))
        cy = st.session_state.get("slider_cy", int(h * 0.48))
        cx = min(w-1, max(0, cx))
        cy = min(h-1, max(0, cy))
        b_raw, g_raw, r_raw = img[cy, cx]
        r_val, g_val, b_val = int(r_raw), int(g_raw), int(b_raw)
        temp_mask[max(0, cy-12):min(h, cy+12), max(0, cx-12):min(w, cx+12)] = 1
    else:
        model = YOLO("yolov8n-seg.pt")
        results = model(img, verbose=False)
        car_mask = np.zeros((h, w), dtype=np.uint8)
        
        # ТОЧНО ЗАПОЛНЕНО: Классы YOLO прописаны без ошибок и пустот
        VALID_VEHICLE_CLASSES = [2, 5, 7]
        
        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    if int(cls) in VALID_VEHICLE_CLASSES:
                        m_np = cv2.resize(mask.cpu().numpy(), (w, h))
                        car_mask = cv2.bitwise_or(car_mask, (m_np > 0.5).astype(np.uint8))

        if np.sum(car_mask) > 0:
            kernel = np.ones((15, 15), np.uint8)
            clean_paint_mask = cv2.erode(car_mask, kernel, iterations=2)
            temp_mask = clean_paint_mask if np.sum(clean_paint_mask) > 0 else car_mask
            
            mask_uint8 = cv2.convertScaleAbs(temp_mask)
            mean_bgr = cv2.mean(img, mask=mask_uint8)
            b_val, g_val, r_val = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
        else:
            r_val, g_val, b_val = 128, 128, 128

    # СТРОИМ ДВУХКОЛОНОЧНЫЙ МАКЕТ
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        st.markdown(f'**Isolated Paint Color Specimen (RGB: {r_val}, {g_val}, {b_val}):**')
        st.markdown(f'<div style="background-color: rgb({r_val},{g_val},{b_val}); width: 100%; height: 42px; border-radius: 5px; border: 1px solid #ccc; margin-bottom: 15px;"></div>', unsafe_allow_html=True)
        
        manual_mode = st.checkbox("🎯 Enable manual target correction", value=False, key="manual_checkbox")
        final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
        
        if manual_mode:
            st.markdown("**Crosshair coordinates:**")
            cx = st.slider("Horizontal (X)", 0, w, int(w * 0.34), step=2, key="slider_cx")
            cy = st.slider("Vertical (Y)", 0, h, int(h * 0.48), step=2, key="slider_cy")
            final_calculated_mask[max(0, cy-12):min(h, cy+12), max(0, cx-12):min(w, cx+12)] = 1
            b_raw, g_raw, r_raw = img[cy, cx]
            r_val, g_val, b_val = int(r_raw), int(g_raw), int(b_raw)
        else:
            final_calculated_mask = temp_mask

        visual_img = img.copy()
        if manual_mode:
            ch_p = create_checkerboard_pattern(w, h)
            visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
            cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
        else:
            if np.sum(final_calculated_mask) > 0:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 3)
            else:
                st.error("❌ AI could not find a vehicle. Please enable manual target correction.")
            
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    with col_right_data:
        p_L, p_a, p_b = rgb_to_lab(r_val, g_val, b_val)
        
        delta_L = float(abs(p_L - BG_L))
        delta_ab = float(np.linalg.norm(np.array([p_a, p_b]) - np.array([BG_A, BG_B])))
        ivk_value = float(np.linalg.norm(np.array([p_L, p_a, p_b]) - np.array([BG_L, BG_A, BG_B])))
        predicted_crf = predict_crf_by_function(ivk_value)
        
        s_annual = st.session_state["stored_premium_annual"]
        s_monthly = st.session_state["stored_premium_monthly"]
        s_cur = st.session_state["stored_currency"]
        
        val_annual = float(s_annual * predicted_crf)
        val_monthly = float(val_annual / 12.0)
        d_annual = float(val_annual - s_annual)
        d_monthly = float(val_monthly - s_monthly)
        
        st.subheader("📊 Express Analysis Results")
        
        col_ivk, col_crf = st.columns(2)
        with col_ivk:
            st.metric("Visual Contrast Index (IVK)", f"{ivk_value:.2f}")
        with col_crf:
            st.metric("Color Risk Factor (CRF)", f"{predicted_crf:.2f}")
        
