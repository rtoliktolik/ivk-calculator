import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import plotly.graph_objects as go

# Fixed reference road background constant (Asphalt)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

# Точки для интерполяции (наша кривая риска)
XP_POINTS = [12.5, 33.5, 47.0, 58.5, 80.0]
FP_POINTS = [1.19, 1.03, 1.00, 0.975, 0.93]

# ---------------------------------------------------------------------------
# Mathematical function for predicting risk (CRF) by IVK values
# ---------------------------------------------------------------------------
def predict_crf_by_function(target_ivk: float) -> float:
    predicted_crf = float(np.interp(target_ivk, XP_POINTS, FP_POINTS))
    return float(np.round(predicted_crf, 2))

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
# Web Interface
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

# Стилизация шрифтов, чтобы показатели не уходили в троеточия
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

# --- ОСНОВНОЙ КОНТЕНТ ---
uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        manual_mode = st.checkbox("🎯 Enable manual target correction", value=True)
        final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
        dominant_bgr = None
        
        if manual_mode:
            st.markdown("**Crosshair coordinates:**")
            cx = st.slider("Horizontal (X)", 0, w, int(w * 0.34), step=2)
            cy = st.slider("Vertical (Y)", 0, h, int(h * 0.48), step=2)
            x1, y1 = max(0, cx - 10), max(0, cy - 10)
            x2, y2 = min(w, cx + 10), min(h, cy + 10)
            final_calculated_mask[y1:y2, x1:x2] = 1
            dominant_bgr = img[cy, cx]
        else:
            with st.spinner("AI is isolating clean paintwork..."):
                model = YOLO("yolov8n-seg.pt")
                results = model(img, verbose=False)
                
                car_mask = np.zeros((h, w), dtype=np.uint8)
                VALID_VEHICLE_CLASSES = [2, 5, 7] # 2: легковая машина, 5: автобус, 7: грузовик
                
                for result in results:
                    if result.masks is not None:
                        for mask, cls in zip(result.masks.data, result.boxes.cls):
                            class_idx = int(cls)
                            if class_idx in VALID_VEHICLE_CLASSES:
                                m_np = cv2.resize(mask.cpu().numpy(), (w, h))
                                m_bin = (m_np > 0.5).astype(np.uint8)
                                car_mask = cv2.bitwise_or(car_mask, m_bin)

                if np.sum(car_mask) > 0:
                    kernel = np.ones((21, 21), np.uint8)
                    clean_paint_mask = cv2.erode(car_mask, kernel, iterations=3)
                    
                    car_pixels_bgr = img[clean_paint_mask == 1]
                    if len(car_pixels_bgr) > 0:
                        final_calculated_mask[clean_paint_mask == 1] = 1
                        dominant_bgr = np.median(car_pixels_bgr, axis=0)
                    else:
                        final_calculated_mask[car_mask == 1] = 1
                        dominant_bgr = np.median(img[car_mask == 1], axis=0)
                else:
                    st.error("❌ AI could not find a vehicle. Please enable manual target correction.")

        if dominant_bgr is not None:
            visual_img = img.copy()
            ch_p = create_checkerboard_pattern(w, h)
            visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
            
            if manual_mode:
                cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)
            else:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
                
            st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    with col_right_data:
        if dominant_bgr is not None:
            # Преобразование замерянного пикселя в Lab
            pixel_bgr = np.uint8([[list(dominant_bgr)]])
            pixel_rgb = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2RGB)
            pixel_rgb_f32 = pixel_rgb.astype(np.float32) / 255.0
            pixel_lab = cv2.cvtColor(pixel_rgb_f32, cv2.COLOR_RGB2Lab).flatten()
            
            bg_bgr = np.uint8([[list(CONSTANT_ROAD_BACKGROUND_RGB[::-1])]])
            bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            bg_rgb_f32 = bg_rgb.astype(np.float32) / 255.0
            bg_lab = cv2.cvtColor(bg_rgb_f32, cv2.COLOR_RGB2Lab).flatten()
            
            # Живые расчеты из точки прицела ползунков
            delta_L = float(abs(float(pixel_lab[0]) - float(bg_lab[0])))
            delta_ab = float(np.linalg.norm(pixel_lab[1:] - bg_lab[1:]))
            ivk_value = float(np.linalg.norm(pixel_lab - bg_lab))
            predicted_crf = predict_crf_by_function(ivk_value)
            
            # Чтение каналов цвета RGB
            rgb_flat = pixel_rgb.flatten()
            r_val = int(rgb_flat[0])
            g_val = int(rgb_flat[1])
            b_val = int(rgb_flat[2])
            
            # Финансовая математика
            adjusted_premium_annual = base_premium_annual * predicted_crf
            adjusted_premium_monthly = adjusted_premium_annual / 12.0
            delta_annual = adjusted_premium_annual - base_premium_annual
            delta_monthly = adjusted_premium_monthly - base_premium_monthly
            
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
            
            col_fin_y, col_fin_m = st.columns(2)
            with col_fin_y:
                st.metric(
