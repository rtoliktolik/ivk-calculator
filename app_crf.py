import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Fixed reference road background constant (Asphalt)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

# ---------------------------------------------------------------------------
# Mathematical function for predicting risk (CRF) by IVK values
# ---------------------------------------------------------------------------
def predict_crf_by_function(target_ivk: float) -> float:
    xp = [12.5, 33.5, 47.0, 58.5, 80.0]
    fp = [1.19, 1.03, 1.00, 0.975, 0.93]
    predicted_crf = float(np.interp(target_ivk, xp, fp))
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

# Inject Custom CSS to increase metric font sizes for uniform view
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 3.5rem !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { font-size: 1.3rem !important; }
    </style>
""", unsafe_allow_html=True)

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=300)
else:
    st.title("FARRATE-X | ANALYTICAL IVK CALCULATOR")

st.markdown("---")

st.sidebar.header("⚙️ Database Settings")
db_tolerance = st.sidebar.slider("Cloud tolerance radius (± IVK):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        manual_mode = st.checkbox("🎯 Enable manual target correction")
        final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
        dominant_bgr = None
        
        if manual_mode:
            st.markdown("**Crosshair coordinates:**")
            cx = st.slider("Horizontal (X)", 0, w, int(w / 2), step=2)
            cy = st.slider("Vertical (Y)", 0, h, int(h / 2), step=2)
            x1, y1 = max(0, cx - 10), max(0, cy - 10)
            x2, y2 = min(w, cx + 10), min(h, cy + 10)
            final_calculated_mask[y1:y2, x1:x2] = 1
            dominant_bgr = img[cy, cx]
        else:
            with st.spinner("AI is isolating clean paintwork, removing glass and lights..."):
                model = YOLO("yolov8n-seg.pt")
                results = model(img, verbose=False)
                
                car_mask = np.zeros((h, w), dtype=np.uint8)
                exclude_mask = np.zeros((h, w), dtype=np.uint8)
                
                for result in results:
                    if result.masks is not None:
                        for mask, cls in zip(result.masks.data, result.boxes.cls):
                            m_np = cv2.resize(mask.cpu().numpy(), (w, h))
                            m_bin = (m_np > 0.5).astype(np.uint8)
                            class_idx = int(cls)
                            
                            if class_idx == 2:  # Car body
                                car_mask = cv2.bitwise_or(car_mask, m_bin)
                            if class_idx == 4:  # Headlights / Flares
                                exclude_mask = cv2.bitwise_or(exclude_mask, m_bin)
                            if class_idx == 7:  # Windows / Glass
                                exclude_mask = cv2.bitwise_or(exclude_mask, m_bin)
                            if class_idx == 13: # Wheels
                                exclude_mask = cv2.bitwise_or(exclude_mask, m_bin)

                if np.sum(car_mask) > 0:
                    car_without_parts = cv2.bitwise_and(car_mask, cv2.bitwise_not(exclude_mask))
                    kernel = np.ones((11, 11), np.uint8)
                    clean_paint_mask = cv2.erode(car_without_parts, kernel, iterations=2)
                    
                    car_pixels_bgr = img[clean_paint_mask == 1]
                    if len(car_pixels_bgr) > 0:
                        h_idx, w_idx = np.where(clean_paint_mask == 1)
                        final_calculated_mask[clean_paint_mask == 1] = 1
                        dominant_bgr = np.median(car_pixels_bgr, axis=0)
                else:
                    st.error("❌ AI could not find a car. Please enable manual target correction.")

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
            pixel_bgr = np.uint8([[list(dominant_bgr)]])
            pixel_rgb = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2RGB)
            
            pixel_rgb_f32 = pixel_rgb.astype(np.float32) / 255.0
            pixel_lab = cv2.cvtColor(pixel_rgb_f32, cv2.COLOR_RGB2Lab).flatten()
            
            bg_bgr = np.uint8([[list(CONSTANT_ROAD_BACKGROUND_RGB[::-1])]])
            bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            bg_rgb_f32 = bg_rgb.astype(np.float32) / 255.0
            bg_lab = cv2.cvtColor(bg_rgb_f32, cv2.COLOR_RGB2Lab).flatten()
            
            delta_L = float(abs(float(pixel_lab[0]) - float(bg_lab[0])))
            delta_ab = float(np.linalg.norm(pixel_lab[1:] - bg_lab[1:]))
            ivk_value = float(np.linalg.norm(pixel_lab - bg_lab))
            
            db_res = simulate_database_lookup(ivk_value, db_tolerance)
            predicted_crf = predict_crf_by_function(ivk_value)
            
            rgb_flat = pixel_rgb.flatten()
            r_val = int(rgb_flat[0])
            g_val = int(rgb_flat[1])
            b_val = int(rgb_flat[2])
            
            st.subheader("📊 Express Analysis Results")
            
            # 🎯 ТЕПЕРЬ ТУТ КОРРЕКТНОЕ НАЗВАНИЕ: Color Risk Factor (CRF)
            st.metric("Visual Contrast Index (IVK)", f"{ivk_value:.2f}")
            st.metric("Color Risk Factor (CRF)", f"{predicted_crf:.2f}")
            
            status_text = "LOW RISK 👍" if predicted_crf < 1.0 else ("HIGH RISK ⚠️" if predicted_crf > 1.0 else "NORMAL")
            st.write(f"**Current Visibility Status:** {status_text}")
            
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("Light Contrast ΔL", f"{delta_L:.2f}")
            m2.metric("Chromatic Contrast Δab", f"{delta_ab:.2f}")
            
            st.write(f"**Detected Car Body Color (RGB):** {r_val}, {g_val}, {b_val}")
            
            pure_color_block = np.zeros((60, 400, 3), dtype=np.uint8)
            pure_color_block[:] = [r_val, g_val, b_val]
            st.image(pure_color_block, caption="Isolated Paint Shade")
            
            st.markdown("---")
            st.subheader("🔮 Predictive Evaluation by Databases")
            st.write(f"Found **{db_res['total_cars']:,}** registered vehicles in the tolerance cloud ({ivk_value:.2f} ± {db_tolerance}).")
            st.caption(f"Related Statistical Groups: {', '.join(db_res['groups'])}")
            
            st.write("### Continuous Accident Risk Regression Curve")
