import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Fixed reference road background constant (Asphalt)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

def predict_crf_by_function(target_ivk: float) -> float:
    xp = [12.5, 33.5, 47.0, 58.5, 80.0]
    fp = [1.19, 1.03, 1.00, 0.975, 0.93]
    res = np.interp(float(target_ivk), xp, fp)
    return float(np.round(float(res), 2))

def simulate_database_lookup(target_ivk: float, tolerance: float) -> dict:
    # Безопасный текстовый формат хранения для 100% защиты от склеивания строк
    n_str = "Grey,Black,Blue,Others,Red,White,Yellow"
    c_str = "3597270,2634864,1382228,772997,654054,1639041,96277"
    min_str = "0.0,25.0,42.0,48.0,52.0,57.0,65.0"
    max_str = "25.0,42.0,48.0,52.0,57.0,65.0,150.0"
    
    db_names = n_str.split(",")
    db_counts = [int(x) for x in c_str.split(",")]
    db_mins = [float(x) for x in min_str.split(",")]
    db_maxs = [float(x) for x in max_str.split(",")]
    
    ivk_min = max(0.0, target_ivk - tolerance)
    ivk_max = target_ivk + tolerance
    total_cars_in_cloud = 0
    matched_groups = []
    
    for i in range(len(db_names)):
        overlap_min = max(ivk_min, db_mins[i])
        overlap_max = min(ivk_max, db_maxs[i])
        if overlap_min < overlap_max:
            group_span = db_maxs[i] - db_mins[i]
            overlap_span = overlap_max - overlap_min
            ratio = overlap_span / group_span if group_span > 0 else 1.0
            cars_in_sample = int(db_counts[i] * ratio)
            if cars_in_sample > 0:
                total_cars_in_cloud += cars_in_sample
                matched_groups.append(db_names[i])
                
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

# --- ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

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

st.sidebar.markdown("---")
st.sidebar.header("💰 Insurance Profile")
currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])
base_premium_annual = st.sidebar.number_input(label="Base Annual Premium:", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

sidebar_calc_space = st.sidebar.empty()

uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    col_left_img, col_right_data = st.columns(2)
    raw_dominant_color = None
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    
    with col_left_img:
        manual_mode = st.checkbox("🎯 Enable manual target correction")
        
        if manual_mode:
            st.markdown("**🎯 Координатная панель прицеливания:**")
            cx = st.slider("Сдвиг прицела по ГОРИЗОНТАЛИ (X)", 0, w - 1, int(w * 0.5), step=1)
            cy = st.slider("Сдвиг прицела по ВЕРТИКАЛИ (Y)", 0, h - 1, int(h * 0.65), step=1)
            
            cx = max(0, min(w - 1, int(cx)))
            cy = max(0, min(h - 1, int(cy)))
            
            x1, y1 = max(0, cx - 10), max(0, cy - 10)
            x2, y2 = min(w, cx + 10), min(h, cy + 10)
            final_calculated_mask[y1:y2, x1:x2] = 1
            raw_dominant_color = img[cy, cx]
        else:
            with st.spinner("AI is isolating clean paintwork..."):
                model = YOLO("yolov8n-seg.pt")
                results = model(img, verbose=False)
                car_mask = np.zeros((h, w), dtype=np.uint8)
                exclude_mask = np.zeros((h, w), dtype=np.uint8)
                for result in results:
                    if result.masks is not None:
                        for mask, cls in zip(result.masks.data, result.boxes.cls):
                            m_np = cv2.resize(mask.cpu().numpy(), (w, h))
                            m_bin = (m_np > 0.5).astype(np.uint8)
                            c_idx = int(cls)
                            if c_idx == 2:
                                car_mask = cv2.bitwise_or(car_mask, m_bin)
                            if c_idx == 4 or c_idx == 7 or c_idx == 13:
                                exclude_mask = cv2.bitwise_or(exclude_mask, m_bin)
                if np.sum(car_mask) > 0:
                    car_without_parts = cv2.bitwise_and(car_mask, cv2.bitwise_not(exclude_mask))
                    kernel = np.ones((11, 11), np.uint8)
                    clean_paint_mask = cv2.erode(car_without_parts, kernel, iterations=2)
                    car_pixels_bgr = img[clean_paint_mask == 1]
                    if len(car_pixels_bgr) > 0:
                        final_calculated_mask[clean_paint_mask == 1] = 1
                        raw_dominant_color = np.median(car_pixels_bgr, axis=0)
                else:
                    st.error("❌ AI could not find a car. Please enable manual target correction.")

        if raw_dominant_color is not None:
            visual_img = img.copy()
            ch_p = create_checkerboard_pattern(w, h)
            visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
            
            if manual_mode:
                cv2.drawMarker(visual_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 90, 10) 
                cv2.drawMarker(visual_img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 70, 6)     
                cv2.drawMarker(visual_img, (cx, cy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 6) 
            else:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
                
            st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    # --- НАДЕЖНЫЙ СТАБИЛЬНЫЙ МНОГОПОТОЧНЫЙ РАСЧЕТ ---
    if raw_dominant_color is not None:
        # ФИКС ТИПОВ ДАННЫХ: Принудительное округление медианы YOLO до uint8 перед получением каналов
        dominant_bgr = np.round(raw_dominant_color).astype(np.uint8)
        
        b_channel = int(dominant_bgr.item(0))
        g_channel = int(dominant_bgr.item(1))
        r_channel = int(dominant_bgr.item(2))
        
        pixel_bgr_mat = np.uint8([[[b_channel, g_channel, r_channel]]])
        pixel_rgb_mat = cv2.cvtColor(pixel_bgr_mat, cv2.COLOR_BGR2RGB)
        pixel_rgb_f32 = pixel_rgb_mat.astype(np.float32) / 255.0
        
        lab_matrix = cv2.cvtColor(pixel_rgb_f32, cv2.COLOR_RGB2Lab)
        val_L = float(lab_matrix.item(0))
        val_a = float(lab_matrix.item(1))
        val_b = float(lab_matrix.item(2))
        
        bg_bgr_mat = np.uint8([[list(CONSTANT_ROAD_BACKGROUND_RGB[::-1])]])
        bg_rgb_mat = cv2.cvtColor(bg_bgr_mat, cv2.COLOR_BGR2RGB)
        bg_rgb_f32 = bg_rgb_mat.astype(np.float32) / 255.0
        
        bg_lab_matrix = cv2.cvtColor(bg_rgb_f32, cv2.COLOR_RGB2Lab)
        bg_L = float(bg_lab_matrix.item(0, 0, 0))
        bg_a = float(bg_lab_matrix.item(0, 0, 1))
        bg_b = float(bg_lab_matrix.item(0, 0, 2))
        
        delta_L = float(abs(val_L - bg_L))
        delta_ab = float(np.sqrt(max(0.0, (val_a - bg_a)**2 + (val_b - bg_b)**2)) + 1e-5)
        ivk_value = float(np.sqrt(max(0.0, (val_L - bg_L)**2 + (val_a - bg_a)**2 + (val_b - bg_b)**2)) + 1e-5)
        
        db_res = simulate_database_lookup(ivk_value, db_tolerance)
        predicted_crf = float(predict_crf_by_function(ivk_value))
        
        bm = float(base_premium_annual / 12.0)
        va = float(base_premium_annual * predicted_crf)
        vm = float(va / 12.0)
        da = float(va - base_premium_annual)
        dm = float(vm - bm)
        
        txt_annual = f"{va:.2f} {currency_symbol}/yr"
        txt_delta_a = f"{da:.2f} {currency_symbol}/yr"
        txt_monthly = f"{vm:.2f} {currency_symbol}/mo"
        txt_delta_m = f"{dm:.2f} {currency_symbol}/mo"

        with sidebar_calc_space.container():
            st.write("**🧮 Live Premium Calculation**")
            st.write(f"Base: {base_premium_annual:.2f} {currency_symbol}/yr")
            st.metric(label="Adjusted Annual Premium", value=txt_annual, delta=txt_delta_a, delta_color="inverse")
            st.metric(label="Adjusted Monthly Premium", value=txt_monthly, delta=txt_delta_m, delta_color="inverse")
        
        with col_right_data:
            st.subheader("📊 Express Analysis Results")
            st.metric("Visual Contrast Index (IVK)", f"{ivk_value:.2f}")
            st.metric("Color Risk Factor (CRF)", f"{predicted_crf:.2f}")
            
            status_text = "LOW RISK 👍" if predicted_crf < 1.0 else ("HIGH RISK ⚠️" if predicted_crf > 1.0 else "NORMAL")
