import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import streamlit.components.v1 as components

# Fixed reference road background constant (Asphalt)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

def predict_crf_by_function(target_ivk: float) -> float:
    xp = [12.5, 33.5, 47.0, 58.5, 80.0]
    fp = [1.19, 1.03, 1.00, 0.975, 0.93]
    # Гарантируем возврат чистого нативного float без типов NumPy
    res = np.interp(float(target_ivk), xp, fp)
    return float(np.round(float(res), 2))

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
                matched_groups.append(group["name"])
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

# --- СЕКЦИЯ НАСТРОЕК В БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("⚙️ Database Settings")
db_tolerance = st.sidebar.slider("Cloud tolerance radius (± IVK):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("💰 Insurance Profile")
currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])
base_premium_annual = st.sidebar.number_input(label="Base Annual Premium:", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

sidebar_calc_space = st.sidebar.empty()

# --- ОСНОВНОЙ КОНТЕНТ ПРИЛОЖЕНИЯ ---
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
            # Инициализация переменных координат в сессии
            if "target_cx" not in st.session_state:
                st.session_state.target_cx = int(w / 2)
            if "target_cy" not in st.session_state:
                st.session_state.target_cy = int(h / 2)
                
            # Чтение параметров из адресной строки для связи с HTML-слайдером Y
            query_params = st.query_params
            if "html_cy" in query_params:
                st.session_state.target_cy = int(query_params["html_cy"])
            
            # Слайдер Х строго горизонтально над изображением
            cx = st.slider("Horizontal Position (X Target)", 0, w, st.session_state.target_cx, step=2)
            st.session_state.target_cx = cx
            
            # Сетка для честного вертикального слайдера слева и картинки справа
            slider_layout_col1, slider_layout_col2 = st.columns([1, 12])
            
            with slider_layout_col1:
                st.write("<div style='text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px;'>Y</div>", unsafe_allow_html=True)
                # HTML5 Вертикальный ползунок, занимающий всю высоту рамки изображения
                html_slider = f"""
                <body style='margin:0; padding:0; background:transparent;'>
                <input type='range' min='0' max='{h}' value='{st.session_state.target_cy}' step='2' 
                style='transform: rotate(180deg); writing-mode: bt-lr; appearance: slider-vertical; width:100%; height:320px; cursor:pointer;'
                onchange='parent.window.location.search = "?html_cy=" + this.value;'>
                </body>
                """
                components.html(html_slider, height=340)
            
            cy = st.session_state.target_cy
            # Инвертируем Y для соответствия направления системы координат OpenCV (ноль сверху)
            cy_corrected = h - cy
            cy_corrected = max(0, min(h - 1, cy_corrected))
            
            x1, y1 = max(0, cx - 10), max(0, cy_corrected - 10)
            x2, y2 = min(w, cx + 10), min(h, cy_corrected + 10)
            final_calculated_mask[y1:y2, x1:x2] = 1
            raw_dominant_color = img[cy_corrected, cx]
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
                # Рисуем полицветный прицел по скорректированной координате Y
                cv2.drawMarker(visual_img, (cx, cy_corrected), (255, 255, 255), cv2.MARKER_CROSS, 45, 5) 
                cv2.drawMarker(visual_img, (cx, cy_corrected), (255, 0, 0), cv2.MARKER_CROSS, 35, 3)     
                cv2.drawMarker(visual_img, (cx, cy_corrected), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 3) 
                with slider_layout_col2:
                    st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)
            else:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    # --- СТАБИЛЬНЫЙ РАСЧЕТ И ОТРИСОВКА ПРАВОЙ КОЛОНКИ ---
    if raw_dominant_color is not None:
        dominant_bgr = np.round(raw_dominant_color).astype(np.uint8)
        
        pixel_bgr = np.uint8([[list(dominant_bgr)]])
        pixel_rgb = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2RGB)
        pixel_rgb_f32 = pixel_rgb.astype(np.float32) / 255.0
        
        lab_matrix = cv2.cvtColor(pixel_rgb_f32, cv2.COLOR_RGB2Lab)
        val_L = float(lab_matrix.item(0, 0, 0))
        val_a = float(lab_matrix.item(0, 0, 1))
        val_b = float(lab_matrix.item(0, 0, 2))
        
        bg_bgr = np.uint8([[list(CONSTANT_ROAD_BACKGROUND_RGB[::-1])]])
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
