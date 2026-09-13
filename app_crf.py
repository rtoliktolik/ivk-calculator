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
    COLOR_STATS_DATABASE = [
        {"name": "Grey", "count": 3597270, "ivk_min": 0.0, "ivk_max": 25.0},
        {"name": "Black", "count": 2634864, "import streamlit as st
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
    COLOR_STATS_DATABASE = [
        {"name": "Grey", "count": 3597270, "ivk_min": 0.0, "ivk_max": 25.0},
        {"name": "Black", "count": 2634864, "ivk_min": 25.0, "ivk_max": 42.0},
        {"name": "Blue", "count": 1382228, "ivk_min": 42.0, "ivk_max": 48.0},
        {"name": "Others", "count": 772997,  "ivk_min": 48.0, "ivk_max": 52.0},
        {"name": "Red", "count": 654054,  "ivk_min": 52.0, "ivk_max": 57.0},
        {"name": "White", "count": 1639041, "ivk_min": 57.0, "ivk_max": 65.0},
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
    /* Стили для кастомного чистого HTML-вертикального бегунка */
    .html-vertical-slider {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 350px;
    }
    .html-vertical-slider input[type=range] {
        writing-mode: vertical-lr;
        direction: rtl;
        appearance: slider-vertical;
        width: 25px;
        height: 320px;
        padding: 0;
        cursor: pointer;
    }
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
            # 1. Горизонтальный бегунок X строго НАД картинкой во всю длину
            cx = st.slider("Horizontal Position (X Target)", 0, w - 1, int(w * 0.5), step=1)
            
            # Разметка: слайдер Y слева (колонка 1), фото автомобиля справа (колонка 15)
            inner_col_slider, inner_col_img = st.columns([1, 15])
            
            with inner_col_slider:
                st.write("<div style='text-align:center; font-weight:bold; font-size:14px; margin-bottom:2px;'>Y</div>", unsafe_allow_html=True)
                
                # Инициализация сессии для передачи значения Y из HTML
                if "manual_cy_val" not in st.session_state:
                    st.session_state.manual_cy_val = int(h * 0.65)
                
                # Скрытый нативный ввод, чтобы Streamlit ловил изменения без падения разметки
                cy_holder = st.empty()
                
                # Рендерим настоящий изолированный HTML5 вертикальный слайдер
                st.markdown(f"""
                    <div class="html-vertical-slider">
                        <input type="range" min="0" max="{h-1}" value="{st.session_state.manual_cy_val}" step="1" id="html_y_slider">
                    </div>
                    <script>
                    var slider = document.getElementById("html_y_slider");
                    slider.oninput = function() {{
                        // Записываем данные в скрытый инпут Streamlit для синхронизации
                        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: this.value}}, '*');
                    }}
                    </script>
                """, unsafe_allow_html=True)
                
                # Безопасный перехват значения из HTML слайдера
                try:
                    cy_captured = cy_holder.number_input("hidden_y", min_value=0, max_value=h-1, value=st.session_state.manual_cy_val, label_visibility="collapsed")
                    st.session_state.manual_cy_val = int(cy_captured)
                except:
                    pass
                
                cy = st.session_state.manual_cy_val
            
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
                # Огромный составной полицветный прицел (Увеличен ровно в 2 раза)
                cv2.drawMarker(visual_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 90, 10) 
                cv2.drawMarker(visual_img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 70, 6)     
                cv2.drawMarker(visual_img, (cx, cy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 6) 
                with inner_col_img:
                    st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)
            else:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    # --- СТАБИЛЬНЫЙ ИЗОЛИРОВАННЫЙ ВЫВОД ПРАВОЙ КОЛОНКИ ---
    if raw_dominant_color is not None:ivk_min": 25.0, "ivk_max": 42.0},
        {"name": "Blue", "count": 1382228, "ivk_min": 42.0, "ivk_max": 48.0},
        {"name": "Others", "count": 772997,  "ivk_min": 48.0, "ivk_max": 52.0},
        {"name": "Red", "count": 654054,  "ivk_min": 52.0, "ivk_max": 57.0},
        {"name": "White", "count": 1639041, "ivk_min": 57.0, "ivk_max": 65.0},
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
    
    /* Стилизация вертикального ползунка через CSS-трансформацию */
    .vertical-slider div[data-testid="stSlider"] > div {
        writing-mode: vertical-lr !important;
        direction: rtl !important;
        height: 340px !important;
        padding-left: 10px !important;
        margin: 0 auto !important;
    }
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
    
    # Главная сетка приложения (50/50)
    col_left_img, col_right_data = st.columns(2)
    raw_dominant_color = None
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Инициализация переменных ручного режима
    manual_mode = col_left_img.checkbox("🎯 Enable manual target correction")
    
    if manual_mode:
        with col_left_img:
            # Горизонтальный бегунок X располагается строго НАД картинкой во всю её длину
            cx = st.slider("Horizontal Position (X Target)", 0, w - 1, int(w * 0.5), step=1)
            
            # Внутренняя пропорциональная сетка: бегунок Y слева (1 часть), фото справа (11 частей)
            inner_col_slider, inner_col_img = st.columns([1, 11])
            
            with inner_col_slider:
                st.write("<div style='text-align:center; font-weight:bold; font-size:12px; margin-bottom:5px;'>Y</div>", unsafe_allow_html=True)
                st.markdown('<div class="vertical-slider">', unsafe_allow_html=True)
                cy = st.slider("Y Tracker", 0, h - 1, int(h * 0.65), step=1, label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
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

    # Генерация изображения с прицелом/контуром
    if raw_dominant_color is not None:
        visual_img = img.copy()
        ch_p = create_checkerboard_pattern(w, h)
        visual_img[final_calculated_mask == 0] = cv2.addWeighted(img, 0.5, ch_p, 0.5, 0)[final_calculated_mask == 0]
        
        if manual_mode:
            # Крупный полицветный прицел высокой видимости (увеличен)
            cv2.drawMarker(visual_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 90, 10) 
            cv2.drawMarker(visual_img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 70, 6)     
            cv2.drawMarker(visual_img, (cx, cy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 6) 
            with inner_col_img:
                st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)
        else:
            with col_left_img:
                cnts, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
                st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)

    # --- ИЗОЛИРОВАННЫЙ РАСЧЕТ И ПОЛНЫЙ ВЫВОД ПРАВОЙ КОЛОНКИ ---
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
        bg_rgb_f32 = bg_rgb.astype(np.float32) / 255.0
        
        bg_lab_matrix = cv2.cvtColor(bg_rgb_f32, cv2.COLOR_RGB2Lab)
        bg_L = float(bg_lab_matrix.item(0, 0, 0))
        bg_a = float(bg_lab_matrix.item(0, 0, 1))
        bg_b = float(bg_lab_matrix.item(0, 0, 2))
        
        delta_L = float(abs(val_L - bg_L))
        delta_ab = float(np.sqrt((val_a - bg_a)**2 + (val_b - bg_b)**2))
        ivk_value = float(np.sqrt((val_L - bg_L)**2 + (val_a - bg_a)**2 + (val_b - bg_b)**2))
        
        db_res = simulate_database_lookup(ivk_value, db_tolerance)
        predicted_crf = float(predict_crf_by_function(ivk_value))
        
        bm = float(base_premium_annual / 12.0)
        va = float(base_premium_annual * predicted_crf)
        vm = float(va / 12.0)
        da = float(va - base_premium_annual)
        dm = float(vm - bm)
        
        txt_annual = f"{va:.2f} {currency_symbol}/yr"
