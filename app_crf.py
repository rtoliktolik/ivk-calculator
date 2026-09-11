import streamlit as st
import cv2
import numpy as np
import os

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
        {"name": "Серый",   "count": 3597270, "ivk_min": 0.0,  "ivk_max": 25.0},
        {"name": "Черный",  "count": 2634864, "ivk_min": 25.0, "ivk_max": 42.0},
        {"name": "Синий",   "count": 1382228, "ivk_min": 42.0, "ivk_max": 48.0},
        {"name": "Другие",  "count": 772997,  "ivk_min": 48.0, "ivk_max": 52.0},
        {"name": "Красный", "count": 654054,  "ivk_min": 52.0, "ivk_max": 57.0},
        {"name": "Белый",   "count": 1639041, "ivk_min": 57.0, "ivk_max": 65.0},
        {"name": "Желтый",  "count": 96277,   "ivk_min": 65.0, "ivk_max": 250.0},
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
        return {"total_cars": 2450, "groups": ["Индивидуальный тон"]}
    return {"total_cars": total_cars_in_cloud, "groups": matched_groups}

# ---------------------------------------------------------------------------
# Настройка веб-интерфейса
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FARRATE-X | Калькулятор ИВК")

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=260)
else:
    st.title("FARRATE-X | АНАЛИТИЧЕСКИЙ КАЛЬКУЛЯТОР ИВК")

st.markdown("---")

# --- СЕКЦИЯ НАСТРОЕК В БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("⚙️ Настройки базы данных")
db_tolerance = st.sidebar.slider("Радиус допуска облака (± ИВК):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("💰 Страховой профиль")
currency_symbol = st.sidebar.selectbox("Выберите валюту:", ["\u20ac", "$", "\u00a3", "\u00a5", "руб."])
base_premium_annual = st.sidebar.number_input(label=f"Базовая годовая премия ({currency_symbol}):", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

# Переключатель режимов замера цвета кузова
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Управление замером")
analysis_mode = st.sidebar.radio(
    "Выберите метод детекции:",
    ("Автоматический ИИ", "Ручной маркер (Ползунки осей)"),
    index=0
)

# Контейнер в боковой панели для расчетов премии
sidebar_calc_space = st.sidebar.empty()

# --- ОСНОВНОЙ КОНТЕНТ ПРИЛОЖЕНИЯ ---
uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото автомобиля", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_h, raw_w, _ = img_raw.shape
    
    # Оптимизация размера под вёрстку экрана
    display_w = 520
    display_h = int((display_w / raw_w) * raw_h)
    img = cv2.resize(img_raw, (display_w, display_h))
    
    b_val, g_val, r_val = 54, 53, 136
    visual_img = img.copy()
    
    # Слайдеры осей инициализируются в основном теле для гарантированной стабильности
    # Сжаты в аккуратную панель управления, которая раскрыта по умолчанию в ручном режиме
    with st.expander("🎛️ Настройка положения прицела инспекции эмали кузова", expanded=(analysis_mode == "Ручной маркер (Ползунки осей)")):
        pct_x = st.slider("Смещение прицела по горизонтали (Ось X в %)", 0, 100, 35, step=1)
        pct_y = st.slider("Смещение прицела по вертикали (Ось Y в %)", 0, 100, 60, step=1)
        
    cx = int((pct_x / 100.0) * display_w)
    cy = int((pct_y / 100.0) * display_h)

    # -----------------------------------------------------------------------
    # МАТЕМАТИЧЕСКАЕ ЯДРО ВЫЧИСЛЕНИЙ
    # -----------------------------------------------------------------------
    if analysis_mode == "Ручной маркер (Ползунки осей)":
        # Накладываем инвертированный бирюзовый прицел (XOR-эффект) строго по процентам слайдеров
        cross_mask = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        cv2.line(cross_mask, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 3)
        cv2.line(cross_mask, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 3)
        cv2.circle(cross_mask, (cx, cy), 4, (255, 255, 255), -1)
        visual_img = cv2.bitwise_xor(img.copy(), cross_mask)
        
        # Замер цвета вокруг прицела
        color_mask = np.zeros((display_h, display_w), dtype=np.uint8)
        cv2.circle(color_mask, (cx, cy), 8, 255, -1)
        mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=color_mask)
        b_val, g_val, r_val = int(mean_b), int(mean_g), int(mean_r)
    else:
        # Автоматический режим ИИ (Использует загруженный локальный файл весов)
        car_mask = np.zeros((display_h, display_w), dtype=np.uint8)
        try:
            from ultralytics import YOLO
            model = YOLO("./yolov8n-seg.pt")
            results = model(img, verbose=False)
            for result in results:
                if result.masks is not None:
                    for mask, cls in zip(result.masks.data, result.boxes.cls):
                        c_id = int(cls)
                        if (c_id == 2 or c_id == 5 or c_id == 7):
                            m_np = cv2.resize(mask.cpu().numpy(), (display_w, display_h))
                            car_mask = cv2.bitwise_or(car_mask, (m_np > 0.5).astype(np.uint8))
        except Exception:
            pass
            
        if np.sum(car_mask) == 0:
            cv2.rectangle(car_mask, (int(display_w*0.25), int(display_h*0.35)), (int(display_w*0.75), int(display_h*0.65)), 1, -1)
            
        kernel = np.ones((35, 35), np.uint8)
        clean_paint_mask = cv2.erode(car_mask, kernel, iterations=2)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_noise_mask = cv2.threshold(gray_img, 35, 255, cv2.THRESH_BINARY)
        _, bright_glare_mask = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY_INV)
        valid_tones = cv2.bitwise_and(dark_noise_mask, bright_glare_mask)
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv_img)
        _, chromatic_mask = cv2.threshold(s_ch, 40, 255, cv2.THRESH_BINARY)
        
        paint_filter = cv2.bitwise_and(valid_tones, chromatic_mask)
        final_calculated_mask = cv2.bitwise_and(clean_paint_mask, paint_filter)
        if np.sum(final_calculated_mask) == 0:
            final_calculated_mask = clean_paint_mask
            
        mask_uint8 = cv2.convertScaleAbs(final_calculated_mask)
        
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
        
        mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=mask_uint8)
        b_val, g_val, r_val = int(mean_b), int(mean_g), int(mean_r)

    # Общие сквозные расчеты индексов ИВК и CRF рисков
    p_L, p_a, p_b = rgb_to_lab(r_val, g_val, b_val)
    ivk_value = float(np.linalg.norm(np.array([p_L, p_a, p_b]) - np.array([BG_L, BG_A, BG_B])))
    predicted_crf = predict_crf_by_function(ivk_value)
    
    base_premium_monthly = float(base_premium_annual / 12.0)
    val_annual = float(base_premium_annual * predicted_crf)
    val_monthly = float(val_annual / 12.0)
    get_d_annual = float(val_annual - base_premium_annual)
    get_d_monthly = float(val_monthly - base_premium_monthly)
    
    db_res = simulate_database_lookup(ivk_value, db_tolerance)

    # -----------------------------------------------------------------------
    # ИНТЕРФЕЙСНАЯ ОТРИСОВКА (БЕЗОПАСНЫЙ СТАБИЛЬНЫЙ ВЫВОД)
    # -----------------------------------------------------------------------
    col_left_img, col_right_data = st.columns([1.1, 0.9])
    
    with col_left_img:
        # Вывод изображения автомобиля (стабильно работает в обоих режимах)
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Зона лакокрасочного покрытия автомобиля", width=520)
            
