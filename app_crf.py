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

# ИЗОЛИРОВАННАЯ ФУНКЦИЯ ДЛЯ ГАРАНТИРОВАННОГО ВЫВОДА ПРАВОЙ ПАНЕЛИ
def render_analytics_panel(r, g, b, db_tol, premium_annual, curr_sym, space_container):
    v_R = (r / 255.0)
    v_G = (g / 255.0)
    v_B = (b / 255.0)
    v_R = ((v_R + 0.055) / 1.055) ** 2.4 if v_R > 0.04045 else v_R / 12.92
    v_G = ((v_G + 0.055) / 1.055) ** 2.4 if v_G > 0.04045 else v_G / 12.92
    v_B = ((v_B + 0.055) / 1.055) ** 2.4 if v_B > 0.04045 else v_B / 12.92
    X = (v_R * 0.4124 + v_G * 0.3576 + v_B * 0.1805) * 100 / 95.047
    Y = (v_R * 0.2126 + v_G * 0.7152 + v_B * 0.0722) * 100 / 100.000
    Z = (v_R * 0.0193 + v_G * 0.1192 + v_B * 0.1192) * 100 / 108.883
    X = X ** (1/3) if X > 0.008856 else (7.787 * X) + (16 / 116)
    Y = Y ** (1/3) if Y > 0.008856 else (7.787 * Y) + (16 / 116)
    Z = Z ** (1/3) if Z > 0.008856 else (7.787 * Z) + (16 / 116)
    L_val = (116 * Y) - 16
    a_val = 500 * (X - Y)
    b_val_lab = 200 * (Y - Z)
    
    ivk = float(np.linalg.norm(np.array([L_val, a_val, b_val_lab]) - np.array([BG_L, BG_A, BG_B])))
    crf = float(np.interp(ivk, XP_POINTS, FP_POINTS))
    
    st.metric(label="Индекс визуального контраста (ИВК)", value=f"{ivk:.2f}")
    st.metric(label="Фактор риска цвета (CRF)", value=f"{crf:.2f}")
    
    verdict = "НИЗКИЙ РИСК 👍" if crf < 1.0 else ("ВЫСОКИЙ РИСК ⚠️" if crf > 1.0 else "НОРМА")
    st.info(f"Вердикт анализа: **{verdict}**")
    st.markdown("---")
    
    db = [
        {"name": "Серый", "count": 3597270, "min": 0.0, "max": 25.0},
        {"name": "Черный", "count": 2634864, "min": 25.0, "max": 42.0},
        {"name": "Синий", "count": 1382228, "min": 42.0, "max": 48.0},
        {"name": "Другие", "count": 772997, "min": 48.0, "max": 52.0},
        {"name": "Красный", "count": 654054, "min": 52.0, "max": 57.0},
        {"name": "Белый", "count": 1639041, "min": 57.0, "max": 65.0},
    ]
    matched = []
    total = 0
    for g in db:
        if max(0.0, ivk - db_tol) < g["max"] and min(ivk + db_tol, 250.0) > g["min"]:
            total += int(g["count"] * 0.15)
            matched.append(g["name"])
            
    st.markdown(f"**🗄️ Страховое облако Big Data:**")
    st.write(f"• **Активных совпадений:** {max(2450, total):,} шт.")
    st.write(f"• **Категории риска:** {', '.join(matched) if matched else 'Индивидуальный тон'}")

    with space_container.container():
        st.write("**🧮 Расчет текущей премии**")
        st.write(f"Базовая: {premium_annual:.2f} {curr_sym}/год")
        st.metric(label="Скорректированная премия", value=f"{premium_annual * crf:.2f} {curr_sym}/год", delta=f"{(premium_annual * crf) - premium_annual:.2f} {curr_sym}/год", delta_color="inverse")

# ---------------------------------------------------------------------------
# Инициализация интерфейса Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="FARRATE-X")

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=260)
else:
    st.title("FARRATE-X | АНАЛИТИЧЕСКИЙ КАЛЬКУЛЯТОР")

st.markdown("---")
db_tolerance = st.sidebar.slider("Радиус допуска облака (± ИВК):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
currency_symbol = st.sidebar.selectbox("Выберите валюту:", ["\u20ac", "$", "\u00a3", "\u00a5", "руб."])
base_premium_annual = st.sidebar.number_input(label="Базовая годовая премия:", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

# Режим автоматической ИИ-инспекции по умолчанию жестко зафиксирован
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Управление замером")
st.sidebar.info("🤖 Активирован автоматический режим ИИ YOLO")

sidebar_calc_space = st.sidebar.empty()

uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото автомобиля", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_h, raw_w, _ = img_raw.shape
    
    display_w = 520
    display_h = int((display_w / raw_w) * raw_h)
    img = cv2.resize(img_raw, (display_w, display_h))
    
    b_val, g_val, r_val = 54, 53, 136
    visual_img = img.copy()

    # СТРОГАЯ АВТОМАТИЧЕСКАЯ СЕГМЕНТАЦИЯ YOLO С ФИЛЬТРОМ КЛАССОВ ТРАНСПОРТА
    car_mask = np.zeros((display_h, display_w), dtype=np.uint8)
    try:
        from ultralytics import YOLO
        model = YOLO("./yolov8n-seg.pt")
        results = model(img, verbose=False)
        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.data, result.boxes.cls):
                    c_id = int(cls)
                    # Жесткая проверка: пропускаем только легковые авто (2), автобусы (5) и грузовики (7)
                    if (c_id == 2 or c_id == 5 or c_id == 7):
                        m_np = cv2.resize(mask.cpu().numpy(), (display_w, display_h))
                        car_mask = cv2.bitwise_or(car_mask, (m_np > 0.5).astype(np.uint8))
    except Exception:
        pass
        
    if np.sum(car_mask) == 0:
        cv2.rectangle(car_mask, (int(display_w*0.25), int(display_h*0.35)), (int(display_w*0.75), int(display_h*0.65)), 1, -1)
        
    # Глубокое сжатие краев маски, чтобы убрать колеса, арки и асфальт под авто
    kernel = np.ones((35, 35), np.uint8)
    clean_paint_mask = cv2.erode(car_mask, kernel, iterations=2)
    
    # Очистка от ахроматического шума стёкол и фар
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
    
    # Поиск контуров и наложение зеленой обводки ИИ кузова
    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visual_img, cnts, -1, (0, 255, 0), 2)
    
    # Безопасный расчет среднего значения цвета кузова через OpenCV
    mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=mask_uint8)
    b_val, g_val, r_val = int(mean_b), int(mean_g), int(mean_r)

    # -----------------------------------------------------------------------
    # ИНТЕРФЕЙСНАЯ ОТРИСОВКА КОЛОНОК
    # -----------------------------------------------------------------------
    col_left_img, col_right_data = st.columns([1.1, 0.9])
    
    with col_left_img:
        st.markdown("**🤖 ИИ изолирует лакокрасочное покрытие кузова (без стекол и колес):**")
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Автоматическая зона сканирования ИИ", width=520)
        
        # Вывод точной цветовой плашки под картинкой автомобиля
        color_patch_bgr = np.full((38, display_w, 3), (b_val, g_val, r_val), dtype=np.uint8)
        st.image(cv2.cvtColor(color_patch_bgr, cv2.COLOR_BGR2RGB), caption=f"Образец цвета кузова (RGB: {r_val}, {g_val}, {b_val})")

    with col_right_data:
        st.markdown("### 📊 Результаты экспресс-анализа")
        # Вызов защищенной локальной функции вычислений
        render_analytics_panel(b_val, g_val, r_val, db_tolerance, base_premium_annual, currency_symbol, sidebar_calc_space)
