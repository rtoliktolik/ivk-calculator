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
currency_symbol = st.sidebar.selectbox("Выберите валюту:", ["€", "$", "£", "¥", "руб."])
base_premium_annual = st.sidebar.number_input(label=f"Базовая годовая премия ({currency_symbol}):", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

# Контейнер в боковой панели для расчетов премии
sidebar_calc_space = st.sidebar.empty()

# --- ОСНОВНОЙ КОНТЕНТ ПРИЛОЖЕНИЯ ---
uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото автомобиля", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # Жесткая и гарантированная инициализация ползунков в процентной шкале
    with st.expander("🎛️ Панель точечного выбора зоны сканирования кузова", expanded=True):
        pct_x = st.slider("Положение маркера по горизонтали (X в %)", 0, 100, 45, step=1)
        pct_y = st.slider("Положение маркера по вертикали (Y в %)", 0, 100, 68, step=1)
        
    # Моментальный пересчет процентов ползунков в реальные пиксели фотографии кузова
    cx = int((pct_x / 100.0) * w)
    cy = int((pct_y / 100.0) * h)
    
    # Создаем локальную маску 30х30 пикселей вокруг точки прицела
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(final_calculated_mask, (cx, cy), 15, 255, -1)
    
    # Отрисовываем жирный красный перекрестный маркер прицела на изображении кузова
    visual_img = img.copy()
    cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 45, 4)
    
    mask_uint8 = cv2.convertScaleAbs(final_calculated_mask)
    mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=mask_uint8)
    b_val, g_val, r_val = int(mean_b), int(mean_g), int(mean_r)

    # МАТЕМАТИЧЕСКИЙ РАСЧЕТ ИНДЕКСОВ И ПРЕМИЙ
    p_L, p_a, p_b = rgb_to_lab(r_val, g_val, b_val)
    ivk_value = float(np.linalg.norm(np.array([p_L, p_a, p_b]) - np.array([BG_L, BG_A, BG_B])))
    predicted_crf = predict_crf_by_function(ivk_value)
    
    base_premium_monthly = float(base_premium_annual / 12.0)
    val_annual = float(base_premium_annual * predicted_crf)
    val_monthly = float(val_annual / 12.0)
    get_d_annual = float(val_annual - base_premium_annual)
    get_d_monthly = float(val_monthly - base_premium_monthly)

    # ОБНОВЛЕНИЕ БОКОВОЙ ПАНЕЛИ
    with sidebar_calc_space.container():
        st.write("**🧮 Расчет текущей премии**")
        st.write(f"Базовая: {base_premium_annual:.2f} {currency_symbol}/год")
        st.metric(label="Скорректированная годовая премия", value=f"{val_annual:.2f} {currency_symbol}/год", delta=f"{get_d_annual:.2f} {currency_symbol}/год", delta_color="inverse")
        st.metric(label="Скорректированная месячная премия", value=f"{val_monthly:.2f} {currency_symbol}/мес", delta=f"{get_d_monthly:.2f} {currency_symbol}/мес", delta_color="inverse")

    # КОМПАКТНЫЙ ДВУХКОЛОНОЧНЫЙ МАКЕТ
    col_left_img, col_right_data = st.columns([1.1, 0.9])
    
    with col_left_img:
        st.markdown(f"### 📋 Результаты экспресс-анализа кузова")
        
        # Создаем плашку цвета из матрицы NumPy и переводим в BGR -> RGB
        color_patch_bgr = np.full((38, 520, 3), (b_val, g_val, r_val), dtype=np.uint8)
        color_patch_rgb = cv2.cvtColor(color_patch_bgr, cv2.COLOR_BGR2RGB)
        st.image(color_patch_rgb, caption=f"Выделенный образец цвета кузова (RGB: {r_val}, {g_val}, {b_val})")
            
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Зона сканирования лакокрасочного покрытия", width=520)

    with col_right_data:
        st.markdown("### 📊 Результаты экспресс-анализа")
        
        st.metric(label="Индекс визуального контраста (ИВК)", value=f"{ivk_value:.2f}")
        st.metric(label="Фактор риска цвета (CRF)", value=f"{predicted_crf:.2f}")
        
        status_text = "НИЗКИЙ РИСК 👍" if predicted_crf < 1.0 else ("ВЫСОКИЙ РИСК ⚠️" if predicted_crf > 1.0 else "НОРМА")
        st.info(f"Вердикт анализа: **{status_text}**")
        
        db_res = simulate_database_lookup(ivk_value, db_tolerance)
        st.markdown("---")
        st.markdown(f"**🗄️ Страховое облако Big Data:**")
        st.write(f"• **Активных совпадений в кластере:** {db_res['total_cars']:,} шт.")
        st.write(f"• **Категории риска из базы:** {', '.join(db_res['groups'])}")
