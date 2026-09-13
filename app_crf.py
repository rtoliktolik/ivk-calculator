import streamlit as st
import cv2
import numpy as np

# Фиксированный фон дороги (Асфальт)
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

def predict_crf_by_function(target_ivk: float) -> float:
    xp = [12.5, 33.5, 47.0, 58.5, 80.0]
    fp = [1.19, 1.03, 1.00, 0.975, 0.93]
    return float(np.interp(target_ivk, xp, fp))

def simulate_database_lookup(target_ivk: float, tolerance: float) -> dict:
    db_names = ["Grey", "Black", "Blue", "Others", "Red", "White", "Yellow"]
    db_counts = [3597270, 2634864, 1382228, 772997, 654054, 1639041, 96277]
    db_mins = [0.0, 25.0, 42.0, 48.0, 52.0, 57.0, 65.0]
    db_maxs = [25.0, 42.0, 48.0, 52.0, 57.0, 65.0, 150.0]
    
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

# --- ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(layout="wide", page_title="FARRATE-X | IVK Calculator")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 3.5rem !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { font-size: 1.3rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("FARRATE-X | ANALYTICAL IVK CALCULATOR")
st.markdown("---")

# --- НАСТРОЙКИ В БОКОВОЙ ПАНЕЛИ ---
st.sidebar.header("⚙️ Database Settings")
db_tolerance = st.sidebar.slider("Cloud tolerance radius (± IVK):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("💰 Insurance Profile")
currency_symbol = st.sidebar.selectbox("Select Currency Symbol:", ["€", "$", "£", "¥", "u.e."])
base_premium_annual = st.sidebar.number_input(label="Base Annual Premium:", min_value=1.0, max_value=1000000.0, value=850.0, step=10.0)

uploaded_file = st.file_uploader("Step 1 — Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    # Слайдеры наведения
    st.markdown("**🎯 Координатная панель прицеливания (двойной ползунок):**")
    cx = st.slider("Сдвиг прицела по ГОРИЗОНТАЛИ (X)", 0, w - 1, int(w * 0.5), step=1)
    cy = st.slider("Сдвиг прицела по ВЕРТИКАЛИ (Y)", 0, h - 1, int(h * 0.65), step=1)
    
    # Извлечение цвета BGR из выбранной точки
    pixel_color_bgr = img[cy, cx]
    b_val, g_val, r_val = int(pixel_color_bgr[0]), int(pixel_color_bgr[1]), int(pixel_color_bgr[2])
    
    # Конвертация в LAB для математического расчета контраста
    pixel_bgr_matrix = np.uint8([[[b_val, g_val, r_val]]])
    pixel_lab = cv2.cvtColor(pixel_bgr_matrix, cv2.COLOR_BGR2Lab)[0][0]
    val_L, val_a, val_b = float(pixel_lab[0]), float(pixel_lab[1]), float(pixel_lab[2])
    
    # Фон дороги в LAB
    bg_bgr_matrix = np.uint8([[[CONSTANT_ROAD_BACKGROUND_RGB[2], CONSTANT_ROAD_BACKGROUND_RGB[1], CONSTANT_ROAD_BACKGROUND_RGB[0]]]])
    bg_lab = cv2.cvtColor(bg_bgr_matrix, cv2.COLOR_BGR2Lab)[0][0]
    bg_L, bg_a, bg_b = float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2])
    
    # Расчет индексов
    delta_L = float(abs(val_L - bg_L))
    delta_ab = float(np.sqrt((val_a - bg_a)**2 + (val_b - bg_b)**2))
    ivk_value = float(np.sqrt((val_L - bg_L)**2 + (val_a - bg_a)**2 + (val_b - bg_b)**2))
    
    predicted_crf = predict_crf_by_function(ivk_value)
    db_res = simulate_database_lookup(ivk_value, db_tolerance)
    
    # Расчет финансовой панели
    bm = float(base_premium_annual / 12.0)
    va = float(base_premium_annual * predicted_crf)
    vm = float(va / 12.0)
    da = float(va - base_premium_annual)
    dm = float(vm - bm)
    
    # Отрисовка финансового блока в сайдбаре
    st.sidebar.markdown("---")
    st.sidebar.write("**🧮 Live Premium Calculation**")
    st.sidebar.write(f"Base: {base_premium_annual:.2f} {currency_symbol}/yr")
    st.sidebar.metric(label="Adjusted Annual Premium", value=f"{va:.2f} {currency_symbol}/yr", delta=f"{da:.2f} {currency_symbol}/yr", delta_color="inverse")
    st.sidebar.metric(label="Adjusted Monthly Premium", value=f"{vm:.2f} {currency_symbol}/mo", delta=f"{dm:.2f} {currency_symbol}/mo", delta_color="inverse")

    # Основная сетка интерфейса (50/50)
    col_left_img, col_right_data = st.columns(2)
    
    with col_left_img:
        visual_img = img.copy()
        # Огромный контрастный прицел (Увеличен в 2 раза)
        cv2.drawMarker(visual_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 90, 10) 
        cv2.drawMarker(visual_img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 70, 6)     
        cv2.drawMarker(visual_img, (cx, cy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 6) 
        st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Body Paintwork Scanning Zone", use_container_width=True)
        
    with col_right_data:
        st.subheader("📊 Express Analysis Results")
        st.metric("Visual Contrast Index (IVK)", f"{ivk_value:.2f}")
        st.metric("Color Risk Factor (CRF)", f"{predicted_crf:.2f}")
        
        status_text = "LOW RISK 👍" if predicted_crf < 1.0 else ("HIGH RISK ⚠️" if predicted_crf > 1.0 else "NORMAL")
        st.write(f"**Current Visibility Status:** {status_text}")
        st.markdown("---")
        
        st.write(f"**Detected Car Body Color (RGB):** {r_val}, {g_val}, {b_val}")
        
        # НАСТОЯЩИЙ И БЕЗОТКАЗНЫЙ ЦВЕТНОЙ ПРЯМОУГОЛЬНИК (нативная матрица NumPy)
        pure_color_block = np.zeros((150, 400, 3), dtype=np.uint8)
        pure_color_block[:, :] = [r_val, g_val, b_val]
        st.image(pure_color_block, caption="Isolated Paint Shade", use_container_width=True)
        st.markdown("---")
        
        m1, m2 = st.columns(2)
        m1.metric("Light Contrast ΔL", f"{delta_L:.2f}")
        m2.metric("Chromatic Contrast Δab", f"{delta_ab:.2f}")
        st.markdown("---")
        
        st.subheader("🔮 Predictive Evaluation by Databases")
        st.write(f"Found **{db_res['total_cars']:,}** registered vehicles in the tolerance cloud.")
        st.caption(f"Related Statistical Groups: {', '.join(db_res['groups'])}")
        st.write("### Continuous Accident Risk Regression Curve")
