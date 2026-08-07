import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Жесткая константа эталона по вашему требованию
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

# ---------------------------------------------------------------------------
# Чистая математика цвета
# ---------------------------------------------------------------------------
def rgb_to_lab_opencv(bgr_image: np.ndarray) -> np.ndarray:
    rgb_f32 = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2Lab)

def lab_to_rgb_pure(lab_color: np.ndarray) -> tuple:
    L, a, b = float(lab_color[0]), float(lab_color[1]), float(lab_color[2])
    
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    
    delta = 6.0 / 29.0
    
    x = fx**3 if fx > delta else (fx - 16.0/116.0) * (3 * delta**2)
    y = fy**3 if fy > delta else (fy - 16.0/116.0) * (3 * delta**2)
    z = fz**3 if fz > delta else (fz - 16.0/116.0) * (3 * delta**2)
    
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = x * Xn, y * Yn, z * Zn
    
    r_l =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_l = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_l =  0.0556434 * X - 2.0402590 * Y + 1.0572252 * Z
    
    def gamma(c):
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
        
    r = int(np.clip(gamma(r_l) * 255.0, 0, 255))
    g = int(np.clip(gamma(g_l) * 255.0, 0, 255))
    b = int(np.clip(gamma(b_l) * 255.0, 0, 255))
    
    return (r, g, b)

def calculate_ivk_lab(car_lab: np.ndarray, bg_rgb=CONSTANT_ROAD_BACKGROUND_RGB) -> dict:
    car_lab = car_lab.flatten()
    bg_bgr = np.uint8([[list(bg_rgb[::-1])]])
    bg_lab = cv2.cvtColor(bg_bgr.astype(np.float32)/255.0, cv2.COLOR_BGR2Lab).flatten()
    
    delta_L = abs(car_lab[0] - bg_lab[0])
    delta_ab = np.linalg.norm(car_lab[1:] - bg_lab[1:])
    ivk = np.linalg.norm(car_lab - bg_lab)
    
    return {
        "car_lab": tuple(float(np.round(v, 2)) for v in car_lab),
        "background_lab": tuple(float(np.round(v, 2)) for v in bg_lab),
        "delta_L": float(np.round(delta_L, 2)),
        "delta_ab": float(np.round(delta_ab, 2)),
        "ivk": float(np.round(ivk, 2))
    }

def get_text_rating(ivk_value: float) -> tuple:
    if ivk_value < 15.0:
        return "Очень плохо 🔴", "#FF4B4B"
    elif ivk_value < 30.0:
        return "Плохо 🟠", "#FFA500"
    elif ivk_value < 50.0:
        return "Удовлетворительно 🟡", "#F0D300"
    elif ivk_value < 75.0:
        return "Хорошо 🟢", "#2EA043"
    else:
        return "Отлично 🔵", "#007BFF"

def create_checkerboard_pattern(width, height, square_size=15):
    base = np.zeros((square_size * 2, square_size * 2, 3), dtype=np.uint8)
    base[0:square_size, 0:square_size] = (240, 240, 240)
    base[square_size:, square_size:] = (240, 240, 240)
    base[0:square_size, square_size:] = (200, 200, 200)
    base[square_size:, 0:square_size] = (200, 200, 200)
    
    reps_y = int(np.ceil(height / (square_size * 2)))
    reps_x = int(np.ceil(width / (square_size * 2)))
    pattern = np.tile(base, (reps_y, reps_x, 1))
    return pattern[0:height, 0:width]

# ---------------------------------------------------------------------------
# Веб-интерфейс Streamlit
# ---------------------------------------------------------------------------
st.title("🚗 Автоматический ИВК-калькулятор")
st.write("По умолчанию нейросеть определяет кузов сама и убирает колеса. Если вы недовольны, включите ручную корректировку ползунками.")

uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото машины", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    # Кнопка ручного режима
    manual_mode = st.checkbox("🎯 Включить ручную точечную корректировку")
    
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    dominant_car_lab = None
    
    if manual_mode:
        st.subheader("Настройка точки контроля")
        # Ползунки для выбора точных координат
        cx = st.slider("Позиция по горизонтали (X)", 0, w, int(w / 2))
        cy = st.slider("Позиция по вертикали (Y)", 0, h, int(h / 2))
        
        # Квадрат 20х20 вокруг выбранной точки
        x1, y1 = max(0, cx - 10), max(0, cy - 10)
        x2, y2 = min(w, cx + 10), min(h, cy + 10)
        final_calculated_mask[y1:y2, x1:x2] = 1
        
        lab_img = rgb_to_lab_opencv(img)
        user_pixels = lab_img[y1:y2, x1:x2]
        dominant_car_lab = np.median(user_pixels, axis=(0, 1))
        
    else:
        # Автоматический режим через ИИ
        with st.spinner("Нейросеть YOLOv8 изолирует металл кузова и убирает колеса..."):
            model = YOLO("yolov8n-seg.pt")
            results = model(img, verbose=False)
            
            car_mask = np.zeros((h, w), dtype=np.uint8)
            wheels_mask = np.zeros((h, w), dtype=np.uint8)
            
            for result in results:
                if result.masks is not None:
                    for mask, cls in zip(result.masks.data, result.boxes.cls):
                        class_idx = int(cls)
                        mask_np = mask.cpu().numpy()
                        mask_np = cv2.resize(mask_np, (w, h))
                        mask_binary = (mask_np > 0.5).astype(np.uint8)
                        
                        if class_idx == 2:    # Кузов
                            car_mask = cv2.bitwise_or(car_mask, mask_binary)
                        elif class_idx == 13: # Колеса
                            wheels_mask = cv2.bitwise_or(wheels_mask, mask_binary)

            if np.sum(car_mask) > 0:
                car_without_wheels_mask = cv2.bitwise_and(car_mask, cv2.bitwise_not(wheels_mask))
                kernel = np.ones((11, 11), np.uint8)
                clean_paint_mask = cv2.erode(car_without_wheels_mask, kernel, iterations=2)
                
                lab_img = rgb_to_lab_opencv(img)
                car_pixels_lab = lab_img[clean_paint_mask == 1]
                
                if len(car_pixels_lab) > 0:
                    L_channel = car_pixels_lab[:, 0]
                    valid_indices = (L_channel > 20) & (L_channel < 85)
                    
                    if len(valid_indices) > 0:
                        car_mask_indices = np.argwhere(clean_paint_mask == 1)
                        valid_car_pixels = car_mask_indices[valid_indices]
                        for pt in valid_car_pixels:
                            final_calculated_mask[pt[0], pt[1]] = 1
                    
                    filtered_car_pixels = car_pixels_lab[valid_indices]
                    if len(filtered_car_pixels) == 0:
                        filtered_car_pixels = car_pixels_lab
                        final_calculated_mask = clean_paint_mask
                        
                    dominant_car_lab = np.median(filtered_car_pixels, axis=0)
            else:
                st.error("❌ ИИ не нашел машину. Включите ручную корректировку.")

    if dominant_car_lab is not None:
        dominant_car_rgb = lab_to_rgb_pure(dominant_car_lab)
        
        # Визуализация слоев
        visual_img = img.copy()
        checkerboard = create_checkerboard_pattern(w, h)
        non_calculated_mask = (final_calculated_mask == 0)
        
        visual_img[non_calculated_mask] = cv2.addWeighted(img, 0.5, checkerboard, 0.5, 0)[non_calculated_mask]
        
        # Обводка контура зоны анализа
        contours, _ = cv2.findContours(final_calculated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visual_img, contours, -1, (0, 255, 0), 3)
        
        if manual_mode:
            # Рисуем прицел в ручном режиме
            cv2.drawMarker(visual_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Ручной выбор (Крестик — центр анализа ЛКП)", use_container_width=True)
        else:
            st.image(cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB), caption="Результат контроля ИИ (Колеса автоматически отсечены)", use_container_width=True)

        # Вывод чистых метрик
        res = calculate_ivk_lab(dominant_car_lab)
        rating_text, rating_color = get_text_rating(res["ivk"])
        
        st.subheader("📊 Результат анализа")
        st.markdown(f"### Визуальная заметность автомобиля: <span style='color:{rating_color}; font-weight:bold;'>{rating_text}</span>", unsafe_allow_html=True)
        st.write("")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Индекс ИВК (Полный)", f"{res['ivk']:.2f}")
        with col2:
            st.metric("Разница яркости (ΔL)", f"{res['delta_L']:.2f}")
        with col3:
            st.metric("Разница тона (Δab)", f"{res['delta_ab']:.2f}")
            
        st.markdown(f"**Цвет выбранной зоны кузова:** RGB{dominant_car_rgb}")
        st.markdown(f'<div style="background-color: rgb{dominant_car_rgb}; width: 100px; height: 30px; border-radius: 5px; border: 1px solid #000;"></div>', unsafe_allow_html=True)
        st.info(f"Эталон фона зафиксирован: RGB{CONSTANT_ROAD_BACKGROUND_RGB} (асфальт, облачно)")