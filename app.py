import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Жесткая константа эталона по вашему требованию
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

# ---------------------------------------------------------------------------
# Чистая математика цвета (CIE LAB без использования модулей cv2)
# ---------------------------------------------------------------------------
def rgb_to_lab_pure(rgb_color: tuple) -> np.ndarray:
    r, g, b = [v / 255.0 for v in rgb_color]
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    X, Y, Z = X / 0.95047, Y / 1.00000, Z / 1.08883
    
    def f(t):
        return t ** (1.0/3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)
        
    fx, fy, fz = f(X), f(Y), f(Z)
    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b])

def lab_to_rgb_pure(lab_color: np.ndarray) -> tuple:
    L, a, b = float(lab_color), float(lab_color), float(lab_color)
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    
    delta = 6.0 / 29.0
    x = fx**3 if fx > delta else (fx - 16.0/116.0) * (3 * delta**2)
    y = fy**3 if fy > delta else (fy - 16.0/116.0) * (3 * delta**2)
    z = fz**3 if fz > delta else (fz - 16.0/116.0) * (3 * delta**2)
    
    X, Y, Z = x * 0.95047, y * 1.00000, z * 1.08883
    r_l =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_l = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_l =  0.0556434 * X - 2.0402590 * Y + 1.0572252 * Z
    
    def gamma(c):
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
        
    return (int(np.clip(gamma(r_l) * 255.0, 0, 255)),
            int(np.clip(gamma(g_l) * 255.0, 0, 255)),
            int(np.clip(gamma(b_l) * 255.0, 0, 255)))

def calculate_ivk_lab(car_lab: np.ndarray, bg_rgb=CONSTANT_ROAD_BACKGROUND_RGB) -> dict:
    bg_lab = rgb_to_lab_pure(bg_rgb)
    delta_L = abs(car_lab - bg_lab)
    delta_ab = np.linalg.norm(car_lab[1:] - bg_lab[1:])
    ivk = np.linalg.norm(car_lab - bg_lab)
    
    return {
        "delta_L": float(np.round(delta_L, 2)),
        "delta_ab": float(np.round(delta_ab, 2)),
        "ivk": float(np.round(ivk, 2))
    }

def get_text_rating(ivk_value: float) -> tuple:
    if ivk_value < 15.0: return "Очень плохо 🔴", "#FF4B4B"
    elif ivk_value < 30.0: return "Плохо 🟠", "#FFA500"
    elif ivk_value < 50.0: return "Удовлетворительно 🟡", "#F0D300"
    elif ivk_value < 75.0: return "Хорошо 🟢", "#2EA043"
    else: return "Отлично 🔵", "#007BFF"

def create_checkerboard_pattern(width, height, square_size=15):
    base = np.zeros((square_size * 2, square_size * 2, 3), dtype=np.uint8)
    base[0:square_size, 0:square_size] = 240
    base[square_size:, square_size:] = 240
    base[0:square_size, square_size:] = 200
    base[square_size:, 0:square_size] = 200
    reps_y = int(np.ceil(height / (square_size * 2)))
    reps_x = int(np.ceil(width / (square_size * 2)))
    pattern = np.tile(base, (reps_y, reps_x, 1))
    return pattern[0:height, 0:width]

# ---------------------------------------------------------------------------
# Веб-интерфейс
# ---------------------------------------------------------------------------
st.title("🚗 Автоматический ИВК-калькулятор")
st.write("Программа работает в облаке. ИИ находит кузов автомобиля и автоматически убирает колеса.")

uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото машины", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # АВТОМАТИЧЕСКОЕ СЖАТИЕ ТЯЖЕЛЫХ ФОТО (Ограничиваем макс. сторону до 1200 пикселей)
    max_size = 1200
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    img_np = np.array(pil_img)
    h, w, _ = img_np.shape
    
    manual_mode = st.checkbox("🎯 Включить ручную точечную корректировку")
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    dominant_car_lab = None
    
    if manual_mode:
        st.subheader("Настройка точки контроля")
        cx = st.slider("Позиция по горизонтали (X)", 0, w, int(w / 2))
        cy = st.slider("Позиция по вертикали (Y)", 0, h, int(h / 2))
        x1, y1 = max(0, cx - 10), max(0, cy - 10)
        x2, y2 = min(w, cx + 10), min(h, cy + 10)
        final_calculated_mask[y1:y2, x1:x2] = 1
        
        selected_pixels = img_np[y1:y2, x1:x2].reshape(-1, 3)
        lab_pixels = [rgb_to_lab_pure(tuple(p)) for p in selected_pixels]
        dominant_car_lab = np.median(lab_pixels, axis=0)
    else:
        with st.spinner("Нейросеть YOLOv8 изолирует кузов..."):
            model = YOLO("yolov8n-seg.pt")
            results = model(img_np, verbose=False)
            
            car_mask = np.zeros((h, w), dtype=np.uint8)
            wheels_mask = np.zeros((h, w), dtype=np.uint8)
            
            for result in results:
                if result.masks is not None:
                    for mask, cls in zip(result.masks.data, result.boxes.cls):
                        c_idx = int(cls)
                        m_np = mask.cpu().numpy()
                        m_img = Image.fromarray((m_np * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
                        m_binary = (np.array(m_img) > 127).astype(np.uint8)
                        
                        if c_idx == 2: car_mask = np.maximum(car_mask, m_binary)
                        elif c_idx == 13: wheels_mask = np.maximum(wheels_mask, m_binary)

            if np.sum(car_mask) > 0:
                clean_paint_mask = np.clip(car_mask.astype(int) - wheels_mask.astype(int), 0, 1).astype(np.uint8)
                
                # Эрозия маски стандартным NumPy для очистки краев от травы
                struct_el = np.ones((9, 9), dtype=bool)
                from scipy.ndimage import binary_erosion
                clean_paint_mask = binary_erosion(clean_paint_mask, structure=struct_el).astype(np.uint8)
                
                car_indices = np.argwhere(clean_paint_mask == 1)
                valid_pixels_lab = []
                valid_coords = []
                
                for r, c in car_indices:
                    rgb = tuple(img_np[r, c])
                    lab = rgb_to_lab_pure(rgb)
                    if 20 < lab[0] < 85:  # Проверка яркости L
                        valid_pixels_lab.append(lab)
                        valid_coords.append((r, c))
                        
                if len(valid_pixels_lab) > 0:
                    dominant_car_lab = np.median(valid_pixels_lab, axis=0)
                    for r, c in valid_coords:
                        final_calculated_mask[r, c] = 1
                else:
                    flat_pixels = img_np[clean_paint_mask == 1].reshape(-1, 3)
                    if len(flat_pixels) > 0:
                        valid_pixels_lab = [rgb_to_lab_pure(tuple(p)) for p in flat_pixels]
                        dominant_car_lab = np.median(valid_pixels_lab, axis=0)
                    final_calculated_mask = clean_paint_mask
            else:
                st.error("❌ ИИ не нашел машину. Включите ручную корректировку.")

    if dominant_car_lab is not None:
        dominant_car_rgb = lab_to_rgb_pure(dominant_car_lab)
        
        checkerboard = create_checkerboard_pattern(w, h)
        visual_img = img_np.copy()
        
        non_calculated_mask = (final_calculated_mask == 0)
        visual_img[non_calculated_mask] = (img_np[non_calculated_mask] * 0.5 + checkerboard[non_calculated_mask] * 0.5).astype(np.uint8)
        
        output_pil = Image.fromarray(visual_img)
        draw = ImageDraw.Draw(output_pil)
        
        if manual_mode:
            draw.line([(cx-15, cy), (cx+15, cy)], fill=(255, 0, 0), width=3)
            draw.line([(cx, cy-15), (cx, cy+15)], fill=(255, 0, 0), width=3)
            
        st.image(output_pil, caption="Зона анализа в облаке (Колеса автоматически убраны)", use_container_width=True)

        res = calculate_ivk_lab(dominant_car_lab)
        rating_text, rating_color = get_text_rating(res["ivk"])
        
        st.subheader("📊 Результат анализа")
        st.markdown(f"### Визуальная заметность автомобиля: <span style='color:{rating_color}; font-weight:bold;'>{rating_text}</span>", unsafe_allow_html=True)
        st.write("")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Индекс ИВК (Полный)", f"{res['ivk']:.2f}")
        with col2: st.metric("Разница яркости (ΔL)", f"{res['delta_L']:.2f}")
        with col3: st.metric("Разница тона (Δab)", f"{res['delta_ab']:.2f}")
            
        st.markdown(f"**Цвет кузова (без бликов):** RGB{dominant_car_rgb}")
        st.markdown(f'<div style="background-color: rgb{dominant_car_rgb}; width: 100px; height: 30px; border-radius: 5px; border: 1px solid #000;"></div>', unsafe_allow_html=True)
        st.info(f"Эталон фона зафиксирован: RGB{CONSTANT_ROAD_BACKGROUND_RGB} (асфальт, облачно)")
