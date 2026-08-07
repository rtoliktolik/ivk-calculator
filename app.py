import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from ultralytics import YOLO

# Жесткая константа эталона по вашему требованию
CONSTANT_ROAD_BACKGROUND_RGB = (105, 105, 105)

def rgb_to_lab_opencv_single(rgb_color: tuple) -> np.ndarray:
    img_bgr = np.uint8([[list(rgb_color[::-1])]])
    img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return img_lab.flatten()

def lab_to_rgb_opencv_single(lab_color: np.ndarray) -> tuple:
    lab_pixel = np.array([[lab_color]], dtype=np.float32)
    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2RGB)
    rgb_clipped = np.clip(rgb_pixel * 255.0, 0, 255).astype(np.uint8)
    return tuple(int(x) for x in rgb_clipped.flatten())

def calculate_ivk_lab(car_lab: np.ndarray, bg_rgb=CONSTANT_ROAD_BACKGROUND_RGB) -> dict:
    bg_lab = rgb_to_lab_opencv_single(bg_rgb)
    car_lab = np.array(car_lab).flatten()
    
    L1, a1, b1 = float(car_lab[0]), float(car_lab[1]), float(car_lab[2])
    L2, a2, b2 = float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2])
    
    delta_L = abs(L1 - L2)
    delta_ab = np.sqrt((a1 - a2)**2 + (b1 - b2)**2)
    
    # Адаптация под человеческий глаз: вес цветности увеличен в 1.8 раза
    ivk = np.sqrt((delta_L * 1.0) ** 2 + (delta_ab * 1.8) ** 2)
    
    return {
        "delta_L": round(delta_L, 2),
        "delta_ab": round(delta_ab, 2),
        "ivk": round(ivk, 2)
    }

def get_text_rating(ivk_value: float) -> tuple:
    if ivk_value < 15.0: return "Очень плохо 🔴", "#FF4B4B"
    elif ivk_value < 25.0: return "Плохо 🟠", "#FFA500"
    elif ivk_value < 35.0: return "Удовлетворительно 🟡", "#F0D300"
    elif ivk_value < 55.0: return "Хорошо 🟢", "#2EA043"
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
# Веб-интерфейс Streamlit
# ---------------------------------------------------------------------------
st.title("🚗 Автоматический ИВК-калькулятор")
st.write("Программа адаптирована под человеческое зрение. ИИ находит чистый пигмент ЛКП, игнорируя тени и колеса.")

uploaded_file = st.file_uploader("Шаг 1 — Загрузите фото машины", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    
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
        lab_pixels = [rgb_to_lab_opencv_single(tuple(p)) for p in selected_pixels]
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
                
                # РАСШИРИЛИ ОБЛАСТЬ: Уменьшили отступ от краев кузова до (7, 7) вместо (11, 11)
                struct_el = np.ones((7, 7), dtype=bool)
                from scipy.ndimage import binary_erosion
                clean_paint_mask = binary_erosion(clean_paint_mask, structure=struct_el).astype(np.uint8)
                
                car_indices = np.argwhere(clean_paint_mask == 1)
                valid_pixels_lab = []
                valid_coords = []
                saturations = []
                
                for r, c in car_indices:
                    rgb = tuple(img_np[r, c])
                    lab = rgb_to_lab_opencv_single(rgb)
                    
                    if 25 < lab[0] < 90:
                        color_saturation = np.linalg.norm(lab[1:])
                        # УМЕНЬШИЛИ ЧУВСТВИТЕЛЬНОСТЬ: Порог насыщенности 7.0 вместо 10.0
                        if color_saturation > 7.0:
                            valid_pixels_lab.append(lab)
                            valid_coords.append((r, c))
                            saturations.append(color_saturation)
                        
                if len(valid_pixels_lab) > 0:
                    valid_pixels_lab = np.array(valid_pixels_lab)
                    cutoff = np.percentile(saturations, 65)
                    pure_paint_indices = [i for i, sat in enumerate(saturations) if sat >= cutoff]
                    
                    pure_labs = valid_pixels_lab[pure_paint_indices]
                    avg_pure_lab = np.median(pure_labs, axis=0)
                    
                    distances = np.linalg.norm(pure_labs - avg_pure_lab, axis=1)
                    best_idx = np.argmin(distances)
                    dominant_car_lab = pure_labs[best_idx]
                    
                    for idx in pure_paint_indices:
                        r, c = valid_coords[idx]
                        final_calculated_mask[r, c] = 1
                else:
                    flat_pixels = img_np[clean_paint_mask == 1].reshape(-1, 3)
                    if len(flat_pixels) > 0:
                        valid_pixels_lab = [rgb_to_lab_opencv_single(tuple(p)) for p in flat_pixels]
                        dominant_car_lab = np.median(valid_pixels_lab, axis=0)
                    final_calculated_mask = clean_paint_mask
            else:
                st.error("❌ ИИ не нашел машину. Включите ручную корректировку.")

    if dominant_car_lab is not None:
        dominant_car_rgb = lab_to_rgb_opencv_single(dominant_car_lab)
        
        checkerboard = create_checkerboard_pattern(w, h)
        visual_img = img_np.copy()
        
        non_calculated_mask = (final_calculated_mask == 0)
        visual_img[non_calculated_mask] = (img_np[non_calculated_mask] * 0.5 + checkerboard[non_calculated_mask] * 0.5).astype(np.uint8)
        
        output_pil = Image.fromarray(visual_img)
        draw = ImageDraw.Draw(output_pil)
        
        # ВОЗВРАЩАЕМ ТОНКУЮ ЗЕЛЕНУЮ ЛИНИЮ КОНТУРА
        if not manual_mode and np.sum(final_calculated_mask) > 0:
            # Находим границы расчетной зоны математическим методом (вычитание размытия)
            mask_pil = Image.fromarray((final_calculated_mask * 255).astype(np.uint8))
            edges = mask_pil.filter(ImageFilter.FIND_EDGES)
            edges_np = np.array(edges)
            # Там где край - рисуем тонкую зеленую линию (толщина 1-2 пикселя)
            visual_img[edges_np > 100] = [0, 255, 0]
            output_pil = Image.fromarray(visual_img)
            draw = ImageDraw.Draw(output_pil)

        if manual_mode:
            draw.line([(cx-15, cy), (cx+15, cy)], fill=(255, 0, 0), width=3)
            draw.line([(cx, cy-15), (cx, cy+15)], fill=(255, 0, 0), width=3)
            
        st.image(output_pil, caption="Зона анализа (Контур и чувствительность ЛКП сбалансированы)", use_container_width=True)

        res = calculate_ivk_lab(dominant_car_lab)
        rating_text, rating_color = get_text_rating(res["ivk"])
        
        st.subheader("📊 Результат анализа")
        st.markdown(f"### Визуальная заметность автомобиля: <span style='color:{rating_color}; font-weight:bold;'>{rating_text}</span>", unsafe_allow_html=True)
        st.write("")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Индекс ИВК (Человеческий)", f"{res['ivk']:.2f}")
        with col2: st.metric("Разница яркости (ΔL)", f"{res['delta_L']:.2f}")
        with col3: st.metric("Разница тона (Δab)", f"{res['delta_ab']:.2f}")
            
        st.markdown(f"**Цвет кузова (чистый пигмент):** RGB{dominant_car_rgb}")
        st.markdown(f'<div style="background-color: rgb{dominant_car_rgb}; width: 100px; height: 30px; border-radius: 5px; border: 1px solid #000;"></div>', unsafe_allow_html=True)
        st.info(f"Эталон фона зафиксирован: RGB{CONSTANT_ROAD_BACKGROUND_RGB} (асфальт, облачно)")
