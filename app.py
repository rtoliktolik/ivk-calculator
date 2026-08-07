import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from scipy.ndimage import binary_erosion
from ultralytics import YOLO

# Constant road background reference (Asphalt, overcast day)
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
    
    # Human eye adaptation: color difference weight multiplied by 1.8
    ivk = np.sqrt((delta_L * 1.0) ** 2 + (delta_ab * 1.8) ** 2)
    
    return {
        "delta_L": round(delta_L, 2),
        "delta_ab": round(delta_ab, 2),
        "ivk": round(ivk, 2)
    }

def get_text_rating(ivk_value: float) -> tuple:
    if ivk_value < 15.0: return "Very Poor 🔴", "#FF4B4B"
    elif ivk_value < 25.0: return "Poor 🟠", "#FFA500"
    elif ivk_value < 35.0: return "Satisfactory 🟡", "#F0D300"
    elif ivk_value < 55.0: return "Good 🟢", "#2EA043"
    else: return "Excellent 🔵", "#007BFF"

def create_checkerboard_pattern(width, height, square_size=15):
    base = np.zeros((square_size * 2, square_size * 2, 3), dtype=np.uint8)
    base[0:square_size, 0:square_size] = 240
    import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    if os.path.exists("logo.png"):
        encoded_img = get_base64_image("logo.png")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{encoded_img}" style="max-width: 280px; height: auto;">
            </div>
            """,
            unsafe_allow_html=True
        )
except Exception:
    pass
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <div style="font-family: 'Arial Black', Gadget, sans-serif; font-size: 36px; font-weight: 900; letter-spacing: 2px; color: #1E3A8A; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); border: 3px solid #1E3A8A; padding: 5px 20px; border-radius: 8px;">
            FAIRRATE<span style="color: #FF4B4B;">-X</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Automated VIC Calculator")
st.write("The app is optimized for human visual perception. AI detects the vehicle body paint, automatically filtering out wheels, windows, and deep shadows.")

uploaded_file = st.file_uploader("Step 1 — Upload a vehicle photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    max_size = 1200
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    img_np = np.array(pil_img)
    h, w, _ = img_np.shape
    
    manual_mode = st.checkbox("🎯 Enable manual region override")
    final_calculated_mask = np.zeros((h, w), dtype=np.uint8)
    dominant_car_lab = None
    
    if manual_mode:
        st.subheader("Manual Control Point Settings")
        cx = st.slider("Horizontal position (X)", 0, w, int(w / 2))
        cy = st.slider("Vertical position (Y)", 0, h, int(h / 2))
        x1, y1 = max(0, cx - 10), max(0, cy - 10)
        x2, y2 = min(w, cx + 10), min(h, cy + 10)
        final_calculated_mask[y1:y2, x1:x2] = 1
        
        selected_pixels = img_np[y1:y2, x1:x2].reshape(-1, 3)
        lab_pixels = [rgb_to_lab_opencv_single(tuple(p)) for p in selected_pixels]
        dominant_car_lab = np.median(lab_pixels, axis=0)
    else:
        with st.spinner("YOLOv8 AI is segmenting the vehicle body..."):
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
                
                struct_el = np.ones((5, 5), dtype=bool)
                clean_paint_mask = binary_erosion(clean_paint_mask, structure=struct_el).astype(np.uint8)
                
                car_indices = np.argwhere(clean_paint_mask == 1)
                valid_pixels_lab = []
                valid_coords = []
                
                for r, c in car_indices:
                    rgb = tuple(img_np[r, c])
                    lab = rgb_to_lab_opencv_single(rgb)
                    
                    if 25 < lab[0] < 92:
                        color_saturation = np.linalg.norm(lab[1:])
                        if color_saturation > 2.0:
                            valid_pixels_lab.append(lab)
                            valid_coords.append((r, c))
                        
                if len(valid_pixels_lab) > 0:
                    valid_pixels_lab = np.array(valid_pixels_lab)
                    avg_pure_lab = np.median(valid_pixels_lab, axis=0)
                    
                    distances = np.linalg.norm(valid_pixels_lab - avg_pure_lab, axis=1)
                    best_idx = np.argmin(distances)
                    dominant_car_lab = valid_pixels_lab[best_idx]
                    
                    for r, c in valid_coords:
                        final_calculated_mask[r, c] = 1
                else:
                    flat_pixels = img_np[clean_paint_mask == 1].reshape(-1, 3)
                    if len(flat_pixels) > 0:
                        valid_pixels_lab = [rgb_to_lab_opencv_single(tuple(p)) for p in flat_pixels]
                        dominant_car_lab = np.median(valid_pixels_lab, axis=0)
                    final_calculated_mask = clean_paint_mask
            else:
                st.error("❌ Vehicle not detected by AI. Please enable manual override above.")

    if dominant_car_lab is not None:
        dominant_car_rgb = lab_to_rgb_opencv_single(dominant_car_lab)
        
        checkerboard = create_checkerboard_pattern(w, h)
        visual_img = img_np.copy()
        
        non_calculated_mask = (final_calculated_mask == 0)
        visual_img[non_calculated_mask] = (img_np[non_calculated_mask] * 0.5 + checkerboard[non_calculated_mask] * 0.5).astype(np.uint8)
        
        output_pil = Image.fromarray(visual_img)
        draw = ImageDraw.Draw(output_pil)
        
        if not manual_mode and np.sum(final_calculated_mask) > 0:
            mask_pil = Image.fromarray((final_calculated_mask * 255).astype(np.uint8))
            edges = mask_pil.filter(ImageFilter.FIND_EDGES)
            edges_np = np.array(edges)
            visual_img[edges_np > 100] = (0, 255, 0)
            output_pil = Image.fromarray(visual_img)
            draw = ImageDraw.Draw(output_pil)

        if manual_mode:
            draw.line([(cx-15, cy), (cx+15, cy)], fill=(255, 0, 0), width=3)
            draw.line([(cx, cy-15), (cx, cy+15)], fill=(255, 0, 0), width=3)
            
        st.image(output_pil, caption="Analysis Zone (Maximum paint coverage matrix)", use_container_width=True)

        res = calculate_ivk_lab(dominant_car_lab)
        rating_text, rating_color = get_text_rating(res["ivk"])
        
        st.subheader("📊 Analysis Results")
        st.markdown(f"### Vehicle Conspicuity Rating: <span style='color:{rating_color}; font-weight:bold;'>{rating_text}</span>", unsafe_allow_html=True)
        st.write("")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("VIC Index (Human Eye)", f"{res['ivk']:.2f}")
        with col2: st.metric("Lightness Contrast (ΔL)", f"{res['delta_L']:.2f}")
        with col3: st.metric("Chromatic Contrast (Δab)", f"{res['delta_ab']:.2f}")
            
        st.markdown(f"**Extracted Body Color (Pure Pigment):** RGB{dominant_car_rgb}")
        st.markdown(f'<div style="background-color: rgb{dominant_car_rgb}; width: 100px; height: 30px; border-radius: 5px; border: 1px solid #000;"></div>', unsafe_allow_html=True)
        st.info(f"Constant Background Reference: RGB{CONSTANT_ROAD_BACKGROUND_RGB} (Asphalt, Overcast)")
