# streamlit_app.py
# Using ONNX Runtime - no OpenCV dependency

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import time
import os

# Page configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Automated Urban Infrastructure Inspection")
st.markdown("### Real-Time Pothole Detection using YOLOv8n (ONNX)")
st.markdown("Upload an image to detect potholes and road damage.")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

CLASS_NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']
COLOR_MAP = {'Pothole': (255, 0, 0)}

@st.cache_resource
def load_model():
    model_path = "pothole_detector.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    else:
        st.error("Model file not found")
        return None

def preprocess_image(image, target_size=640):
    img = image.resize((target_size, target_size))
    img_array = np.array(img).astype(np.float32)
    img_array = img_array.transpose(2, 0, 1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def draw_boxes(image, detections, conf_thresh):
    draw = ImageDraw.Draw(image)
    pothole_count = 0
    for det in detections:
        if det['confidence'] >= conf_thresh:
            x1, y1, x2, y2 = det['bbox']
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            draw.text((x1, y1-15), f"{det['class_name']}: {det['confidence']:.2f}", fill="red")
            if det['class_name'] == 'Pothole':
                pothole_count += 1
    return image, pothole_count

def main():
    model = load_model()
    if model is None:
        return
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Potholes"):
            with st.spinner("Running detection..."):
                input_tensor = preprocess_image(image)
                start = time.time()
                outputs = model.run(None, {'images': input_tensor})
                inference_time = (time.time() - start) * 1000
                
                # Parse outputs (format depends on model)
                detections = []
                # Note: Output parsing depends on ONNX export format
                # You may need to adjust this based on actual output shape
                
                result_image, pothole_count = draw_boxes(image.copy(), detections, confidence_threshold)
                
                st.image(result_image, caption="Detection Results", use_container_width=True)
                st.metric("Inference Time", f"{inference_time:.1f} ms")
                st.metric("Potholes Detected", pothole_count)

if __name__ == "__main__":
    main()
