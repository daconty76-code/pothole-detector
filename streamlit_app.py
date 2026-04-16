# streamlit_app.py
# Automated Urban Infrastructure Inspection: Real-Time Pothole Detection
# Deployed on Streamlit Community Cloud
# NO OpenCV - uses PIL for image processing

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time

# Page configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🛣️",
    layout="wide"
)

# Title
st.title("🛣️ Automated Urban Infrastructure Inspection")
st.markdown("### Real-Time Pothole Detection using YOLOv8n")
st.markdown("Upload an image to detect potholes and road damage.")

# Sidebar
st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
    help="Lower values detect more potholes but increase false positives."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("**Model:** YOLOv8n")
st.sidebar.write("**Dataset:** RDD2022")
st.sidebar.write("**Training Epochs:** 50")

# Class names from RDD2022 dataset
CLASS_NAMES = [
    'longitudinal crack',
    'transverse crack',
    'alligator crack',
    'other corruption',
    'Pothole'
]

# Color mapping for bounding boxes (RGB format for PIL)
COLOR_MAP = {
    'longitudinal crack': (255, 0, 0),      # Red
    'transverse crack': (0, 255, 0),        # Green
    'alligator crack': (255, 255, 0),       # Yellow
    'other corruption': (0, 255, 255),      # Cyan
    'Pothole': (255, 0, 0)                  # Red for potholes
}

@st.cache_resource
def load_model():
    """Load the trained YOLOv8n model"""
    model_path = "pothole_detector.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error("Model file not found. Please check deployment.")
        return None

def draw_boxes_pil(image, detections, confidence_thresh):
    """Draw bounding boxes using PIL (no OpenCV)"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    detected_potholes = 0
    
    if detections and len(detections) > 0:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            if conf < confidence_thresh:
                continue
            
            if class_name == 'Pothole':
                detected_potholes += 1
            
            color = COLOR_MAP.get(class_name, (255, 0, 0))
            
            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            draw.rectangle([(x1, y1 - 18), (x1 + len(label) * 7, y1)], fill=color)
            draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255))
    
    return img_copy, detected_potholes

def process_image(image, model, confidence_thresh):
    """Run inference on an image"""
    start_time = time.time()
    results = model(image, conf=confidence_thresh, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_name': class_name
            })
    
    return detections, inference_time

def main():
    model = load_model()
    if model is None:
        return
    
    st.success("✅ Model loaded successfully")
    
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Potholes", type="primary"):
            with st.spinner("Running detection..."):
                detections, inference_time = process_image(image, model, confidence_threshold)
                
                result_image, pothole_count = draw_boxes_pil(image, detections, confidence_threshold)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result_image, caption="Detection Results", use_container_width=True)
                
                with col2:
                    st.metric("Inference Time", f"{inference_time:.1f} ms")
                    st.metric("FPS", f"{1000/inference_time:.1f}")
                    st.metric("Potholes Detected", pothole_count)
                    
                    if pothole_count > 0:
                        st.warning(f"⚠️ {pothole_count} pothole(s) detected!")
                    else:
                        st.success("✅ No potholes detected")
                
                if len(detections) > 0:
                    with st.expander("Detailed Detections"):
                        for det in detections:
                            if det['confidence'] >= confidence_threshold:
                                st.write(f"- {det['class_name']}: {det['confidence']:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instructions**
1. Upload an image
2. Adjust confidence threshold (optional)
3. Click Detect button
4. Review results
""")

if __name__ == "__main__":
    main()
