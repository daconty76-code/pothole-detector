# streamlit_app.py
# Using ONNX Runtime - no OpenCV dependency

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import time
import os

st.set_page_config(page_title="Pothole Detection System", page_icon="🛣️", layout="wide")

st.title("🛣️ Automated Urban Infrastructure Inspection")
st.markdown("### Real-Time Pothole Detection using YOLOv8n (ONNX)")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

CLASS_NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']

@st.cache_resource
def load_model():
    model_path = "best.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    else:
        st.error(f"Model file not found. Looking for: {model_path}")
        st.write("Files in directory:", os.listdir("."))
        return None

def preprocess_image(image, target_size=640):
    """Preprocess image for ONNX model"""
    # Resize
    img = image.resize((target_size, target_size))
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    # Transpose to CHW format (Channel, Height, Width)
    img_array = img_array.transpose(2, 0, 1)
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)

def draw_boxes(image, detections, conf_thresh, original_size):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    pothole_count = 0
    
    # Get scale factors to map from 640x640 back to original image size
    scale_x = original_size[0] / 640
    scale_y = original_size[1] / 640
    
    for det in detections:
        conf = det['confidence']
        if conf >= conf_thresh:
            # Scale bounding boxes back to original image size
            x1 = int(det['bbox'][0] * scale_x)
            y1 = int(det['bbox'][1] * scale_y)
            x2 = int(det['bbox'][2] * scale_x)
            y2 = int(det['bbox'][3] * scale_y)
            
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            draw.text((x1, y1-15), f"{det['class_name']}: {conf:.2f}", fill="red")
            
            if det['class_name'] == 'Pothole':
                pothole_count += 1
    
    return image, pothole_count

def main():
    model = load_model()
    if model is None:
        return
    
    st.success("✅ Model loaded successfully")
    
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        original_size = image.size
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Potholes", type="primary"):
            with st.spinner("Running detection..."):
                # Preprocess
                input_tensor = preprocess_image(image)
                
                # Run inference
                start = time.time()
                outputs = model.run(None, {'images': input_tensor})
                inference_time = (time.time() - start) * 1000
                
                # Parse outputs
                # YOLO ONNX output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
                detections = []
                
                if len(outputs) > 0:
                    output = outputs[0]
                    
                    # Handle different output shapes
                    if len(output.shape) == 3:
                        # Shape: [batch, num_detections, 6]
                        for det in output[0]:
                            conf = float(det[4])
                            if conf > 0.1:  # Low threshold for testing
                                class_id = int(det[5])
                                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                                detections.append({
                                    'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                    'confidence': conf,
                                    'class_name': class_name
                                })
                
                # Draw results
                result_image, pothole_count = draw_boxes(image.copy(), detections, confidence_threshold, original_size)
                
                # Display results
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
                
                if detections:
                    with st.expander("Detailed Detections"):
                        for det in detections:
                            if det['confidence'] >= confidence_threshold:
                                st.write(f"- {det['class_name']}: {det['confidence']:.2f}")

if __name__ == "__main__":
    main()
