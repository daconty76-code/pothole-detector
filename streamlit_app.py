# streamlit_app.py
# Debug version - shows exact error and model output

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
        session = ort.InferenceSession(model_path)
        # Get model input/output info for debugging
        st.sidebar.write("Model Info:")
        st.sidebar.write(f"Inputs: {[x.name for x in session.get_inputs()]}")
        st.sidebar.write(f"Outputs: {[x.name for x in session.get_outputs()]}")
        return session
    else:
        st.error(f"Model file not found: {model_path}")
        st.write("Files in directory:", os.listdir("."))
        return None

def preprocess_image(image, target_size=640):
    """Preprocess image for ONNX model"""
    img = image.resize((target_size, target_size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    return np.expand_dims(img_array, axis=0)

def main():
    model = load_model()
    if model is None:
        return
    
    st.success(" Model loaded successfully")
    
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        original_size = image.size
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Detect Potholes", type="primary"):
            try:
                with st.spinner("Running detection..."):
                    # Preprocess
                    input_tensor = preprocess_image(image)
                    
                    # Get input name
                    input_name = model.get_inputs()[0].name
                    
                    # Run inference
                    start = time.time()
                    outputs = model.run(None, {input_name: input_tensor})
                    inference_time = (time.time() - start) * 1000
                    
                    # Debug: Show output info
                    st.write(f"Number of outputs: {len(outputs)}")
                    for i, out in enumerate(outputs):
                        st.write(f"Output {i} shape: {out.shape}, dtype: {out.dtype}")
                    
                    # Try to parse outputs
                    detections = []
                    output = outputs[0]
                    
                    # Different possible output formats
                    if len(output.shape) == 3:
                        # Format: [batch, num_detections, 6]
                        for det in output[0]:
                            conf = float(det[4])
                            if conf > 0.1:
                                class_id = int(det[5])
                                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                                detections.append({
                                    'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                    'confidence': conf,
                                    'class_name': class_name
                                })
                    elif len(output.shape) == 2:
                        # Format: [num_detections, 6]
                        for det in output:
                            conf = float(det[4])
                            if conf > 0.1:
                                class_id = int(det[5])
                                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
                                detections.append({
                                    'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                    'confidence': conf,
                                    'class_name': class_name
                                })
                    elif len(output.shape) == 1:
                        st.write("Unexpected output format - needs investigation")
                        st.write(f"Output values: {output[:50]}")  # Show first 50 values
                    
                    st.write(f"Detections found: {len(detections)}")
                    
                    # Draw results
                    def draw_boxes(image, detections, conf_thresh, original_size):
                        draw = ImageDraw.Draw(image)
                        pothole_count = 0
                        scale_x = original_size[0] / 640
                        scale_y = original_size[1] / 640
                        
                        for det in detections:
                            if det['confidence'] >= conf_thresh:
                                x1 = int(det['bbox'][0] * scale_x)
                                y1 = int(det['bbox'][1] * scale_y)
                                x2 = int(det['bbox'][2] * scale_x)
                                y2 = int(det['bbox'][3] * scale_y)
                                
                                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                                draw.text((x1, y1-15), f"{det['class_name']}: {det['confidence']:.2f}", fill="red")
                                
                                if det['class_name'] == 'Pothole':
                                    pothole_count += 1
                        
                        return image, pothole_count
                    
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
                            st.warning(f" {pothole_count} pothole(s) detected!")
                        else:
                            st.success("No potholes detected")
                    
                    if detections:
                        with st.expander("Detailed Detections"):
                            for det in detections:
                                if det['confidence'] >= confidence_threshold:
                                    st.write(f"- {det['class_name']}: {det['confidence']:.2f}")
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                st.write("Please check the logs for more details.")

if __name__ == "__main__":
    main()
