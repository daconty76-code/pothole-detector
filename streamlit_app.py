# streamlit_app.py
# With debug mode to show all detections

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import time
import os

st.set_page_config(page_title="Pothole Detection System", page_icon="🛣️", layout="wide")

st.title("🛣️ Automated Urban Infrastructure Inspection")
st.markdown("### Real-Time Pothole Detection using YOLOv8n (ONNX)")

# Sidebar controls
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# Add debug toggle
debug_mode = st.sidebar.checkbox("Show Debug Info", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**")
st.sidebar.markdown("- Lower confidence threshold = more detections")
st.sidebar.markdown("- Try 0.1 or 0.15 if nothing is detected")

CLASS_NAMES = ['longitudinal crack', 'transverse crack', 'alligator crack', 'other corruption', 'Pothole']

@st.cache_resource
def load_model():
    model_path = "best.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
        return None

def preprocess_image(image, target_size=640):
    img = image.resize((target_size, target_size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    return np.expand_dims(img_array, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_yolo_output(output, conf_threshold=0.25):
    output = output[0].T  # Shape: (8400, 9)
    
    detections = []
    all_raw_detections = []  # For debugging
    
    for detection in output:
        x = detection[0]
        y = detection[1]
        w = detection[2]
        h = detection[3]
        box_conf = detection[4]
        
        class_scores = sigmoid(detection[5:10])
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        confidence = box_conf * max_class_score
        
        # Store all detections for debugging
        all_raw_detections.append({
            'confidence': float(confidence),
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}",
            'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)
        })
        
        if confidence > conf_threshold:
            x1 = max(0, min(640, x - w / 2))
            y1 = max(0, min(640, y - h / 2))
            x2 = max(0, min(640, x + w / 2))
            y2 = max(0, min(640, y + h / 2))
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class_name': class_name,
                'class_id': class_id
            })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    all_raw_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections, all_raw_detections

def draw_boxes(image, detections, original_size):
    draw = ImageDraw.Draw(image)
    pothole_count = 0
    scale_x = original_size[0] / 640
    scale_y = original_size[1] / 640
    
    for det in detections:
        x1 = int(det['bbox'][0] * scale_x)
        y1 = int(det['bbox'][1] * scale_y)
        x2 = int(det['bbox'][2] * scale_x)
        y2 = int(det['bbox'][3] * scale_y)
        
        if det['class_name'] == 'Pothole':
            color = "red"
            pothole_count += 1
        elif 'crack' in det['class_name']:
            color = "blue"
        else:
            color = "green"
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        draw.text((x1, y1-15), f"{det['class_name']}: {det['confidence']:.2f}", fill=color)
    
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
            try:
                with st.spinner("Running detection..."):
                    input_tensor = preprocess_image(image)
                    
                    start = time.time()
                    outputs = model.run(None, {'images': input_tensor})
                    inference_time = (time.time() - start) * 1000
                    
                    # Decode with current threshold
                    detections, raw_detections = decode_yolo_output(outputs[0], confidence_threshold)
                    
                    # Draw results
                    result_image, pothole_count = draw_boxes(image.copy(), detections, original_size)
                    
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
                            st.info("💡 Try lowering the confidence threshold (0.1-0.15)")
                    
                    # Debug information
                    if debug_mode:
                        st.markdown("---")
                        st.subheader("🔍 Debug Information")
                        
                        # Show top 10 raw detections (before threshold)
                        st.write(f"**Top raw detections (before threshold):**")
                        raw_data = []
                        for i, det in enumerate(raw_detections[:15]):
                            raw_data.append({
                                'Confidence': f"{det['confidence']:.4f}",
                                'Class': det['class_name'],
                                'Threshold Applied': det['confidence'] >= confidence_threshold
                            })
                        st.table(raw_data)
                        
                        st.write(f"**Detections after threshold ({confidence_threshold}):** {len(detections)}")
                        
                        # Show histogram of confidence values
                        if raw_detections:
                            confidences = [d['confidence'] for d in raw_detections[:50]]
                            st.write(f"**Confidence range:** min={min(confidences):.4f}, max={max(confidences):.4f}")
                            st.write(f"**Recommended threshold:** try {max(confidences) * 0.7:.2f} or lower")
                        
                        # Show what the model sees
                        pothole_detections = [d for d in raw_detections if d['class_name'] == 'Pothole']
                        if pothole_detections:
                            st.write(f"**Pothole-specific detections:** {len(pothole_detections)}")
                            for d in pothole_detections[:5]:
                                st.write(f"  - Confidence: {d['confidence']:.4f}")
                        else:
                            st.write("**No pothole-specific detections found at any confidence level**")
                            st.write("The model may not have learned potholes well during training.")
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
