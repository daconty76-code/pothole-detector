# streamlit_app.py
# Fixed for YOLO ONNX output format (1, 9, 8400)

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
        st.error(f"Model file not found: {model_path}")
        return None

def preprocess_image(image, target_size=640):
    """Preprocess image for ONNX model"""
    img = image.resize((target_size, target_size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    return np.expand_dims(img_array, axis=0)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def decode_yolo_output(output, conf_threshold=0.25):
    """
    Decode YOLO ONNX output format (1, 9, 8400)
    Format: [batch, features, detections]
    Features: [x, y, w, h, box_conf, class_conf0, class_conf1, ...]
    """
    # Remove batch dimension
    output = output[0]  # Shape: (9, 8400)
    
    # Transpose to (8400, 9) for easier processing
    output = output.T  # Shape: (8400, 9)
    
    detections = []
    
    for detection in output:
        # Extract values
        x = detection[0]
        y = detection[1]
        w = detection[2]
        h = detection[3]
        box_conf = detection[4]
        
        # Get class scores (last 5 values for 5 classes)
        class_scores = detection[5:10]
        
        # Apply sigmoid to class scores
        class_scores = sigmoid(class_scores)
        
        # Get max class score and class id
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        
        # Calculate overall confidence
        confidence = box_conf * max_class_score
        
        if confidence > conf_threshold:
            # Convert from center-x, center-y, width, height to x1, y1, x2, y2
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Ensure coordinates are valid (x1 <= x2, y1 <= y2)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Clip to image boundaries (0-640)
            x1 = max(0, min(640, x1))
            y1 = max(0, min(640, y1))
            x2 = max(0, min(640, x2))
            y2 = max(0, min(640, y2))
            
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class_name': class_name,
                'class_id': class_id
            })
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS (Non-Maximum Suppression) to remove overlapping boxes
    def nms(detections, iou_threshold=0.45):
        filtered = []
        for det in detections:
            keep = True
            for existing in filtered:
                # Calculate IoU
                x1 = max(det['bbox'][0], existing['bbox'][0])
                y1 = max(det['bbox'][1], existing['bbox'][1])
                x2 = min(det['bbox'][2], existing['bbox'][2])
                y2 = min(det['bbox'][3], existing['bbox'][3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
                    area2 = (existing['bbox'][2] - existing['bbox'][0]) * (existing['bbox'][3] - existing['bbox'][1])
                    iou = intersection / (area1 + area2 - intersection)
                    
                    if iou > iou_threshold:
                        keep = False
                        break
            if keep:
                filtered.append(det)
        return filtered
    
    return nms(detections)

def draw_boxes(image, detections, conf_thresh, original_size):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    pothole_count = 0
    
    # Scale factors from 640x640 to original image size
    scale_x = original_size[0] / 640
    scale_y = original_size[1] / 640
    
    for det in detections:
        if det['confidence'] >= conf_thresh:
            # Scale bounding boxes back to original image size
            x1 = int(det['bbox'][0] * scale_x)
            y1 = int(det['bbox'][1] * scale_y)
            x2 = int(det['bbox'][2] * scale_x)
            y2 = int(det['bbox'][3] * scale_y)
            
            # Choose color based on class
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
                    
                    # Run inference
                    start = time.time()
                    outputs = model.run(None, {'images': input_tensor})
                    inference_time = (time.time() - start) * 1000
                    
                    # Decode YOLO output
                    detections = decode_yolo_output(outputs[0], confidence_threshold)
                    
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
                            st.warning(f" {pothole_count} pothole(s) detected!")
                        else:
                            st.success("No potholes detected")
                    
                    if detections:
                        with st.expander("Detailed Detections"):
                            for det in detections[:10]:  # Show top 10
                                if det['confidence'] >= confidence_threshold:
                                    st.write(f"- {det['class_name']}: {det['confidence']:.2f}")
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
