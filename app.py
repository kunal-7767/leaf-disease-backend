from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Load model
try:
    model = YOLO("best.pt")
    print("✅ Model loaded successfully")
    print(f"📊 Model classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        file = request.files["image"]
        
        # Read and decode image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        print(f"📐 Input image shape: {img.shape}")
        
        # 🔥 CRITICAL FIX: Use model.predict() instead of direct call
        # This ensures proper preprocessing like during training
        results = model.predict(
            img, 
            conf=0.25,    # Confidence threshold
            iou=0.45,     # NMS IOU threshold
            imgsz=640,    # Inference size
            augment=False  # Disable augmentation for consistent results
        )
        
        print(f"🔍 Number of results: {len(results)}")
        
        detections = []
        result = results[0]  # Get first result
        
        if result.boxes is not None and len(result.boxes) > 0:
            for j, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detection = {
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": confidence,
                    "confidence_percentage": round(confidence * 100, 2),
                    "bbox": bbox
                }
                detections.append(detection)
                
                print(f"🎯 Detection {j}: {model.names[cls_id]} - {confidence:.2%}")
        else:
            print("❌ No detections found")
            # Try with lower confidence threshold
            results_low_conf = model.predict(img, conf=0.1)
            if results_low_conf[0].boxes is not None and len(results_low_conf[0].boxes) > 0:
                print("✅ Found detections with lower confidence threshold")
                for j, box in enumerate(results_low_conf[0].boxes):
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    detection = {
                        "class_id": cls_id,
                        "class_name": model.names[cls_id],
                        "confidence": confidence,
                        "confidence_percentage": round(confidence * 100, 2),
                        "bbox": bbox
                    }
                    detections.append(detection)
                    print(f"🎯 Low-conf Detection {j}: {model.names[cls_id]} - {confidence:.2%}")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "image_shape": img.shape
        })
        
    except Exception as e:
        print(f"💥 Detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/high-confidence-detect", methods=["POST"])
def high_confidence_detect():
    """Alternative endpoint with higher confidence threshold"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        file = request.files["image"]
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Use higher confidence threshold
        results = model.predict(
            img, 
            conf=0.6,    # Higher confidence threshold
            iou=0.45,
            imgsz=640
        )
        
        detections = []
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detection = {
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": confidence,
                    "confidence_percentage": round(confidence * 100, 2),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)
                print(f"🎯 High-Conf Detection: {model.names[cls_id]} - {confidence:.2%}")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "confidence_threshold_used": 0.6
        })
        
    except Exception as e:
        print(f"💥 High-confidence detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    """Endpoint to check model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_loaded": True,
        "classes": model.names,
        "num_classes": len(model.names),
        "model_type": "YOLO"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)