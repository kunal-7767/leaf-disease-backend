from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import logging
import os

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
        
        results = model.predict(
            img, 
            conf=0.25,
            iou=0.45,
            imgsz=640,
            augment=False
        )
        
        print(f"🔍 Number of results: {len(results)}")
        
        detections = []
        result = results[0]
        
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
        
        return jsonify({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "image_shape": img.shape
        })
        
    except Exception as e:
        print(f"💥 Detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)