from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def load_model():
    """Load YOLO model with multiple fallback strategies"""
    global model
    try:
        # Try loading custom model first
        from ultralytics import YOLO
        model = YOLO("best.pt")
        logger.info("✅ Custom model loaded successfully")
        logger.info(f"📊 Model classes: {model.names}")
        return True
    except Exception as e:
        logger.error(f"❌ Custom model failed: {e}")
        
        # Load model - ONLY OUR CUSTOM LEAF DISEASE MODEL
        try:
            from ultralytics import YOLO
            model = YOLO("best.pt")
            logger.info("✅ OUR CUSTOM LEAF DISEASE MODEL LOADED SUCCESSFULLY!")
            logger.info(f"📊 Our model classes: {model.names}")
            logger.info("🚀 READY FOR LEAF DISEASE DETECTION!")
        except Exception as e:
            logger.error(f"❌❌❌ OUR CUSTOM MODEL FAILED: {e}")
            logger.error("🔥 THIS IS CRITICAL - NO FALLBACK!")
            # Crash so we know it's broken
            raise Exception(f"OUR LEAF DISEASE MODEL FAILED: {e}")

# Load model on startup
model_loaded = load_model()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Leaf Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/health", "/detect", "/model-info"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "service": "Leaf Disease Detection API"
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    if model is None:
        return jsonify({
            "model_loaded": False,
            "message": "Model not available - running in API mode"
        })
    
    return jsonify({
        "model_loaded": True,
        "classes": model.names if hasattr(model, 'names') else [],
        "num_classes": len(model.names) if hasattr(model, 'names') else 0
    })

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded", 
            "message": "API is running but detection is unavailable"
        }), 503
    
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400
    
    try:
        file = request.files["image"]
        
        # Validate file
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Read and decode image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Invalid image file"}), 400
        
        logger.info(f"📐 Processing image: {img.shape}")
        
        # Run detection
        results = model.predict(
            img, 
            conf=0.25,
            iou=0.45,
            imgsz=640,
            augment=False
        )
        
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
                logger.info(f"🎯 Detection {j}: {model.names[cls_id]} - {confidence:.2%}")
        else:
            logger.info("❌ No detections found")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "total_detections": len(detections),
            "image_shape": img.shape,
            "model_used": "custom" if "best.pt" in str(model) else "standard"
        })
        
    except Exception as e:
        logger.error(f"💥 Detection error: {e}")
        return jsonify({
            "success": False,
            "error": "Detection failed",
            "details": str(e)
        }), 500

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint"""
    return jsonify({
        "message": "API is working!",
        "timestamp": os.times().elapsed,
        "model_status": "loaded" if model else "not loaded"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"🔧 Model loaded: {model is not None}")
    app.run(host="0.0.0.0", port=port, debug=False)