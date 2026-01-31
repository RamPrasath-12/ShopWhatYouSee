import base64, cv2, numpy as np
from models.yolo_detector import YoloDetector
from config import YOLO_MODELS, YOLO_CONF_THRESH
from models.agman_extractor import process_crop_base64
from models.scene_context import SceneContextDetector
from models.llm_reasoner import LLMReasoner
from flask_cors import CORS,cross_origin
import psycopg2
from models.product_retrieval import search_products
from flask import Flask, request, jsonify, send_from_directory
from utils.color_utils import hex_to_color_name
import os


def get_db():
    return psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="123456"
    )



app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static"
)
CORS(app, resources={r"/*": {"origins": "*"}})

scene_detector = SceneContextDetector()

llm = LLMReasoner(model_name="google/flan-t5-small")  # or the model you have loaded
yolo = YoloDetector(YOLO_MODELS)  # Ensemble detection with multiple models

def b64_to_cv2(img_b64):
    header, data = img_b64.split(',', 1)
    imgbytes = base64.b64decode(data)
    arr = np.frombuffer(imgbytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    # Preflight
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS OK"}), 200

    data = request.get_json()
    img_b64 = data.get("image")

    if not img_b64:
        return jsonify({"error": "No image received"}), 400

    frame = b64_to_cv2(img_b64)
    
    # DEBUG: Log image dimensions
    h, w = frame.shape[:2]
    print(f"\n📸 DETECTION REQUEST:")
    print(f"   Image size: {w}x{h} pixels")
    print(f"   Aspect ratio: {w/h:.2f}")

    detections = yolo.infer(frame)
    detections = [d for d in detections if d['conf'] >= YOLO_CONF_THRESH]
    
    # DEBUG: Log detection results
    print(f"   Detections found: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"   [{i+1}] {det['class']} (conf={det['conf']:.2f})")
    print()

    return jsonify({"detections": detections})

@app.route('/extract-attributes', methods=['POST'])
def extract_attributes():
    data = request.get_json()
    b64 = data.get("image")
    category = data.get("category", "unknown")  # Default if missing
    if not b64:
        return jsonify({"error":"no image provided"}), 400
    result = process_crop_base64(b64, category)
    return jsonify(result)

@app.route('/scene', methods=['POST'])
def scene():
    data = request.get_json()
    b64 = data.get("image")
    if not b64:
        return jsonify({"error":"no frame provided"}), 400
    result = scene_detector.infer(b64)
    return jsonify(result)
# --------------------------------------------------
# LLM INTENT REASONING ROUTE
# --------------------------------------------------
@app.route("/llm", methods=["POST"])
def llm_route():
    body = request.get_json() or {}

    item = body.get("item", {})
    scene = body.get("scene")              # full scene object
    user_query = body.get("user_query")    # may be None / ""

    # 🔒 Category is ALWAYS locked (YOLO ground truth)
    category = item.get("category")

    try:
        # -----------------------------
        # Call LLM reasoner
        # -----------------------------
        result = llm.generate_filters(
            item=item,
            scene=scene,
            user_query=user_query
        )

        # -----------------------------
        # Enforce category (non-negotiable)
        # -----------------------------
        if "filters" not in result:
            result["filters"] = {}

        result["filters"]["category"] = category

        return jsonify(result)

    except Exception as e:
        print("❌ LLM ROUTE ERROR:", e)
        print("Item:", item)
        print("Scene:", scene)
        print("Query:", user_query)

        # -----------------------------
        # SAFE FALLBACK (still usable)
        # -----------------------------
        return jsonify({
            "filters": {
                "category": category,
                "color": None,
                "pattern": item.get("pattern"),
                "sleeve_length": item.get("sleeve_length"),
                "style": None,
                "price_max": None
            },
            "llm_error": str(e)
        }), 200


# --------------------------------------------------
# PRODUCT RETRIEVAL ROUTE
# --------------------------------------------------
@app.route("/search", methods=["POST"])
def search_api():
    body = request.get_json() or {}

    filters = body.get("filters", {})
    detected_category = body.get("detected_category")

    # 🔒 Category enforcement (FINAL GUARANTEE)
    if not filters.get("category"):
        filters["category"] = detected_category

    # -----------------------------
    # Sanity cleanup (avoid junk)
    # -----------------------------
    clean_filters = {
        k: v for k, v in filters.items()
        if v is not None and v != ""
    }

    # -----------------------------
    # Search using progressive retrieval
    # -----------------------------
    products = search_products(clean_filters)

    return jsonify({
        "products": products,
        "applied_filters": clean_filters
    })


# --------------------------------------------------
# APP ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
