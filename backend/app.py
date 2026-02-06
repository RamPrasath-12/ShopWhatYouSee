import base64, cv2, numpy as np, json
from models.yolo_detector import YoloDetector
from config import YOLO_MODELS,YOLO_CONF_THRESH
from models.agman_extractor import process_crop_base64
from models.scene_context import SceneContextDetector
from models.gemini_reasoner import GeminiReasoner
from flask_cors import CORS,cross_origin

from models.product_retrieval import search_products
from flask import Flask, request, jsonify  , send_from_directory
from utils.color_utils import hex_to_color_name
import json
from flask import request, jsonify
import threading
import sys

# Force unbuffered output for Windows consoles
sys.stdout.reconfigure(line_buffering=True)

# ================================================
# PRELOAD PHI-3 IN BACKGROUND AT STARTUP - DISABLED FOR MEMORY SAFETY
# ================================================
# threading.Thread(target=preload_phi3, daemon=True).start()
# print("[Startup] [WARN] Phi-3 preload disabled to save RAM as per request")
from models.rule_parser import extract_numeric_filters



from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import os

# Define data directory relative to backend
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Centralized LLM Manager (Global to persist cache)
from models.unified_llm import UnifiedLLM
unified_llm_instance = UnifiedLLM()

# Serve dataset images - Route 1: /static/images/
@app.route('/static/images/<path:filename>')
def serve_static_images(filename):
    return send_from_directory(IMAGES_DIR, filename)

# Serve dataset images - Route 2: /images/ (preferred)
@app.route('/images/<path:filename>')
def serve_images(filename):
    print(f"[Images] Serving: {filename} from {IMAGES_DIR}")
    return send_from_directory(IMAGES_DIR, filename)

scene_detector = SceneContextDetector()

llm = GeminiReasoner()
yolo = YoloDetector(YOLO_MODELS) 

# Pre-load removed to save memory. LLM is lazy-loaded in unified_llm.py
# try:
#     from models.local_llm_efficient import preload_llm
#     print("[Startup] Pre-loading Local LLM...")
#     preload_llm()
# except Exception as e:
#     print(f"[Startup] LLM preload skipped: {e}")

def b64_to_cv2(img_b64):
    header, data = img_b64.split(',', 1)
    imgbytes = base64.b64decode(data)
    arr = np.frombuffer(imgbytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# @app.route('/detect', methods=['POST', 'OPTIONS'])
# def detect():
#     # Preflight
#     if request.method == 'OPTIONS':
#         return jsonify({"message": "CORS OK"}), 200

#     data = request.get_json()
#     img_b64 = data.get("image")

#     if not img_b64:
#         return jsonify({"error": "No image received"}), 400

#     frame = b64_to_cv2(img_b64)

#     detections = yolo.infer(frame)
#     detections = [d for d in detections if d['conf'] >= YOLO_CONF_THRESH]

#     return jsonify({"detections": detections})



###############
#yolo review 2 code
###############

# --------------------------------------------------
# LOGGING HELPER
# --------------------------------------------------
def log_step(step_name, data):
    print("\n" + "="*50)
    print(f"[BACKEND] STEP: {step_name}")
    print("="*50)
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)
    print("="*50 + "\n")

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    # Preflight
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS OK"}), 200

    data = request.get_json()
    img_b64 = data.get("image")

    if not img_b64:
        return jsonify({"error": "No image received"}), 400

    log_step("YOLO DETECTION REQUEST RECEIVED", "Processing image...")
    
    frame = b64_to_cv2(img_b64)

    # MEMORY OPTIMIZATION: Ensure loaded, infer, then UNLOAD
    yolo.ensure_loaded()
    
    try:
        detections = yolo.infer(frame)
        detections = [d for d in detections if d['conf'] >= YOLO_CONF_THRESH]
    except Exception as e:
        print(f"Detection failed: {e}")
        detections = []
    finally:
        # CRITICAL: Unload YOLO immediately to free 1GB+ RAM for LLM
        yolo.unload()

    log_step("YOLO DETECTION RESULT", f"Found {len(detections)} items")
    for i, d in enumerate(detections):
        print(f"  [{i}] {d['class']} ({d['conf']:.2f})")

    return jsonify({"detections": detections})

@app.route('/extract-attributes', methods=['POST'])
def extract_attributes():
    data = request.get_json()
    b64 = data.get("image")
    category = data.get("category")
    
    log_step("AGMAN EXTRACTION REQUEST", f"Category: {category}")
    
    if not b64:
        return jsonify({"error":"no image provided"}), 400
    if not category:
        return jsonify({"error":"no category provided"}), 400
        
    result = process_crop_base64(b64, category)
    
    log_step("AGMAN RESULT", result.get('attributes'))
    
    return jsonify(result)

@app.route('/scene', methods=['POST'])
def scene():
    data = request.get_json()
    b64 = data.get("image")
    
    log_step("SCENE DETECTION REQUEST", "Processing...")
    
    if not b64:
        return jsonify({"error":"no frame provided"}), 400
    result = scene_detector.infer(b64)
    
    log_step("SCENE RESULT", result)
    
    return jsonify(result)

# --------------------------------------------------
# LLM INTENT REASONING ROUTE
# --------------------------------------------------
@app.route("/llm", methods=["POST"])
def llm_route():
    body = request.get_json() or {}

    item = body.get("item", {})
    scene = body.get("scene")
    user_query = body.get("user_query") or ""
    # Accept session_history from frontend (with fallback to 'history' for compatibility)
    history = body.get("session_history") or body.get("history") or []

    log_step("LLM REASONING REQUEST", {
        "user_query": user_query,
        "history_depth": len(history),
        "session_history": history[-3:] if history else [],  # Show last 3 queries
        "visual_attributes": item,
        "scene": scene
    })

    # Category is ALWAYS locked (YOLO output)
    category = item.get("category")

    # ---------- RULE-BASED NUMERIC PARSING ----------
    numeric_filters = extract_numeric_filters(user_query)

    # Remove numeric intent before sending to LLM
    clean_user_query = user_query
    if numeric_filters:
        clean_user_query = (
            "Ignore price and numbers. "
            "Focus only on visual style, formality, and clothing attributes."
        )

    try:
        # Scene as string (use scene label if provided)
        scene_str = scene if isinstance(scene, str) else (
            scene.get("scene_label") if isinstance(scene, dict) else str(scene or "unknown")
        )

        # AG-MAN attributes (visual truth) - include color_name if available
        agman_attributes = {
            "color_hex": item.get("color_hex"),
            "color_name": item.get("color_name"),  # Pre-mapped color name
            "pattern": item.get("pattern"),
            "sleeve": item.get("sleeve_length")
        }

        # ---------- SCENE CONTEXT (Passed to LLM) ----------
        print(f"[Scene Context] Input to LLM: {scene_str}")

        # ---------- UNIFIED LLM REASONING (Phi-3 Local -> Groq Fallback) ----------
        from models.unified_llm import generate_filters
        
        normalized_history = history if isinstance(history, list) else []
        
        # Unified call - handles local/external switching internally
        llm_result = generate_filters(
            category=category,
            attributes=agman_attributes,
            scene=scene_str,
            query=clean_user_query,
            session_history=normalized_history,
            prefer_external=False # PRIORITIZE LOCAL PHI-3
        )
        
        filters = llm_result.get("filters", {})
        confidence = llm_result.get("confidence", 0.0)
        source = llm_result.get("source", "unknown")
        
        log_step(f"LLM RESULT ({source.upper()})", llm_result)

        # ---------- SMART FILTER MERGE ----------
        # Use LLM category if provided and user explicitly asked for it, else use YOLO
        llm_category = filters.get("category")
        if llm_category and llm_category.lower() != category.lower():
            # LLM detected user wants different category - use it
            print(f"[Category Override] YOLO: {category} -> LLM: {llm_category}")
            filters["category"] = llm_category
        else:
            filters["category"] = category
        
        # Ensure AGMAN attributes are in filters if not already set by LLM
        if agman_attributes.get("color_hex") and not filters.get("color"):
            filters["color"] = agman_attributes["color_hex"]
        if agman_attributes.get("color_name") and not filters.get("color_name"):
            filters["color_name"] = agman_attributes["color_name"]
        if agman_attributes.get("pattern") and not filters.get("pattern"):
            filters["pattern"] = agman_attributes["pattern"]
        if agman_attributes.get("sleeve") and not filters.get("sleeve"):
            filters["sleeve"] = agman_attributes["sleeve"]
            
        filters.update(numeric_filters) # numeric via rules only

        log_step("LLM FINAL FILTERS", filters)

        return jsonify({
            "filters": filters,
            "confidence": confidence,
            "llm_source": source,
            "llm_failed": False,
            "llm_error": None
        })

    except Exception as e:
        print(f"❌ LLM ERROR: {e}")
        # ---------- HARD FAIL SAFE FALLBACK ----------
        fallback_filters = {
            "category": category,
            "color_hex": item.get("color_hex"),
            "pattern": item.get("pattern"),
            "sleeve": item.get("sleeve_length")
        }
        fallback_filters.update(numeric_filters)

        return jsonify({
            "filters": fallback_filters,
            "confidence": 0.0,
            "llm_failed": True,
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
    embedding = body.get("embedding")  # For FAISS similarity search

    log_step("SEARCH REQUEST", {
        "filters": filters,
        "has_embedding": embedding is not None and len(embedding) > 0 if embedding else False,
        "embedding_dim": len(embedding) if embedding else 0
    })

    # CRITICAL MEMORY SAFETY: Unload YOLO before loading LLM
    # In case detection didn't unload it or previous request left it
    try:
        yolo.unload()
        import gc, time
        gc.collect()
        time.sleep(1.0) # Give OS time to reclaim memory
    except:
        pass

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
    
    # Add embedding to filters for FAISS search
    if embedding and len(embedding) > 0:
        clean_filters["embedding"] = embedding
        print(f"[Search] 🧬 Using embedding ({len(embedding)} dims) for FAISS similarity")

    # -----------------------------
    # Search using progressive retrieval
    # -----------------------------
    products = search_products(clean_filters)
    
    log_step("SEARCH RESULTS", f"Found {len(products)} products")

    return jsonify({
        "products": products,
        "applied_filters": {k: v for k, v in clean_filters.items() if k != "embedding"}  # Don't send embedding back
    })



# --------------------------------------------------
# APP ENTRY
# --------------------------------------------------

# --------------------------------------------------
# RATING COLLECTION ENDPOINT
# --------------------------------------------------
@app.route("/rating", methods=["POST"])
def rating_route():
    """Collect user ratings for ML/UX improvement"""
    from rating_system import save_rating
    
    data = request.get_json() or {}
    
    # Validate rating
    rating = data.get("rating")
    if not rating or not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({"error": "Rating must be 1-5"}), 400
    
    success = save_rating(data)
    
    if success:
        return jsonify({"message": "Rating saved", "rating": rating})
    else:
        return jsonify({"error": "Failed to save rating"}), 500

# --------------------------------------------------
# INSIGHTS ENDPOINT FOR STAKEHOLDERS
# -------------------------------------------------- 
@app.route("/insights", methods=["GET"])
def insights_route():
    """Get e-commerce insights + Deep LLM Analysis based on user ratings"""
    from rating_system import get_insights, get_latest_rating
    from models.insights_engine import InsightsEngine
    
    # 1. General Stats
    stats = get_insights()
    
    # 2. Deep Analysis of Latest Session
    latest = get_latest_rating() or {}
    print(f"[Insights] Analyzing latest session: {latest.get('id', 'None')}")
    
    engine = InsightsEngine(unified_llm_instance)
    analysis = engine.generate_report(latest)
    
    return jsonify({
        "stats": stats,
        "analysis": analysis,
        "session_data": latest
    })


if __name__ == "__main__":
    # use_reloader=False prevents double model loading crash
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

