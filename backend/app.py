import base64, cv2, numpy as np, json, uuid, logging
from collections import deque
from models.yolo_detector import YoloDetector
from config import YOLO_MODELS, YOLO_CONF_THRESH
from models.agman_extractor import process_crop_base64
from models.scene_context import SceneContextDetector
from models.gemini_reasoner import GeminiReasoner
from flask_cors import CORS, cross_origin

from models.product_retrieval import search_products_v2
from services.filter_schema import filter_schema
from flask import Flask, request, jsonify, send_from_directory, g
from utils.color_utils import hex_to_color_name
import threading
import sys
import re
import time as _time

# Force unbuffered output for Windows consoles
sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────────────────────────────────
# STRUCTURED LOGGING + ROLLING METRICS
# ──────────────────────────────────────────────────
_request_log = deque(maxlen=500)   # Rolling window of last 500 requests
_log_lock = threading.Lock()

# Structured JSON logger (file + console)
_structured_logger = logging.getLogger("shopwhatyousee")
_structured_logger.setLevel(logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(message)s"))
if not _structured_logger.handlers:
    _structured_logger.addHandler(_sh)

# ================================================
# PRELOAD PHI-3 IN BACKGROUND AT STARTUP - DISABLED FOR MEMORY SAFETY
# ================================================
# threading.Thread(target=preload_phi3, daemon=True).start()



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

# ==================================================
# DEMO-ONLY: In-memory session persistence.
# For production, replace with Redis or DB-backed sessions.
# ==================================================
SESSION_FILTERS = {}   # session_id -> {"filters": {...}, "hard": {...}, "soft": {...}, "timestamp": float}
MAX_SESSIONS = 1000
SESSION_TTL_SECONDS = 1800  # 30 minutes

# Word-boundary regex for gender detection in user query
_GENDER_REGEX = re.compile(
    r'\b(men|women|male|female|boy|girl|boys|girls|mens|womens|man|woman)\b',
    re.IGNORECASE
)
_GENDER_KEYWORD_MAP = {
    "men": "Men", "mens": "Men", "male": "Men", "man": "Men",
    "women": "Women", "womens": "Women", "female": "Women", "woman": "Women",
    "boy": "Boys", "boys": "Boys",
    "girl": "Girls", "girls": "Girls",
}


def _cleanup_sessions():
    """Remove expired sessions. Evict oldest if exceeding MAX_SESSIONS."""
    now = _time.time()
    # Remove expired
    expired = [sid for sid, data in SESSION_FILTERS.items()
               if now - data.get("timestamp", 0) > SESSION_TTL_SECONDS]
    for sid in expired:
        del SESSION_FILTERS[sid]
    # Evict oldest if still over limit
    while len(SESSION_FILTERS) > MAX_SESSIONS:
        oldest_sid = min(SESSION_FILTERS, key=lambda s: SESSION_FILTERS[s].get("timestamp", 0))
        del SESSION_FILTERS[oldest_sid]


def _detect_gender_from_query(query):
    """Extract gender from user query using word-boundary regex.
    Returns DB-canonical gender string or None.
    """
    if not query:
        return None
    match = _GENDER_REGEX.search(query)
    if match:
        keyword = match.group(1).lower()
        return _GENDER_KEYWORD_MAP.get(keyword)
    return None

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

# Warmup YOLO: run a dummy inference to pre-load CPU/CUDA kernels.
# This ensures the first real request gets fast inference, not cold-start.
try:
    _warmup_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = yolo.infer(_warmup_frame)
    del _warmup_frame
    print("[YOLO] Warmup complete — models resident in memory")
except Exception as e:
    print(f"[YOLO] Warmup failed (non-critical): {e}")

# LLM is lazy-loaded in unified_llm.py (Gemini API, not local)

def b64_to_cv2(img_b64):
    header, data = img_b64.split(',', 1)
    imgbytes = base64.b64decode(data)
    arr = np.frombuffer(imgbytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


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

    # YOLO singleton — models stay loaded, no ensure_loaded/unload cycle
    try:
        detections = yolo.infer(frame)
        detections = [d for d in detections if d['conf'] >= YOLO_CONF_THRESH]
    except Exception as e:
        print(f"Detection failed: {e}")
        detections = []

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
# LLM INTENT REASONING ROUTE (Phase 2 - Flat Filters)
# --------------------------------------------------
# Pipeline order:
#   1. LLM call (Groq, flat JSON)
#   2. Price normalization (numeric -> bucket)
#   3. filter_schema.validate() (reject invalid values)
#   4. Category precedence (YOLO > LLM for image search)
#   5. Return validated flat filters
# --------------------------------------------------
@app.route("/llm", methods=["POST"])
def llm_route():
    body = request.get_json() or {}

    # Accept both key names
    item = body.get("item") or body.get("visual_attributes", {})
    scene = body.get("scene")
    user_query = body.get("user_query") or ""
    history = body.get("session_history") or body.get("history") or []
    is_image_search = body.get("is_image_search", False)

    # YOLO-detected category (from image)
    yolo_category = item.get("category")

    log_step("LLM REQUEST (Phase 2)", {
        "user_query": user_query,
        "yolo_category": yolo_category,
        "is_image_search": is_image_search,
        "history_depth": len(history),
    })

    try:
        # Scene as string
        scene_str = scene if isinstance(scene, str) else (
            scene.get("scene_label") if isinstance(scene, dict) else str(scene or "unknown")
        )

        # AG-MAN attributes for context
        agman_attributes = {
            "color_hex": item.get("color_hex"),
            "color_name": item.get("color_name"),
            "pattern": item.get("pattern"),
            "sleeve": item.get("sleeve_length")
        }

        # ---- Step 1: LLM Call ----
        from models.unified_llm import generate_filters
        normalized_history = history if isinstance(history, list) else []

        llm_result = generate_filters(
            category=yolo_category,
            attributes=agman_attributes,
            scene=scene_str,
            query=user_query,
            session_history=normalized_history,
        )

        raw_filters = llm_result.get("filters", {})
        price_max = llm_result.get("price_max")
        confidence = llm_result.get("confidence", 0.0)
        source = llm_result.get("source", "unknown")

        log_step(f"LLM RAW ({source.upper()})", {
            "filters": raw_filters,
            "price_max": price_max,
            "confidence": confidence
        })

        # ---- Step 2: Price Normalization ----
        # Convert numeric price_max to price_bucket (deterministic)
        if price_max is not None and "price_bucket" not in raw_filters:
            bucket = filter_schema.price_to_bucket(price_max)
            if bucket:
                raw_filters["price_bucket"] = bucket
                print(f"  [Price] {price_max} -> {bucket}")

        # ---- Step 3: Validate against DB-sourced allowed values ----
        validated_filters = filter_schema.validate(raw_filters)

        # ---- Step 4: Category Precedence ----
        llm_category = validated_filters.get("category")

        if is_image_search and yolo_category:
            # Image search: YOLO wins.
            # LLM override only if confidence > 0.9 AND same category group.
            if (llm_category
                    and llm_category != yolo_category
                    and confidence > 0.9
                    and filter_schema.same_category_group(llm_category, yolo_category)):
                validated_filters["category"] = llm_category
                print(f"  [Category] LLM override accepted: {yolo_category} -> {llm_category} (conf={confidence})")
            else:
                validated_filters["category"] = yolo_category
                if llm_category and llm_category != yolo_category:
                    print(f"  [Category] LLM override REJECTED: LLM={llm_category}, YOLO={yolo_category} (conf={confidence})")
        elif llm_category:
            # Text-only: LLM defines category
            validated_filters["category"] = llm_category
        # If neither specifies category, do NOT inject one

        log_step("LLM VALIDATED FILTERS", validated_filters)

        # ---- Log confidence for monitoring ----
        print(f"  [LLM Confidence] {confidence} (source={source})")

        return jsonify({
            "filters": validated_filters,
            "price_max": price_max,
            "confidence": confidence,
            "llm_source": source,
            "reasoning": llm_result.get("reasoning", ""),
            "llm_failed": False,
            "llm_error": None
        })

    except Exception as e:
        print(f"LLM ERROR: {e}")
        # Hard fail-safe: return YOLO category only
        fallback = {}
        if yolo_category:
            fallback["category"] = yolo_category

        return jsonify({
            "filters": fallback,
            "confidence": 0.0,
            "llm_source": "error",
            "reasoning": "",
            "llm_failed": True,
            "llm_error": str(e)
        }), 200


# --------------------------------------------------
# DEBUG: Filter Schema Inspector
# --------------------------------------------------
@app.route("/debug/filter-schema", methods=["GET"])
def debug_filter_schema():
    """Inspect loaded allowed values for debugging. Service restart required if DB changes."""
    if not filter_schema.is_ready:
        return jsonify({"error": "FilterSchema not initialized"}), 503
    return jsonify(filter_schema.get_debug_info())


# --------------------------------------------------
# PRODUCT RETRIEVAL ROUTE (via product_retrieval.py)
# --------------------------------------------------
@app.route("/search", methods=["POST"])
def search_api():
    body = request.get_json() or {}

    embedding = body.get("embedding")
    filters = body.get("filters", {})
    detected_category = body.get("detected_category")
    target_category = body.get("target_category")
    top_k = body.get("top_k", 20)
    price_max = body.get("price_max")  # Numeric price cap from LLM
    
    # Detected attributes from AG-MAN (original visual detection)
    detected_attributes = body.get("detected_attributes", {})
    
    # Ensure detected_attributes has at least the category
    if not detected_attributes.get("category") and detected_category:
        detected_attributes["category"] = detected_category

    # Build user_filters from various sources
    # Priority: explicit user_filters > flat filters > hard/soft constraints
    user_filters = body.get("user_filters", {})
    if not user_filters:
        # Convert legacy hard_constraints / soft_preferences to flat user_filters
        hc = body.get("hard_constraints", {})
        sp = body.get("soft_preferences", {})
        if hc or sp:
            for key, val in {**hc, **sp}.items():
                if isinstance(val, dict):
                    user_filters[key] = val.get("value", "")
                elif val:
                    user_filters[key] = val
        elif filters:
            # Fallback to flat filters dict
            user_filters = {k: v for k, v in filters.items() if v}
    
    # Category override: target_category > user_filters.category > filters.category
    if target_category:
        user_filters["category"] = target_category
    elif not user_filters.get("category") and filters.get("category"):
        user_filters["category"] = filters["category"]

    log_step("SEARCH REQUEST", {
        "has_embedding": embedding is not None and len(embedding) > 0 if embedding else False,
        "detected_attrs": detected_attributes,
        "user_filters": {k: v for k, v in user_filters.items() if v},
        "top_k": top_k
    })

    if not embedding or len(embedding) == 0:
        return jsonify({"error": "No embedding provided"}), 400

    # Build query_context for new retrieval v2
    effective_category = user_filters.get("category") or detected_category
    
    query_context = {
        "category": effective_category,
        "embedding": embedding,
        "detected_attributes": detected_attributes,
        "user_filters": user_filters,
        "price_max": price_max,
    }

    t0 = _time.time()
    result = search_products_v2(query_context, top_k=top_k)
    search_ms = (_time.time() - t0) * 1000

    products = result.get("products", []) if isinstance(result, dict) else result
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}

    log_step("SEARCH RESULTS", f"Found {len(products)} products in {search_ms:.1f}ms")

    return jsonify({
        "products": products,
        "metadata": {
            "result_count": len(products),
            "search_time_ms": round(search_ms, 1),
            "mode": metadata.get("mode", ""),
            "weights": metadata.get("weights", {}),
            "overrides": metadata.get("overrides", {}),
            "pool_size": metadata.get("pool_size", 0),
        },
        "retrieval_meta": {
            "query_type": "text",
            "overrides": metadata.get("overrides", {}),
            "preserved_attrs": metadata.get("preserved_attrs", {}),
            "relaxation_log": metadata.get("relaxation_log", []),
            "top_5_categories": metadata.get("top_5_categories", []),
        }
    })


# --------------------------------------------------
# SEARCH BY IMAGE — Full Pipeline Endpoint
# --------------------------------------------------
@app.route("/search-by-image", methods=["POST"])
def search_by_image():
    """End-to-end: image -> YOLO -> AG-MAN -> FAISS -> results."""
    body = request.get_json() or {}
    img_b64 = body.get("image")
    user_filters = body.get("filters", {})
    top_k = body.get("top_k", 20)

    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    timings = {}

    # -- Step 1: YOLO detection --------------------------------
    t0 = _time.time()
    frame = b64_to_cv2(img_b64)
    try:
        detections = yolo.infer(frame)
        detections = [d for d in detections if d["conf"] >= YOLO_CONF_THRESH]
    except Exception as e:
        print(f"[search-by-image] YOLO error: {e}")
        detections = []
    timings["yolo_ms"] = round((_time.time() - t0) * 1000, 1)

    if not detections:
        return jsonify({"error": "No fashion items detected", "timings": timings}), 400

    # Use highest-confidence detection
    best = max(detections, key=lambda d: d["conf"])
    detected_category = best["class"]

    # P4: Debug log ALL detections for selection validation
    all_dets = [{"class": d["class"], "conf": round(d["conf"], 3)} for d in detections]

    log_step("SEARCH-BY-IMAGE: YOLO", {
        "detected": detected_category,
        "confidence": best["conf"],
        "all_detections": len(detections),
        "all_classes": all_dets,
        "selection": "highest_confidence",
        "yolo_ms": timings["yolo_ms"]
    })

    # -- Step 2: AG-MAN extraction -----------------------------
    t1 = _time.time()

    # YOLO already provides cropped_image as base64 — use it directly
    if "cropped_image" in best and best["cropped_image"]:
        crop_b64 = best["cropped_image"]
    else:
        # Fallback: manual crop using x1/y1/x2/y2 (absolute pixel coordinates)
        x1 = int(best.get("x1", 0))
        y1 = int(best.get("y1", 0))
        x2 = int(best.get("x2", frame.shape[1]))
        y2 = int(best.get("y2", frame.shape[0]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame  # fallback to full image
        _, buf = cv2.imencode(".jpg", crop)
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    agman_result = process_crop_base64(crop_b64, detected_category)
    embedding = agman_result["embedding"]
    timings["agman_ms"] = round((_time.time() - t1) * 1000, 1)

    log_step("SEARCH-BY-IMAGE: AG-MAN", {
        "embedding_dim": len(embedding),
        "agman_ms": timings["agman_ms"]
    })

    # -- Step 3: Retrieval -------------------------------------
    # Merge detected category into filters (unless user overrides)
    filters = dict(user_filters)
    if "category" not in filters:
        filters["category"] = detected_category

    # Build query_context for search_products_v2 (new v2 interface)
    # For image search: AG-MAN detection = detected_attributes, user filters = user_filters
    detected_attributes = {
        "category": detected_category,
    }
    
    query_context = {
        "category": filters.get("category", detected_category),
        "embedding": embedding,
        "detected_attributes": detected_attributes,
        "user_filters": {k: v for k, v in filters.items() if v},
    }

    t2 = _time.time()
    result = search_products_v2(query_context, top_k=top_k)
    products = result.get("products", []) if isinstance(result, dict) else result
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    timings["faiss_ms"] = round((_time.time() - t2) * 1000, 1)
    timings["total_ms"] = round(timings["yolo_ms"] + timings["agman_ms"] + timings["faiss_ms"], 1)

    log_step("SEARCH-BY-IMAGE: RESULTS", {
        "result_count": len(products),
        "detected_category": detected_category,
        "filters": filters,
        "timings": timings,
    })

    return jsonify({
        "products": products,
        "detected_category": detected_category,
        "detection_confidence": best["conf"],
        "timings": timings,
        "metadata": {
            "result_count": len(products),
            "filters_applied": {k: v for k, v in filters.items() if v},
        },
        "retrieval_meta": {
            "query_type": "image",
            "yolo_class": detected_category,
            "yolo_confidence": round(best["conf"], 3),
            "applied_filters": {k: v for k, v in filters.items() if v},
            "relaxation_log": metadata.get("relaxation_log", []),
        }
    })


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

# --------------------------------------------------
# STRUCTURED LOGGING MIDDLEWARE
# --------------------------------------------------
@app.before_request
def _before_request():
    """Attach request_id and start time to each request."""
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = _time.time()

@app.after_request
def _after_request(response):
    """Log structured JSON for every request. Store in rolling deque."""
    latency_ms = round((_time.time() - getattr(g, 'start_time', _time.time())) * 1000, 1)
    request_id = getattr(g, 'request_id', 'unknown')

    # Build structured log entry
    entry = {
        "request_id": request_id,
        "endpoint": request.path,
        "method": request.method,
        "latency_ms": latency_ms,
        "status_code": response.status_code,
    }

    # Try to extract filter/relaxation info from response
    try:
        if response.content_type and "json" in response.content_type:
            data = response.get_json(silent=True)
            if data and isinstance(data, dict):
                rmeta = data.get("retrieval_meta", {})
                entry["filter_count"] = len(rmeta.get("applied_filters", {}))
                entry["relaxation_steps"] = rmeta.get("relaxation_steps", 0)
    except Exception:
        pass

    # Log as structured JSON (avoid plain text)
    _structured_logger.info(json.dumps(entry))

    # Store in rolling window (thread-safe)
    with _log_lock:
        _request_log.append(entry)

    return response

# --------------------------------------------------
# METRICS ENDPOINT — Rolling Window (deque 500)
# --------------------------------------------------
@app.route("/metrics", methods=["GET"])
def metrics_route():
    """Return rolling-window metrics from last 500 requests."""
    from utils.retrieval_metrics import get_metrics
    m = get_metrics()
    aggregate = m.get_summary()

    # Rolling window stats
    with _log_lock:
        entries = list(_request_log)

    total = len(entries)
    if total == 0:
        rolling = {
            "window_size": 0,
            "avg_latency_ms": 0,
            "relaxation_frequency_pct": 0,
            "empty_filter_rate_pct": 0,
            "endpoints": {},
        }
    else:
        latencies = [e["latency_ms"] for e in entries]
        relaxation_count = sum(1 for e in entries if e.get("relaxation_steps", 0) > 0)
        empty_filter_count = sum(1 for e in entries if e.get("filter_count", 0) == 0)

        # Per-endpoint breakdown
        endpoint_stats = {}
        for e in entries:
            ep = e["endpoint"]
            if ep not in endpoint_stats:
                endpoint_stats[ep] = {"count": 0, "total_ms": 0}
            endpoint_stats[ep]["count"] += 1
            endpoint_stats[ep]["total_ms"] += e["latency_ms"]
        for ep in endpoint_stats:
            s = endpoint_stats[ep]
            s["avg_ms"] = round(s["total_ms"] / s["count"], 1)
            del s["total_ms"]

        rolling = {
            "window_size": total,
            "avg_latency_ms": round(sum(latencies) / total, 1),
            "p95_latency_ms": round(sorted(latencies)[int(total * 0.95)], 1) if total > 1 else round(latencies[0], 1),
            "relaxation_frequency_pct": round(relaxation_count / total * 100, 1),
            "empty_filter_rate_pct": round(empty_filter_count / total * 100, 1),
            "endpoints": endpoint_stats,
        }

    return jsonify({
        "aggregate": aggregate,
        "rolling": rolling,
    })


if __name__ == "__main__":
    # Initialize services at startup
    print("[Startup] Initializing Filter Schema...")
    filter_schema.init()
    print("SERVER READY -- product_retrieval.py + Filter Schema")
    # use_reloader=False prevents double model loading crash
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
