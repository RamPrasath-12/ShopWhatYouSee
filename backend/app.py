from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, cv2, numpy as np

from models.yolo_detector import YoloDetector
from config import YOLO_WEIGHTS, YOLO_CONF_THRESH
from models.agman_extractor import process_crop_base64
from models.scene_context import SceneContextDetector
from models.llm_reasoner import LLMReasoner
from flask_cors import CORS,cross_origin
from flask import Flask, request, jsonify
import psycopg2
from models.product_retrieval import search_products
from flask import send_from_directory


def get_db():
    return psycopg2.connect(
        host="localhost",
        database="shopwhatyousee",
        user="postgres",
        password="postgres123@"
    )







app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

scene_detector = SceneContextDetector()

llm = LLMReasoner(model_name="google/flan-t5-small")  # or the model you have loaded
yolo = YoloDetector(YOLO_WEIGHTS)

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

    detections = yolo.infer(frame)
    detections = [d for d in detections if d['conf'] >= YOLO_CONF_THRESH]

    return jsonify({"detections": detections})

@app.route('/extract-attributes', methods=['POST'])
def extract_attributes():
    data = request.get_json()
    b64 = data.get("image")
    if not b64:
        return jsonify({"error":"no image provided"}), 400
    result = process_crop_base64(b64)
    return jsonify(result)

@app.route('/scene', methods=['POST'])
def scene():
    data = request.get_json()
    b64 = data.get("image")
    if not b64:
        return jsonify({"error":"no frame provided"}), 400
    result = scene_detector.infer(b64)
    return jsonify(result)


@app.route('/llm', methods=['POST'])
def llm_route():
    body = request.get_json() or {}
    item = body.get("item", {})                # { color_hex, pattern, sleeve_length }
    scene_label = body.get("scene_label", None)
    user_query = body.get("user_query", "")    # string
    result = llm.generate_filters(item, scene_label, user_query)
    # ensure response is JSON-serializable
    return jsonify(result)

@app.route("/search", methods=["POST"])
def search_api():
    filters = request.json
    products = search_products(filters)
    return jsonify({"products": products})




if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)
