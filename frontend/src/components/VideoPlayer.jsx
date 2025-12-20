import React, { useRef, useState } from "react";
import axios from "axios";
const moduleBox = {
  background: "#f9f9f9",
  padding: "16px",
  borderRadius: "10px",
  border: "1px solid #ddd",
  marginTop: "20px",
  maxWidth: "700px"
};

const moduleTitle = {
  marginBottom: "10px",
  color: "#333"
};

const buttonStyle = {
  padding: "8px 16px",
  background: "#1976d2",
  color: "#fff",
  border: "none",
  borderRadius: "6px",
  cursor: "pointer"
};


function VideoPlayer() {
  const videoRef = useRef(null);

  // ---------------- STATES ----------------
  const [capturedFrame, setCapturedFrame] = useState(null);

  const [detections, setDetections] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);

  const [attributes, setAttributes] = useState(null);
  const [embedding, setEmbedding] = useState(null);

  const [scene, setScene] = useState(null);

  const [userText, setUserText] = useState("");
  const [llmOutput, setLlmOutput] = useState(null);

  const [products, setProducts] = useState([]);
  const [attrLoading, setAttrLoading] = useState(false);

  // =================================================
  // MODULE 1: FRAME CAPTURE (RAW FRAME)
  // =================================================
  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      console.error("Video not ready");
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frameBase64 = canvas.toDataURL("image/jpeg", 0.9);

    console.log("Frame captured", {
      width: canvas.width,
      height: canvas.height,
      timestamp: video.currentTime.toFixed(2)
    });

    setCapturedFrame(frameBase64);
    return frameBase64;
  };

  // =================================================
  // MODULE 2: YOLOv8 DETECTION
  // =================================================
  const detectItems = async () => {
    const frameB64 = captureFrame();
    if (!frameB64) return;

    try {
      const res = await axios.post("http://localhost:5000/detect", {
        image: frameB64
      });

      setDetections(res.data.detections || []);

      // reset downstream states
      setSelectedItem(null);
      setAttributes(null);
      setEmbedding(null);
      setScene(null);
      setLlmOutput(null);
      setProducts([]);
      setUserText("");
    } catch (err) {
      console.error("DETECTION ERROR:", err);
    }
  };

  // =================================================
  // MODULE 3: AG-MAN ATTRIBUTE EXTRACTION
  // =================================================
  const handleItemClick = async (item) => {
    setSelectedItem(item);
    setAttributes(null);
    setEmbedding(null);
    setLlmOutput(null);
    setProducts([]);
    setUserText("");

    setAttrLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/extract-attributes", {
        image: item.cropped_image
      });

      setAttributes(res.data.attributes);
      setEmbedding(res.data.embedding);
    } catch (err) {
      console.error("AG-MAN ERROR:", err);
    }

    setAttrLoading(false);
  };

  // =================================================
  // MODULE 4: SCENE CONTEXT (PLACES365)
  // =================================================
  const detectScene = async () => {
    const frameB64 = captureFrame();
    if (!frameB64) return;

    try {
      const res = await axios.post("http://localhost:5000/scene", {
        image: frameB64
      });
      setScene(res.data);
    } catch (err) {
      console.error("SCENE ERROR:", err);
    }
  };

  // =================================================
  // MODULE 5: LLM INTENT REASONING
  // =================================================
 
const sendToLLM = async () => {
  if (!selectedItem || !attributes) {
    alert("Select an item first");
    return;
  }

  // AUTO MODE if input empty
  const query =
    userText && userText.trim().length > 0
      ? userText.trim()
      : null;

  const payload = {
    item: {
      category: selectedItem.class.toLowerCase(), // 🔒 YOLO ground truth
      color_hex: attributes.color_hex,
      pattern: attributes.pattern,
      sleeve_length: attributes.sleeve_length
    },
    scene: scene || null,        // full scene object
    user_query: query            // null → AUTO MODE
  };

  try {
    const res = await axios.post(
      "http://localhost:5000/llm",
      payload
    );

    console.log("LLM OUTPUT:", res.data);
    setLlmOutput(res.data);

  } catch (err) {
    console.error("LLM ERROR:", err);
    alert("LLM failed");
  }
};

  // =================================================
  // MODULE 6: PRODUCT RETRIEVAL
  // =================================================
  
const searchProducts = async () => {
  if (!selectedItem) {
    alert("Select an item first");
    return;
  }

  if (!llmOutput || !llmOutput.filters) {
    alert("Generate filters using LLM first");
    return;
  }

  const payload = {
    filters: llmOutput.filters,              // 🔥 LLM decides
    detected_category: selectedItem.class.toLowerCase()
  };

  console.log("SEARCH PAYLOAD:", payload);

  try {
    const res = await axios.post(
      "http://localhost:5000/search",
      payload
    );

    setProducts(res.data.products || []);

  } catch (err) {
    console.error("PRODUCT SEARCH ERROR:", err);
  }
};

  // =================================================
  // UI
  // =================================================
  return (
  <div
  style={{
    padding: "24px",
    maxWidth: "1100px",
    margin: "0 auto",
    fontFamily: "Arial, sans-serif"
  }}
>
  <h2
    style={{
     // textAlign: "center",
      marginBottom: "25px",
      color: "#222",
      letterSpacing: "0.5px"
    }}
  >
    ShopWhatYouSee – Review-1 Prototype
  </h2>

    {/* VIDEO INPUT */}
    <video
      ref={videoRef}
      src="/sample.mp4"
      controls
      autoPlay
      muted
      style={{ width: "600px", borderRadius: "10px", border: "1px solid #ccc" }}
    />

    {/* MODULE 1: FRAME CAPTURE */}
    {capturedFrame && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}>Frame Capture</h3>
        <img
          src={capturedFrame}
          alt="Captured Frame"
          style={{ width: "320px", borderRadius: "6px", border: "1px solid #444" }}
        />
        {/* <p style={{ fontSize: "13px", marginTop: "6px" }}>
          Raw video frame captured and sent to backend for processing.
        </p> */}
      </div>
    )}

    {/* ACTION BUTTONS */}
    <div style={{ marginTop: "15px" }}>
      <button style={buttonStyle} onClick={detectItems}>
        Detect Fashion Items
      </button>
      <button
        style={{ ...buttonStyle, marginLeft: "10px", background: "#388e3c" }}
        onClick={detectScene}
      >
        Identify Scene
      </button>
    </div>

    {/* MODULE 2: YOLOv8 DETECTION */}
    {detections.length > 0 && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}> YOLOv8 Fashion Detection</h3>

        <div style={{ display: "flex", gap: "15px", flexWrap: "wrap" }}>
          {detections.map((item, idx) => (
            <div key={idx} style={{ textAlign: "center" }}>
              <img
                src={`data:image/jpeg;base64,${item.cropped_image}`}
                alt="detected"
                style={{
                  width: "140px",
                  height: "140px",
                  border:
                    selectedItem === item
                      ? "3px solid red"
                      : "2px solid #000",
                  borderRadius: "8px",
                  cursor: "pointer"
                }}
                onClick={() => handleItemClick(item)}
              />
              <p style={{ marginTop: "6px", fontSize: "14px" }}>
                {item.class}
                <br />
                <small>
                  Confidence: {(item.conf * 100).toFixed(1)}%
                </small>
              </p>
            </div>
          ))}
        </div>
      </div>
    )}

    {/* MODULE 3: AG-MAN ATTRIBUTE EXTRACTION */}
    {attributes && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}>
          AG-MAN Attribute Extraction
        </h3>

        <p>
          <strong>Dominant Color:</strong> {attributes.color_hex}
        </p>

        <div
          style={{
            background: attributes.color_hex,
            width: "90px",
            height: "26px",
            borderRadius: "4px",
            border: "1px solid #000",
            marginBottom: "10px"
          }}
        />

        <p>
          <strong>Pattern:</strong> {attributes.pattern}
        </p>
        <p>
          <strong>Sleeve Length:</strong> {attributes.sleeve_length}
        </p>

        {embedding && (
          <p>
            <strong>Embedding Dimension:</strong> {embedding.length}
          </p>
        )}
      </div>
    )}

    {/* MODULE 4: SCENE CONTEXT */}
    {scene && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}>Scene Context Detection</h3>
        <pre style={{ fontSize: "13px" }}>
          {JSON.stringify(scene, null, 2)}
        </pre>
      </div>
    )}

    {/* MODULE 5: LLM INTENT REASONING */}
    {selectedItem && attributes && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}>LLM Intent Reasoning</h3>

        <input
          type="text"
          placeholder="Optional: refine results (e.g. under 2000, casual)"
          value={userText}
          onChange={(e) => setUserText(e.target.value)}
          style={{
            padding: "8px",
            width: "380px",
            borderRadius: "6px",
            border: "1px solid #ccc"
          }}
        />
        {/* text for llm */}
        <p style={{ fontSize: "12px", color: "#555", marginTop: "4px" }}>
  Leave empty to let AI infer style from visual attributes and scene context.
</p>


        <br />
        <br />

        <button style={buttonStyle} onClick={sendToLLM}>
          Reason & Generate Filters
        </button>

        {llmOutput && (
          <pre style={{ marginTop: "10px", fontSize: "13px" }}>
            {JSON.stringify(llmOutput, null, 2)}
          </pre>
        )}
      </div>
    )}

    {/* MODULE 6: PRODUCT RETRIEVAL */}
    {products.length > 0 && (
      <div style={moduleBox}>
        <h3 style={moduleTitle}>Products Retrieved</h3>

        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
          {products.map((p) => (
            <div
              key={p.id}
              style={{
                width: "180px",
                border: "1px solid #ccc",
                borderRadius: "8px",
                padding: "8px"
              }}
            >
              <img
                src={p.image_url}
                alt={p.name}
                style={{
                  width: "100%",
                  height: "160px",
                  objectFit: "cover",
                  borderRadius: "6px"
                }}
              />
              <p style={{ fontSize: "14px", marginTop: "6px" }}>
                {p.name}
              </p>
              <p style={{ fontWeight: "bold" }}>₹{p.price}</p>
            </div>
          ))}
        </div>
      </div>
    )}

   {llmOutput && (
  <div style={{ marginTop: "15px" }}>
    <button
      style={{ ...buttonStyle, background: "#ff5722" }}
      onClick={searchProducts}
    >
      Retrieve Matching Products
    </button>
  </div>
 )}

  </div>
);
}
export default VideoPlayer;