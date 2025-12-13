import React, { useRef, useState } from "react";
import axios from "axios";

function VideoPlayer() {
  const videoRef = useRef(null);

  // STATES
  const [detections, setDetections] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [attributes, setAttributes] = useState(null);
  const [embedding, setEmbedding] = useState(null);
  const [scene, setScene] = useState(null);
  const [userText, setUserText] = useState("");
  const [llmOutput, setLlmOutput] = useState(null);
  const [attrLoading, setAttrLoading] = useState(false);
  const [products, setProducts] = useState([]);

  // ------------------------------------------------
  // Capture frame as Base64
  // ------------------------------------------------
  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      console.error("VIDEO NOT READY");
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/jpeg");
  };

  // ------------------------------------------------
  // 1️⃣ YOLO DETECTION
  // ------------------------------------------------
  const detectItems = async () => {
    const frameB64 = captureFrame();
    if (!frameB64) return;

    try {
      const res = await axios.post("http://localhost:5000/detect", {
        image: frameB64,
      });

      console.log("YOLO DETECTIONS:", res.data);
      setDetections(res.data.detections || []);

      // reset states
      setSelectedItem(null);
      setAttributes(null);
      setEmbedding(null);
      setScene(null);
      setLlmOutput(null);
      setUserText("");
      
    } catch (err) {
      console.error("DETECTION ERROR:", err);
    }
  };

  // ------------------------------------------------
  // 2️⃣ AG-MAN Attribute Extraction
  // ------------------------------------------------
  const handleItemClick = async (item) => {
    setSelectedItem(item);
    setAttributes(null);
    setEmbedding(null);
    setLlmOutput(null);
    setUserText("");

    setAttrLoading(true);
    setProducts([]); 


    try {
      const res = await axios.post("http://localhost:5000/extract-attributes", {
        image: item.cropped_image,
      });

      console.log("AG-MAN ATTRIBUTES:", res.data);
      setAttributes(res.data.attributes);
      setEmbedding(res.data.embedding);

    } catch (err) {
      console.error("AG-MAN ERROR:", err);
    }

    setAttrLoading(false);
  };

  // ------------------------------------------------
  // 3️⃣ Scene Context Detection
  // ------------------------------------------------
  const detectScene = async () => {
    const frameB64 = captureFrame();
    if (!frameB64) return;

    try {
      const res = await axios.post("http://localhost:5000/scene", {
        image: frameB64,
      });

      console.log("SCENE RESULT:", res.data);
      setScene(res.data);
    } catch (err) {
      console.error("SCENE ERROR:", err);
    }
  };

  // ------------------------------------------------
  // 4️⃣ LLM Reasoning
 const sendToLLM = async () => {
  if (!attributes) {
    console.log("Missing attributes for LLM");
    return;
  }

  if (!userText || userText.trim() === "") {
    alert("Please enter a request (e.g., 'make it formal and price less than 1000')");
    return;
  }

  const payload = {
    item: {
      color_hex: attributes.color_hex,
      pattern: attributes.pattern,
      sleeve_length: attributes.sleeve_length
    },
    scene_label: scene?.scene_label || null,
    user_query: userText.trim()
  };

  console.log("LLM PAYLOAD:", payload);

  try {
    const res = await axios.post("http://localhost:5000/llm", payload, {
      headers: { "Content-Type": "application/json" }
    });

    let out = res.data;
    if (typeof out === "string") {
      try { out = JSON.parse(out); } catch(e){}
    }

    console.log("LLM OUTPUT:", out);
    setLlmOutput(out);
  } catch (err) {
    console.error("LLM ERROR:", err);
    alert("LLM request failed. Check backend logs.");
  }
};
const searchProducts = async () => {
  try {
    const res = await axios.post(
      "http://localhost:5000/search",
      llmOutput
    );
    setProducts(res.data.products);
  } catch (err) {
    console.error("PRODUCT SEARCH ERROR:", err);
  }
};



  // ------------------------------------------------
  // UI
  // ------------------------------------------------
  return (
    <div style={{ padding: "20px" }}>
      <h2>Video Player – ShopWhatYouSee Prototype</h2>

      {/* VIDEO */}
      <video
        ref={videoRef}
        src="/sample.mp4"
        controls
        autoPlay
        muted
        style={{ width: "600px", borderRadius: "10px" }}
      />

      <br />

      {/* BUTTONS */}
      <button
        onClick={detectItems}
        style={{ marginTop: "10px", padding: "10px 20px", background: "#ff5722", color: "#fff", borderRadius: "5px" }}
      >
        Detect Fashion Items
      </button>

      <button
        onClick={detectScene}
        style={{ marginTop: "10px", marginLeft: "10px", padding: "10px 20px", background: "#2196F3", color: "#fff", borderRadius: "5px" }}
      >
        Identify Scene
      </button>

      {/* YOLO RESULTS */}
      {detections.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3>Detected Items</h3>

          <div style={{ display: "flex", gap: "15px", flexWrap: "wrap" }}>
            {detections.map((item, idx) => (
              <div key={idx} style={{ textAlign: "center", cursor: "pointer" }}>
                <img
                  src={`data:image/jpeg;base64,${item.cropped_image}`}
                  alt="detected"
                  style={{
                    width: "140px",
                    height: "140px",
                    border: selectedItem === item ? "3px solid red" : "2px solid black",
                    borderRadius: "10px",
                  }}
                  onClick={() => handleItemClick(item)}
                />
                <p>{item.class}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AG-MAN OUTPUT */}
      {attributes && (
        <div style={{ marginTop: "20px" }}>
          <h3>Item Attributes (AG-MAN)</h3>
          <div style={{ background: "#f5f5f5", padding: "12px", borderRadius: "8px", width: "380px" }}>
            <p><strong>Color:</strong> {attributes.color_hex}</p>
            <div style={{ background: attributes.color_hex, width: "50px", height: "20px", borderRadius: "4px" }}></div>
            <p><strong>Pattern:</strong> {attributes.pattern}</p>
            <p><strong>Sleeve Length:</strong> {attributes.sleeve_length}</p>
            <p><strong>Embedding Size:</strong> {embedding?.length} dims</p>
          </div>
        </div>
      )}

      {/* SCENE OUTPUT */}
      {scene && (
        <div style={{ marginTop: "20px" }}>
          <h3>Scene Context</h3>
          <pre style={{ background: "#eee", padding: "10px", borderRadius: "8px", width: "350px" }}>
            {JSON.stringify(scene, null, 2)}
          </pre>
        </div>
      )}

      {/* LLM SECTION */}
      {selectedItem && attributes && (
        <div style={{ marginTop: "20px" }}>
          <h3>Ask AI to Modify the Item</h3>

          <input
            type="text"
            placeholder="e.g., make it red and formal"
            value={userText}
            onChange={(e) => setUserText(e.target.value)}
            style={{ padding: "10px", width: "350px", marginRight: "10px" }}
          />

          <button
            disabled={!attributes || attrLoading}
            onClick={sendToLLM}
            style={{
              padding: "10px 20px",
              background: "#4CAF50",
              color: "white",
              borderRadius: "5px",
              cursor: attributes ? "pointer" : "not-allowed",
            }}
          >
            Apply LLM
          </button>

      {/*product retrieval button */}
          <button
  disabled={!llmOutput}
  onClick={searchProducts}
  style={{
    padding: "10px 20px",
    background: "#2196F3",
    color: "white",
    borderRadius: "5px",
    marginLeft: "10px"
  }}
>
  Search Products
</button>

{llmOutput && (
  <pre style={{whiteSpace:"pre-wrap"}}>{JSON.stringify(llmOutput, null, 2)}</pre>
)}

{products && products.length > 0 && (
  <div style={{ marginTop: "20px" }}>
    <h3>Recommended Products</h3>
    <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
      {products.map((p) => (
        <div key={p.id} style={{
          border: "1px solid #ccc",
          padding: "10px",
          width: "180px",
          borderRadius: "8px"
        }}>
          <img
            src={p.image_url}
            alt={p.name}
            style={{ width: "100%", height: "160px", objectFit: "cover" }}
          />
          <h4 style={{ fontSize: "16px" }}>{p.name}</h4>
          <p>₹{p.price}</p>
        </div>
      ))}
    </div>
  </div>
)}


        </div>
      )}
    </div>
  );
}

export default VideoPlayer;
