import React, { useRef, useState } from "react";
import axios from "axios";

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

    const payload = {
      item: {
        category: selectedItem.class.toLowerCase(),
        color_hex: attributes.color_hex,
        pattern: attributes.pattern,
        sleeve_length: attributes.sleeve_length
      },
      scene: scene || null,
      user_query: userText.trim() || null
    };

    try {
      const res = await axios.post("http://localhost:5000/llm", payload);
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

    const payload = {
      filters: llmOutput?.filters || {},
      detected_category: selectedItem.class.toLowerCase()
    };

    try {
      const res = await axios.post("http://localhost:5000/search", payload);
      setProducts(res.data.products || []);
    } catch (err) {
      console.error("PRODUCT SEARCH ERROR:", err);
    }
  };

  // =================================================
  // STYLED COMPONENTS
  // =================================================
  const styles = {
    container: {
      minHeight: "100vh",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      padding: "40px 20px"
    },
    contentWrapper: {
      maxWidth: "1200px",
      margin: "0 auto"
    },
    header: {
      textAlign: "center",
      marginBottom: "40px"
    },
    title: {
      fontSize: "42px",
      fontWeight: "bold",
      color: "white",
      marginBottom: "10px",
      textShadow: "2px 2px 8px rgba(0,0,0,0.3)"
    },
    subtitle: {
      fontSize: "16px",
      color: "rgba(255,255,255,0.9)",
      fontWeight: "300"
    },
    videoContainer: {
      background: "rgba(255,255,255,0.95)",
      borderRadius: "20px",
      padding: "30px",
      boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
      marginBottom: "30px"
    },
    video: {
      width: "100%",
      maxWidth: "700px",
      borderRadius: "16px",
      boxShadow: "0 10px 30px rgba(0,0,0,0.2)",
      display: "block",
      margin: "0 auto"
    },
    buttonGroup: {
      display: "flex",
      gap: "15px",
      justifyContent: "center",
      marginTop: "25px",
      flexWrap: "wrap"
    },
    primaryButton: {
      padding: "14px 32px",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      color: "white",
      border: "none",
      borderRadius: "50px",
      cursor: "pointer",
      fontSize: "15px",
      fontWeight: "600",
      boxShadow: "0 8px 20px rgba(102, 126, 234, 0.4)",
      transition: "all 0.3s ease",
      transform: "translateY(0)"
    },
    secondaryButton: {
      padding: "14px 32px",
      background: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
      color: "white",
      border: "none",
      borderRadius: "50px",
      cursor: "pointer",
      fontSize: "15px",
      fontWeight: "600",
      boxShadow: "0 8px 20px rgba(245, 87, 108, 0.4)",
      transition: "all 0.3s ease",
      transform: "translateY(0)"
    },
    successButton: {
      padding: "14px 32px",
      background: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
      color: "white",
      border: "none",
      borderRadius: "50px",
      cursor: "pointer",
      fontSize: "15px",
      fontWeight: "600",
      boxShadow: "0 8px 20px rgba(79, 172, 254, 0.4)",
      transition: "all 0.3s ease",
      transform: "translateY(0)"
    },
    moduleBox: {
      background: "white",
      borderRadius: "20px",
      padding: "30px",
      marginBottom: "25px",
      boxShadow: "0 15px 40px rgba(0,0,0,0.15)",
      border: "none",
      transition: "transform 0.3s ease, box-shadow 0.3s ease"
    },
    moduleTitle: {
      fontSize: "24px",
      fontWeight: "700",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
      marginBottom: "20px",
      display: "flex",
      alignItems: "center",
      gap: "10px"
    },
    badge: {
      display: "inline-block",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      color: "white",
      padding: "6px 16px",
      borderRadius: "20px",
      fontSize: "12px",
      fontWeight: "600",
      marginBottom: "15px"
    },
    capturedImage: {
      width: "100%",
      maxWidth: "400px",
      borderRadius: "12px",
      boxShadow: "0 10px 25px rgba(0,0,0,0.15)",
      border: "3px solid #667eea",
      display: "block",
      margin: "15px auto"
    },
    detectionGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
      gap: "20px",
      marginTop: "20px"
    },
    detectionItem: {
      textAlign: "center",
      transition: "transform 0.3s ease"
    },
    detectionImage: {
      width: "100%",
      height: "160px",
      objectFit: "cover",
      borderRadius: "12px",
      cursor: "pointer",
      transition: "all 0.3s ease",
      boxShadow: "0 5px 15px rgba(0,0,0,0.1)"
    },
    detectionImageSelected: {
      width: "100%",
      height: "160px",
      objectFit: "cover",
      borderRadius: "12px",
      cursor: "pointer",
      transition: "all 0.3s ease",
      boxShadow: "0 10px 30px rgba(102, 126, 234, 0.5)",
      border: "4px solid #667eea",
      transform: "scale(1.05)"
    },
    detectionLabel: {
      marginTop: "10px",
      fontSize: "15px",
      fontWeight: "600",
      color: "#333"
    },
    attributeCard: {
      background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
      padding: "20px",
      borderRadius: "12px",
      marginBottom: "15px"
    },
    attributeRow: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "12px",
      fontSize: "15px"
    },
    attributeLabel: {
      fontWeight: "600",
      color: "#555"
    },
    attributeValue: {
      fontWeight: "500",
      color: "#667eea"
    },
    colorSwatch: {
      width: "120px",
      height: "40px",
      borderRadius: "8px",
      boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
      border: "2px solid white",
      margin: "10px 0"
    },
    input: {
      width: "100%",
      padding: "14px 20px",
      borderRadius: "12px",
      border: "2px solid #e0e0e0",
      fontSize: "15px",
      marginBottom: "15px",
      transition: "all 0.3s ease",
      outline: "none"
    },
    codeBlock: {
      background: "#1e1e1e",
      color: "#d4d4d4",
      padding: "20px",
      borderRadius: "12px",
      fontSize: "13px",
      fontFamily: "monospace",
      overflowX: "auto",
      marginTop: "15px",
      boxShadow: "inset 0 2px 8px rgba(0,0,0,0.3)"
    },
    productGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
      gap: "25px",
      marginTop: "20px"
    },
    productCard: {
      background: "white",
      borderRadius: "16px",
      overflow: "hidden",
      boxShadow: "0 8px 20px rgba(0,0,0,0.1)",
      transition: "all 0.3s ease",
      cursor: "pointer"
    },
    productImage: {
      width: "100%",
      height: "220px",
      objectFit: "cover"
    },
    productInfo: {
      padding: "15px"
    },
    productName: {
      fontSize: "15px",
      fontWeight: "600",
      color: "#333",
      marginBottom: "8px"
    },
    productPrice: {
      fontSize: "20px",
      fontWeight: "700",
      color: "#667eea"
    },
    loadingSpinner: {
      display: "inline-block",
      width: "30px",
      height: "30px",
      border: "4px solid rgba(102, 126, 234, 0.2)",
      borderTop: "4px solid #667eea",
      borderRadius: "50%",
      animation: "spin 1s linear infinite"
    }
  };

  // =================================================
  // UI
  // =================================================
  return (
    <div style={styles.container}>
      <div style={styles.contentWrapper}>
        {/* HEADER */}
        <div style={styles.header}>
          <h1 style={styles.title}>🛍️ ShopWhatYouSee</h1>
          <p style={styles.subtitle}>
            AI-Powered Fashion Discovery • Review-1 Prototype
          </p>
        </div>

        {/* VIDEO SECTION */}
        <div style={styles.videoContainer}>
          <video
            ref={videoRef}
            src="/sample.mp4"
            controls
            autoPlay
            muted
            style={styles.video}
          />

          <div style={styles.buttonGroup}>
            <button
              style={styles.primaryButton}
              onClick={detectItems}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 12px 28px rgba(102, 126, 234, 0.6)";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 8px 20px rgba(102, 126, 234, 0.4)";
              }}
            >
              🔍 Detect Fashion Items
            </button>
            <button
              style={styles.secondaryButton}
              onClick={detectScene}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 12px 28px rgba(245, 87, 108, 0.6)";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 8px 20px rgba(245, 87, 108, 0.4)";
              }}
            >
              🌍 Identify Scene
            </button>
          </div>
        </div>

        {/* MODULE 1: FRAME CAPTURE */}
        {capturedFrame && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 1</span>
            <h3 style={styles.moduleTitle}>📸 Frame Capture</h3>
            <img
              src={capturedFrame}
              alt="Captured Frame"
              style={styles.capturedImage}
            />
            <p style={{ color: "#666", fontSize: "14px", textAlign: "center", marginTop: "15px" }}>
              ✓ Raw video frame captured and ready for processing
            </p>
          </div>
        )}

        {/* MODULE 2: YOLOv8 DETECTION */}
        {detections.length > 0 && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 2</span>
            <h3 style={styles.moduleTitle}>🎯 YOLOv8 Fashion Detection</h3>
            <p style={{ color: "#666", fontSize: "14px", marginBottom: "20px" }}>
              Click on any detected item to extract attributes
            </p>

            <div style={styles.detectionGrid}>
              {detections.map((item, idx) => (
                <div key={idx} style={styles.detectionItem}>
                  <img
                    src={`data:image/jpeg;base64,${item.cropped_image}`}
                    alt="detected"
                    style={
                      selectedItem === item
                        ? styles.detectionImageSelected
                        : styles.detectionImage
                    }
                    onClick={() => handleItemClick(item)}
                    onMouseOver={(e) => {
                      if (selectedItem !== item) {
                        e.target.style.transform = "scale(1.05)";
                        e.target.style.boxShadow = "0 10px 25px rgba(0,0,0,0.2)";
                      }
                    }}
                    onMouseOut={(e) => {
                      if (selectedItem !== item) {
                        e.target.style.transform = "scale(1)";
                        e.target.style.boxShadow = "0 5px 15px rgba(0,0,0,0.1)";
                      }
                    }}
                  />
                  <p style={styles.detectionLabel}>
                    {item.class}
                    <br />
                    <small style={{ color: "#888", fontSize: "13px" }}>
                      {(item.conf * 100).toFixed(1)}% confidence
                    </small>
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* MODULE 3: AG-MAN ATTRIBUTE EXTRACTION */}
        {attributes && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 3</span>
            <h3 style={styles.moduleTitle}>✨ AG-MAN Attribute Extraction</h3>

            <div style={styles.attributeCard}>
              <div style={styles.attributeRow}>
                <span style={styles.attributeLabel}>Dominant Color:</span>
                <span style={styles.attributeValue}>{attributes.color_hex}</span>
              </div>

              <div
                style={{
                  ...styles.colorSwatch,
                  background: attributes.color_hex
                }}
              />

              <div style={styles.attributeRow}>
                <span style={styles.attributeLabel}>Pattern:</span>
                <span style={styles.attributeValue}>{attributes.pattern}</span>
              </div>

              <div style={styles.attributeRow}>
                <span style={styles.attributeLabel}>Sleeve Length:</span>
                <span style={styles.attributeValue}>{attributes.sleeve_length}</span>
              </div>

              {embedding && (
                <div style={styles.attributeRow}>
                  <span style={styles.attributeLabel}>Embedding Dimension:</span>
                  <span style={styles.attributeValue}>{embedding.length}D</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* MODULE 4: SCENE CONTEXT */}
        {scene && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 4</span>
            <h3 style={styles.moduleTitle}>🌆 Scene Context Detection</h3>
            <pre style={styles.codeBlock}>
              {JSON.stringify(scene, null, 2)}
            </pre>
          </div>
        )}

        {/* MODULE 5: LLM INTENT REASONING */}
        {selectedItem && attributes && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 5</span>
            <h3 style={styles.moduleTitle}>🧠 LLM Intent Reasoning</h3>

            <input
              type="text"
              placeholder="Optional: refine results (e.g., under 2000, casual, blue color)"
              value={userText}
              onChange={(e) => setUserText(e.target.value)}
              style={styles.input}
              onFocus={(e) => {
                e.target.style.borderColor = "#667eea";
                e.target.style.boxShadow = "0 0 0 3px rgba(102, 126, 234, 0.1)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "#e0e0e0";
                e.target.style.boxShadow = "none";
              }}
            />

            <button
              style={styles.primaryButton}
              onClick={sendToLLM}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 12px 28px rgba(102, 126, 234, 0.6)";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 8px 20px rgba(102, 126, 234, 0.4)";
              }}
            >
              🎯 Generate Filters
            </button>

            {llmOutput && (
              <pre style={styles.codeBlock}>
                {JSON.stringify(llmOutput, null, 2)}
              </pre>
            )}
          </div>
        )}

        {/* MODULE 6: PRODUCT RETRIEVAL */}
        {products.length > 0 && (
          <div style={styles.moduleBox}>
            <span style={styles.badge}>Module 6</span>
            <h3 style={styles.moduleTitle}>🛒 Product Retrieval</h3>
            <p style={{ color: "#666", fontSize: "14px", marginBottom: "20px" }}>
              Found {products.length} similar products
            </p>

            <div style={styles.productGrid}>
              {products.map((p) => (
                <div
                  key={p.id}
                  style={styles.productCard}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = "translateY(-8px)";
                    e.currentTarget.style.boxShadow = "0 15px 35px rgba(0,0,0,0.2)";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 8px 20px rgba(0,0,0,0.1)";
                  }}
                >
                  <img
                    src={p.image_url}
                    alt={p.name}
                    style={styles.productImage}
                  />
                  <div style={styles.productInfo}>
                    <p style={styles.productName}>{p.name}</p>
                    <p style={styles.productPrice}>₹{p.price}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SEARCH PRODUCTS BUTTON */}
        {selectedItem && (
          <div style={{ textAlign: "center", marginTop: "30px" }}>
            <button
              style={styles.successButton}
              onClick={searchProducts}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 12px 28px rgba(79, 172, 254, 0.6)";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 8px 20px rgba(79, 172, 254, 0.4)";
              }}
            >
              🔎 Search Products
            </button>
          </div>
        )}
      </div>

      <style>
        {`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}

export default VideoPlayer;