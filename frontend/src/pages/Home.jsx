import React, { useState } from "react";
import VideoPlayer from "../components/VideoPlayer";
import axios from "axios";

function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectionResults, setDetectionResults] = useState([]);
  const [similarProducts, setSimilarProducts] = useState([]);
  const [showVideoPlayer, setShowVideoPlayer] = useState(false);

  // Triggered when user uploads an image
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setSelectedImage(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("image", file);

    axios.post("http://localhost:5000/detect", formData)
      .then((res) => {
        setDetectionResults(res.data.detections);
      })
      .catch((err) => console.error(err));
  };

  // Triggered when user clicks on a detected item
  const handleItemClick = (cropImageUrl) => {
    axios.post("http://localhost:5000/retrieve", { image: cropImageUrl })
      .then((res) => {
        setSimilarProducts(res.data.results);
      })
      .catch((err) => console.error(err));
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>ShopWhatYouSee</h1>

      {!showVideoPlayer && (
        <>
          <h3>Upload Image to Detect Fashion Items</h3>
          <input type="file" accept="image/*" onChange={handleImageUpload} />

          {selectedImage && (
            <div>
              <img
                src={selectedImage}
                alt="Uploaded"
                style={{ width: "400px", marginTop: "10px", borderRadius: "10px" }}
              />
            </div>
          )}

          {/* Detected Items */}
          {detectionResults.length > 0 && (
            <div style={{ marginTop: "20px" }}>
              <h3>Detected Fashion Items</h3>
              <div style={{ display: "flex", gap: "10px" }}>
                {detectionResults.map((item, idx) => (
                  <img
                    key={idx}
                    src={`data:image/jpeg;base64,${item.cropped_image}`}
                    alt="crop"
                    style={{
                      width: "120px",
                      height: "120px",
                      border: "2px solid black",
                      cursor: "pointer",
                    }}
                    onClick={() =>
                      handleItemClick(`data:image/jpeg;base64,${item.cropped_image}`)
                    }
                  />
                ))}
              </div>
            </div>
          )}

          {/* Similar Products */}
          {similarProducts.length > 0 && (
            <div style={{ marginTop: "30px" }}>
              <h3>Similar Products</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "20px" }}>
                {similarProducts.map((item, idx) => (
                  <div key={idx} style={{ border: "1px solid #ddd", padding: "10px" }}>
                    <img
                      src={item.image_url}
                      alt="product"
                      style={{ width: "150px", height: "150px" }}
                    />
                    <p>{item.name}</p>
                    <p>₹{item.price}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Button to Switch to Video Player */}
          <button
            style={{
              marginTop: "30px",
              padding: "10px 20px",
              backgroundColor: "#4a4aff",
              color: "#fff",
              border: "none",
              borderRadius: "5px",
            }}
            onClick={() => setShowVideoPlayer(true)}
          >
            Try Video Mode →
          </button>
        </>
      )}

      {showVideoPlayer && <VideoPlayer />}
    </div>
  );
}

export default Home;
