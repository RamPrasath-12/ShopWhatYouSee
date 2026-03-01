import React from "react";
import VideoPlayer from "../components/VideoPlayer";

function Home() {
  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#f2f2f2",
        padding: "30px 0"
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          background: "#ffffff",
          borderRadius: "12px",
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          padding: "20px 30px"
        }}
      >
        <VideoPlayer />
      </div>
    </div>
  );
}

export default Home;
