
import React, { useRef, useState, useEffect } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import { MOVIES } from "../api/mockData";

// Icons
const PlayIcon = () => (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 5V19L19 12L8 5Z" />
    </svg>
);

const PauseIcon = () => (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
        <path d="M6 19H10V5H6V19ZM14 5V19H18V5H14Z" />
    </svg>
);

const Forward10Icon = () => (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M13 5l9 7-9 7V5z"></path>
        <path d="M4 5l9 7-9 7V5z"></path>
    </svg>
);

const Rewind10Icon = () => (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 19l-9-7 9-7v14z"></path>
        <path d="M20 19l-9-7 9-7v14z"></path>
    </svg>
);

const VolumeIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
    </svg>
);

const SettingsIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
        <circle cx="12" cy="12" r="3"></circle>
    </svg>
);

const FullscreenIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
    </svg>
);

const CloseIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
);

const CaptionsIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2" ry="2"></rect>
        <path d="M6 8h4v8H6zm8 0h4v8h-4z"></path>
    </svg>
);

const MagicIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 16L22.5 13.5L20 11L17.5 13.5L20 16ZM20 2L22.5 4.5L20 7L17.5 4.5L20 2ZM10 17L12.5 13L10 9L7.5 13L10 17Z" />
        <path d="M5 2L7.5 6L5 10L2.5 6L5 2Z" />
    </svg>
);

// New Lens Icon
const LensIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M8 12h8"></path>
        <path d="M12 8v8"></path>
        {/* Make it look like a viewfinder */}
        <path d="M8 8l-2-2"></path>
        <path d="M16 16l2 2"></path>
        <path d="M16 8l2-2"></path>
        <path d="M8 16l-2 2"></path>
    </svg>
);

// Scene Detection Icon (Camera)
const SceneIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
        <circle cx="12" cy="13" r="4"></circle>
    </svg>
);

const Watch = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const location = useLocation();

    // Support both demo videos and uploaded videos
    const uploadedMovie = location.state?.movie;
    const demoMovie = MOVIES.find(m => m.id === parseInt(id));
    const movie = uploadedMovie || demoMovie;

    const videoRef = useRef(null);
    const containerRef = useRef(null);
    const [paused, setPaused] = useState(false);
    const [showControls, setShowControls] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [detections, setDetections] = useState([]);
    const [loading, setLoading] = useState(false); // Restore loading state
    const [loadingType, setLoadingType] = useState('items'); // 'items' or 'scene'
    const [muted, setMuted] = useState(false); // Add muted state

    const toggleMute = () => {
        if (videoRef.current) {
            videoRef.current.muted = !videoRef.current.muted;
            setMuted(!muted);
        }
    };


    const [isSelectionMode, setIsSelectionMode] = useState(false);
    const [selectionStart, setSelectionStart] = useState(null); // {x, y}
    const [selectionCurrent, setSelectionCurrent] = useState(null); // {x, y}

    // AG-MAN & LLM Pipeline State
    const [selectedItem, setSelectedItem] = useState(null);
    const [attributes, setAttributes] = useState(null);
    const [loadingAttributes, setLoadingAttributes] = useState(false);
    const [scene, setScene] = useState(null);
    const [userQuery, setUserQuery] = useState("");

    if (!movie) {
        return <div style={{ color: 'white', padding: 20 }}>Movie not found</div>;
    }

    // Format scene label: "science_museum" → "Science Museum"
    const formatSceneLabel = (label) => {
        if (!label) return '';
        return label
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const onPause = () => setPaused(true);
        const onPlay = () => {
            setPaused(false);
            setDetections([]);
            setIsSelectionMode(false); // Reset selection on play
        };
        const onTimeUpdate = () => setCurrentTime(video.currentTime);
        const onLoadedMetadata = () => setDuration(video.duration);

        video.addEventListener("pause", onPause);
        video.addEventListener("play", onPlay);
        video.addEventListener("timeupdate", onTimeUpdate);
        video.addEventListener("loadedmetadata", onLoadedMetadata);

        return () => {
            video.removeEventListener("pause", onPause);
            video.removeEventListener("play", onPlay);
            video.removeEventListener("timeupdate", onTimeUpdate);
            video.removeEventListener("loadedmetadata", onLoadedMetadata);
        };
    }, []);

    // Auto-hide scene badge after 10 seconds
    useEffect(() => {
        if (scene) {
            const timer = setTimeout(() => {
                setScene(null);
            }, 10000); // 10 seconds

            return () => clearTimeout(timer);
        }
    }, [scene]);

    // Full Frame Capture
    const captureFrame = () => {
        const video = videoRef.current;
        if (!video) return null;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas; // Return CANVAS element, not base64 string immediately
    };

    // Crop Logic with Context Padding
    const captureCroppedFrame = (selectionRect) => {
        // 1. Get full resolution frame
        const fullCanvas = captureFrame();
        if (!fullCanvas) return null;

        const video = videoRef.current;
        // 2. Calculate displayed video dimensions (handling object-fit: contain)
        const videoRatio = video.videoWidth / video.videoHeight;
        const elementRatio = video.offsetWidth / video.offsetHeight;

        let displayedW, displayedH, displayedX, displayedY;

        if (elementRatio > videoRatio) {
            // Pillarbars (Black bars on sides)
            displayedH = video.offsetHeight;
            displayedW = displayedH * videoRatio;
            displayedX = (video.offsetWidth - displayedW) / 2;
            displayedY = 0;
        } else {
            // Letterbox (Black bars top/bottom)
            displayedW = video.offsetWidth;
            displayedH = displayedW / videoRatio;
            displayedX = 0;
            displayedY = (video.offsetHeight - displayedH) / 2;
        }

        // 3. Map selection coordinates (screen relative) to video coordinates (intrinsic)
        const relativeX = selectionRect.x - displayedX;
        const relativeY = selectionRect.y - displayedY;

        // Scale factor
        const scale = video.videoWidth / displayedW;

        let sourceX = relativeX * scale;
        let sourceY = relativeY * scale;
        let sourceW = selectionRect.w * scale;
        let sourceH = selectionRect.h * scale;

        // 4. ADD PADDING (15% on each side for context)
        const PADDING_PERCENT = 0.15;
        const padW = sourceW * PADDING_PERCENT;
        const padH = sourceH * PADDING_PERCENT;

        sourceX = Math.max(0, sourceX - padW);
        sourceY = Math.max(0, sourceY - padH);
        sourceW = Math.min(video.videoWidth - sourceX, sourceW + 2 * padW);
        sourceH = Math.min(video.videoHeight - sourceY, sourceH + 2 * padH);

        if (sourceW <= 0 || sourceH <= 0) return null;

        // 5. Draw to new canvas
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = sourceW;
        cropCanvas.height = sourceH;
        const cropCtx = cropCanvas.getContext("2d");
        cropCtx.drawImage(fullCanvas, sourceX, sourceY, sourceW, sourceH, 0, 0, sourceW, sourceH);

        return cropCanvas.toDataURL("image/jpeg", 0.9);
    };

    // ---------------------------------------------
    // Selection Handlers
    // ---------------------------------------------
    const handleMouseDown = (e) => {
        if (!isSelectionMode) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setSelectionStart({ x, y });
        setSelectionCurrent({ x, y });
    };

    const handleMouseMove = (e) => {
        if (!isSelectionMode || !selectionStart) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setSelectionCurrent({ x, y });
    };

    const handleMouseUp = async () => {
        if (!isSelectionMode || !selectionStart || !selectionCurrent) return;

        // Finalize selection
        const x = Math.min(selectionStart.x, selectionCurrent.x);
        const y = Math.min(selectionStart.y, selectionCurrent.y);
        const w = Math.abs(selectionStart.x - selectionCurrent.x);
        const h = Math.abs(selectionStart.y - selectionCurrent.y);

        // Clear selection box IMMEDIATELY (before API call)
        setSelectionStart(null);
        setSelectionCurrent(null);
        setIsSelectionMode(false);

        if (w > 20 && h > 20) {
            // Valid selection -> Trigger Detection
            setLoadingType('items');
            setLoading(true);
            const croppedB64 = captureCroppedFrame({ x, y, w, h });
            if (croppedB64) {
                try {
                    const res = await axios.post("http://localhost:5000/detect", {
                        image: croppedB64
                    });
                    setDetections(res.data.detections || []);
                } catch (err) {
                    console.error("Lens Detection Error:", err);
                }
                setLoading(false);
            }
        }
    };

    // Standard Magic Click (Full Frame)
    const handleMagicClick = async () => {
        setLoadingType('items');
        setLoading(true);
        const fullCanvas = captureFrame();
        const frameB64 = fullCanvas.toDataURL("image/jpeg", 0.9);

        try {
            const res = await axios.post("http://localhost:5000/detect", {
                image: frameB64
            });
            setDetections(res.data.detections || []);
        } catch (err) {
            console.error("Detection Error:", err);
        }
        setLoading(false);
    };

    const toggleLensMode = () => {
        videoRef.current.pause(); // Ensure paused
        setIsSelectionMode(!isSelectionMode);
        setDetections([]); // Clear previous
    };

    // Scene Detection
    const handleSceneDetect = async () => {
        setLoadingType('scene');
        setLoading(true);
        const fullCanvas = captureFrame();
        if (!fullCanvas) {
            setLoading(false);
            return;
        }
        const frameB64 = fullCanvas.toDataURL("image/jpeg", 0.9);

        try {
            const res = await axios.post("http://localhost:5000/scene", {
                image: frameB64
            });
            setScene(res.data);
            console.log("Scene detected:", res.data);
        } catch (err) {
            console.error("Scene Detection Error:", err);
        }
        setLoading(false);
    };

    // AG-MAN Attribute Extraction
    const handleItemClick = async (item) => {
        setSelectedItem(item);
        setAttributes(null);
        setUserQuery("");
        setLoadingAttributes(true);

        try {
            const res = await axios.post("http://localhost:5000/extract-attributes", {
                image: item.cropped_image,
                category: item.class
            });
            setAttributes(res.data.attributes);
        } catch (err) {
            console.error("AG-MAN Attribute Error:", err);
        }
        setLoadingAttributes(false);
    };

    // Navigate to ProductPage with LLM-enhanced data
    const navigateToProducts = async () => {
        if (!selectedItem || !attributes) return;

        try {
            // Call LLM to generate smart filters
            const llmRes = await axios.post("http://localhost:5000/llm", {
                item: {
                    category: selectedItem.class.toLowerCase(),
                    color_hex: attributes.color_hex,
                    pattern: attributes.pattern,
                    sleeve_length: attributes.sleeve_length
                },
                scene: scene,
                user_query: userQuery.trim() || null
            });

            // Navigate with enriched data
            navigate(`/product/${selectedItem.class}`, {
                state: {
                    item: { ...selectedItem, attributes },
                    llmFilters: llmRes.data.filters,
                    scene: scene,
                    movie: movie
                }
            });
        } catch (err) {
            console.error("LLM Error:", err);
            // Fallback: navigate with basic data
            navigate(`/product/${selectedItem.class}`, {
                state: {
                    item: { ...selectedItem, attributes },
                    scene: scene,
                    movie: movie
                }
            });
        }
    };

    const togglePlay = () => {
        if (isSelectionMode) return; // Disable play during selection
        if (videoRef.current.paused) {
            videoRef.current.play();
        } else {
            videoRef.current.pause();
        }
    };

    const skipTime = (amount) => {
        if (videoRef.current) {
            videoRef.current.currentTime += amount;
        }
    };

    const handleSeek = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        videoRef.current.currentTime = percent * duration;
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Calculate Selection Box Style
    const getSelectionStyle = () => {
        if (!selectionStart || !selectionCurrent) return {};
        const left = Math.min(selectionStart.x, selectionCurrent.x);
        const top = Math.min(selectionStart.y, selectionCurrent.y);
        const width = Math.abs(selectionStart.x - selectionCurrent.x);
        const height = Math.abs(selectionStart.y - selectionCurrent.y);
        return {
            left, top, width, height,
            position: 'absolute',
            border: '2px solid #00a8e1',
            backgroundColor: 'rgba(0, 168, 225, 0.2)',
            zIndex: 50,
            boxShadow: '0 0 10px rgba(0,0,0,0.5)'
        };
    };

    return (
        <>
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                `}
            </style>
            <div style={styles.container}>
                <div
                    style={{
                        ...styles.playerWrapper,
                        cursor: isSelectionMode ? 'crosshair' : 'default'
                    }}
                    ref={containerRef}
                    onMouseEnter={() => setShowControls(true)}
                    onMouseLeave={() => setShowControls(false)}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                >
                    <video
                        ref={videoRef}
                        src={movie.videoSrc}
                        style={styles.video}
                        onClick={togglePlay}
                        loop
                        muted={muted}
                        crossOrigin="anonymous"
                    />

                    {/* Selection Box Overlay */}
                    {isSelectionMode && selectionStart && (
                        <div style={getSelectionStyle()}></div>
                    )}

                    {/* Visual indicator for Lens Mode */}
                    {isSelectionMode && !selectionStart && (
                        <div style={styles.lensInstruction}>
                            Click and drag to select an item
                        </div>
                    )}

                    {/* Top Overlay */}
                    <div style={{
                        ...styles.topOverlay,
                        opacity: (showControls || paused || isSelectionMode) ? 1 : 0
                    }}>
                        <div style={styles.topLeft}>
                            <div style={styles.xrayBadge}>X-Ray</div>
                            <div style={styles.movieTitle}>{movie.title}</div>
                        </div>

                        <div style={styles.topRight}>
                            {/* Action Buttons */}
                            {(paused || isSelectionMode) && !loading && detections.length === 0 && (
                                <>
                                    {/* Lens Button */}
                                    <button
                                        style={{
                                            ...styles.magicButtonIcon,
                                            borderColor: isSelectionMode ? '#00a8e1' : '#ccc',
                                            color: isSelectionMode ? '#00a8e1' : '#white',
                                            background: isSelectionMode ? 'rgba(0, 168, 225, 0.2)' : 'rgba(255,255,255,0.1)'
                                        }}
                                        onClick={toggleLensMode}
                                        title="Google Lens Region Search"
                                    >
                                        <LensIcon />
                                    </button>

                                    {/* Magic Button (Full Frame) */}
                                    <button
                                        style={styles.magicButtonIcon}
                                        onClick={handleMagicClick}
                                        title="Scan Full Scene"
                                    >
                                        <MagicIcon />
                                    </button>

                                    {/* Scene Detection Button */}
                                    <button
                                        style={{
                                            ...styles.magicButtonIcon,
                                            borderColor: scene ? '#4caf50' : '#ccc',
                                            color: scene ? '#4caf50' : 'white',
                                            background: scene ? 'rgba(76, 175, 80, 0.2)' : 'rgba(255,255,255,0.1)'
                                        }}
                                        onClick={handleSceneDetect}
                                        title="Detect Scene Context"
                                    >
                                        <SceneIcon />
                                    </button>
                                </>
                            )}
                            <button style={styles.iconBtn}><CaptionsIcon /></button>
                            <button style={styles.iconBtn}><SettingsIcon /></button>
                            <button style={styles.iconBtn} onClick={toggleMute}>
                                {muted ? (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                        <line x1="23" y1="9" x2="17" y2="15"></line>
                                        <line x1="17" y1="9" x2="23" y2="15"></line>
                                    </svg>
                                ) : (
                                    <VolumeIcon />
                                )}
                            </button>
                            <button style={styles.iconBtn}><FullscreenIcon /></button>
                            <button style={styles.iconBtn} onClick={() => navigate('/')}><CloseIcon /></button>
                        </div>
                    </div>

                    {/* Scene Context Badge (Permanent) */}
                    {scene && (
                        <div style={styles.sceneOverlay}>
                            {formatSceneLabel(scene.scene_label)}
                        </div>
                    )}

                    {/* Center Controls */}
                    <div style={{
                        ...styles.centerControls,
                        opacity: (showControls || paused) && !isSelectionMode ? 1 : 0,
                        pointerEvents: isSelectionMode ? 'none' : 'auto'
                    }}>
                        <button style={styles.skipBtn} onClick={() => skipTime(-10)}>
                            <Rewind10Icon />
                        </button>

                        <button style={styles.playPauseBtn} onClick={togglePlay}>
                            {paused ? <PlayIcon /> : <PauseIcon />}
                        </button>

                        <button style={styles.skipBtn} onClick={() => skipTime(10)}>
                            <Forward10Icon />
                        </button>
                    </div>

                    {/* Loading State */}
                    {loading && (
                        <div style={styles.centerOverlay}>
                            <div style={styles.loaderSpinner}></div>
                            <div style={styles.loadingText}>
                                {loadingType === 'scene' ? 'detecting the scene...' : 'detecting items...'}
                            </div>
                        </div>
                    )}

                    {/* Bottom Bar */}
                    <div style={{
                        ...styles.bottomBar,
                        opacity: (showControls || paused) ? 1 : 0
                    }}>
                        <div style={styles.timeDisplay}>
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </div>
                        <div style={styles.progressBarContainer} onClick={handleSeek}>
                            <div style={{
                                ...styles.progressBar,
                                width: `${(currentTime / duration) * 100}%`
                            }}></div>
                        </div>
                    </div>

                    {/* Detection Results Sidebar */}
                    {detections.length > 0 && (
                        <div style={styles.sidebar}>
                            <div style={styles.sidebarHeader}>
                                <h3 style={styles.sidebarTitle}>Lens Results</h3>
                                <button style={styles.closeBtn} onClick={() => setDetections([])}>
                                    <CloseIcon />
                                </button>
                            </div>

                            {/* Scene Context Badge */}
                            {scene && (
                                <div style={styles.sceneBadge}>
                                    {formatSceneLabel(scene.scene_label)}
                                </div>
                            )}

                            <div style={styles.itemsGrid}>
                                {detections.map((item, idx) => (
                                    <div key={idx}>
                                        <div
                                            style={{
                                                ...styles.itemCard,
                                                borderColor: selectedItem === item ? '#00a8e1' : 'transparent'
                                            }}
                                            onClick={() => handleItemClick(item)}
                                        >
                                            <div style={styles.imageWrapper}>
                                                <img
                                                    src={`data:image/jpeg;base64,${item.cropped_image}`}
                                                    alt={item.class}
                                                    style={styles.itemImage}
                                                />
                                            </div>
                                            <div style={styles.itemInfo}>
                                                <div style={styles.itemClass}>{item.class}</div>
                                                <div style={styles.shopLink}>
                                                    {selectedItem === item ? 'Selected ✓' : 'Tap to analyze ›'}
                                                </div>
                                            </div>
                                        </div>

                                        {/* AG-MAN Attributes (shown when selected) */}
                                        {selectedItem === item && (
                                            <div style={styles.attributesBox}>
                                                {loadingAttributes ? (
                                                    <div style={styles.attrLoading}>Analyzing...</div>
                                                ) : attributes ? (
                                                    <>
                                                        <div style={styles.attrTitle}>Visual Attributes</div>
                                                        <div style={styles.attrRow}>
                                                            <span style={styles.attrLabel}>Color:</span>
                                                            <div style={styles.colorRow}>
                                                                <div style={{
                                                                    ...styles.colorSwatch,
                                                                    background: attributes.color_hex
                                                                }}></div>
                                                                <span style={{ color: 'white', fontWeight: '500' }}>
                                                                    {attributes.color_name || attributes.color_hex}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        {attributes.pattern && (
                                                            <div style={styles.attrRow}>
                                                                <span style={styles.attrLabel}>Pattern:</span>
                                                                <span style={{ color: 'white', fontWeight: '500' }}>
                                                                    {attributes.pattern}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {attributes.sleeve_length && (
                                                            <div style={styles.attrRow}>
                                                                <span style={styles.attrLabel}>Sleeve:</span>
                                                                <span style={{ color: 'white', fontWeight: '500' }}>
                                                                    {attributes.sleeve_length}
                                                                </span>
                                                            </div>
                                                        )}

                                                        {/* LLM Query Input */}
                                                        <div style={styles.llmSection}>
                                                            <div style={styles.attrLabel}>Refine Search (Optional)</div>
                                                            <input
                                                                type="text"
                                                                placeholder="e.g., casual, under ₹2000"
                                                                value={userQuery}
                                                                onChange={(e) => setUserQuery(e.target.value)}
                                                                style={styles.llmInput}
                                                            />
                                                            <button
                                                                style={styles.shopButton}
                                                                onClick={navigateToProducts}
                                                            >
                                                                Shop Similar Items →
                                                            </button>
                                                        </div>
                                                    </>
                                                ) : null}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

const styles = {
    container: {
        backgroundColor: "#000",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Amazon Ember', Arial, sans-serif"
    },
    playerWrapper: {
        position: "relative",
        width: "100%",
        height: "100%",
        backgroundColor: "#000",
        overflow: "hidden",
        userSelect: 'none'
    },
    video: {
        width: "100%",
        height: "100%",
        objectFit: "contain"
    },
    topOverlay: {
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        padding: "30px 40px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        background: "linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)",
        transition: "opacity 0.3s ease",
        zIndex: 20
    },
    topLeft: {
        display: "flex",
        alignItems: "center",
        gap: "20px"
    },
    xrayBadge: {
        color: "#ccc",
        fontWeight: "bold",
        fontSize: "16px",
        letterSpacing: "1px"
    },
    movieTitle: {
        color: "white",
        fontSize: "20px",
        fontWeight: "500"
    },
    topRight: {
        display: "flex",
        alignItems: "center",
        gap: "25px"
    },
    iconBtn: {
        background: "none",
        border: "none",
        color: "white",
        cursor: "pointer",
        opacity: 0.8,
        transition: "opacity 0.2s"
    },
    magicButtonIcon: {
        background: "rgba(255, 215, 0, 0.2)",
        border: "1px solid #ffd700",
        borderRadius: "50%",
        width: "40px",
        height: "40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#ffd700",
        cursor: "pointer",
        marginRight: "10px",
        backdropFilter: "blur(5px)"
    },
    centerControls: {
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        display: "flex",
        alignItems: "center",
        gap: "60px",
        zIndex: 20,
        transition: "opacity 0.3s ease"
    },
    playPauseBtn: {
        background: "none",
        border: "none",
        color: "white",
        cursor: "pointer",
        padding: 0,
        filter: "drop-shadow(0 4px 6px rgba(0,0,0,0.5))"
    },
    skipBtn: {
        background: "none",
        border: "none",
        color: "white",
        cursor: "pointer",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        opacity: 0.8,
        gap: "5px"
    },
    skipText: {
        fontSize: "12px",
        fontWeight: "bold",
        position: "absolute",
        top: "14px"
    },
    centerOverlay: {
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "25px 35px",
        background: "rgba(0, 0, 0, 0.75)",
        borderRadius: "16px",
        backdropFilter: "blur(8px)",
        border: "1px solid rgba(76, 175, 80, 0.3)",
        zIndex: 25
    },
    loaderSpinner: {
        width: "40px",
        height: "40px",
        border: "3px solid rgba(76, 175, 80, 0.15)",
        borderTop: "3px solid #4caf50",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite"
    },
    loadingText: {
        color: "#4caf50",
        fontSize: "13px",
        fontWeight: "600",
        marginTop: "12px",
        textAlign: "center",
        letterSpacing: "0.3px",
        opacity: 0.95
    },
    bottomBar: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        padding: "30px 40px",
        background: "linear-gradient(to top, rgba(0,0,0,0.9), transparent)",
        transition: "opacity 0.3s ease",
        zIndex: 20
    },
    timeDisplay: {
        color: "#ccc",
        fontSize: "14px",
        marginBottom: "10px",
        fontFamily: "monospace"
    },
    progressBarContainer: {
        width: "100%",
        height: "4px",
        backgroundColor: "rgba(255,255,255,0.3)",
        cursor: "pointer",
        position: "relative",
        borderRadius: "2px"
    },
    progressBar: {
        height: "100%",
        backgroundColor: "#00a8e1",
        position: "relative",
        borderRadius: "2px"
    },
    sidebar: {
        position: "absolute",
        top: 0,
        right: 0,
        width: "350px",
        height: "100%",
        background: "rgba(10, 10, 10, 0.95)",
        backdropFilter: "blur(15px)",
        padding: "20px",
        overflowY: "auto",
        zIndex: 30,
        borderLeft: "1px solid rgba(255,255,255,0.1)",
        animation: "slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1)"
    },
    sidebarHeader: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "25px",
        paddingBottom: "15px",
        borderBottom: "1px solid rgba(255,255,255,0.1)"
    },
    sidebarTitle: {
        color: "white",
        fontSize: "18px",
        fontWeight: "600",
        margin: 0
    },
    closeBtn: {
        background: "none",
        border: "none",
        color: "#ccc",
        cursor: "pointer",
        padding: "5px"
    },
    itemsGrid: {
        display: "flex",
        flexDirection: "column",
        gap: "15px"
    },
    itemCard: {
        background: "rgba(255,255,255,0.05)",
        borderRadius: "8px",
        padding: "12px",
        display: "flex",
        gap: "15px",
        alignItems: "center",
        cursor: "pointer",
        transition: "all 0.2s ease",
        border: "1px solid transparent"
    },
    imageWrapper: {
        width: "80px",
        height: "80px",
        borderRadius: "4px",
        overflow: "hidden",
        backgroundColor: "#000"
    },
    itemImage: {
        width: "100%",
        height: "100%",
        objectFit: "contain"
    },
    itemInfo: {
        flex: 1
    },
    itemClass: {
        color: "white",
        fontWeight: "600",
        fontSize: "15px",
        textTransform: "capitalize",
        marginBottom: "6px"
    },
    shopLink: {
        color: "#00a8e1",
        fontSize: "13px",
        fontWeight: "500"
    },
    lensInstruction: {
        position: 'absolute',
        top: '20%',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.6)',
        color: 'white',
        padding: '10px 20px',
        borderRadius: '20px',
        fontSize: '14px',
        pointerEvents: 'none',
        zIndex: 15
    },
    sceneBadge: {
        background: 'rgba(76, 175, 80, 0.2)',
        border: '1px solid #4caf50',
        borderRadius: '20px',
        padding: '8px 14px',
        color: '#4caf50',
        fontSize: '13px',
        marginBottom: '15px',
        textAlign: 'center'
    },
    sceneOverlay: {
        position: 'absolute',
        top: '100px',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(76, 175, 80, 0.9)',
        backdropFilter: 'blur(10px)',
        border: '1px solid #4caf50',
        borderRadius: '20px',
        padding: '10px 20px',
        color: 'white',
        fontSize: '14px',
        fontWeight: '600',
        zIndex: 25,
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        animation: 'slideDown 0.3s ease'
    },
    attributesBox: {
        background: 'rgba(0, 168, 225, 0.1)',
        borderRadius: '8px',
        padding: '12px',
        marginTop: '8px',
        marginBottom: '12px',
        border: '1px solid rgba(0, 168, 225, 0.3)'
    },
    attrTitle: {
        color: '#00a8e1',
        fontSize: '13px',
        fontWeight: '600',
        marginBottom: '10px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
    },
    attrRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '8px',
        fontSize: '13px'
    },
    attrLabel: {
        color: '#aaa',
        fontSize: '12px',
        fontWeight: '500',
        marginBottom: '4px'
    },
    colorRow: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        color: 'white'
    },
    colorSwatch: {
        width: '24px',
        height: '24px',
        borderRadius: '4px',
        border: '1px solid rgba(255,255,255,0.3)'
    },
    attrLoading: {
        color: '#00a8e1',
        textAlign: 'center',
        padding: '10px',
        fontSize: '13px'
    },
    llmSection: {
        marginTop: '12px',
        paddingTop: '12px',
        borderTop: '1px solid rgba(255,255,255,0.1)'
    },
    llmInput: {
        width: '100%',
        boxSizing: 'border-box', // Fix overflow
        padding: '8px 12px',
        borderRadius: '6px',
        border: '1px solid rgba(255,255,255,0.2)',
        background: 'rgba(0,0,0,0.3)',
        color: 'white',
        fontSize: '13px',
        marginTop: '6px',
        marginBottom: '10px',
        outline: 'none'
    },
    shopButton: {
        width: '100%',
        padding: '10px',
        background: 'linear-gradient(135deg, #00a8e1, #0080b3)',
        color: 'white',
        border: 'none',
        borderRadius: '6px',
        fontSize: '14px',
        fontWeight: '600',
        cursor: 'pointer',
        transition: 'transform 0.2s',
        ':hover': {
            transform: 'scale(1.02)'
        }
    }
};

export default Watch;
