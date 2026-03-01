
import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const ProductPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { item, movie, llmFilters, scene, embedding, sessionHistory } = location.state || {};

    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentFilters, setCurrentFilters] = useState(llmFilters || {});

    // Store ORIGINAL AGMAN attributes separately (never overwritten by LLM)
    const [originalAttributes, setOriginalAttributes] = useState(null);

    // Search generation counter — prevents stale results from overwriting newer ones
    const searchGenRef = useRef(0);
    // Prevent double-firing of handleQuerySubmit (React concurrent rendering)
    const queryInFlightRef = useRef(false);

    // Product detail modal state
    const [selectedProduct, setSelectedProduct] = useState(null);

    // Insights Modal State
    const [showInsights, setShowInsights] = useState(false);
    const [insightsData, setInsightsData] = useState(null);

    // Dynamic mock details based on detected class
    const category = item ? item.class.toLowerCase() : "unknown";
    const attributes = item?.attributes;

    // Initialize originalAttributes on first load
    useEffect(() => {
        if (item && item.attributes && !originalAttributes) {
            setOriginalAttributes({
                category: item.class,
                color_hex: item.attributes.color_hex,
                color_name: item.attributes.color_name,
                pattern: item.attributes.pattern,
                sleeve_length: item.attributes.sleeve
            });
            console.log("[ProductPage] Stored original AGMAN attributes:", item.attributes);
        }
    }, [item]);

    const mockTitles = {
        "shirt": "Men's Classic Regular Fit Cotton Formal Shirt",
        "tshirt": "Premium Cotton Crew Neck T-Shirt - Urban Style",
        "jacket": "Vintage Denim Trucker Jacket with Sherpa Lining",
        "dress": "Women's Elegant A-Line Evening Dress",
        "pants": "Slim Fit Chinos | Stretchable Fabric",
        "cap": "Sports Baseball Cap - Adjustable Strap"
    };

    const mockPrice = {
        "shirt": 499, "tshirt": 299, "jacket": 1299,
        "dress": 899, "pants": 699, "cap": 199
    };

    // Initial Search (uses V2 with visual attributes as soft preferences)
    useEffect(() => {
        if (item) {
            setLoading(true);
            // Default filters if none provided
            const initialFilters = llmFilters || {
                category: item.class,
                color: attributes?.color_name || attributes?.color_hex,
                pattern: attributes?.pattern
            };
            setCurrentFilters(initialFilters);

            // Build detected_attributes from AG-MAN detection
            const detectedAttrs = {
                category: item.class,
                color_name: attributes?.color_name || "",
                sleeve: attributes?.sleeve || "",
                pattern: attributes?.pattern || "",
                gender: llmFilters?.gender || ""
            };

            // Increment search generation — stale results will be ignored
            const gen = ++searchGenRef.current;
            console.log(`[ProductPage] Initial search gen=${gen} — PURE_SIMILARITY mode`, { detectedAttrs });

            axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                embedding: embedding,
                detected_attributes: detectedAttrs,
                user_filters: { category: item.class },
            })
                .then(res => {
                    // Only apply if this is still the latest search
                    if (searchGenRef.current !== gen) {
                        console.log(`[ProductPage] ❌ Ignoring stale initial search gen=${gen} (current=${searchGenRef.current})`);
                        return;
                    }
                    if (res.data.products && res.data.products.length > 0) {
                        setProducts(res.data.products);
                    }
                    if (res.data.metadata) {
                        console.log("[ProductPage] V2 Metadata:", res.data.metadata);
                    }
                    setLoading(false);
                })
                .catch(err => {
                    console.error("❌ Search failed:", err);
                    if (searchGenRef.current === gen) setLoading(false);
                });
        }
    }, [item]);

    // Handle User Query (Iterative Refinement)
    const [userQuery, setUserQuery] = useState("");
    const [queryLoading, setQueryLoading] = useState(false);

    const handleQuerySubmit = async (e) => {
        e.preventDefault();
        if (!userQuery.trim()) return;

        // ━━━ GUARD 1: Prevent double-firing ━━━
        if (queryInFlightRef.current) {
            console.log("[ProductPage] ⚠️ Query already in flight — skipping duplicate");
            return;
        }
        queryInFlightRef.current = true;

        // ━━━ GUARD 2: IMMEDIATELY invalidate ALL pending/stale searches ━━━
        // Increment BEFORE the LLM call so that any initial/stale search
        // completing after this point will be discarded.
        const gen = ++searchGenRef.current;
        console.log(`[ProductPage] 🔄 Query submitted gen=${gen}: "${userQuery}"`);

        setQueryLoading(true);
        try {
            const currentAttributesForLLM = {
                category: item.class,
                color_name: currentFilters.color || originalAttributes?.color_name || attributes?.color_name,
                color_hex: currentFilters.color_hex || originalAttributes?.color_hex || attributes?.color_hex,
                pattern: currentFilters.pattern || originalAttributes?.pattern || attributes?.pattern,
                sleeve_length: currentFilters.sleeve || originalAttributes?.sleeve_length || attributes?.sleeve,
                original_color: originalAttributes?.color_name,
                original_pattern: originalAttributes?.pattern
            };

            // 1. Call LLM for filter generation
            const llmRes = await axios.post('http://localhost:5000/llm', {
                user_query: userQuery,
                item: currentAttributesForLLM,
                scene: scene,
                session_history: sessionHistory || [],
            });

            const newFilters = llmRes.data.filters || {};
            const llmConfidence = llmRes.data.confidence || 0;
            const priceMax = llmRes.data.price_max || null;

            console.log(`[ProductPage] LLM response gen=${gen}:`, {
                filters: newFilters,
                confidence: llmConfidence,
                price_max: priceMax,
                source: llmRes.data.source
            });

            // ━━━ GUARD 3: Skip search if LLM returned FALLBACK / empty ━━━
            if (!newFilters || Object.keys(newFilters).length === 0 || llmConfidence < 0.1) {
                console.log(`[ProductPage] ⚠️ LLM FALLBACK (confidence=${llmConfidence}) — skipping search`);
                return;
            }

            // Check if we're still the active query after LLM wait
            if (searchGenRef.current !== gen) {
                console.log(`[ProductPage] ❌ Stale after LLM — gen=${gen} vs current=${searchGenRef.current}`);
                return;
            }

            // 2. Merge LLM filters with current state (LLM keys override)
            const mergedFilters = { ...currentFilters, ...newFilters };
            setCurrentFilters(mergedFilters);

            // 3. Build detected_attributes from original AG-MAN
            const detectedAttrs = {
                category: originalAttributes?.category || item.class,
                color_name: originalAttributes?.color_name || attributes?.color_name || "",
                sleeve: originalAttributes?.sleeve_length || attributes?.sleeve || "",
                pattern: originalAttributes?.pattern || attributes?.pattern || "",
                gender: llmFilters?.gender || ""
            };

            console.log(`[ProductPage] 🚀 Sending search gen=${gen}`, {
                detected_attributes: detectedAttrs,
                user_filters: mergedFilters,
            });

            // 4. Fire search with correct LLM-merged filters
            const searchRes = await axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                embedding: embedding,
                detected_attributes: detectedAttrs,
                user_filters: mergedFilters,
                price_max: priceMax,
            });

            // Check staleness after search completes
            if (searchGenRef.current !== gen) {
                console.log(`[ProductPage] ❌ Stale after search — gen=${gen} vs current=${searchGenRef.current}`);
                return;
            }

            if (searchRes.data.products && searchRes.data.products.length > 0) {
                setProducts(searchRes.data.products);
                console.log(`[ProductPage] ✅ Applied ${searchRes.data.products.length} results gen=${gen}`, {
                    metadata: searchRes.data.metadata,
                    retrieval: searchRes.data.retrieval_meta
                });
            } else {
                console.log(`[ProductPage] ⚠️ No products returned for gen=${gen}`);
            }
            setUserQuery("");
        } catch (err) {
            console.error("Query failed:", err);
            alert("Refinement failed. Try again.");
        } finally {
            queryInFlightRef.current = false;
            setQueryLoading(false);
        }
    };

    // Load Insights Logic
    const fetchInsights = async () => {
        try {
            const res = await axios.get('http://localhost:5000/insights');
            setInsightsData(res.data);
            setShowInsights(true);
        } catch (err) {
            console.error(err);
            alert("Failed to load insights");
        }
    };

    if (!item) return <div style={{ color: 'white', padding: 20 }}>No product selected.</div>;

    const mainProduct = products.length > 0 ? products[0] : null;
    const title = mainProduct ? mainProduct.name : (mockTitles[category] || `Premium ${item.class}`);
    const price = mainProduct ? mainProduct.price : (mockPrice[category] || 499);

    return (
        <div style={styles.container}>
            {/* Header */}
            <header style={styles.header}>
                <div style={styles.backBar} onClick={() => navigate(-1)}>‹ Back</div>

                {/* AI Query Bar */}
                <form onSubmit={handleQuerySubmit} style={styles.queryForm}>
                    <input
                        type="text"
                        value={userQuery}
                        onChange={(e) => setUserQuery(e.target.value)}
                        placeholder="Refine... (e.g. 'blue one')"
                        style={styles.queryInput}
                    />
                    <button type="submit" disabled={queryLoading} style={styles.queryBtn}>
                        {queryLoading ? 'Thinking...' : 'Ask AI'}
                    </button>
                </form>

                {/* Insights Button */}
                <button onClick={fetchInsights} style={styles.insightsBtn}>📊 Insights</button>
            </header>

            {/* Layout: No Body Scroll, Internal Scroll */}
            <div style={styles.contentArea}>

                {/* Visual Analysis (Left) */}
                <div style={styles.visualColumn}>
                    {/* Visual Match Image (Highlight Top Match) - NOW AT TOP & LARGE */}
                    {mainProduct && (
                        <div
                            style={styles.matchContainer}
                            onClick={() => setSelectedProduct(mainProduct)}
                            title="Click to view details"
                        >
                            <img
                                src={mainProduct.image_url}
                                alt="Match"
                                onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/400?text=No+Img"; }}
                                style={styles.matchImage}
                            />
                            <div style={styles.matchLabelOverlay}>BEST MATCH</div>
                        </div>
                    )}

                    {/* Detected Image - NOW AT BOTTOM & SMALLER */}
                    <div style={styles.imageContainer}>
                        <div style={styles.smallLabel}>Your Selection</div>
                        <img
                            src={`data:image/jpeg;base64,${item.cropped_image}`}
                            alt="Detected"
                            style={styles.detectedImage}
                        />
                    </div>
                </div>

                {/* Product Details (Right - Scrollable) */}
                <div style={styles.detailsColumn}>
                    <h1 style={styles.title}>{title}</h1>

                    {/* Rating */}
                    <div style={styles.ratingBlock}>
                        {[1, 2, 3, 4, 5].map(star => (
                            <span key={star} style={{ cursor: 'pointer', fontSize: '20px', color: '#ffa41c', marginRight: 2 }}
                                onClick={() => axios.post('http://localhost:5000/rating', {
                                    rating: star, product_id: mainProduct?.product_id, query: userQuery, filters: currentFilters
                                }).then(() => alert(`Rated ${star} ⭐`))}
                            >⭐</span>
                        ))}
                        <span style={{ fontSize: 13, color: '#007185', marginLeft: 8, cursor: 'pointer' }}>1,240 ratings</span>
                    </div>

                    <div style={styles.priceRow}>
                        <sup style={{ fontSize: '14px', top: '-0.5em' }}>₹</sup>
                        <span style={{ fontSize: '28px', fontWeight: '500' }}>{price}</span>
                    </div>

                    <div style={styles.desc}>
                        <p>• <b>Visual Match:</b> AI matched this item based on color, pattern, and style from your selection.</p>
                        <p>• <b>Category:</b> {category.charAt(0).toUpperCase() + category.slice(1)}</p>
                        {attributes?.color_name && <p>• <b>Detected Color:</b> {attributes.color_name}</p>}
                    </div>

                    {/* Similar Items */}
                    {products.length > 1 && (
                        <div style={styles.similarSection}>
                            <h3>Similar Options</h3>
                            <div style={styles.similarGrid}>
                                {products.slice(1).map(prod => (
                                    <div
                                        key={prod.id}
                                        style={styles.similarCard}
                                        onClick={() => setSelectedProduct(prod)}
                                    >
                                        <div style={styles.similarImgWrapper}>
                                            <img
                                                src={prod.image_url}
                                                style={styles.similarImage}
                                                onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/100?text=No+Img" }}
                                            />
                                        </div>
                                        <div style={styles.similarName}>{prod.name}</div>
                                        <div style={{ fontWeight: '700', color: '#B12704', fontSize: 13 }}>₹{prod.price}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* PRODUCT DETAIL MODAL */}
            {selectedProduct && (
                <div style={styles.modalOverlay} onClick={() => setSelectedProduct(null)}>
                    <div style={{ ...styles.modalContent, maxWidth: 600 }} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eee', paddingBottom: 10 }}>
                            <h2 style={{ margin: 0, fontSize: 18 }}>Product Details</h2>
                            <button onClick={() => setSelectedProduct(null)} style={{ border: 'none', background: 'none', fontSize: 20, cursor: 'pointer', color: '#555' }}>✕</button>
                        </div>
                        <div style={{ marginTop: 20, display: 'flex', gap: 30 }}>
                            <div style={{ flex: '0 0 240px', textAlign: 'center' }}>
                                <img
                                    src={selectedProduct.image_url}
                                    style={{ width: '100%', maxHeight: 320, objectFit: 'contain', borderRadius: 4 }}
                                    onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/240?text=No+Img" }}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <h3 style={{ margin: '0 0 10px', color: '#0F1111', fontSize: 20, lineHeight: 1.3 }}>{selectedProduct.name}</h3>
                                <div style={{ fontSize: 24, fontWeight: '500', color: '#B12704', marginBottom: 15 }}>₹{selectedProduct.price}</div>

                                <div style={{ fontSize: 14, lineHeight: 1.6, color: '#333' }}>
                                    {selectedProduct.brand && <div style={{ marginBottom: 5 }}><b>Brand:</b> {selectedProduct.brand}</div>}
                                    {selectedProduct.color && <div style={{ marginBottom: 5 }}><b>Color:</b> {selectedProduct.color}</div>}
                                    {selectedProduct.pattern && <div style={{ marginBottom: 5 }}><b>Pattern:</b> {selectedProduct.pattern}</div>}
                                    {selectedProduct.category && <div style={{ marginBottom: 5 }}><b>Category:</b> {selectedProduct.category}</div>}
                                    {selectedProduct.product_id && <div style={{ color: '#888', marginTop: 15, fontSize: 12 }}>ID: {selectedProduct.product_id}</div>}
                                </div>

                                {selectedProduct.final_score && (
                                    <div style={{ marginTop: 20, padding: 12, background: '#f0f2f2', borderRadius: 4, border: '1px solid #e7e7e7', fontSize: 13 }}>
                                        <b>Visual Match Score:</b> {(selectedProduct.final_score * 100).toFixed(0)}%
                                        <br />
                                        <span style={{ color: '#565959' }}>Based on deep learning embeddings</span>
                                    </div>
                                )}

                                <button
                                    onClick={() => {
                                        const url = selectedProduct.product_url || `https://www.myntra.com/${selectedProduct.product_id}`;
                                        window.open(url, '_blank');
                                    }}
                                    style={{ marginTop: 20, padding: '10px 20px', background: '#FF3F6C', border: 'none', borderRadius: 20, cursor: 'pointer', fontWeight: 'bold', width: '100%', color: 'white', fontSize: 15 }}
                                >
                                    Buy Now on Myntra
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* INSIGHTS MODAL OVERLAY */}
            {showInsights && (
                <div style={styles.modalOverlay} onClick={() => setShowInsights(false)}>
                    <div style={styles.modalContent} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eee', paddingBottom: 10 }}>
                            <h2 style={{ margin: 0 }}>System Evaluation</h2>
                            <button onClick={() => setShowInsights(false)} style={{ border: 'none', background: 'none', fontSize: 20, cursor: 'pointer' }}>✖</button>
                        </div>
                        {insightsData && (
                            <div style={{ marginTop: 20 }}>
                                <div style={styles.scoreRow}>
                                    <div style={styles.scoreCard}>
                                        <span style={{ fontSize: 12, color: '#666' }}>Avg Satisfaction</span>
                                        <div style={{ fontSize: 24, fontWeight: 'bold', color: '#007185' }}>{insightsData.stats.average_rating} ⭐</div>
                                    </div>
                                    <div style={styles.scoreCard}>
                                        <span style={{ fontSize: 12, color: '#666' }}>Relevance</span>
                                        <div style={{ fontSize: 18, fontWeight: 'bold', color: 'green' }}>
                                            {insightsData.analysis?.relevance_level || 'Calculating...'}
                                        </div>
                                    </div>
                                </div>
                                <div style={{ background: '#f9f9f9', padding: 15, borderRadius: 8, fontSize: 14 }}>
                                    <strong>🤖 LLM Analysis:</strong>
                                    <ul style={{ paddingLeft: 20, marginTop: 5 }}>
                                        {insightsData.analysis?.strengths?.map((s, i) => <li key={i}>{s}</li>)}
                                    </ul>
                                    {insightsData.analysis?.improvement_suggestion && (
                                        <div style={{ marginTop: 10, color: '#007185' }}>
                                            💡 <b>Suggestion:</b> {insightsData.analysis.improvement_suggestion}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        {!insightsData && <div>Loading analysis...</div>}
                    </div>
                </div>
            )}
        </div>
    );
};

const styles = {
    container: { height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column', background: 'white', fontFamily: '"Amazon Ember", Arial, sans-serif' },
    header: { height: '60px', background: '#131921', display: 'flex', alignItems: 'center', padding: '0 20px', gap: 20, color: 'white' },
    backBar: { color: 'white', cursor: 'pointer', fontWeight: 'bold', fontSize: 14, display: 'flex', alignItems: 'center' },
    queryForm: { flex: 1, display: 'flex', maxWidth: 800, margin: '0 20px' },
    queryInput: { flex: 1, padding: '10px 15px', borderRadius: '4px 0 0 4px', border: 'none', outline: 'none', fontSize: 15 },
    queryBtn: { padding: '0 25px', borderRadius: '0 4px 4px 0', border: 'none', background: '#febd69', cursor: 'pointer', fontWeight: 'bold', color: '#111' },
    insightsBtn: { padding: '8px 15px', borderRadius: 4, background: '#232f3e', color: 'white', cursor: 'pointer', border: '1px solid #555', fontSize: 13 },

    contentArea: { flex: 1, display: 'flex', overflow: 'hidden', maxWidth: '1400px', margin: '0 auto', width: '100%' },

    // Left Column: Visuals
    visualColumn: {
        flex: '0 0 450px',
        background: 'white',
        padding: '30px 20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        borderRight: '1px solid #f0f0f0',
        overflowY: 'auto'
    },

    // Main Product (Best Match) - Top + Large
    matchContainer: {
        position: 'relative',
        width: '100%',
        border: 'none',
        padding: 10,
        cursor: 'pointer',
        textAlign: 'center',
        marginBottom: 30
    },
    matchImage: {
        maxWidth: '100%',
        maxHeight: '450px',
        width: 'auto',
        height: 'auto',
        objectFit: 'contain',
        transition: 'transform 0.2s'
    },
    matchLabelOverlay: {
        position: 'absolute',
        top: 0,
        left: 0,
        background: '#CC0C39', // Amazon "Best Seller" red looks professional
        color: 'white',
        padding: '4px 10px',
        fontSize: 12,
        fontWeight: 'bold',
        borderRadius: 2
    },

    // Detected Image (User Selection) - Bottom + Small
    imageContainer: {
        marginTop: 20,
        textAlign: 'center',
        borderTop: '1px solid #eee',
        paddingTop: 15,
        width: '100%'
    },
    smallLabel: { fontSize: 12, color: '#565959', marginBottom: 5, textTransform: 'uppercase', letterSpacing: 0.5 },
    detectedImage: {
        height: '100px',
        maxWidth: '100px',
        objectFit: 'cover',
        border: '1px solid #ddd',
        borderRadius: 4,
        boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
    },

    // Right Column: Details
    detailsColumn: { flex: 1, padding: '30px 40px', overflowY: 'auto' },
    title: { fontSize: 24, lineHeight: 1.3, margin: '0 0 10px 0', color: '#0F1111', fontWeight: 500 },
    ratingBlock: { marginBottom: 15, display: 'flex', alignItems: 'center' },
    priceRow: { marginBottom: 20, color: '#B12704', lineHeight: 1 },
    desc: { fontSize: 14, lineHeight: 1.6, color: '#333', marginBottom: 30 },

    similarSection: { borderTop: '1px solid #eee', paddingTop: 30, marginTop: 20 },
    similarGrid: { display: 'flex', gap: 20, flexWrap: 'wrap' },

    similarCard: {
        width: 160,
        border: '1px solid #eee',
        borderRadius: 8,
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'box-shadow 0.2s',
        padding: '10px'
    },
    similarImgWrapper: { height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
    similarImage: { maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' },
    similarName: { fontSize: 13, height: 38, overflow: 'hidden', marginBottom: 5, lineHeight: 1.4, color: '#007185' },

    modalOverlay: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 },
    modalContent: { background: 'white', padding: 30, borderRadius: 8, width: 600, maxWidth: '90%', maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 10px 25px rgba(0,0,0,0.2)' },
    scoreRow: { display: 'flex', gap: 20, marginBottom: 20 },
    scoreCard: { flex: 1, border: '1px solid #eee', padding: 15, borderRadius: 6, textAlign: 'center', background: '#FAFAFA' }
};

export default ProductPage;
