
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const ProductPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { item, movie, llmFilters, scene, embedding, sessionHistory } = location.state || {};

    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentFilters, setCurrentFilters] = useState(llmFilters || {});

    // Insights Modal State
    const [showInsights, setShowInsights] = useState(false);
    const [insightsData, setInsightsData] = useState(null);

    // Dynamic mock details based on detected class
    const category = item ? item.class.toLowerCase() : "unknown";
    const attributes = item?.attributes;

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

    // Initial Search
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

            axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                filters: initialFilters,
                embedding: embedding,
                session_history: sessionHistory || []
            })
                .then(res => {
                    if (res.data.products && res.data.products.length > 0) {
                        setProducts(res.data.products);
                    }
                    setLoading(false);
                })
                .catch(err => {
                    console.error("❌ Search failed:", err);
                    setLoading(false);
                });
        }
    }, [item]);

    // Handle User Query (Iterative Refinement)
    const [userQuery, setUserQuery] = useState("");
    const [queryLoading, setQueryLoading] = useState(false);

    const handleQuerySubmit = async (e) => {
        e.preventDefault();
        if (!userQuery.trim()) return;

        setQueryLoading(true);
        try {
            // 1. Get refined filters from LLM
            const llmRes = await axios.post('http://localhost:5000/llm', {
                user_query: userQuery,
                visual_attributes: currentFilters, // Send CURRENT filters to refine
                scene: scene,
                session_history: sessionHistory || []
            });

            const newFilters = llmRes.data.filters;
            // CRITICAL FIX: Merge new filters with existing ones (additive behavior)
            // Unless LLM explicitly removes them, we assume refinement adds specificity
            const mergedFilters = { ...currentFilters, ...newFilters };
            setCurrentFilters(mergedFilters);

            console.log("LLM Refined Filters:", mergedFilters);

            // 2. Search with merged filters
            const searchRes = await axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                filters: mergedFilters,
                embedding: embedding,
                session_history: sessionHistory || []
            });

            if (searchRes.data.products && searchRes.data.products.length > 0) {
                setProducts(searchRes.data.products);
            }
            setUserQuery("");
        } catch (err) {
            console.error("Query failed:", err);
            alert("Refinement failed. Try again.");
        } finally {
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
                    <div style={styles.imageContainer}>
                        {/* REDUCED IMAGE SIZE */}
                        <img
                            src={`data:image/jpeg;base64,${item.cropped_image}`}
                            alt="Detected"
                            style={styles.detectedImage}
                        />
                        <div style={styles.tag}>IDENTIFIED IN VIDEO</div>
                    </div>

                    {/* Visual Match Image (Highlight Top Match) */}
                    {mainProduct && (
                        <div style={styles.matchContainer}>
                            <p style={styles.matchLabel}>BEST MATCH</p>
                            <img
                                src={mainProduct.image_url}
                                alt="Match"
                                onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/150?text=No+Img"; }}
                                style={styles.matchImage}
                            />
                        </div>
                    )}
                </div>

                {/* Product Details (Right - Scrollable) */}
                <div style={styles.detailsColumn}>
                    <h1 style={styles.title}>{title}</h1>

                    {/* Rating */}
                    <div style={styles.ratingBlock}>
                        {[1, 2, 3, 4, 5].map(star => (
                            <span key={star} style={{ cursor: 'pointer', fontSize: '24px', color: '#ffa41c' }}
                                onClick={() => axios.post('http://localhost:5000/rating', {
                                    rating: star, product_id: mainProduct?.product_id, query: userQuery, filters: currentFilters
                                }).then(() => alert(`Rated ${star} ⭐`))}
                            >⭐</span>
                        ))}
                    </div>

                    <div style={styles.priceRow}>
                        <sup style={{ fontSize: '14px' }}>₹</sup><span style={{ fontSize: '32px', fontWeight: '500' }}>{price}</span>
                    </div>

                    <div style={styles.desc}>
                        <p>• <b>Visual Filter:</b> {currentFilters.color ? `Color: ${currentFilters.color}` : ''} {currentFilters.pattern ? `, Pattern: ${currentFilters.pattern}` : ''}</p>
                        <p>• <b>Shop the Look:</b> System matched this items visual embeddings with our catalog.</p>
                    </div>

                    {/* Similar Items */}
                    {products.length > 1 && (
                        <div style={styles.similarSection}>
                            <h3>Similar Options</h3>
                            <div style={styles.similarGrid}>
                                {products.slice(1).map(prod => (
                                    <div key={prod.id} style={styles.similarCard}>
                                        <img
                                            src={prod.image_url}
                                            style={styles.similarImage}
                                            onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/100?text=No+Img" }}
                                        />
                                        <div style={styles.similarName}>{prod.name}</div>
                                        <div style={{ fontWeight: 'bold', color: '#B12704' }}>₹{prod.price}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

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
    container: { height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column', background: 'white', fontFamily: 'Arial' },
    header: { height: '60px', background: '#131921', display: 'flex', alignItems: 'center', padding: '0 20px', gap: 20 },
    backBar: { color: 'white', cursor: 'pointer', fontWeight: 'bold' },
    queryForm: { flex: 1, display: 'flex', maxWidth: 600 },
    queryInput: { flex: 1, padding: '8px 15px', borderRadius: '20px 0 0 20px', border: 'none', outline: 'none' },
    queryBtn: { padding: '0 20px', borderRadius: '0 20px 20px 0', border: 'none', background: '#febd69', cursor: 'pointer' },
    insightsBtn: { padding: '8px 15px', borderRadius: 4, background: '#232f3e', color: 'white', cursor: 'pointer', border: '1px solid #555' },

    contentArea: { flex: 1, display: 'flex', overflow: 'hidden' },

    visualColumn: { flex: '0 0 350px', background: '#f5f5f5', padding: 20, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', borderRight: '1px solid #ddd' },
    imageContainer: { position: 'relative', marginBottom: 20 },
    detectedImage: { maxHeight: '200px', maxWidth: '100%', objectFit: 'contain', border: '1px solid #ccc', borderRadius: 4 },
    tag: { position: 'absolute', bottom: -10, left: '50%', transform: 'translateX(-50%)', background: '#000', color: '#fff', fontSize: 10, padding: '2px 8px', borderRadius: 10 },

    matchContainer: { textAlign: 'center', border: '2px solid #007600', padding: 10, borderRadius: 8, background: 'white', width: '100%' },
    matchLabel: { margin: '0 0 10px 0', color: '#007600', fontWeight: 'bold', fontSize: 12 },
    matchImage: { width: 150, height: 150, objectFit: 'contain' },

    detailsColumn: { flex: 1, padding: '30px', overflowY: 'auto' }, // SCROLLABLE RIGHT SIDE
    title: { fontSize: 24, lineHeight: 1.2, margin: '0 0 10px 0' },
    ratingBlock: { marginBottom: 15 },
    priceRow: { marginBottom: 20, color: '#B12704' },
    desc: { fontSize: 14, lineHeight: 1.5, color: '#333', marginBottom: 30 },

    similarSection: { borderTop: '1px solid #eee', paddingTop: 20 },
    similarGrid: { display: 'flex', gap: 15, flexWrap: 'wrap' },
    similarCard: { width: 120, border: '1px solid #eee', padding: 8, borderRadius: 4, textAlign: 'center' },
    similarImage: { width: '100%', height: 100, objectFit: 'contain', marginBottom: 5 },
    similarName: { fontSize: 11, height: 28, overflow: 'hidden', marginBottom: 4 },

    modalOverlay: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 },
    modalContent: { background: 'white', padding: 25, borderRadius: 8, width: 500, maxWidth: '90%', maxHeight: '80vh', overflowY: 'auto', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' },
    scoreRow: { display: 'flex', gap: 20, marginBottom: 20 },
    scoreCard: { flex: 1, border: '1px solid #eee', padding: 10, borderRadius: 6, textAlign: 'center' }
};

export default ProductPage;
